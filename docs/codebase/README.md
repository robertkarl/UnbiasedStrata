Codebase docs

# UnbiasedStrata codebase documentation

This is the top of a hierarchical guide to the UnbiasedStrata source tree. The project builds **one binary**,
`unbiased-strata`, that serves two mixture-of-experts models over an OpenAI-style HTTP API from a single NVIDIA
card plus host RAM:

- **Qwen3.8-Flash-Next**: 125B parameters, IQ3_XXS, one 24 GB+ card and 128 GB RAM.
- **Qwen3.6-35B-A3B**: 35B parameters, 3B active, Q4_K_XL or IQ3_XXS, any card from 6 to 32 GB, Pascal or newer.

The binary has two halves. The **engine** is a fork of [Strata](https://github.com/Niko1221/Strata) 0.1.39. It
splits every token's work between experts cached in VRAM, experts pinned in RAM and experts read from the GGUF on a
miss. The **front end** is C++ built on llama.cpp's tokenizer, chat-template and HTTP code from a pinned submodule.

For what the project is, how to run it and how it measures, start with the repository [README](../../README.md).
For where it came from, see [UPSTREAM.md](../../UPSTREAM.md). These pages explain **how the code works**.

## How this documentation is organised

The pages form a tree with three levels:

| Level | What it is | Example |
| --- | --- | --- |
| 0 | This page: the whole system, the map of areas, reading paths | `docs/codebase/README.md` |
| 1 | One area's `README.md`: summary, contents, source map (every file with line count), concepts and data flow | [core/README.md](core/README.md) |
| 2 (and 3) | A component or a group of related files: types and functions with `path:line` citations, control flow, flags and environment variables, tests, known gotchas | [core/verify.md](core/verify.md) |

Every page starts with a breadcrumb back up the tree. Line numbers cite the tree at the commit the docs were
written against, so they may drift as the code changes; use them as a starting point and search from there.

## The system at a glance

```
             HTTP client (curl, coding agent, chat UI)
                              |  POST /v1/chat/completions, GET /v1/models, GET /health
                              v
 +-------------------------- unbiased-strata (front end, frontend/) -------------------------+
 |  cpp-httplib server -> Jinja chat template -> tokenize    (llama.cpp common, submodule)   |
 |  optional --mmproj: CPU image encoder (mtmd) -> embeddings file (SVE1)                    |
 |  common_chat_parse: reasoning / content / tool calls  <-  detokenize  <-  token ids       |
 +-----------------------|-----------------------------------------------^-------------------+
                         | stdin:  GEN / GENI / STOP / STATS / QUIT      | stdout: READY / T <id> / DONE / ERR
                         v   (same binary re-executed with --engine)     |
 +-------------------------- engine child (src/program/generate.cpp, --serve) -------------------+
 |  startup: GGUF geometry -> VRAM plan -> dense weights -> draft head -> expert source -> graphs |
 |  per request: resume a conversation checkpoint, prefill the new prompt tail, then decode:      |
 |                                                                                                |
 |   prefill (src/prefill)            decode (src/core)                 drafts (src/core, src/spec)|
 |   chunked batched forward:         CUDA graph per token / window:    MTP draft head +          |
 |   cuBLAS GEMM, GDN chunks,         attention or GDN -> router ->     prompt-lookup suffix      |
 |   prompt attention, MMQ / fused    experts -> combine; Verifier      drafter; DraftPolicy      |
 |   MoE, experts streamed to card    checks T<=8 drafted tokens        picks the window          |
 |                                                                                                |
 |   expert tiers (src/core/expert_*):  VRAM cache slots  >  pinned RAM  >  GGUF file on a miss   |
 |   GPU kernels (src/kernels/cuda) for hits and PCIe-fetched misses; CPU AVX2/AVX-512 kernels    |
 |   (src/kernels/cpu) on a worker pool for the rest, in parallel with the GPU                    |
 +------------------------------------------------------------------------------------------------+
                         ^
   offline, once:  tools/iq_pack.py (pack), tools/mtp_qwen36_pack.py (draft pack), tools/make_profile.py
                   (expert ranking); ref/qwen36.py is the numpy oracle the tests compare against
```

### The two models in one engine

The engine reads the geometry from GGUF metadata. Its `general.architecture` key selects the model. See
[core/model-geometry.md](core/model-geometry.md).

| | Qwen3.8-Flash-Next (`qwen4exp`) | Qwen3.6-35B-A3B (`qwen35moe`) |
| --- | --- | --- |
| Layers | 48 | 40 |
| Mixers | Gated DeltaNet on 3 of 4 layers; sparse **QSA** attention with an indexer on the rest | Gated DeltaNet (SiLU-gated) on 3 of 4 layers; **dense** attention on the rest |
| Residual | 4-stream hyper-connections | plain pre-norm residual |
| Extra tables | PLE / n-gram table (second shard, 27 to 51 GB, streamed from disk or RAM) | none |
| Experts | 512, top-10, plus a shared expert | 256, top-8, plus a shared expert |
| Draft head | MTP head built with upstream's tools | `blk.40` of the -MTP GGUF, cut by `tools/mtp_qwen36_pack.py` |
| Sizing | fixed requirements | header-only VRAM planner chooses context, expert slots and RAM tier |

### The life of a request

1. **Front end.** [frontend/http-api.md](frontend/http-api.md) renders the messages through the GGUF's chat
   template ([frontend/chat-and-parsing.md](frontend/chat-and-parsing.md)). It tokenizes them, clamps
   `max_tokens` to the context minus an 8-token reserve, and writes `GEN` to the engine. A request with images
   goes through the encoder in [frontend/vision.md](frontend/vision.md) and is sent as `GENI`.
2. **Engine serve loop.** [program/serve-mode.md](program/serve-mode.md) matches the prompt against the live
   session, a conversation checkpoint or a parked conversation ([program/prompt-cache.md](program/prompt-cache.md),
   [core/conversation/README.md](core/conversation/README.md)). Only the new tail is read.
3. **Prefill.** [prefill/pipeline.md](prefill/pipeline.md) reads the new tail in chunks of up to 8192 tokens. It
   borrows the top expert-cache slots and streams missing experts to the card through a copy ring
   ([prefill/expert-streaming.md](prefill/expert-streaming.md)). It leaves the GDN states, KV pools and the
   draft head's KV where token-by-token decode would have left them.
4. **Decode.** Each round, the [draft policy](libs/spec.md) chooses between an MTP draft
   ([core/mtp-draft-head.md](core/mtp-draft-head.md)) and a prompt-lookup draft. The
   [Verifier](core/verify.md) runs the window as one CUDA graph ([core/cuda-graphs.md](core/cuda-graphs.md)).
   Inside each layer ([core/layer-forward.md](core/layer-forward.md)), the router's choice is published through
   a pinned "doorbell". Experts resident in VRAM run on the GPU; the rest run on the GPU after a PCIe copy or on
   the CPU pool ([core/experts/dispatch.md](core/experts/dispatch.md),
   [kernels/cpu/pool.md](kernels/cpu/pool.md)). A second graph combines their outputs.
5. **Streaming back.** The engine prints `T <id>` per accepted token and then `DONE`. The front end detokenizes
   and parses reasoning, content and tool calls, and streams them as SSE. Between requests,
   [adaptive swaps](core/experts/resident-ram-and-adaptive.md) move hot experts into VRAM. `/health` reports VRAM
   hits, RAM reads and file reads.

## Areas

| Area | What it covers | Main source |
| --- | --- | --- |
| [Build](build/README.md) | Makefile, CMake options and targets, the llama.cpp submodule and its patch, test fixture scripts, HIP shims | `Makefile`, `CMakeLists.txt`, `cmake/`, `patches/`, `third_party/`, `include/strata/hip_compat/` |
| [Front end](frontend/README.md) | HTTP API, starting and watching the engine child, the line protocol, chat template and output parsing, images | `frontend/` |
| [Program](program/README.md) | The engine's `main`: all 151 flags, start-up, the one-shot and serve loops, speculative decoding, prompt cache, expert tiers, diagnostics, 47 `STRATA_*` variables | `src/program/`, `include/strata/program/` |
| [Core](core/README.md) | Model geometry, weights, session state and KV formats, the per-layer forward pass, CUDA graphs, the Verifier, the MTP draft head, the VRAM planner, device memory | `src/core/`, `include/strata/core/`, `include/strata/plan/vram_plan.hpp` |
| [Core › Experts](core/experts/README.md) | The pack, expert sources (pinned RAM, file), the VRAM expert cache, dispatch of hits and misses, adaptive swaps, multi-GPU tiers | `src/core/expert_*.cpp`, `remote_experts`, `peer_experts` |
| [Core › Conversation](core/conversation/README.md) | Prompt-cache checkpoints and snapshots of running state (KV, GDN, conv) | `src/core/conversation_*` |
| [Kernels](kernels/README.md) | All decode-time compute | `src/kernels/`, `include/strata/kernels/` |
| [Kernels › CUDA](kernels/cuda/README.md) | Attention (QSA and dense), GDN, MoE and expert GEMVs for every quantization, norms, RoPE, KV quantization, sampling | `src/kernels/cuda/` |
| [Kernels › CPU](kernels/cpu/README.md) | AVX2 and AVX-512 expert kernels, the expert layout table, the worker pool, the n-gram table | `src/kernels/cpu/`, `src/kernels/ngram.cpp` |
| [Kernels › Parity tests](kernels/parity/README.md) | Every parity, bench and model test, with its reference and fixtures | `src/kernels/*_parity.cpp`, `tests/cuda/` |
| [Prefill](prefill/README.md) | Batched prompt processing: GEMMs, fused and MMQ MoE, GDN chunks, prompt attention, expert streaming, the ggml bridge | `src/prefill/`, `include/strata/prefill/` |
| [Libraries](libs/README.md) | Speculation policy and suffix drafter, direct and robust file I/O, the older planner, the GGUF reader and dequantizers, the PLE reader | `src/spec/`, `src/platform/`, `src/plan/`, `src/artifact/`, `src/ngram/` |
| [Tools](tools/README.md) | Building packs, profiles, draft vocabularies and draft heads, the on-disk pack format, Qwen3.6 verification tooling, the numpy reference model | `tools/`, `ref/`, `tests/python/` |
| [Guides](guides/README.md) | Shipped data files and their byte formats, an index of every test, a reading guide to the prose notes, a glossary | `data/`, `tests/`, `docs/` |

### Where each directory of the repository is documented

| Directory | Documented in |
| --- | --- |
| `frontend/` | [frontend/](frontend/README.md) |
| `src/program/`, `include/strata/program/` | [program/](program/README.md) |
| `src/core/`, `include/strata/core/` | [core/](core/README.md), [core/experts/](core/experts/README.md), [core/conversation/](core/conversation/README.md) |
| `src/kernels/cuda/`, `include/strata/kernels/*.hpp` | [kernels/cuda/](kernels/cuda/README.md) |
| `src/kernels/cpu/`, `include/strata/kernels/cpu/`, `src/kernels/ngram.cpp` | [kernels/cpu/](kernels/cpu/README.md) |
| `src/kernels/*_parity.cpp`, `*_bench.cpp`, `*.inc`, `tests/cuda/`, `tests/platform/` | [kernels/parity/](kernels/parity/README.md) |
| `src/prefill/`, `include/strata/prefill/` | [prefill/](prefill/README.md) |
| `src/spec/`, `src/platform/`, `src/plan/`, `src/artifact/`, `src/ngram/` and their headers | [libs/](libs/README.md) |
| `include/strata/plan/vram_plan.hpp` | [core/vram-plan.md](core/vram-plan.md) |
| `include/strata/hip_compat/`, `include/strata/platform/hip_compat/`, `cmake/hip_backend.cmake` | [build/hip-compat.md](build/hip-compat.md) |
| `cmake/qwen36_*.cmake` | [build/test-fixtures-cmake.md](build/test-fixtures-cmake.md) |
| `CMakeLists.txt`, `Makefile`, `patches/`, `third_party/`, `.gitmodules` | [build/](build/README.md) |
| `tools/`, `ref/`, `tests/python/` | [tools/](tools/README.md) |
| `tests/core/` | [core/experts/](core/experts/README.md), [core/model-geometry.md](core/model-geometry.md), [core/vram-plan.md](core/vram-plan.md), indexed in [guides/testing.md](guides/testing.md) |
| `data/`, `tests/data/` | [guides/data-files.md](guides/data-files.md) |
| `docs/` (prose notes) | [guides/project-notes.md](guides/project-notes.md) |

## Reading paths

- **New to the project:** [guides/glossary.md](guides/glossary.md), then this page's diagram, then
  [core/README.md](core/README.md) and [program/README.md](program/README.md).
- **Running or operating the server:** [frontend/http-api.md](frontend/http-api.md),
  [frontend/engine-process.md](frontend/engine-process.md), [program/cli-flags.md](program/cli-flags.md),
  [program/env-vars.md](program/env-vars.md), [core/vram-plan.md](core/vram-plan.md).
- **Changing decode speed or correctness:** [core/layer-forward.md](core/layer-forward.md),
  [core/experts/dispatch.md](core/experts/dispatch.md), [kernels/cuda/README.md](kernels/cuda/README.md),
  [kernels/cpu/README.md](kernels/cpu/README.md), [program/speculative-decoding.md](program/speculative-decoding.md),
  then the matching test in [kernels/parity/README.md](kernels/parity/README.md).
- **Changing prompt processing:** [prefill/pipeline.md](prefill/pipeline.md),
  [prefill/expert-streaming.md](prefill/expert-streaming.md), [prefill/moe.md](prefill/moe.md).
- **Preparing model files:** [tools/pack-format.md](tools/pack-format.md),
  [tools/pack-builders.md](tools/pack-builders.md), [tools/mtp-tools.md](tools/mtp-tools.md),
  [guides/data-files.md](guides/data-files.md).
- **Adding a model or porting a layer type:** [guides/project-notes.md](guides/project-notes.md) (the Qwen3.6
  port, step by step), [core/model-geometry.md](core/model-geometry.md),
  [tools/reference-model.md](tools/reference-model.md), [tools/qwen36-verification.md](tools/qwen36-verification.md).
- **Building and testing:** [build/README.md](build/README.md), [build/cmake-options.md](build/cmake-options.md),
  [guides/testing.md](guides/testing.md).

## Things worth knowing before you read the code

- **Upstream code is still present.** The engine source still holds code for AMD (HIP), Intel, Windows,
  multi-GPU and models other than the two supported ones. `make build` does not build it and it is not tested
  ([build/README.md](build/README.md)). Comments often cite upstream docs that this fork does not carry, such as
  `docs/BATCHING.md`, `docs/semantics.md` and `ref/gdn.py`.
- **The `native_*` kernels are the default paths.** They are upstream's line-for-line ports of llama.cpp's
  kernels, not Qwen3.6-specific code. The front end always starts the engine with `--native <model>`, which
  turns them all on ([kernels/cuda/README.md](kernels/cuda/README.md)).
- **Tests are off by default.** `STRATA_BUILD_TESTS` is OFF because upstream's `tests/CMakeLists.txt` and
  `bench/micro/` were not imported. About 44 GPU test programs are still registered whenever CUDA is on. ctest
  in a `make build` tree therefore lists programs that were never built. See
  [build/test-targets.md](build/test-targets.md) and [guides/testing.md](guides/testing.md).
- **The prose notes and the code disagree in a few places.** Each area page records what its writer found. The
  main ones:
  - Content streams token by token since REGINA-110, not after the whole message as the README says
    ([frontend/http-api.md](frontend/http-api.md)).
  - `--kv-resident` (KV streaming) is refused for Qwen3.6's dense attention ([guides/data-files.md](guides/data-files.md),
    [core/session.md](core/session.md)).
  - `spec::Controller` is built and tested but never called ([libs/spec.md](libs/spec.md)).
  - `strata-plan` counts dense and embedding bytes twice ([libs/plan.md](libs/plan.md)).
  - The Makefile ignores its architecture filter when `CUDA_ARCHS` is given on the command line
    ([build/makefile.md](build/makefile.md)).
