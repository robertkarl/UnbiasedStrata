# Tencent Hy4-Flash text inference

`unbiased-strata` can launch a native `strata-hy4` child on NVIDIA CUDA or Apple Silicon Metal. Both read the same checkpoint. The child executes the graph and the front end loads the embedded BPE vocabulary, special-token IDs, merges and Jinja template into its vocabulary-only llama.cpp model. This path serves text, including streaming chat and parsed tool calls.

## Build and serve

From the repository root, initialize the pinned llama.cpp submodule. Install libzstd development files (`libzstd-dev` on Debian or `zstd` in Homebrew). Linux also needs the CUDA toolkit and a supported host C++ compiler. The Hy4 kernels target compute capability 7.0 or newer. Set `CUDA_ARCHS` to the card you plan to run on. An RTX 3090, 3090 Ti or A6000 uses 86.

```sh
git submodule update --init
make hy4 CUDA_ARCHS=86    # Linux
# make hy4                # Apple Silicon
```

Choose one text pack and put the shared `Hy4-Flash-Engram-Q8.strata` next to it on fast local storage. The SQ-64, SQ-96 and SQ-128 text files occupy 62.03, 89.58 and 113.65 GB respectively. The shared lookup table adds 67.58 GB. Strata lists an 8 GiB GPU paired with 64, 96 or 128 GiB of system RAM for the corresponding tier. The runner reads the required rows as it generates. All three published manifests specify the same tokenizer, chat template, text configuration and Engram hash. [Checkpoint files and checksums](https://huggingface.co/strata-engine/Hy4-Flash-Strata/tree/995bac227ab7e0d221772be641e2b5954a9fac06).

```sh
build-frontend/bin/unbiased-strata --engine build-hy4/bin/strata-hy4 \
    -m /models/hy4/Hy4-Flash-SQ-64.strata --jinja --alias hy4 --port 8080 \
    -- --ctx 4096 --chunk 128 --draft auto

curl -sS http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"hy4","messages":[{"role":"user","content":"What is 2 plus 2?"}],"reasoning_effort":"none","temperature":0,"max_tokens":32}'
```

Use `--engram /path/to/Hy4-Flash-Engram-Q8.strata` after the separator if the table is elsewhere. `--draft auto` enables the embedded DFly2 drafter when it can initialize. `--draft off` selects plain decode. The optional vision shard and HTTP image input require a different engine protocol from the fork’s existing image projector adapter.

## Integration choice

| Approach | What the source comparison showed |
| --- | --- |
| Merge PolyStrata | The fork imported the tree of `d0cd84dc` in `0197ae86`, so Git cannot infer their common base. A three-way merge with that explicit base against PolyStrata `3e164216` reported 67 conflicted paths. PolyStrata had advanced 354 commits and changed 867 files since the import. Its main branch contained no Hy4 runner. [PolyStrata source](https://github.com/simplexai-labs/PolyStrata/tree/3e1642161) |
| Merge or cherry-pick Niko’s commits | The first GPU runner commit applies as nine added files, but it targets HIP. Later commits supply the line protocol, CUDA, expert tiering, Metal, and DFly2. 134 non-merge commits touch the family directory. An explicit-base merge from the last common Strata/PolyStrata ancestor (`477ac13c`, rename limit 1,000) reported 2,358 conflicted paths. A full tree diff since the fork’s previous Strata import (`96ca4383`) spans 8,540 files, including Strata’s relocation into `engine/`. [First runner commit](https://github.com/Niko1221/Strata/commit/423a0960f) |
| Port individual operations | A quantized matrix multiply covers one part of the graph. The model also needs four Engram injections, sparse index selection and sliding-window MLA, hyper-connections, routed expert placement, and the matching cache state during decode. llama.cpp PR 32517 implements a GGUF graph with converter and KV-memory changes. Its patch at `fc0b8338` touched 32 files and failed a clean apply on 23 paths at this fork’s llama.cpp pin. Its GGUF loader would also need a separate solution for Strata’s native packs. [llama.cpp implementation](https://github.com/ggml-org/llama.cpp/pull/32517) |
| Import the finished runner | The upstream runner already accepts token IDs and emits `READY`, `PP`, `T` and `DONE` over the fork’s existing child protocol. The import is 59 files and 32,242 lines, including the license. It preserves the checkpoint reader, graph, quantization arithmetic and scheduling. The fork supplies its build targets, vocabulary adapter and C++ HTTP server. [Pinned Strata runner](https://github.com/Niko1221/Strata/tree/6e38f588a646c0ee6948a74afbe670eb75896106/engine/src/hy4flash) |

The imported files keep their paths relative to Niko’s `engine/` except for `metal/src/{mtl_rt.h,mtl_rt.mm,resource_path.h}`, which live beside the Hy4 Metal runner here. The CMake build links system libzstd and includes the identical `stb_image.h` from the fork’s llama.cpp submodule. The C++ front end and standalone build are fork-specific. The upstream license accompanies the runner in `src/hy4flash/LICENSE`.

## Reproduce the comparison

On Apple Silicon, build the oracle from Niko’s checkout at the exact commit. Its build script expects the checkout at `$HY4_DIR/wt`.

```sh
oracle_dir=$(mktemp -d)
git clone https://github.com/Niko1221/Strata "$oracle_dir/wt"
git -C "$oracle_dir/wt" checkout --detach 6e38f588a646c0ee6948a74afbe670eb75896106
HY4_DIR="$oracle_dir" bash "$oracle_dir/wt/engine/src/hy4flash/metal/build_mac.sh" run
```

The comparison script requires Python, NumPy and Hugging Face Tokenizers. Supply a fresh output directory for each run. Raw logits, engine logs, greedy token IDs and timings are saved there.

```sh
STRATA_METAL_EXP_CACHE_GB=12 python3 bench/correctness/hy4_oracle.py \
    --oracle "$oracle_dir/bin/strata-hy4-metal" --candidate build-hy4/bin/strata-hy4 \
    --frontend build-frontend/bin/unbiased-strata \
    --model /models/hy4/Hy4-Flash-SQ-64.strata --engram /models/hy4/Hy4-Flash-Engram-Q8.strata \
    --draft both --long --out /tmp/hy4-results -- --cold-experts
```

The short inputs use opposite ends of the repository’s multilingual prompt corpus. The optional long input crosses the 2,048-entry sparse-index threshold. Both engines export every logit in the final 128-token chunk. Its first 127 positions have an observed next token for perplexity and accuracy. The timing test runs the same 128-token prefill and 32-token greedy decode with prompt reuse disabled, discards one warm-up and reports the median of three trials. `--frontend` also compares the streamed HTTP token IDs and measures time to first token and throughput between the first and last token.

## SQ-64 results on Apple Silicon

The independently built Strata oracle and UnbiasedStrata ran on the same 60-core M3 Ultra with 96 GB of unified memory. Both used the SHA-256-verified SQ-64 and Engram-Q8 files, a 4,096-token context, 128-token chunks and a 12 GB expert cache with on-demand reads. All three 128 × 120,832 float32 logit dumps were byte-identical. The long input issued 139,264 Engram lookups.

| Full-vocabulary metric | Strata | UnbiasedStrata |
| --- | ---: | ---: |
| Perplexity on 381 next tokens | 8.40109 | 8.40109 |
| Observed next-token accuracy | 230/381 (60.37%) | 230/381 (60.37%) |
| Top-1 agreement with oracle | — | 381/381 |
| Mean KL from oracle, nats | — | 0 |

The rates below are median tokens per second. Native rows use engine timings. HTTP rows compare elapsed time observed by the HTTP client with elapsed time observed by a client of Strata’s native protocol.

| Mode and measurement | Strata prompt | UnbiasedStrata prompt | Strata decode | UnbiasedStrata decode |
| --- | ---: | ---: | ---: | ---: |
| Plain, native runner | 38.42 | 39.39 | 11.35 | 11.02 |
| Plain, streamed HTTP | 38.41 | 40.12 | 11.00 | 10.96 |
| DFly2, native runner | 42.35 | 41.72 | 15.08 | 15.00 |
| DFly2, streamed HTTP | 42.34 | 36.98 | 14.61 | 15.07 |

All measured runs returned the same 32 greedy token IDs across engines, HTTP and draft settings. Both HTTP chat requests answered `4` and both servers exited cleanly after the streamed requests. The largest throughput difference was 12.7% on DFly2 HTTP prompt processing. These figures cover SQ-64 on Metal. The SQ-96 and SQ-128 checks covered their manifests. Available CUDA cards were serving other models, so the CUDA checks below cover kernels and builds.

### Backend checks

CUDA 12.4 with GCC 13 built the runner for `sm_75`, `sm_86` and `sm_89`. On an RTX 3090 Ti, upstream’s `sk_test --random-only` passed 74 matrix and embedding checks against CPU dequantization. Upstream’s `mla_test` passed 31 attention checks, including the real SQ-64 layer 5 sparse and layer 2 sliding-window projection, norm and sink tensors. Both tests linked this fork’s imported sources. Only the required tensors were transferred to the CUDA host.

The M3 Ultra passed the upstream attention and routing test and all 24 matrix checks on eight real SQ-64 trunk tensors. `make test` passed the existing adapter, literal handling, picture cache and HTTP regression tests. A controlled Hy4 tool-call reply also produced separate reasoning and a parsed OpenAI function call with the expected argument.
