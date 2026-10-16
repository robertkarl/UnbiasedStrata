[Codebase docs](../README.md) › Kernels

# Kernels

This area holds the arithmetic of the engine: every GPU kernel, the CPU expert kernels and the worker pool that runs
them, the host-side PLE n-gram hash and table reader, and the parity tests that pin all of them to llama.cpp / ggml or
to the numpy oracle. It sits below the model code: the layer, session, verify-window and MTP code in `src/core/`, the
prompt path in `src/prefill/`, and the run-time switches and startup in `src/program/generate.cpp` call these
functions; the kernels themselves call only the CUDA (or HIP) runtime, ggml's block definitions and ggml-cpu's dot
products, and a few platform helpers. It builds into two static libraries: `strata_kernels` (the GPU kernels plus the
host-only `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp` and `src/platform/direct_file.cpp`; only with a GPU
backend) and `strata_kernels_cpu` (the CPU expert kernels and the pool; always built). Both models are served by the
same kernels: Qwen3.8-Flash-Next (2560 wide, hyper-connections, PLE, 512 experts top-10) and Qwen3.6-35B-A3B
(2048 wide, plain block, no PLE, 256 experts top-8), with per-model behaviour selected by runtime geometry or
template parameters.

## Contents

| Page | Covers |
|---|---|
| [cuda/README.md](cuda/README.md) | The 44 CUDA files in `src/kernels/cuda/` and their headers in `include/strata/kernels/`: attention / QSA and KV formats, Gated DeltaNet, MoE routing and GPU expert tiers, GEMVs and quantizers, norms / RoPE / hyper-connections / PLE block, sampling and verify helpers (written separately; summarised below) |
| [cpu/README.md](cpu/README.md) | The CPU expert path: Q2_0 VNNI kernel, native ggml-cpu experts with multi-token AVX-512 / AVX2 kernels, `native_experts.txt` layout and ISA probes, `ExpertPool`, CPU tests, and the PLE n-gram table (`cpu/ngram.md`) |
| [parity/README.md](parity/README.md) | Every parity / bench / unit test at the top level of `src/kernels/` and the model-level harnesses in `tests/cuda/` (plus `tests/platform/direct_file_async.cpp`): what each checks, its reference, CMake target and condition, and fixtures |

## Source map

Per-file maps are on the child pages; this table groups the files.

| Path | Files / lines | Role | Detailed in |
|---|---|---|---|
| [src/kernels/cuda/](../../../src/kernels/cuda/) | 44 `.cu`, 20,435 | GPU kernels, all in `strata_kernels` | [cuda/README.md](cuda/README.md) |
| [include/strata/kernels/](../../../include/strata/kernels/) `*.hpp` (except `ngram.hpp`) | 47, 3,313 | Host-callable declarations of the GPU kernels, plus shared helpers (`bf16_bits.hpp`, `f16_bits.hpp`, `dp4a.hpp`, `q8_1_finite.hpp`, `rope_scaling.hpp`) | [cuda/README.md](cuda/README.md) |
| [src/kernels/cpu/](../../../src/kernels/cpu/) | 18, 4,453 | CPU expert kernels, pool, affinity helpers, CPU tests | [cpu/README.md](cpu/README.md) |
| [include/strata/kernels/cpu/](../../../include/strata/kernels/cpu/) | 8, 688 | CPU kernel and pool headers | [cpu/README.md](cpu/README.md) |
| [src/kernels/ngram.cpp](../../../src/kernels/ngram.cpp) | 1, 521 | PLE n-gram hash and `PleTable` (built into `strata_kernels`) | [cpu/ngram.md](cpu/ngram.md) |
| [include/strata/kernels/ngram.hpp](../../../include/strata/kernels/ngram.hpp) | 1, 216 | Its header | [cpu/ngram.md](cpu/ngram.md) |
| [src/kernels/](../../../src/kernels/) `*_parity.cpp`, `*_test.cpp`, `*_bench.cpp`, `*.inc` | 45, 16,390 | Kernel parity tests, benches and generated oracle vectors | [parity/README.md](parity/README.md) |
| [tests/cuda/](../../../tests/cuda/), [tests/platform/direct_file_async.cpp](../../../tests/platform/direct_file_async.cpp) | 8, 3,478 | Model-level CUDA harnesses, async file I/O test | [parity/model-tests.md](parity/model-tests.md) |

## Key concepts

### Where each kernel family runs in one decode step

```
token id ──> embedding (GPU) ──> per layer l:
   GPU  pre[l]   hyper-connection read (gr / fused_gr; plain norm on Qwen3.6)
                 mixer: Gated DeltaNet (gdn, fused_gdn, native_gdn*)  or  attention (qsa*, native_qsa*, kv_*)
                 [layer 1, Flash-Next] PLE block (ple.cu) on 16 rows from PleTable  <── host: ngram_rows + PleTable
                 router (router_top10 / native_router), shared expert (shared_expert / native MMVQ)
                 doorbell ring ───────────────────────────────────────────────┐
   GPU  hit experts from the VRAM tier (s2_expert_grouped / native_expert_grouped)
   CPU                                              ExpertPool: missed experts ┘ (cpu/)
   GPU  post[l]  moe_combine (router weights) + residual write (gr_write)
──> head, sampler (sampler.cu) ──> next token
```

The verify window of speculative decoding runs the same families on up to 8 tokens at once (multi-column MMVQ,
`verify_kernels.cu`, multi-token CPU experts), and the prompt path (`src/prefill/`) uses GEMMs plus the prompt-side
attention and expert kernels. The CPU / GPU overlap is described in [cpu/pool.md](cpu/pool.md).

### Correctness contract

Kernels are written to produce **the same numbers as llama.cpp's backends** (the `native_*` CUDA adapters and the
ggml-cpu expert path transcribe the pinned llama.cpp commit), and every multi-token or rewritten kernel is required to
be **bitwise** equal to its single-token or previous form, because the speculative verify window must reproduce plain
greedy decode exactly. The parity tests enforce both, each with negative controls; see
[parity/README.md](parity/README.md).

### Build

| Library | Sources | Condition |
|---|---|---|
| `strata_kernels` | `src/kernels/cuda/*.cu` (44), `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp` | `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP` (or `STRATA_HIP_GFX906`); links `strata_core strata_artifact` |
| `strata_kernels_cpu` | `src/kernels/cpu/{expert,pool,expert_layout,q2_avx2}.cpp` (+ `native_expert, iq_avx512, iq_avx2, kq_avx2, kq_avx1` with `STRATA_NATIVE_EXPERTS`) | Always; ISA flags per source file; links `ggml-cpu ggml-base` with native experts |

`strata_engine` links both. ggml (and ggml-cpu) is built from the `third_party/llama.cpp` submodule; configure fails if
it is not checked out while `STRATA_NATIVE_EXPERTS=ON` (the default).

## See also

- [../core/README.md](../core/README.md) (the callers: layer forward, session, experts dispatch),
  [../prefill/README.md](../prefill/README.md), [../build/README.md](../build/README.md)
- [../libs/ngram-ple.md](../libs/ngram-ple.md) (`PleReader`), [../libs/platform.md](../libs/platform.md) (`DirectFile`)
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) ("How it works"),
  [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md), [docs/qwen36-port/](../../qwen36-port/)
