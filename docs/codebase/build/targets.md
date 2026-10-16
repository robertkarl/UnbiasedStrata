[Codebase docs](../README.md) › [Build](README.md) › Targets

# Libraries and programs

## Purpose

Inventory of the non-test targets declared by [`CMakeLists.txt`](../../../CMakeLists.txt) and
[`cmake/hip_backend.cmake`](../../../cmake/hip_backend.cmake): their sources, link dependencies, the
condition under which they exist, and whether `make build` builds them. Test, parity and bench executables
are in [test-targets.md](test-targets.md).

"Default CPU configure" below means `cmake -S . -B build` with no options (no GPU backend). "`make build`"
means the Makefile's configure (CUDA, native experts, portable, K-quant MMQ, tests off) followed by
`--target unbiased-strata`, which builds only that target and its transitive dependencies.

## Libraries

| Target | Kind | Line | Sources | Links (PUBLIC unless noted) | Exists when | `make build` |
| --- | --- | --- | --- | --- | --- | --- |
| `strata_warnings` | INTERFACE | `:194` | none | compile options only (`-Wall -Wextra`, `/W4 /utf-8`, optional `-Werror`) | always | yes (via deps) |
| `strata_artifact` | INTERFACE (header-only) | `:214` | `include/` (GGUF reader, scalar dequantizers) | `strata_warnings` | always | yes |
| `strata_plan` | INTERFACE (header-only) | `:266` | `include/` (VRAM planner) | `strata_warnings` | always | yes |
| `strata_spec` | STATIC | `:1052` | `src/spec/suffix_drafter.cpp` (88), `controller.cpp` (73), `draft_policy.cpp` (101) | none (CPU only) | always | yes |
| `strata_kernels_cpu` | STATIC | `:1080` | `src/kernels/cpu/expert.cpp` (683, AVX-512), `pool.cpp` (783), `expert_layout.cpp` (349), `q2_avx2.cpp` (130, AVX2); with native experts also `native_expert.cpp` (171), `iq_avx512.cpp` (273), `iq_avx2.cpp` (493), `kq_avx2.cpp` (265), `kq_avx1.cpp` (71) (`:1170-1172`) | `Threads::Threads`; with native experts `ggml-cpu`, `ggml-base`; PRIVATE include `third_party/ggml` | always | yes |
| `strata_core` | STATIC | `:282` | `src/core/device.cu` (344), `pinned.cu` (725), `graph.cpp` (146), `weights.cpp` (479), `layout.cpp` (189), `src/platform/memory.cpp` (138) | `strata_plan`, `strata_warnings`, GPU runtime (PUBLIC because `graph.cpp` is host code calling `cudaGraphLaunch`, `:289-292`) | CUDA or HIP | yes |
| `strata_kernels` | STATIC | `:323-369` | 44 `src/kernels/cuda/*.cu` (dequant, GEMV, router, RoPE, sampler, GDN, QSA attention, KV formats, PLE, verify, i-quant kernels, the `native_*` llama.cpp adapters) plus `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp`; 21,868 lines in all | `strata_core`, `strata_artifact`; PRIVATE include `third_party/ggml` (`:396`) | CUDA or HIP | yes |
| `strata_engine` | STATIC | `:450-455` | `src/core/layer.cpp` (1517), `session.cpp` (967), `expert_source.cpp` (3046), `remote_experts.cpp` (341), `remote_expert_opt.cu` (194), `expert_cache.cpp` (696), `native_head.cpp` (198), `native_dense.cpp` (210), `verify.cpp` (2268), `mtp.cpp` (1372), `conversation_snapshot.cpp` (269), `conversation_state.cpp` (340), `conversation_memory.cpp` (51), `peer_experts.cpp` (325) | `strata_core`, `strata_kernels`, `strata_kernels_cpu`, GPU runtime | CUDA or HIP | yes |
| `strata_prefill` | STATIC | `:565-567` | `src/prefill/gemm.cu` (556), `kernels.cu` (1130), `prefill.cpp` (3359) | `strata_engine`, GPU BLAS, GPU runtime; `strata_mmq` (`:1243`, `:1293`); PRIVATE `roc::hipblaslt` when found on HIP (`:568-570`) | CUDA or HIP | yes |
| `strata_mmq` (CUDA) | STATIC | `:1218-1244` | `src/prefill/moe_mmq.cu` (268), `moe_fused.cu` (465), `moe_fused_iq.cu` (646), `ggml_cuda_host.cu` (147), and from the submodule `ggml/src/ggml-cuda/quantize.cu` plus `template-instances/mmq-instance-<t>.cu` for `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0` (+ `q5_0` with `STRATA_ORCA_Q4KS_MMQ`, + `q4_k q5_k q5_1 q6_k` with `STRATA_MMQ_KQUANTS`) | `ggml-base`, GPU runtime; PRIVATE includes `ggml/include`, `ggml/src`, `ggml/src/ggml-cuda` of the submodule | CUDA + native experts | yes (with the K-quant instances) |
| `strata_mmq` (HIP) | STATIC | `:1276-1294` | `moe_mmq.cu`, `ggml_cuda_host.cu`, `quantize.cu` and the instances above (no fused kernels, no K-quants), all compiled as HIP with `GGML_USE_HIP=1` | `ggml-base`, `hip::host`, `roc::hipblas` | HIP + `STRATA_PREFILL_MMQ` + native experts | no |
| `strata_hip_runtime` | INTERFACE | `hip_backend.cmake:73-94` | none | `hip::host`; include `include/strata/hip_compat` BEFORE; force-includes its `cuda_runtime.h` | HIP | no |
| `CUDA::cudart`, `CUDA::cublas` (gfx906 aliases) | IMPORTED INTERFACE | `:113-118` | none | `hip::host` / `roc::hipblas`; include `include/strata/platform/hip_compat` | gfx906 | no |
| `ggml`, `ggml-base`, `ggml-cpu` | STATIC (from submodule) | `:1167` | `third_party/llama.cpp/ggml`, binary dir `build/ggml`, `EXCLUDE_FROM_ALL` | | native experts | yes (needed by `strata_kernels_cpu`, `strata_mmq`, llama) |
| `llama`, `llama-common`, `mtmd`, `cpp-httplib` | from submodule | `:1438` | `third_party/llama.cpp`, binary dir `build/llama.cpp`, `EXCLUDE_FROM_ALL` | | CUDA + native experts | yes |

"GPU runtime" is `_strata_gpu_runtime_target`: `CUDA::cudart` (CUDA and gfx906) or `strata_hip_runtime` (HIP).
"GPU BLAS" is `CUDA::cublas` or `roc::hipblas` (+ `roc::rocblas` for hipBLAS < 1.0).

## Programs

| Target | Line | Sources | Links | Exists when | `make build` | Role |
| --- | --- | --- | --- | --- | --- | --- |
| `unbiased-strata` | `:1440-1443` | `frontend/main.cpp`, `frontend/engine_entry.cpp` | `strata_engine`, `strata_prefill`, `strata_spec`, GPU runtime, `llama-common`, `llama`, `mtmd`, `cpp-httplib`, `Threads::Threads`; include `${STRATA_GGML_DIR}/vendor` | `STRATA_ENABLE_CUDA AND STRATA_NATIVE_EXPERTS` | **the target** | The fork's single binary: HTTP front end and, with `--engine`, the engine child |
| `strata` | `:576-578` | `src/program/generate.cpp` (8843) | `strata_engine`, `strata_prefill`, `strata_spec`, GPU runtime | CUDA or HIP | no | Upstream engine program (`strata generate`); also run by the `qwen36_runner_check` test |
| `strata-gguf` | `:218` | `src/artifact/gguf_reader.cpp` (67) | `strata_artifact` | always | no | GGUF inspection CLI |
| `strata-dequant` | `:221-224` | `src/artifact/dequant.cpp` (123), `STRATA_DEQUANT_SELFTEST` | `strata_artifact` | always | no | Dequantizer CLI |
| `strata-plan` | `:270` | `src/plan/plan_main.cpp` (73) | `strata_plan` | always | no | VRAM plan CLI |
| `strata-device` | `:294` | `src/core/device_main.cpp` (122) | `strata_core` | CUDA or HIP | no | Device check; `--selftest` is the `cuda_device_selftest` / `hip_device_selftest` test |
| `strata-load` | `:297` | `src/core/load_main.cpp` (122) | `strata_core`, `strata_kernels` | CUDA or HIP | no | Expert arena load; the `expert_arena_load` test if `pack/full/experts.bin` exists |
| `strata-concurrent` | `:300-302` | `src/core/concurrent_main.cpp` (197) | `strata_core`, `strata_kernels`, runtime | CUDA or HIP | no | |
| `strata-overlap` | `:304-307` | `src/core/overlap_main.cpp` (169) | `strata_core`, `strata_kernels`, runtime | CUDA or HIP | no | |
| `tune_hipblaslt` | `:572-573` | `tools/hip/tune_hipblaslt.cpp` - **absent from this fork** | `hip::host`, `roc::hipblas`, `roc::hipblaslt` | HIP and hipBLASLt found | no | |

## Link graph

```
unbiased-strata ─┬─ strata_prefill ─┬─ strata_engine ─┬─ strata_core ──── strata_plan, strata_warnings, cudart
                 │                  │                 ├─ strata_kernels ─ strata_core, strata_artifact
                 │                  │                 └─ strata_kernels_cpu ─ Threads, ggml-cpu, ggml-base
                 │                  ├─ cublas, cudart
                 │                  └─ strata_mmq ─── ggml-base, cudart
                 ├─ strata_engine
                 ├─ strata_spec
                 └─ llama-common, llama, mtmd, cpp-httplib, Threads
strata (upstream program) = same minus the llama.cpp libraries and the front end
```

`strata-load` links `strata_kernels` (`:298`) although `strata_kernels` is declared later (`:323`); CMake
resolves target names at generate time, so the order does not matter.

## How the front end and the engine share one executable

[`frontend/engine_entry.cpp`](../../../frontend/engine_entry.cpp) is six lines:

```
#define main strata_engine_main
#include "../src/program/generate.cpp"
```

so the upstream engine program is compiled unmodified into `unbiased-strata` under another name, and
`frontend/main.cpp:651` dispatches to it when the first argument is `--engine`. In a full build
`generate.cpp` is therefore compiled twice (in `strata` and in `unbiased-strata`). UPSTREAM.md describes the
same arrangement ("The engine's own program is compiled into the binary with its entry point renamed; the
front end starts it as a child process").

## What each configuration builds with the default `all` target

| Configuration | Libraries and programs in `all` |
| --- | --- |
| Default CPU configure, `-DSTRATA_NATIVE_EXPERTS=OFF` (verified) | `strata-gguf`, `strata-dequant`, `strata-plan`, `strata_spec`, `strata_kernels_cpu` |
| Default CPU configure with the submodule | the above plus native CPU kernels, ggml as needed, and `router_dot_parity` |
| CUDA (`make build` options) | every library above, `strata`, `strata-*`, `unbiased-strata`, and 44 test, parity and bench executables that are not gated by `STRATA_BUILD_TESTS` ([test-targets.md](test-targets.md#gpu-parity-programs-not-gated-by-strata_build_tests)) |

`make build` avoids the extra programs by building `--target unbiased-strata` only.

## Gotchas

- `unbiased-strata` requires `STRATA_ENABLE_CUDA`; neither AMD backend can produce it. Under
  `STRATA_HIP_GFX906` `STRATA_ENABLE_CUDA` is forced ON, so the target would be declared, but configure
  stops earlier on the missing `tests/hip/router_fast.cpp` (see [test-targets.md](test-targets.md)).
- ggml is added once from the submodule (`:1167`) and llama.cpp's own CMake is then added from the same
  checkout (`:1438`). The comment at `:1430` says llama.cpp's build reuses the ggml configured above; the
  mechanism is in llama.cpp's CMake files (submodule, not checked out here), so it was not verified from
  source.
- The vendored `third_party/ggml/ggml-common.h` is used by `strata_kernels` and `strata_kernels_cpu`;
  `strata_mmq` takes `ggml-common.h` from the submodule's `ggml/src` instead (its include path,
  `:1230`). See [third-party-and-patches.md](third-party-and-patches.md).

## See also

- [cmake-options.md](cmake-options.md), [test-targets.md](test-targets.md), [makefile.md](makefile.md)
- [Front end](../frontend/README.md) (what `frontend/main.cpp` does with the two roles)
- [Program](../program/README.md) (`src/program/generate.cpp`), [Prefill](../prefill/README.md) (`strata_prefill`, `strata_mmq`)
- [UPSTREAM.md](../../../UPSTREAM.md)
