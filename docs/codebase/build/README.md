[Codebase docs](../README.md) › Build

# Build system and repository infrastructure

The build turns the engine sources (`src/`, `include/`), the C++ front end (`frontend/`) and the pinned
llama.cpp submodule (`third_party/llama.cpp`) into one Linux/CUDA binary, `build/unbiased-strata`. The entry
point for users is the top-level `Makefile` (`make build`), which checks the submodule, applies the one
llama.cpp patch, and runs CMake with a fixed set of options for the single target `unbiased-strata`. The
top-level `CMakeLists.txt` (1444 lines, inherited almost whole from upstream Strata 0.1.39) has 127
`add_library` / `add_executable` calls: the engine libraries (`strata_core`, `strata_kernels`, `strata_kernels_cpu`,
`strata_engine`, `strata_prefill`, `strata_mmq`, `strata_spec`), the upstream engine program `strata`, a
long list of parity tests and benchmarks, and the fork's addition, `unbiased-strata`, which links the engine
libraries with llama.cpp's `llama`, `llama-common`, `mtmd` and `cpp-httplib`. ggml (CPU backend, plus the
CUDA MMQ sources) and llama.cpp are taken from the submodule; the build downloads nothing. Two AMD paths
(wave32 RDNA through `cmake/hip_backend.cmake`, wave64 gfx906 through a CUDA-name shim) are still in the
CMake code but are not built by `make build`, and in this fork they cannot configure because their test
sources were not imported (see [test-targets.md](test-targets.md)).

## Contents

| Page | What it covers |
| --- | --- |
| [makefile.md](makefile.md) | `make build`, `clean`, `test-frontend`, `test-chat-parser`, `test-serve`; the CUDA arch list and how it is filtered |
| [cmake-options.md](cmake-options.md) | Every `option()` and cache variable, its default, its effect, the mutual exclusions and the compile definitions they produce |
| [targets.md](targets.md) | The libraries and user-facing programs: sources, link graph, how `unbiased-strata` is assembled, what `make build` builds |
| [test-targets.md](test-targets.md) | Every test, parity and bench executable and every CTest registration, with its gate; which ones reference files absent from this fork |
| [test-fixtures-cmake.md](test-fixtures-cmake.md) | The `cmake -P` fixture scripts (`cmake/qwen36_*.cmake`), the Python/numpy probes, and the CTest fixture chains |
| [third-party-and-patches.md](third-party-and-patches.md) | The llama.cpp submodule, the vendored `ggml-common.h`, the chat-parser patch, `.gitmodules` / `.gitattributes` / `.gitignore` |
| [hip-compat.md](hip-compat.md) | `cmake/hip_backend.cmake` and the two CUDA-to-HIP header shims (`include/strata/hip_compat/`, `include/strata/platform/hip_compat/`) |

## Source map

| File | Lines | Role |
| --- | ---: | --- |
| [CMakeLists.txt](../../../CMakeLists.txt) | 1444 | The whole CMake build: options, backends, every library, program, test and the `unbiased-strata` target |
| [Makefile](../../../Makefile) | 68 | User entry point: `build`, `clean`, `test-frontend`, `test-chat-parser`, `test-serve` |
| [cmake/hip_backend.cmake](../../../cmake/hip_backend.cmake) | 110 | `STRATA_ENABLE_HIP` setup: arch validation, ROCm packages, `strata_hip_runtime`, `.cu` as HIP |
| [cmake/qwen36_synth_fixtures.cmake](../../../cmake/qwen36_synth_fixtures.cmake) | 27 | CTest fixture: four synthetic qwen35moe GGUFs |
| [cmake/qwen36_oracle_dumps.cmake](../../../cmake/qwen36_oracle_dumps.cmake) | 35 | CTest fixture: kernel-geometry GGUF and `ref/qwen36.py` seam dumps (F32 and Q8 activations) |
| [cmake/qwen36_attn_fixtures.cmake](../../../cmake/qwen36_attn_fixtures.cmake) | 39 | CTest fixture: three attention GGUFs (G = 8, 12, 2) and their dumps |
| [cmake/qwen36_block_fixtures.cmake](../../../cmake/qwen36_block_fixtures.cmake) | 28 | CTest fixture: oracle dumps of `toy.gguf` on 8 tokens |
| [patches/llama.cpp-qwen3-coder-any-argument-order.patch](../../../patches/llama.cpp-qwen3-coder-any-argument-order.patch) | 26 | The one change to llama.cpp: Qwen3-Coder tool-call arguments in any order |
| [.gitmodules](../../../.gitmodules) | 4 | Declares the shallow `third_party/llama.cpp` submodule |
| [.gitattributes](../../../.gitattributes) | 2 | `*.sh` keep LF line endings |
| [.gitignore](../../../.gitignore) | 2 | Ignores `/build*/` and `*.log` |
| `third_party/llama.cpp` (submodule, gitlink `3cf03257`) | - | ggml and llama.cpp sources; not checked out in a fresh clone |
| [third_party/ggml/ggml-common.h](../../../third_party/ggml/ggml-common.h) | 1911 | Vendored ggml block layouts and i-quant grids for Strata's own kernels |
| [third_party/ggml/VERSION.txt](../../../third_party/ggml/VERSION.txt) | 1 | The llama.cpp commit the vendored header comes from |
| [third_party/ggml/LICENSE](../../../third_party/ggml/LICENSE) | 21 | ggml's MIT license |
| [include/strata/hip_compat/cuda_runtime.h](../../../include/strata/hip_compat/cuda_runtime.h) | 143 | wave32 HIP shim: CUDA runtime names to HIP (force-included) |
| [include/strata/hip_compat/intrinsics.hpp](../../../include/strata/hip_compat/intrinsics.hpp) | 162 | wave32 HIP shim: `__dp4a`, `__byte_perm`, SIMD byte ops, warp intrinsics |
| [include/strata/hip_compat/cublas_v2.h](../../../include/strata/hip_compat/cublas_v2.h) | 60 | wave32 HIP shim: cuBLAS names to hipBLAS, or rocBLAS for hipBLAS 0.x |
| [include/strata/hip_compat/cuda_fp16.h](../../../include/strata/hip_compat/cuda_fp16.h) | 2 | Includes `hip/hip_fp16.h` |
| [include/strata/hip_compat/cuda_runtime_api.h](../../../include/strata/hip_compat/cuda_runtime_api.h) | 2 | Includes the shim's `cuda_runtime.h` |
| [include/strata/hip_compat/math_constants.h](../../../include/strata/hip_compat/math_constants.h) | 4 | `CUDART_INF_F` |
| [include/strata/platform/hip_compat/strata_hip.h](../../../include/strata/platform/hip_compat/strata_hip.h) | 295 | gfx906 (wave64) shim: runtime names, coherent mapped memory, logical 32-lane warps, device intrinsics |
| [include/strata/platform/hip_compat/cublas_v2.h](../../../include/strata/platform/hip_compat/cublas_v2.h) | 26 | gfx906 shim: cuBLAS names to hipBLAS |
| [include/strata/platform/hip_compat/cuda_runtime.h](../../../include/strata/platform/hip_compat/cuda_runtime.h) | 2 | Includes `strata_hip.h` |
| [include/strata/platform/hip_compat/cuda_fp16.h](../../../include/strata/platform/hip_compat/cuda_fp16.h) | 2 | Includes `strata_hip.h` |
| [include/strata/platform/hip_compat/math_constants.h](../../../include/strata/platform/hip_compat/math_constants.h) | 4 | `CUDART_INF_F`, `CUDART_NAN_F` |

## Key concepts and data flow

### What `make build` does

```
make build
  |-- check third_party/llama.cpp/ggml/CMakeLists.txt exists      (else: "run git submodule update --init")
  |-- git apply patches/llama.cpp-...patch (skipped if already applied)
  |-- cmake -S . -B build  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON
  |                        -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=<list>
  |                        [-DSTRATA_EXPERIMENTAL_SM60=ON if an arch < 75]
  |        configure: CUDA toolkit, arch guard, ggml from the submodule (patch applied again if needed),
  |                   llama.cpp added with its tools/tests/server off
  `-- cmake --build build --target unbiased-strata          (only this target and its dependencies)
```

### How `unbiased-strata` is assembled (`CMakeLists.txt:1431-1444`)

```
frontend/main.cpp ----------------------------+
frontend/engine_entry.cpp                     |
   #define main strata_engine_main            +--> unbiased-strata
   #include "../src/program/generate.cpp"     |
                                              |
strata_engine --+-- strata_core -- strata_plan, strata_warnings, CUDA::cudart
                +-- strata_kernels -- strata_core, strata_artifact   (44 .cu + 3 .cpp)
                +-- strata_kernels_cpu -- Threads, ggml-cpu, ggml-base
strata_prefill -- strata_engine, CUDA::cublas, strata_mmq (ggml-cuda MMQ instances from the submodule)
strata_spec    (CPU-only speculation pieces)
llama-common, llama, mtmd, cpp-httplib   (from third_party/llama.cpp, EXCLUDE_FROM_ALL)
Threads::Threads
```

One process image serves both roles: `frontend/main.cpp` runs `strata_engine_main` when its first argument
is `--engine` (`frontend/main.cpp:651`), otherwise it is the HTTP front end and starts itself again as the
engine child. The upstream `strata` executable (`src/program/generate.cpp` alone) is still declared but is
not built by `make build`.

### Backends

| Backend | Switch | Status in this fork |
| --- | --- | --- |
| CPU only | none (default `cmake` configure) | Builds the artifact tools, planner, `strata_spec`, `strata_kernels_cpu`; no engine, no `unbiased-strata`. Needs the submodule unless `-DSTRATA_NATIVE_EXPERTS=OFF` |
| NVIDIA CUDA | `STRATA_ENABLE_CUDA=ON` | The supported build (`make build`) |
| AMD wave32 (RDNA2/3/4) | `STRATA_ENABLE_HIP=ON` | Still in CMake; configure fails on absent `tests/hip/*.cpp` sources; never builds `unbiased-strata` (it needs `STRATA_ENABLE_CUDA`) |
| AMD gfx906 (wave64) | `STRATA_HIP_GFX906=ON` (forces `STRATA_ENABLE_CUDA=ON`) | Still in CMake; configure fails on absent `tests/hip/router_fast.cpp` |
| Intel Arc (SYCL) | `STRATA_ENABLE_SYCL=ON` | `add_subdirectory(sycl)`, but `sycl/` was removed from this fork |

UPSTREAM.md says the same at the policy level: "Code paths for other models, AMD cards and Windows are still
present in the engine source. They are not built by `make build`, not tested and not supported here."

### Default vs test builds

`STRATA_BUILD_TESTS` defaults ON only when both `tests/CMakeLists.txt` and `bench/micro/` exist
(`CMakeLists.txt:88-92`); neither exists in this fork, so it defaults OFF, and `make build` passes OFF
explicitly. However, many parity executables in the GPU section are not gated by `STRATA_BUILD_TESTS` at
all, so `cmake --build build` (the `all` target) builds them in any CUDA build. `make build` avoids them by
building only `--target unbiased-strata`. The full breakdown is in [test-targets.md](test-targets.md).

## See also

- [README.md](../../../README.md), section "Build" (user-facing build instructions)
- [UPSTREAM.md](../../../UPSTREAM.md) (what the fork changed in the build; the pinned commits)
- [docs/qwen36-port/pascal-build.md](../../qwen36-port/pascal-build.md) (an sm_61 build, ctest run and the Makefile's Pascal switch)
- [docs/qwen36-port/step2-notes.md](../../qwen36-port/step2-notes.md) section 4 (CPU-only and CUDA configure lines with tests)
- [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) (what the snapshot left out of `tests/` and `bench/micro/`)
- Other areas whose code these targets compile: [Core](../core/README.md), [Libraries](../libs/README.md),
  [Prefill](../prefill/README.md), [Program](../program/README.md), [Front end](../frontend/README.md),
  [Tools](../tools/README.md)
