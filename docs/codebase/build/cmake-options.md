[Codebase docs](../README.md) › [Build](README.md) › CMake options

# CMake options and cache variables

## Purpose

Every user-settable switch of [`CMakeLists.txt`](../../../CMakeLists.txt) and
[`cmake/hip_backend.cmake`](../../../cmake/hip_backend.cmake): its default, what it changes, which
combinations are refused, and the compile definitions that reach the C++/CUDA sources.

Project: `project(strata VERSION 0.1.39 LANGUAGES CXX)` with `cmake_minimum_required(VERSION 3.24)`
(`CMakeLists.txt:10-11`). C++20, required, no extensions (`CMakeLists.txt:51-53`).

## `option()` switches

| Option | Default | `make build` | Line | Effect |
| --- | --- | --- | --- | --- |
| `STRATA_ENABLE_SYCL` | OFF | OFF | `CMakeLists.txt:14` | Experimental Intel Arc engine: `add_subdirectory(sycl)` and `return()` before anything else is configured. `sycl/` does not exist in this fork, so ON fails |
| `STRATA_PORTABLE` | OFF | **ON** | `:34` | "build for other PCs": ggml-cpu built for an AVX2 baseline (`GGML_NATIVE OFF`, `GGML_AVX/AVX2/FMA/F16C/BMI2 ON`, `:1119-1123`) instead of the build host; static MSVC runtime on Windows (`:47-49`) |
| `STRATA_ENABLE_CUDA` | OFF | **ON** | `:58` | Enables CUDA, the GPU section (`strata_core` ... `strata`), and `unbiased-strata` (which also needs `STRATA_NATIVE_EXPERTS`) |
| `STRATA_HIP_GFX906` | OFF | OFF | `:64` | AMD MI50 (wave64): forces `STRATA_ENABLE_CUDA=ON` (`:69`) and compiles the CUDA tree as HIP through `include/strata/platform/hip_compat/` ([hip-compat.md](hip-compat.md)) |
| `STRATA_ENABLE_HIP` | OFF | OFF | `:71` | AMD wave32 (gfx1100, gfx1201; others warned) through `cmake/hip_backend.cmake` |
| `STRATA_PREFILL_MMQ` | OFF | OFF | `:72` | HIP only: build ggml-cuda's MMQ prompt kernels as HIP (`:1264-1303`) |
| `STRATA_MMQ_KQUANTS` | OFF | **ON** | `:76` | CUDA: adds `q4_k q5_k q5_1 q6_k` MMQ instances to `strata_mmq` (`:1224-1226`) and defines `STRATA_MMQ_KQUANTS=1` there. Comment: off upstream because every kernel is loaded eagerly and these cost about 5 MB of VRAM |
| `STRATA_ORCA_Q4KS_MMQ` | OFF | OFF | `:77` | Adds the `q5_0` MMQ instance (OrcaRouter Q4_K_S) and `STRATA_ORCA_Q4KS_MMQ=1` |
| `STRATA_BUILD_TESTS` | ON only if `tests/CMakeLists.txt` **and** `bench/micro/` exist, so **OFF in this fork** | OFF | `:88-92` | Gates most CPU tests and some GPU tests; see [test-targets.md](test-targets.md) |
| `STRATA_BUILD_CONVERSATION_TESTS` | OFF | OFF | `:93` | Conversation cache/memory/snapshot/validation tests (`:96-104`, `:541-559`, `:1065`) |
| `STRATA_WERROR` | OFF | OFF | `:94` | Adds `-Werror` / `/WX` to `strata_warnings` |
| `STRATA_EXPERIMENTAL_SM60` | OFF | ON when an arch < 75 | `:138` | Declared inside the CUDA branch. Allows sm_60/sm_70 (refused otherwise, `:156-161`); defines `STRATA_EXPERIMENTAL_SM60=1` globally (`:172-176`) and on `strata_core` (`:285-288`); on MSVC switches CUDA to the shared runtime (`:131-133`) |
| `STRATA_NATIVE_EXPERTS` | **ON** | ON | `:1102` | Native (IQ / K-quant) experts through ggml-cpu: configures ggml from the submodule, applies the patch, adds the native CPU kernels to `strata_kernels_cpu`, builds `strata_mmq` on CUDA. Also required for `unbiased-strata` |
| `STRATA_GFX1012_PORTABLE_DOT` | OFF | n/a | `cmake/hip_backend.cmake:77` | HIP only: on gfx1012 use the portable signed-byte dot instead of the SDWA asm (`intrinsics.hpp:29`) |

## Cache variables (`set(... CACHE ...)`) and plain `-D` variables

| Variable | Default | Line | Effect |
| --- | --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` (forced when empty on single-config generators) | `:54-56` | |
| `STRATA_ISA_FLOOR` | `""` (`avx`, `none` accepted) | `:39-46` | Experimental older-CPU build: ggml-cpu built for SSE4.2 (+AVX for `avx`) with every newer `GGML_*` ISA flag OFF (`:1124-1139`); `STRATA_ISA_FLOOR_AVX=1` / `STRATA_ISA_FLOOR_NONE=1` on `strata_kernels_cpu` (`:1187-1191`, read by `src/kernels/cpu/expert_layout.cpp`) |
| `CMAKE_CUDA_ARCHITECTURES` | `120` if undefined | `:139-141` | Every entry checked (`:142-171`): `native`/`all`/`all-major` skipped; < 75 needs `STRATA_EXPERIMENTAL_SM60`; < 60 always refused; >= 120 with CUDA < 13.0 warns (issues #220, #224). `make build` passes `61;70;75;86;89;120` minus what `nvcc` lacks |
| `CMAKE_CUDA_RUNTIME_LIBRARY` | CMake default (static); `Shared` on MSVC + SM60 | `:131-133` | Avoids duplicate cudart symbols (LNK1169, #585) |
| `CMAKE_HIP_ARCHITECTURES` | `gfx906` under `STRATA_HIP_GFX906` (`:107-109`); `gfx1100` under `STRATA_ENABLE_HIP` (`hip_backend.cmake:4-6`) | | Validated list in [hip-compat.md](hip-compat.md) |
| `STRATA_ORACLE_SOURCE_DIR` | `""` (PATH) | `:420` | With `STRATA_ORACLE_BUILD_DIR` and CUDA: builds 16 `*_parity` programs from `bench/micro/` against a shared llama.cpp build (`:423-443`). **`bench/micro/` is absent**, so setting both fails at generate |
| `STRATA_ORACLE_BUILD_DIR` | `""` (PATH) | `:421` | See above; `find_library(... REQUIRED)` for `ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda` |
| `STRATA_PARITY_PROMPT_ATTN` | unset (not an `option()`) | `:372`, `:389` | Builds `qsa_prompt_attn_parity` (CUDA) / `qsa_select_bench` (HIP) without the full test suite |
| `IQ_FIXTURE_PYTHON` | unset | `:683` | Overrides the interpreter search for the i-quant fixtures |
| `STRATA_PLE_FIXTURE_DIR` | `${CMAKE_SOURCE_DIR}/bench/micro` (PATH) | `:937-938` | Where `ple_parity` reads `ple_in.bin` / `ple_out.bin`. The default directory does not exist in this fork |
| `STRATA_GGML_DIR` | `""` -> `third_party/llama.cpp` | `:1103`, `:1144-1146` | llama.cpp checkout to take ggml (and, for `unbiased-strata`, llama.cpp itself) from. The patch is applied only when this is the submodule (`:1155`) |

### Forced settings of the subprojects

Under `STRATA_NATIVE_EXPERTS` (`:1110-1142`), before `add_subdirectory(<ggml>)`:
`BUILD_SHARED_LIBS OFF`, `GGML_STATIC ON`, `GGML_CUDA OFF`, `GGML_OPENMP OFF`, `GGML_BACKEND_DL OFF`,
`GGML_BUILD_TESTS OFF`, `GGML_BUILD_EXAMPLES OFF`, and one of: `STRATA_PORTABLE` -> `GGML_NATIVE OFF` + AVX2 set;
`STRATA_ISA_FLOOR` -> SSE4.2 floor; otherwise `GGML_NATIVE ON` (host CPU). ggml is therefore always a static
CPU-only library; the CUDA side of ggml is used only as source files compiled into `strata_mmq`.

Under `STRATA_ENABLE_CUDA AND STRATA_NATIVE_EXPERTS` (`:1431-1437`), before `add_subdirectory(<llama.cpp>)`:
`LLAMA_BUILD_MTMD ON`, `LLAMA_BUILD_COMMON ON`, and OFF for `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_TOOLS`,
`LLAMA_BUILD_EXAMPLES`, `LLAMA_BUILD_SERVER`, `LLAMA_BUILD_APP`, `LLAMA_BUILD_UI`, `LLAMA_USE_PREBUILT_UI`,
`LLAMA_OPENSSL`, `LLAMA_LLGUIDANCE`, `LLAMA_ALL_WARNINGS`.

## Refused combinations (FATAL_ERROR)

| Condition | Line |
| --- | --- |
| `STRATA_ENABLE_SYCL` with `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP` | `:16-18` |
| `STRATA_ISA_FLOOR` not empty/`avx`/`none` | `:41-43` |
| `STRATA_ISA_FLOOR` with `STRATA_PORTABLE` | `:44-46` |
| `STRATA_HIP_GFX906` with `STRATA_ENABLE_HIP` | `:66-68` |
| `STRATA_PREFILL_MMQ` without `STRATA_ENABLE_HIP` | `:78-80` |
| `STRATA_ENABLE_CUDA` with `STRATA_ENABLE_HIP` (unless gfx906) | `:81-83` |
| A CUDA arch < 75 without `STRATA_EXPERIMENTAL_SM60`; any arch < 60 | `:156-165` |
| `STRATA_PREFILL_MMQ` or `STRATA_ORCA_Q4KS_MMQ` without `STRATA_NATIVE_EXPERTS` | `:1104-1109` |
| `${STRATA_GGML_DIR}/ggml/CMakeLists.txt` missing ("run `git submodule update --init` first") | `:1147-1149` |
| The llama.cpp patch neither already applied nor appliable | `:1161-1163` |
| HIP: an arch outside the known lists, or an empty list | `hip_backend.cmake:30-38` |
| HIP: `STRATA_PREFILL_MMQ` but the ggml checkout lacks `ggml-cuda/common.cuh` or `quantize.cu` | `:1273-1275` |

## Compile definitions and flags set by the build

| Definition / flag | Where | Set when | Read by |
| --- | --- | --- | --- |
| `STRATA_VERSION="0.1.39"` | global, `:23` | always | `src/program/generate.cpp` |
| `/Zi`, `/DEBUG /OPT:REF /OPT:ICF` | global, `:26-29` | MSVC Release (PDB beside the exe, issue #31) | |
| `STRATA_EXPERIMENTAL_SM60=1` | global `:175`; `strata_core` `:287` | `STRATA_EXPERIMENTAL_SM60` | `src/core/device.cu`, `src/kernels/cuda/qsa_decode_attn.cu`, `src/kernels/cuda/qsa_prompt_attn.cu`, `src/prefill/gemm.cu` |
| `STRATA_HIP_GFX906=1`, `__HIP_PLATFORM_AMD__=1` | global `:121`; `CUDA::cudart` interface `:115` | gfx906 | 17 files under `src/` and `include/` (kernels, device, verify, expert cache) |
| `STRATA_USE_HIP=1`, `STRATA_HIP_ARCHS="gfx1100,..."` | `strata_hip_runtime` interface, `hip_backend.cmake:76` | `STRATA_ENABLE_HIP` | `src/core/device.cu` (arch check, `#error` without it), `prefill.cpp`, `generate.cpp`, `expert_cache`, `mtp`, `verify`, ... |
| `--use_fast_math` | per-source, the 14 `src/kernels/cuda/native_*.cu` files, `:398-412` | CUDA and not gfx906 | "Match the pinned llama.cpp CUDA arithmetic contract for this adapter only" |
| `-ffp-contract=off` | `elementwise.cu`, `quantize_act.cu`, `:413-418` | `STRATA_ENABLE_HIP` | keeps Q8_K multiply and rounding as separate FP32 ops |
| `STRATA_DEQUANT_SELFTEST` | `strata-dequant`, `:224` | always | `src/artifact/dequant.cpp` (its CLI entry point) |
| `STRATA_NATIVE_EXPERTS=1` | `strata_kernels_cpu` PUBLIC, `:1193` | `STRATA_NATIVE_EXPERTS` | `src/kernels/cpu/expert_layout.cpp`, `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_MMQ=1`, `STRATA_PREFILL_FUSED=1` | `strata_prefill` PUBLIC, `:1244` | CUDA + native experts | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_MMQ=1` | `strata_prefill` PUBLIC, `:1294` | HIP + `STRATA_PREFILL_MMQ` | `src/prefill/prefill.cpp` |
| `STRATA_MMQ_KQUANTS=1` | `strata_mmq`, `:1238` | `STRATA_MMQ_KQUANTS` | `src/prefill/moe_mmq.cu`, `include/strata/prefill/moe_mmq.hpp` |
| `STRATA_ORCA_Q4KS_MMQ=1` | `strata_mmq`, `:1241`, `:1290` | `STRATA_ORCA_Q4KS_MMQ` | `src/prefill/moe_mmq.cu` |
| `GGML_USE_HIP` | `strata_mmq`, `:1288`, `:1423` | HIP MMQ / gfx906 | ggml-cuda sources |
| `STRATA_HIPBLASLT_AVAILABLE=1` | `strata_prefill`, `:569` | HIP and hipBLASLt found | `src/prefill/gemm.cu` |
| `--extended-lambda -use_fast_math` (+ `/bigobj` on MSVC) | `strata_mmq` CUDA sources, `:1232-1235` | CUDA + native experts | |
| `-Wall -Wextra` (`/W4 /utf-8` on MSVC), `-Werror` with `STRATA_WERROR` | `strata_warnings` INTERFACE, `:194-208` | always, only for targets that link it | Kept off global flags so third-party code is not affected |

### Per-file CPU ISA flags (GCC/Clang spelling; MSVC uses `/arch:`)

| File | Flags | Line |
| --- | --- | --- |
| `src/kernels/cpu/expert.cpp` | `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma` | `:1090-1091` |
| `src/kernels/cpu/q2_avx2.cpp` | `-mavx2 -mfma -mf16c` | `:1092` |
| `src/kernels/cpu/iq_avx512.cpp` | same as `expert.cpp` | `:1181-1182` |
| `src/kernels/cpu/iq_avx2.cpp`, `kq_avx2.cpp` | `-mavx2 -mfma -mf16c` | `:1183-1184` |
| `src/kernels/cpu/kq_avx1.cpp` | `-mavx` | `:1185` |

The flags are on the source files, not the target, so only those translation units may use the wider ISA;
the engine picks kernels at run time (comment at `:1042-1050`, `:1083-1084`).

## Internal variables other parts of the file read

| Variable | Set at | Meaning |
| --- | --- | --- |
| `_strata_gpu_runtime_target` | `:123`, `:178`, `:182` | `CUDA::cudart` (CUDA, gfx906) or `strata_hip_runtime` (HIP) |
| `_strata_gpu_blas_target` | `:124`, `:179`, `:183` | `CUDA::cublas`, or `STRATA_HIP_BLAS_TARGETS` (`roc::hipblas` [+ `roc::rocblas`]) |
| `_strata_gpu_include_directories` | `:125`, `:180`, `:184` | CUDA toolkit includes, or the HIP compat dir |
| `_strata_tests_default` | `:88-91` | default of `STRATA_BUILD_TESTS` |
| `_q36_python`, `_iq_python` | `:239-249`, `:678-703` | interpreters with numpy (see [test-fixtures-cmake.md](test-fixtures-cmake.md)) |
| `STRATA_CUDA_TESTS` | `:1035-1037` | read by `tests/CMakeLists.txt` (absent in this fork) |
| `STRATA_HIPBLASLT_AVAILABLE`, `STRATA_HIP_ARCHS`, `STRATA_HIP_BLAS_TARGETS` | `hip_backend.cmake` | |
| `_strata_mmq_types` | `:1201-1205` | MMQ instance list: `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0` (+ `q5_0`) |

## Gotchas

- The comment on `STRATA_GGML_DIR` (`:1101`, `:1103`) still says an empty value fetches the pinned upstream
  commit. In this fork an empty value means the submodule (`:1143-1146`); nothing is fetched. The fallback
  branches that use `strata_llamacpp_SOURCE_DIR` (`:1212-1213`, `:1269-1270`) are unreachable because
  `STRATA_GGML_DIR` is always set by then.
- `STRATA_EXPERIMENTAL_SM60` is read on MSVC (`:131`) before its `option()` is declared (`:138`); a value
  passed with `-D` is in the cache already, so this works for `-D` use.
- A plain `cmake -S . -B build` (no options) with an empty submodule fails at `:1148` because
  `STRATA_NATIVE_EXPERTS` defaults ON. `-DSTRATA_NATIVE_EXPERTS=OFF` configures without the submodule (verified:
  it yields `strata-gguf`, `strata-dequant`, `strata-plan`, `strata_spec`, `strata_kernels_cpu`).
- Comments refer to `docs/INTEL_ARC.md`, `docs/AMD_HIP.md`, `docs/UNSLOTH_Q4.md`, `docs/ORCA_Q4_K_S.md`,
  `docs/kv-streaming-design.md` and `docs/native-mmvq.md`; none of them exist in this fork (UPSTREAM.md: all
  upstream docs except `DETAILS.md` and the paper were removed).

## See also

- [makefile.md](makefile.md), [targets.md](targets.md), [test-targets.md](test-targets.md), [hip-compat.md](hip-compat.md)
- [docs/qwen36-port/pascal-build.md](../../qwen36-port/pascal-build.md) (`STRATA_EXPERIMENTAL_SM60` sites in the tree)
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) (upstream's hardware and build notes)
