[Codebase docs](../README.md) › [Build](README.md) › HIP compatibility

# AMD (HIP) backends and the CUDA-to-HIP shims

## Purpose

The engine's GPU code is written once, in CUDA. Two AMD paths compile the same `.cu` sources with HIP by
putting headers named `cuda_runtime.h`, `cuda_fp16.h`, `cublas_v2.h` and `math_constants.h` first on the include
path, mapping CUDA runtime, cuBLAS and device-intrinsic names onto HIP/hipBLAS. This page documents
[`cmake/hip_backend.cmake`](../../../cmake/hip_backend.cmake), the gfx906 branch of
[`CMakeLists.txt`](../../../CMakeLists.txt), and both header sets.

**Status in this fork:** neither AMD path is built by `make build`, neither can produce `unbiased-strata`
(it requires `STRATA_ENABLE_CUDA` and native experts, `CMakeLists.txt:1431`), and both fail to generate
because unguarded `tests/hip/*.cpp` sources were not imported (see [test-targets.md](test-targets.md)). The
code is kept from upstream; UPSTREAM.md: "not built by `make build`, not tested and not supported here".

| Path | Switch | Hardware | Shim directory | How the shim is reached |
| --- | --- | --- | --- | --- |
| wave32 | `STRATA_ENABLE_HIP=ON` | RDNA2/3/4: gfx1100, gfx1201 validated; gfx1101, gfx1200 community-validated; gfx1012, gfx1102, gfx1030, gfx1031 unvalidated | `include/strata/hip_compat/` | `strata_hip_runtime` INTERFACE target: include dir `BEFORE`, plus `-include cuda_runtime.h` (`/FI` on MSVC) for every CXX and HIP source of a target that links it (`hip_backend.cmake:72-94`) |
| wave64 | `STRATA_HIP_GFX906=ON` (forces `STRATA_ENABLE_CUDA=ON`) | gfx906 (MI50) | `include/strata/platform/hip_compat/` | `include_directories(BEFORE ...)` for the whole project, `CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES`, and the include dir of the `CUDA::cudart` alias (`CMakeLists.txt:106-125`); no force-include |

## `cmake/hip_backend.cmake` (wave32, 110 lines)

Included from `CMakeLists.txt:84-86` when `STRATA_ENABLE_HIP` is ON.

1. **Architectures** (`:4-42`): `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1100`. Spaces are accepted as separators;
   a `:xnack-`-style suffix is stripped. Each arch must be in one of three lists: validated (`gfx1100 gfx1201`,
   silent), community (`gfx1101 gfx1200`, status message), unvalidated (`gfx1012 gfx1102 gfx1030 gfx1031`,
   warning); anything else is a `FATAL_ERROR`. The comma-joined list becomes `STRATA_HIP_ARCHS`.
2. **Packages** (`:44-67`): `enable_language(HIP)`, `find_package(hip CONFIG REQUIRED)`, `hipblas` (REQUIRED),
   `rocblas` (REQUIRED only if `hipblas_VERSION < 1.0`, appended to `STRATA_HIP_BLAS_TARGETS`), `hipblaslt`
   (QUIET; sets `STRATA_HIPBLASLT_AVAILABLE`). Requires the `hip::host` and `roc::hipblas` targets.
3. `CMAKE_POSITION_INDEPENDENT_CODE ON` (`:70`), because HIP links a PIE.
4. **`strata_hip_runtime`** (`:72-94`): defines `STRATA_USE_HIP=1` and `STRATA_HIP_ARCHS="gfx1100,..."`, optional
   `STRATA_GFX1012_PORTABLE_DOT=1` (`option` at `:77`), links `hip::host`, and force-includes the shim's
   `cuda_runtime.h`.
5. **Language** (`:96-108`): every `src/*.cu`, `bench/*.cu`, `tests/*.cu` is marked `LANGUAGE HIP`, and
   `tests/hip/intrinsics.cpp` / `native_qsa_score.cpp` if present.

`src/core/device.cu:27-29` refuses to compile under `STRATA_USE_HIP` without `STRATA_HIP_ARCHS`, and uses the
list at run time to stop with a clear message on a card the binary has no code for (`device.cu:39-56`).

## gfx906 branch of `CMakeLists.txt` (`:106-125`, `:1416-1426`)

- `CMAKE_HIP_ARCHITECTURES` defaults to `gfx906`; `enable_language(HIP)`, `find_package(hip)`, `find_package(hipblas)`.
- Creates imported INTERFACE targets named `CUDA::cudart` (-> `hip::host`, definitions `STRATA_HIP_GFX906=1`,
  `__HIP_PLATFORM_AMD__=1`, include dir `include/strata/platform/hip_compat` BEFORE) and `CUDA::cublas`
  (-> `roc::hipblas`), so every "CUDA" target in the file is configured unchanged.
- At the end of the file every `src/*.cu`, `bench/micro/*.cu`, `tools/vision/*.cu` and, when MMQ is built, every
  ggml-cuda `.cu` is marked `LANGUAGE HIP`, and `strata_mmq` gets `GGML_USE_HIP`.
- CUDA-specific branches: `--use_fast_math` is not applied to the `native_*.cu` files on gfx906 (`:412`);
  `gdn_rec_parity`, `gemm_bf16_parity`, `prefill_qwen36_test`, `qwen36_mtp_test` are CUDA-only (`:738`);
  `mmvq_multi_parity_cuda_layout` runs the same test with `STRATA_MMVQ_WAVE=0` (`:724-727`).

## `include/strata/hip_compat/` (wave32)

| File | Lines | Content |
| --- | ---: | --- |
| `cuda_runtime.h` | 143 | `#include <hip/hip_runtime.h>` (with `WIN32_LEAN_AND_MEAN`/`NOMINMAX` on Windows); undefines HIP's `__noinline__` macro (`:15-17`); 87 `#define cuda... hip...` mappings (`:18-120`) with HIP 7 spellings for `cudaFreeHost` and the `cudaHostAlloc*` flags (`:43-47`, `:63-71`); `cudaMallocHost` -> `strata::hip_compat::malloc_host` (`:80`, `:122-127`); on Windows `cudaMemGetInfo` -> `strata::hip_compat::mem_get_info` (declared `:86-89`, defined in `src/core/device.cu:163`, lowers the free figure by the WDDM budget unless `STRATA_WDDM_BUDGET=0`); `cudaFuncSetAttribute` and both `cudaGraphInstantiate` overloads as inline functions (`:129-139`); `__trap()`, `cudaMemcpyToSymbol` (`:140-141`); then includes `intrinsics.hpp` |
| `intrinsics.hpp` | 162 | Device code, only under `__HIPCC__`. `dp4a` (`:18-53`): `__builtin_amdgcn_sudot4` on gfx1100/1101/1102/1200/1201, `__builtin_amdgcn_sdot4` on gfx1030/1031/1032, SDWA `v_mul_i32_i24` asm on gfx1012 (unless `STRATA_GFX1012_PORTABLE_DOT`), portable byte loop otherwise. `byte_perm` via `__builtin_amdgcn_perm` (`:59-62`, avoids HIP's scratch-memory version). SWAR `vsub4`, `vsubss4`, `vcmpne4` (`:67-95`). Warp ops `shfl_*_sync`, `ballot_sync` that `__builtin_trap()` on a mask other than `0xffffffff` (`:99-141`); `syncwarp` (fences + wave barrier) for HIP < 7 (`:105-112`). Macros map the CUDA names (`:145-157`); `__nanosleep(n)` -> `__builtin_amdgcn_s_sleep(1)` (`:160`) |
| `cublas_v2.h` | 60 | For hipBLAS major version 0 (e.g. Ubuntu's 0.54, no workspace API): maps onto rocBLAS directly, with inline `cublasSetMathMode` (accepts only the default mode) and `cublasGemmEx` -> `rocblas_gemm_ex` (`:5-41`). Otherwise `#define`s onto hipBLAS (`:42-60`) |
| `cuda_fp16.h` | 2 | `#include <hip/hip_fp16.h>` |
| `cuda_runtime_api.h` | 2 | `#include "cuda_runtime.h"` |
| `math_constants.h` | 4 | `CUDART_INF_F` -> `HIP_INF_F` |

## `include/strata/platform/hip_compat/` (gfx906, wave64)

| File | Lines | Content |
| --- | ---: | --- |
| `strata_hip.h` | 295 | Defines `HIP_DISABLE_WARP_SYNC_BUILTINS`, includes HIP runtime and fp16, defines `STRATA_HIP_GFX906 1`. Runtime type/constant/function mappings (`:26-155`), following llama.cpp's `ggml-cuda/vendors/hip.h` where they overlap. Notable functions: `strata_device_get_attribute` falls back to `MaxSharedMemoryPerBlock` when the opt-in limit reads 0 on GCN (`:83-91`); `strata_host_alloc` adds `hipHostMallocCoherent` to every mapped allocation, because the GPU/CPU doorbells spin on mapped words and coarse-grained memory may stay in L2 (`:107-118`); `cudaFuncGetName` returns not-supported (`:153`); both `cudaGraphInstantiate` forms (`:159-165`); `cudaInitDevice` emulated without changing the current device (`:166-177`). Device side (`:180-295`): a CUDA warp is a logical 32-lane half of the 64-lane wavefront: shuffles default to width 32, `__ballot_sync` / `__activemask` return the caller's half (`:182-200`), `__syncwarp` is fences plus a wave barrier (`:202-207`); `__nanosleep` -> `s_sleep(8)` or `s_sleep(1)` (`:211-215`); `__dp4a` via `__builtin_amdgcn_sdot4` on gfx906/908/90a/942/GFX11/GFX12 (`:219-233`); `__vsub4`, `__vsubss4`, `__vcmpeq4`, `__vcmpne4` via ext-vector types (`:236-262`); `__fmul_rn` / `__fadd_rn` / `__fsub_rn` with `#pragma clang fp contract(off)` so they are not fused into FMAs (`:265-282`); `__byte_perm` via `v_perm_b32` (`:285-293`, an IQ4_NL GEMV went from 290 us through scratch memory per the comment) |
| `cublas_v2.h` | 26 | hipBLAS (`HIPBLAS_V2`) mappings; `cublasSetMathMode` and `cublasSetWorkspace` become no-ops returning success (`:24-26`) |
| `cuda_runtime.h`, `cuda_fp16.h` | 2 each | `#include "strata_hip.h"` |
| `math_constants.h` | 4 | `CUDART_INF_F`, `CUDART_NAN_F` via compiler builtins |

## Differences between the two shims

| Aspect | wave32 (`hip_compat/`) | gfx906 (`platform/hip_compat/`) |
| --- | --- | --- |
| Warp model | native wave32; partial masks trap | logical 32-lane halves of wave64; masks ignored |
| Mapped host memory | `hipHostMalloc` as asked | always coherent when mapped |
| `cudaMemGetInfo` | WDDM-budget wrapper on Windows | plain `hipMemGetInfo` |
| BLAS | hipBLAS, or rocBLAS for hipBLAS 0.x | hipBLAS only; math mode / workspace ignored |
| `__fmul_rn` and kin | not mapped | contraction-off wrappers |
| Inclusion | force-included into every source of linked targets | found by `#include <cuda_runtime.h>` (e.g. `include/strata/kernels/dp4a.hpp:23`) |
| Build-wide macro | `STRATA_USE_HIP=1` | `STRATA_HIP_GFX906=1` (both the CMake definition and the header) |

## Gotchas

- The comments in `platform/hip_compat/strata_hip.h:3`, `cublas_v2.h:1` and `math_constants.h:1` say the
  directory is used by a "STRATA_ENABLE_HIP build"; in `CMakeLists.txt` it is wired only for
  `STRATA_HIP_GFX906`, and `STRATA_ENABLE_HIP` uses `include/strata/hip_compat/` (`hip_backend.cmake:72`).
- `hip_backend.cmake:96-100` globs `bench/*.cu` and `tests/*.cu`; neither has `.cu` files in this fork.
- On HIP, `src/kernels/cuda/elementwise.cu` and `quantize_act.cu` get `-ffp-contract=off` to keep Q8_K
  rounding bitwise (`CMakeLists.txt:413-418`); the gfx906 header achieves the same for `__f*_rn` with pragmas.
- Upstream's AMD documentation (`docs/AMD_HIP.md`, referenced from `hip_backend.cmake:26`) and the Windows
  build script `tools/hip/build_windows.bat` (referenced at `:83`) are not in this fork.

## See also

- [cmake-options.md](cmake-options.md) (the switches and the refused combinations)
- [test-targets.md](test-targets.md#amd-only-targets) (the HIP tests and their missing sources)
- [docs/qwen36-port/pascal-build.md](../../qwen36-port/pascal-build.md) (the analogous guard inventory for Pascal)
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) (upstream's AMD hardware list)
