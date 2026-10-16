[Codebase docs](../README.md) › [Prefill](README.md) › GEMM

# Dense projections: `Gemm`

## Purpose

Every dense projection of a chunk is `Y[T, N] = X[T, K] . W[N, K]^T` with `W` row-major (the GGUF / pack layout)
and FP32 output. `Gemm` wraps one cuBLAS handle (hipBLAS on HIP, plus an optional hipBLASLt path) bound to the prompt
stream, a dequantization scratch and a workspace. It serves the attention / GDN / shared-expert projections, the
BF16 tensors (router, hyper-connection, SSM alpha/beta, indexer, shared gate, PLE), the FP16 expert fallback and the
draft layer.

Files: [include/strata/prefill/gemm.hpp](../../../include/strata/prefill/gemm.hpp),
[src/prefill/gemm.cu](../../../src/prefill/gemm.cu),
[src/prefill/hipblaslt_tuning.hpp](../../../src/prefill/hipblaslt_tuning.hpp).

Note: on CUDA this is plain cuBLAS `cublasGemmEx` (`CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT`); there is no
cuBLASLt code in this subsystem. "Lt" appears only as hipBLASLt on HIP builds.

## Main functions

| Function | Where | What it does |
| --- | --- | --- |
| `Gemm::init(stream, scratch_elems, err)` | `src/prefill/gemm.cu:349` | own handle, a 32 MiB workspace and an FP16 scratch it allocates; every failure names the call and status (#240) |
| `Gemm::init_external(stream, scratch, elems, ws, ws_bytes, err)` | `src/prefill/gemm.cu:313` | the same over caller-owned buffers: `Prefill::init` passes 32M-element scratch and 32 MiB workspace carved from the borrowed region |
| `Gemm::rebind(...)` | `src/prefill/gemm.cu:335` | after `Prefill::relayout` moved the buffers |
| `Gemm::bf16(X, W, Y, T, N, K, ldy, beta)` | `src/prefill/gemm.cu:464` | BF16 x BF16 -> FP32; `beta = 1` accumulates (the `STRATA_PREFILL_BF16X2` low part) |
| `Gemm::f16(...)` | `src/prefill/gemm.cu:519` | FP16 x FP16 -> FP32 |
| `Gemm::native(X, ggml_type, W_blocks, Y, T, N, K, ldy, beta)` | `src/prefill/gemm.cu:538` | dequantizes GGUF blocks to FP16 into the scratch with `strata::kernels::dequant_f16` (row slices when `N*K` exceeds it), then `f16` |

All products use the column-major identity `Y^T[N, T] = W . X^T` with `CUBLAS_OP_T` on `W` (`gemm.cu:511-515`).
Errors in a product exit the process (`ck`), setup-call failures are logged and tolerated (`note`).

Callers in `prefill.cpp`: `native_proj` (`:1668`) for quantized projections with FP16 activations (`mixed_h`,
`y_h`, `attn_h`, `sh_h`), `bf16_proj` (`:1674`) for resident BF16 tensors with BF16 activations (`mixed_bf`, `xn16`,
`lo16`), `m.gemm.f16` for the FP16 expert fallback (`:3057`, `:3060`), and `draft_kv`'s `proj` lambdas.

Gotcha: the header comment says quantized weights are dequantized to BF16 (`dequant_bf16`); the code dequantizes to
FP16 and runs the FP16 GEMM (`gemm.cu:547`, `:552`). The prompt path's activations for those products are FP16 images
(`mixed_h`).

## BF16 below sm_80 (`gemm.cu:381-462`)

Turing and Volta have no BF16 tensor cores and cuBLAS falls back to a SIMT kernel; Pascal has no BF16 GEMM at all.
`bf16_path()` (`gemm.cu:439`) chooses per device:

| Path | When | How |
| --- | --- | --- |
| 0: cuBLAS BF16 | cc >= 80, or cc unknown, or 7.5 by default | `cublasGemmEx` with `CUDA_R_16BF` |
| 1: through FP16 | cc 7.0-7.4 by default; 7.5 with `STRATA_BF16_TC=1`; any card with `STRATA_BF16_TC=2` (test mode) | weight converted once, activations in 32 MiB row slices, to FP16 (finite values clamped to +-65504), then `f16`. Not for `N == 1` (GEMV) or `beta != 0` |
| 2: through FP32 | cc < 70 (Pascal), only in a `STRATA_EXPERIMENTAL_SM60` build | both widened exactly to FP32, `cublasSgemm` |

The conversion buffers (`tc_w_`, `tc_x_`) are grown on demand per `Gemm` and freed in the destructor.

## HIP and hipBLASLt

The HIP builds map cuBLAS names to hipBLAS through the compat headers (`CMakeLists.txt:60-63`). Bits that remain in
this file:

- `STRATA_ABSORB_HIPBLAS_STICKY` (`gemm.cu:49-64`): on Windows HIP only, a stale `hipErrorInvalidValue` left by a
  successful hipBLAS GEMM is cleared (#247 / #325); any other sticky error exits.
- hipBLASLt (`gemm.cu:71-296`), compiled only with `__HIPCC__` and `STRATA_HIPBLASLT_AVAILABLE` (CMake defines it on
  `strata_prefill` when `STRATA_ENABLE_HIP` finds roc::hipblaslt, `CMakeLists.txt:568-574`). It is active only when
  `STRATA_HIPBLASLT_TUNING=<file>` names a tuning table whose header matches the device arch and hipBLASLt version.
  For each `(dtype, T, N, K, ldy, beta)` the closest row (same N, K, ldy; nearest `T` bucket) gives a solution index;
  the algorithm is fetched by index, checked with `matmulIsAlgoSupported`, cached, and run with the shared workspace.
  Anything missing falls back to hipBLAS (`cublasGemmEx` via the shim). A failed launch with `beta != 0` exits rather
  than fall back (the output was already accumulated into). `STRATA_HIPBLASLT_VERBOSE` logs each choice and a summary
  of fallback shapes at destruction.
- `hipblaslt_tuning.hpp`: `TuningTable::load` parses `STRATA_HIPBLASLT_TUNING_V1 <arch> <version>` then rows
  `f16|bf16 N K ldy T_bucket solution_id` (comments with `#`, duplicates refused); `closest(...)` picks the nearest
  bucket, ties to the smaller. The generator is `tools/hip/tune_hipblaslt.cpp` per `CMakeLists.txt:572`; that file is
  not in this tree (what it measures is not determined from source).

On the gfx906 (`STRATA_HIP_GFX906`) wave64 build the CUDA code paths are compiled through HIP and these
hipBLASLt bits are not used (no `STRATA_ENABLE_HIP`).

## Configuration

| Knob | Effect |
| --- | --- |
| `STRATA_BF16_TC=0|1|2` | sub-sm_80 BF16 path choice (see table) |
| `STRATA_EXPERIMENTAL_SM60` (CMake) | builds the Pascal FP32 path |
| `STRATA_HIPBLASLT_TUNING=<file>`, `STRATA_HIPBLASLT_VERBOSE` | HIP only |
| `GEMM_SCRATCH`, `GEMM_WS` (`prefill.cpp:674-675`) | 32M FP16 elements and 32 MiB, taken from the prompt buffers |

## Tests

`gemm_bf16_parity` (`src/prefill/gemm_bf16_parity.cu`) checks `Gemm::bf16` against cuBLAS's own BF16 product over
the prompt path's shapes, including slicing, `beta = 1` and `ldy > N`; see [tests.md](tests.md). The HIP tests
registered in CMake (`hip_prefill_gemm` at `CMakeLists.txt:583`, unguarded; `hip_prefill_hipblaslt_gemm`, guarded
by `EXISTS`) point at `tests/hip/`, and `tune_hipblaslt` at `tools/hip/`; neither directory is present in this fork,
so a `STRATA_ENABLE_HIP=ON` configure of this tree is expected to fail on the missing sources (not verified by
building). Per [UPSTREAM.md](../../../UPSTREAM.md), the AMD and Windows code paths are still in the source but
are not built by `make build`, not tested and not supported in this fork (the README lists one NVIDIA card, Pascal
to RTX 50, Linux, as the supported configuration).

## See also

[pipeline.md](pipeline.md), [moe.md](moe.md) (the FP16 expert fallback), [tests.md](tests.md).
