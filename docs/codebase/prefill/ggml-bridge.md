[Codebase docs](../README.md) › [Prefill](README.md) › ggml bridge (MMQ)

# The ggml bridge: llama.cpp's MMQ inside the prompt path

## Purpose

The prompt path's default expert products (and the draft layer's Q8_0 products in `draft_kv`) use llama.cpp's
quantized matrix-multiply kernels (`mmq.cuh`, MIT): weights stay in their GGUF blocks, activations are rounded to
q8_1, the products run on int8 tensor cores (dp4a on Pascal). Instead of linking the whole ggml CUDA backend, the
build compiles only the MMQ template instances for the types the packs use plus ggml's activation quantizer, and
`ggml_cuda_host.cu` supplies the handful of host symbols those sources reference.

Files: [src/prefill/moe_mmq.cu](../../../src/prefill/moe_mmq.cu),
[src/prefill/ggml_cuda_host.cu](../../../src/prefill/ggml_cuda_host.cu),
[include/strata/prefill/moe_mmq.hpp](../../../include/strata/prefill/moe_mmq.hpp), `CMakeLists.txt:1190-1300`.

## API (`strata::prefill::mmq`)

| Function / type | Where | What it does |
| --- | --- | --- |
| `built()` | `src/prefill/moe_mmq.cu:106` | true in a build with the MMQ sources (the stub in `prefill.cpp:48-66` returns false) |
| `supported(type)` | `src/prefill/moe_mmq.cu:108` | Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0; + Q4_K, Q5_K, Q5_1, Q6_K with `STRATA_MMQ_KQUANTS`; + Q5_0 with `STRATA_ORCA_Q4KS_MMQ`. IQ1_M is not covered |
| `fits(type, w_rows)` | `src/prefill/moe_mmq.cu:127` | #420: on every visible GPU some tile size `J` in 8..128 has an MMQ config for this type that fits the card's opt-in shared memory (the "fallback" configs when `w_rows % 128 != 0`); otherwise logs once per type and returns false, so the product takes the FP16 path instead of llama.cpp aborting with `J_best=0` |
| `matrix_bytes(type, rows, cols)` | `:155` | GGUF bytes of a matrix |
| `q8_bytes(rows, cols)` | `:159` | q8_1 MMQ activation bytes (row padded to 512 values, plus 128 blocks of slack) |
| `quantize(x, ids, xq, type, cols, ld, rows, stream)` | `:163` | `quantize_mmq_q8_1_cuda`; row `i` of the output is row `ids[i]` of `x` (or `i`) |
| `Product` | `include/strata/prefill/moe_mmq.hpp:36` | one launch over `n` experts `expert_bytes` apart: per expert the activation rows `[bounds[e], bounds[e+1])` times its matrix into `dst` rows via `ids` |
| `Context::run(p, stream)` | `src/prefill/moe_mmq.cu:177` | fills llama.cpp's `mmq_args` (MoE mode with expert bounds) and calls `mul_mat_q_case<type>` on a `ggml_backend_cuda_context`; an uncovered type exits |
| `gather_native`, `gather_native_group` | `:214`, `:231` | copy a native expert's gate, up and down slices into a group buffer slot (16-byte vector copies when aligned, bytes otherwise; the group form does up to 16 experts in one launch and returns false if anything is unaligned) |
| `gather_strata_q2(blob, gu, d)` | `:250` | the Strata Q2_0 blob (codes and FP16 scales in separate planes, Flash-Next widths 1280 x 2560 and 2560 x 640) into GGUF Q2_0 blocks `{f16 d; 16 code bytes}`; same values |
| `swiglu(gu, h, rows, n_ff, interleaved)` | `:256` | FP32 SwiGLU, interleaved (Strata pack) or split (GGUF) |
| `iota(dst, n)` | `:262` | the identity row map MMQ writes through |

## Host shim (`ggml_cuda_host.cu`)

| Symbol | Line | Notes |
| --- | ---: | --- |
| `ggml_cuda_error` | 12 | prints and aborts |
| `ggml_cuda_get_device` | 17 | `cudaGetDevice` |
| `ggml_cuda_info()` | 62 | per-device `cc`, shared memory per block (opt-in), SM count, warp size, VRAM. CUDA: read with `cudaDeviceGetAttribute` (ABI-stable, #542: a `cudaDeviceProp` from a mismatched libcudart read `sharedMemPerBlockOptin` as 1 and every i-quant lost MMQ), with `strata::cc_major_of` / `smem_optin_of` honouring `STRATA_EMULATE_CC`. HIP: the cc from the gfx arch name in ggml's AMD encoding (`ggml_cuda_parse_amd_id`) |
| `CachingPool` / `new_pool_for_device` | 118, 143 | MMQ's stream-k fixup buffers, kept and reused |
| `~ggml_backend_cuda_context` | 147 | empty |

## Build

- CUDA (`STRATA_ENABLE_CUDA`, with `STRATA_NATIVE_EXPERTS` providing the ggml sources): `strata_mmq` is built from
  `moe_mmq.cu`, `moe_fused.cu`, `moe_fused_iq.cu`, `ggml_cuda_host.cu`, ggml's `ggml-cuda/quantize.cu`, and
  `template-instances/mmq-instance-<t>.cu` for `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0`, + `q5_0`
  (`STRATA_ORCA_Q4KS_MMQ`), + `q4_k q5_k q5_1 q6_k` (`STRATA_MMQ_KQUANTS`). Compiled with `--extended-lambda
  -use_fast_math`. Linking it gives `strata_prefill` `STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1`
  (`CMakeLists.txt:1203-1244`). The ggml directory is `STRATA_GGML_DIR` or the fetched llama.cpp checkout.
- `STRATA_MMQ_KQUANTS` defaults OFF in CMake (the extra instances cost about 5 MB of VRAM under
  `CUDA_MODULE_LOADING=EAGER`, `CMakeLists.txt:73-76`) but the fork's `make build` turns it ON (`Makefile:31`)
  because Qwen3.6's UD-Q4_K_XL file needs Q4_K / Q5_K / Q6_K experts on MMQ.
- HIP (`STRATA_ENABLE_HIP` + `STRATA_PREFILL_MMQ=ON`): `moe_mmq.cu`, `ggml_cuda_host.cu`, `quantize.cu` and the base
  instance list compiled as HIP with `GGML_USE_HIP=1`; the fused kernels are not built there, so only
  `STRATA_PREFILL_MMQ=1` is defined (`CMakeLists.txt:1264-1300`). Not supported in this fork (see
  [gemm.md](gemm.md)).
- gfx906 (`STRATA_HIP_GFX906`): the CUDA branch through HIP; the fused kernels report themselves unavailable.

## Numerics

MMQ rounds the activations and H to q8_1 (a scale per 32), so it is not bit-exact against an FP32 product. On the
Qwen3.6 fixture `ffn_out` sits at about 1e-2 relative L1 against the F32 oracle, the largest seam error, and the FP16
GEMM path (`STRATA_PREFILL_MMQ=0`) is closer to the F32 oracle but slower
([docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md), sections 3.1 and 3.2).

## Tests

`prefill_mmq_kquant_test` (CUDA, with `STRATA_MMQ_KQUANTS`): Q4_K / Q5_K / Q5_1 / Q8_0 / Q6_K expert products on
synthetic ggml-quantized weights against a double-precision product, a screen for layout / stride / bound errors.
The fused tests also run the MMQ sequence as their comparison arm. See [tests.md](tests.md).

## See also

[moe.md](moe.md), [gemm.md](gemm.md), [docs/qwen36-port/step3-experts.md](../../qwen36-port/step3-experts.md)
(Q6_K added to the MMQ list).
