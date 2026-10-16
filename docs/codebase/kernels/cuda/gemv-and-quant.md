[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › GEMV and quantization

# Dense GEMVs, activation quantizers and dequantizers

## Purpose

These are the matrix-vector products and format conversions used outside the MoE expert tiers:

- GEMVs over upstream's canonical S-forms (S2/S4/S8 code planes plus fp32 scales), used by S2 packs;
- BF16-weight GEMVs (router logits, `ssm_alpha` / `ssm_beta`, indexer projections, PLE value, HC weights);
- quantizers that turn an F32 activation into the weight's `vec_dot_type` (Q8_0, Q8_K, q8_1);
- dequantizers from GGUF blocks to BF16/FP16/FP32 for the prompt path's tensor-core GEMMs;
- the arch shims `dp4a.hpp` and `q8_1_finite.hpp`.

The pinned llama.cpp MMVQ for native GGUF weights is in [moe-and-experts.md](moe-and-experts.md).

Files: `src/kernels/cuda/s_gemv.cu`, `s2_gemv.cu`, `s2_gemv_quads.cu`, `s2_gemv_fast.cu`, `s2_gemv_q8.cu`,
`bf16_gemv.cu`, `native_bf16.cu`, `dequant_s2.cu`, `dequant_bf16.cu`, `quantize_act.cu`; headers `s_gemv.hpp`,
`s2_gemv.hpp`, `s2_gemv_q8.hpp`, `bf16_gemv.hpp`, `dequant_s2.hpp`, `dequant_bf16.hpp`, `quantize_act.hpp`,
`dp4a.hpp`, `q8_1_finite.hpp`.

## Formats

### Canonical S-forms (`s_gemv.hpp:17-36`)

One decode for every source type: `value = cb[code] * scale + offset`. `SForm` carries:

| Field | Meaning |
| --- | --- |
| `code_bits` | 2, 4 or 8. Codes are packed LSB-first, so element i of an S2 row sits at byte i/4, bits (i%4)*2 |
| `code_bias` | added to the code in the integer domain before the multiply (S2: -1, which makes `(code - 1) * scale` bit-exact) |
| `group_elems` | elements per fp32 scale (and offset); 64 for S2, a power of two in general |
| `codebook` | `Affine` (`code + bias`) or `Iq4Nl` (ggml's `kvalues_iq4nl` table) |
| `has_offset` | only K-quants with a min (Q4_K, Q5_K) |
| `act_kind` | 0 = the weight's `vec_dot_type` is Q8_0, 1 = Q8_K; carried from the manifest's source type because it cannot be derived from the other fields (Q5_0 vs Q5_K, IQ4_NL vs IQ4_XS) |

A tensor is `[n_in, n_out]` with n_in contiguous (GGUF's ne0 fastest), so row o is a run of `n_in / group_elems`
groups. S2 is Q2_0's canonical form, and the header calls it 31.64 GiB of upstream's 38 GiB pack.

### Activation formats

| Format | Layout | Produced by | Consumed by |
| --- | --- | --- | --- |
| Q8_0 | 34 B / 32: fp16 `d`, int8 `qs[32]` | `quantize_q8_0`, `quantize_q8_0_scaled` (+ fp32 scales) | `s2_gemv_q8`, `s_gemv_q8_0_split`, `s2_expert_grouped` |
| Q8_K | 292 B / 256: f32 `d`, int8 `qs[256]`, int16 `bsums[16]` (not read, but included in the stride) | `quantize_q8_K` | `s_gemv_q8k`, `s_gemv_q8k_split` |
| q8_1 | 36 B / 32: `half2 {d, sum}`, int8 `qs[32]` | `native_quantize_q8_1`, `quantize_q8_1_rows` | native MMVQ, i-quant kernels |
| BF16 | high 16 bits of an f32 | `f32_to_bf16_bulk` | `bf16_gemv`, `bf16_gemv_split` |
| FP16 | IEEE half | `f32_to_f16_bulk` | `s_gemv`, `s2_gemv*` (reference GEMVs) |

## Main entry points

### S-form GEMVs (`s_gemv.cu` and the S2 files)

| Function | Launch | Status |
| --- | --- | --- |
| `s_gemv` (`s_gemv.cu:344`) | one thread per output row, 128 threads, default stream, synchronous | naive reference (FP16 activations) |
| `s_gemv_split` / `_async` (`:533`, `:538`) | one block of `threads_per_row` (power of two <= 1024) per row, shared-memory reduction | split reference; `_async` used by `src/core/overlap_main.cpp` / `concurrent_main.cpp` benches |
| `s_gemv_q8k` (`:580`) | thread per row, 128 threads | Q8_K reference, code_bits 4/8 |
| `s_gemv_q8k_split` (`:607`) | warp per row, 256 threads | engine path for 4/8-bit K-quant / IQ4_XS forms (Q8_K activations) |
| `s_gemv_q8_0_split` (`:647`) | warp per row, 256 threads | engine path for 4/8-bit legacy forms (Q8_0 activations; e.g. `ffn_down_shexp`, n_in 640 = not a multiple of 256) |
| `s2_gemv_q8` (`s2_gemv_q8.cu:77`) | one block of `threads_per_row` per row | engine path for S2 (Q2_0) dense weights; dequantizes the activation in the loop (no dp4a) |
| `s2_gemv` (`s2_gemv.cu:46`) | thread per row, 128 threads, synchronous | naive S2 reference |
| `s2_gemv_quads` (`s2_gemv_quads.cu:76`) | block per row | four elements per code byte, one 64-bit activation load per quad |
| `s2_gemv_fast` (`s2_gemv_fast.cu:124`) | block per row | kept as a recorded negative result: staging x in shared memory measured slower, a constant-memory LUT neutral (`s2_gemv_fast.cu:1-28`) |

The Q8 split kernels load the 16-entry IQ4_NL codebook into shared memory once per block. A divergent
`__constant__` read of it measured 2.12x slower on `attn_qkv` (`s_gemv.cu:18-31`). The codebook load and barrier come
before the early return, so surplus warps do not leave a barrier early (`s_gemv.cu:150-157`). Only code_bits 4 and 8
are instantiated for the Q8_K / Q8_0 split kernels. 2-bit weights go through `s2_gemv_q8`.

The engine's dispatcher is `gemv_quantized` (`src/core/layer.cpp:141-176`). A weight with native GGUF data uses
`native_quantize_q8_1` + `native_mmvq`. Otherwise code_bits 2 uses `s2_gemv_q8`, `act_kind == 0` uses
`s_gemv_q8_0_split`, and anything else uses `s_gemv_q8k_split`. `TPR` (threads per row) is 32 (`layer.cpp:91`).

### BF16 GEMVs

| Function | Launch | Contract |
| --- | --- | --- |
| `bf16_gemv` (`bf16_gemv.cu:84`) | warp per row (256 threads, 8 rows per block) when `n_out >= 64`, else the naive thread-per-row kernel | BF16 activations x BF16 weights; every product exact in f32 |
| `bf16_gemv_split` (`:117`) | one block of `threads_per_row` per row | not bit-identical to `bf16_gemv` (another summation order) |
| `bf16_gemv_fp32_mmvf` (`native_bf16.cu:156`) | one block per output row; block size 32..256 chosen by `mmvf_block_size` (`:121`) to minimise pair iterations | pinned llama.cpp `mmvf`: F32 activation, pairs of BF16 weights with two ordered `__fmaf_rn`, warp XOR then block reduction. n_in even; x 8-byte aligned. Throws on bad geometry |
| `bf16_gemv_fp32_mmvf_multi` (`:136`) | the same with NT = 4 or 8 accumulators | 1..8 rows `x[t * ldx]`, weight read once, each output bitwise its single-row call |
| `bf16_gemv_fp32_mmvf_cols` (`:187`) | slices of 8 columns | contiguous columns form of the above |

`project_bf16` in `src/core/layer.cpp:88-94` picks between them: `bf16_gemv_fp32_mmvf` with `--native-bf16`,
otherwise the split or warp BF16 forms.

### Quantizers (`quantize_act.cu`)

| Function | Launch | Rules transcribed from ggml |
| --- | --- | --- |
| `quantize_q8_0` (`:224`) | one thread per 32-block, 128 threads | `d32 = amax / 127` in FP32; quantize by dividing in double by d32 (not the fp16-rounded d) with `rint` (half to even); store fp16 d. Matches ggml's bytes (`quantize_act.cu:13-23`) |
| `quantize_q8_0_scaled` (`:245`) | same | the same blocks plus the fp32 scale per block, so the GPU expert hit path multiplies by the CPU pool's `ActQ::scale` |
| `quantize_q8_K` (`:281`) | one thread per 256-block, 64 threads | `quantize_row_q8_K_ref`: scale `-127 / max` (not -128), where `max` is the signed value at the largest magnitude; `nearest_int` (round half to even via 12582912.0f) |
| `dequant_q8_0`, `dequant_q8_K` (`:267`, `:299`) | | inverses for round-trip tests |

A null stream makes these calls synchronise. For HIP, CMake compiles `quantize_act.cu` and `elementwise.cu` with
`-ffp-contract=off` (`CMakeLists.txt:413-418`).

### Dequantizers

| Function | Notes |
| --- | --- |
| `dequant_s2` (`dequant_s2.cu:39`) | naive, one thread per S2 block, synchronous; `(code - 1) * scale` |
| `dequant_bf16` / `dequant_f16` / `dequant_f32` (`dequant_bf16.cu:237-252`) | GGUF Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, Q2_0 (`geometry`, `:177`); one thread per 32-element group, 256 threads; rows `[row0, row0 + rows)` of a row-major tensor. `dequant_f16` / `_f32` forward IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S / IQ1_M to `iq_dequant_f16` / `_f32`. Q2_0 values are exact in FP16 |

The prompt path (`src/prefill/gemm.cu`, `src/prefill/kernels.cu`, `src/prefill/prefill.cpp:3048-3052`) uses these to
feed cuBLAS / tensor-core GEMMs.

### Architecture shims

- `dp4a.hpp`: `STRATA_DP4A(a, b, c)` is `__dp4a` on sm_61 and newer (and on HIP). Below sm_61 (`__CUDA_ARCH__ < 610`,
  that is sm_60 in a `STRATA_EXPERIMENTAL_SM60` build) it is llama.cpp's signed byte loop, which is bit-exact.
  `strata_spin_pause()` is `__nanosleep(100)` from sm_70 and a plain spin below. Every q8 dot product in
  `native_mmvq.cu`, `iq_kernels.cu` (`ggml_cuda_dp4a`) and `s2_expert_grouped.cu` goes through the macro. sm_61
  (Pascal GP10x) uses the native instruction.
- `q8_1_finite.hpp` (#606): `q8_1_finite` clamps the q8_1 block's scale and sum to +/-65504 before the fp16 store, so
  one massive activation does not become `inf * 0 = NaN` in the dot product. `q8_1_quant` clamps codes to
  [-127, 127]. Finite blocks are stored bit for bit as before, and a NaN stays NaN (so `STRATA_DBG_NAN` still sees
  it). `q8_1_ds` packs the pair.

## Control and data flow

```
mixed (F32) --quantize_q8_0--> x_q8_0  --s2_gemv_q8 / s_gemv_q8_0_split--> y      (S-form weights)
            --quantize_q8_K--> x_q8k   --s_gemv_q8k_split-->               y
            --native_quantize_q8_1--> x_q8_1 --native_mmvq-->              y      (native GGUF weights)
            --(--native-bf16) bf16_gemv_fp32_mmvf-->                       y      (BF16 weights)
            --f32_to_bf16_bulk--> bf16_gemv(_split)-->                     y      (legacy BF16)
```

`src/core/layer.cpp:237-238` quantizes `mixed` to both Q8_K and Q8_0 once per layer half. The comment at
the comment around `layer.cpp:130` explains why both images are produced.

## Configuration

- CLI: `--native-bf16` (pinned MMVF for SSM/router/indexer projections), `--native-bf16-extra` (PLE / shared gate),
  `--native-dense-gguf` (native projections replace S-forms).
- No environment variables in these files.

## Model-specific behaviour

Qwen3.6 is served from GGUF files (native types), so its dense projections run on `native_mmvq`. The S-form GEMVs
serve upstream's canonical packs for Flash-Next. Qwen3.6's router, `ssm_alpha`, `ssm_beta` and shared gate are F32
in the shipped file and are rounded to BF16 by the pack. The BF16 GEMVs then apply (`docs/QWEN36-35B-A3B.md`
section 3.4).

## Tests

`s_gemv_parity`, `s2_gemv_parity`, `s2_gemv_q8_parity`, `s_gemv_q8k_parity` (constructs exact ties for the Q8_K
rounding), `bf16_gemv_parity`, `dequant_s2_parity`, `quantize_act_parity` (against ggml's bytes and numpy fp16
vectors), `elementwise_parity` (includes `dequant_bf16`), `q5_projection_parity` (Q5_1 via `dequant_bf16` vs ggml).
Oracle-linked: `native_bf16_parity` (source not in this repository).

## Gotchas

- `quantize_q8_0` divides by the FP32 `d32`, not by the fp16 d it stores. Conflating the two moved 18 of 80 blocks
  (`quantize_act.cu:15-18`).
- `s_gemv_q8k_split` used to take a `threads_per_row` it ignored. It was removed so nobody tunes a disconnected knob
  (`s_gemv.hpp:103-112`).
- The naive `bf16_gemv` kernel is uncoalesced and is used only when `n_out < 64` (`bf16_gemv.cu:85-113`).
- `dequant_s2`, `s_gemv`, `s2_gemv` synchronise the device and use the default stream. They are references and must
  not be used on a captured path.
- `s2_gemv_fast.cu` is named "fast" but is not faster (`s2_gemv_fast.cu:3`).
- `pack-format.md` and `activation-contract.md`, cited throughout these headers, are not in this repository. The
  pack format is described in [tools/pack-format.md](../../tools/pack-format.md).

## See also

- [moe-and-experts.md](moe-and-experts.md) (native MMVQ, i-quant formats), [bits-helpers.md](bits-helpers.md)
  (`f16_bits.hpp`, `bf16_bits.hpp`), [norms-rope-elementwise.md](norms-rope-elementwise.md) (`f32_to_*_bulk`)
- `docs/qwen36-port/pascal-build.md` (dp4a / nanosleep on sm_61)
