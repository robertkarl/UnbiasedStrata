[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Bit-exact float conversions

# `bf16_bits.hpp` and `f16_bits.hpp`: the shared float conversions

## Purpose

These two header-only files give the bit-exact f32 <-> bf16 and f32 <-> fp16 conversions that the kernels, their
host references and the parity tests all use. They are `__host__ __device__` (`STRATA_BF16_HD`, `STRATA_HD`), so a
test's host reference calls the same function the kernel does rather than a transcription. They exist because
private copies of these conversions in the kernels were wrong in ways that only certain inputs exposed.

Files: `include/strata/kernels/bf16_bits.hpp` (61 lines), `include/strata/kernels/f16_bits.hpp` (104 lines). There is no
`.cu` file. The arch shims `dp4a.hpp` and `q8_1_finite.hpp` are on [gemv-and-quant.md](gemv-and-quant.md).

## Functions

### `bf16_from_f32(float) -> uint16_t` (`bf16_bits.hpp:37`)

Round to nearest even, as ggml's `ggml_compute_fp32_to_bf16` does: `i = (i + ((i >> 16) & 1) + 0x7FFF) & 0xFFFF0000`.
The `& 1` is on bit 16, the lowest kept bit, and that is what makes an exact tie round to even. Using bit 15 differs
only on exact ties, so the header warns that end-to-end testing would never catch it (`:6-9`). A NaN is tested first
and returned quieted, `(i >> 16) | 64`, keeping the sign. Without that test the rounding add carried a NaN into -0 or
+inf. Every non-NaN input rounds as before.

### `f32_from_bf16(uint16_t) -> float` (`bf16_bits.hpp:50`)

Exact: a 16-bit shift.

### `f16_from_f32(float) -> uint16_t` (`f16_bits.hpp:37`)

Round to nearest even (numpy's `.astype(np.float16)`), with three regimes handled explicitly:

- normal: the mantissa is rounded from 13 surviving bits with a sticky remainder (the carry may bump the exponent);
- subnormal: the implicit 1 is restored and shifted right, so the rounding point moves; anything below 2^-24 / 2
  underflows to a signed zero;
- overflow: an f32 inf/NaN (raw exponent 255) stays inf/NaN (NaN gets the quiet bit). A finite value too large for
  fp16 saturates to inf. The older private copies returned NaN for finite overflow (`:48-53`).

### `f32_from_f16(uint16_t) -> float` (`f16_bits.hpp:73`)

Exact for all 65,536 inputs, subnormals included (the private copies flushed them to zero, `:78-81`).

## Why these replace `__float2half`

`f16_bits.hpp:3-11` records that `__float2half` (or `__half_as_ushort`) produced 0x2600 for a value whose fp16 is
0x26DB, and 0x3C01 for exactly 1.0. The cause was never isolated. The kernels that write fp16 that a reference
compares bit for bit therefore use `f16_from_f32`: the FP16 KV append (`qsa.cu`), the INT8 / Q4_0 KV scales
(`kv_q8.cu`, `kv_q4.cu`), `quantize_q8_0` (`quantize_act.cu`), `f32_to_f16_bulk` (`elementwise.cu`) and the shared
expert.

`bf16_bits.hpp` is separate from `f16_bits.hpp` on purpose. fp16 has 5 exponent / 10 mantissa bits and bf16 has 8 / 7,
so a routine reused across them gives plausible numbers at a precision off by 4x (`bf16_bits.hpp:11-14`). `gr.cu` once
carried a private bf16 copy and now includes this header. Users of `bf16_from_f32` / `f32_from_bf16` include
`bf16_gemv.cu`, `native_bf16.cu`, `gr.cu`, `fused_gr.cu`, `elementwise.cu` (`f32_to_bf16_bulk`), `shared_expert.cu`
and `src/core/layer.cpp`.

## Configuration

None. The conversions are compile-time code with no switches. On HIP device code they use `__builtin_memcpy`.

## Model-specific behaviour

None.

## Tests

- `bf16_bits_test` (`src/kernels/bf16_bits_test.cpp`, CPU only, CTest `bf16_bits_test`): every one of the 2^32 f32
  bit patterns against ggml's function, against the previous rounding for non-NaN inputs, and NaN quieting.
- `qsa_parity` section 0 checks `f16_from_f32` against 395 vectors of numpy's float16 output (an oracle sharing no code
  with the header; `qsa_parity.cpp:359-418`). `quantize_act_parity` and `elementwise_parity` use the header for
  their fp16 references.
- `bf16_gemv_parity` uses both headers for its host references.

## Gotchas

- `f32_from_f16` and `f32_from_bf16` do not round. They exist to compare against references that hold the same bits.
- Any new kernel that writes fp16 that a test compares bit for bit should use `f16_from_f32`, not the CUDA intrinsic.

## See also

- [gemv-and-quant.md](gemv-and-quant.md), [kv-cache.md](kv-cache.md), [attention-qsa.md](attention-qsa.md)
