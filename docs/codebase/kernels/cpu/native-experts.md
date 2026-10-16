[Codebase docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Native experts

# Native (GGUF-form) experts on the CPU

## Purpose

A **native pack** (`tools/iq_pack.py`) keeps each expert as its raw GGUF slices `[gate rows | up rows | down rows]`
in whatever ggml quant type the model file uses (for example IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
Q4_K, Q5_K for gate/up; Q2_0, IQ4_NL, IQ3_XXS, IQ4_XS, Q5_1, Q8_0, Q5_K, Q6_K for down). These values cannot be
re-expressed in the Strata Q2_0 form, so the CPU computes them with **ggml-cpu's own arithmetic**
(`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row is one
`vec_dot`, exactly what llama.cpp's CPU backend computes (`include/strata/kernels/cpu/native_expert.hpp:1-6`).

On top of that baseline, several hand-written **multi-token** kernels decode a block of weights once and apply it to
every token of a speculative verify window that was routed to the same expert. This is the only CPU expert path for
Qwen3.6-35B-A3B, and the path for every Flash-Next native pack.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/native_expert.hpp](../../../../include/strata/kernels/cpu/native_expert.hpp) | 49 | `NativeFmt`, buffer limits, the four row/quant entry points |
| [src/kernels/cpu/native_expert.cpp](../../../../src/kernels/cpu/native_expert.cpp) | 171 | ggml-cpu trait lookup, `native_fmt`, kernel dispatch with env overrides |
| [include/strata/kernels/cpu/iq_avx512.hpp](../../../../include/strata/kernels/cpu/iq_avx512.hpp) | 18 | AVX-512 i-quant API |
| [src/kernels/cpu/iq_avx512.cpp](../../../../src/kernels/cpu/iq_avx512.cpp) | 273 | IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22) rows vs Q8_K, 512-bit, up to 8 tokens |
| [include/strata/kernels/cpu/iq_avx2.hpp](../../../../include/strata/kernels/cpu/iq_avx2.hpp) | 26 | AVX2 i-quant API |
| [src/kernels/cpu/iq_avx2.cpp](../../../../src/kernels/cpu/iq_avx2.cpp) | 493 | The same formats plus IQ4_XS (23) vs Q8_K, and IQ4_NL (20) down rows vs Q8_0, 256-bit |
| [include/strata/kernels/cpu/kq_avx2.hpp](../../../../include/strata/kernels/cpu/kq_avx2.hpp) | 25 | K-quant / BF16 router API |
| [src/kernels/cpu/kq_avx2.cpp](../../../../src/kernels/cpu/kq_avx2.cpp) | 265 | Q4_K (12) gate/up vs Q8_K, Q5_1 (7) / Q8_0 (8) down; bit-exact vs ggml's AVX2 `vec_dot`; also the BF16 router dot |
| [include/strata/kernels/cpu/kq_avx1.hpp](../../../../include/strata/kernels/cpu/kq_avx1.hpp) | 18 | AVX1 router dot API |
| [src/kernels/cpu/kq_avx1.cpp](../../../../src/kernels/cpu/kq_avx1.cpp) | 71 | `bf16_rows_dot_multi_avx1` for AVX-only CPUs (from the Strata_Dirigo fork) |

These files are added to `strata_kernels_cpu` only when `STRATA_NATIVE_EXPERTS=ON` (the default), which also builds
ggml from `third_party/llama.cpp` (or `-DSTRATA_GGML_DIR`) and links `ggml-cpu ggml-base`. Per-source flags:
`iq_avx512.cpp` AVX-512 (+VNNI/VBMI), `iq_avx2.cpp` and `kq_avx2.cpp` `-mavx2 -mfma -mf16c`, `kq_avx1.cpp` `-mavx`
(`kq_avx1.cpp:16-18` has an `#error` if built without AVX). Without `STRATA_NATIVE_EXPERTS`, stubs in
`src/kernels/cpu/expert_layout.cpp:223-231` refuse or abort, and `expert_layout_load` refuses a native pack.

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `kNativeActBytes = 4096`, `kNativeHBytes = 1024` | `native_expert.hpp:17-19` | Largest quantized gate/up activation (2560 as Q8_K = 2,920 B) and down activation the pool's buffers hold |
| `NativeFmt` | `native_expert.hpp:22` | Per-layer types (`gu_type`, `d_type`), their `vec_dot_type`s (`gu_act`, `d_act`), `n_embd`, `n_ff`, row bytes, `up_off`, `down_off`, blob `bytes`, activation sizes |
| `native_fmt(gu, d, n_embd, n_ff, f, err)` | `native_expert.cpp:32` | Fills `NativeFmt` from ggml-cpu traits; refuses a type without `vec_dot`/`from_float`, a geometry wider than `H x FF` (2560 x 640: the pool's buffers), non-whole blocks, or activations bigger than the buffers |
| `native_quant_act` / `native_quant_h` | `native_expert.cpp:76`, `:80` | ggml `from_float` into the layer's activation format |
| `native_gu_rows(f, blob, act, nt, ff, r0, r1)` | `native_expert.cpp:84` | `ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t])`; picks a kernel (below) |
| `native_down_rows(f, blob, hq, nt, out, r0, r1)` | `native_expert.cpp:141` | `out[t][r] = down_r . hq[t]`; picks a kernel (below) |
| `iq512_supported` / `iq512_gu_rows` / `iq512_rows` | `iq_avx512.cpp:245-273` | 512-bit i-quant rows; `Fmt<TY>::decode` decodes 64 values (grid lookups, a 64-bit sign mask, scales) once, each token applies them with a masked subtract + `maddubs` + `madd` |
| `iq256_supported` / `iq256_gu_rows` / `iq256_rows` | `iq_avx2.cpp:410-438` | 256-bit version over 32-value halves; IQ2_XS has a dedicated vectorized sign decode (`row_dot_iq2xs`, `iq_avx2.cpp:266`) |
| `iq4nl256_down_rows` | `iq_avx2.cpp:477` | IQ4_NL down rows vs Q8_0: nibble decode and `\|w\|` once per block, then sign + `maddubs` + `madd` + FMA per token |
| `kq256_supported` / `kq256_gu_rows` / `kq256_rows` | `kq_avx2.cpp:202`, `:239`, `:253` | ggml's `__AVX2__` branches of `ggml_vec_dot_q4_K_q8_K`, `_q5_1_q8_1`, `_q8_0_q8_0` with the token loop moved inside the block loop |
| `bf16_rows_dot_multi` / `bf16_rows_dot` | `kq_avx2.cpp:204`, `:219` | BF16 router rows x fp32 activations for up to 8 tokens; used by the routing-aware prefetch (an estimate, not the real routing) |
| `bf16_rows_dot_multi_avx1` | `kq_avx1.cpp:45` | Same arithmetic with separate mul + add and 128-bit widening for AVX-only CPUs |

## Kernel selection (`native_gu_rows` / `native_down_rows`)

```
gate/up rows (native_expert.cpp:84-139)
  STRATA_KQ256=1 and gu_type == Q4_K and nt >= 2 and AVX2   -> kq256_gu_rows (bit-exact vs ggml)
  nt >= STRATA_IQ_MT_MIN (default 2) and the type has a kernel:
     AVX-512 CPU, not STRATA_NO_IQ512, iq512_supported        -> iq512_gu_rows
     (no AVX-512, or STRATA_NO_IQ512) AVX2, not STRATA_NO_IQ256, iq256_supported -> iq256_gu_rows
  otherwise                                                    -> ggml vec_dot, token by token

down rows (native_expert.cpp:141-169)
  AVX2 and STRATA_KQ256=1 and nt >= 2 and d_type in {Q5_1, Q8_0} -> kq256_rows
  AVX2 and nt >= STRATA_IQ_MT_MIN and d_type == IQ4_NL, not STRATA_NO_IQ4NL -> iq4nl256_down_rows
  otherwise                                                    -> ggml vec_dot, token by token
```

Q2_0 layers of a native pack bypass these two functions: the pool calls the Q2_0 kernels directly when
`q2_native_kernels(type)` is true (type 42 on an AVX2 CPU; `include/strata/kernels/cpu/expert_layout.hpp:65`). See
[q2-expert.md](q2-expert.md).

IQ4_XS (type 23) has only an AVX2 kernel; an AVX-512 CPU keeps ggml-cpu for it (`native_expert.cpp:113-117`) so its
rows do not change rounding. IQ1_M (type 29) has no hand-written kernel; `iq_avx512.cpp:9-10` says "no shipped model
has IQ1_M expert rows", but the IQ2_XS pack's table in `tests/data/native_experts/iq2_xs.txt` lists type 29 gate/up on
three layers, which therefore run on ggml-cpu. `iq512_rows` and `iq256_rows` (plain row dots for these formats) have no caller in the engine;
only `native_expert_parity` calls them.

## Invariants

- The i-quant multi-token kernels use ggml's integer arithmetic but a **different float addition order**, so their
  result differs from ggml's in the last bits; a token's rows therefore round differently alone (ggml path, `nt == 1`)
  than in a group (#152). `STRATA_IQ_MT_MIN=1` routes every group through the multi-token kernel so the output does
  not depend on how many drafts a window held (measured -1..-3% decode on IQ3_S, `native_expert.cpp:100-104`).
- The K-quant kernels (`kq_avx2.cpp`) are bit-exact per token against ggml-cpu's AVX2 `vec_dot` for any group size
  (`kq_avx2.cpp:1-13`), which assumes ggml-cpu was itself built for AVX2 (the portable build) - on an AVX-512 ggml
  build the reference `vec_dot` is a different branch.
- Each kernel is reached only for the types it implements: falling through an empty `switch` would leave `ff`
  unwritten (`native_expert.cpp:113-115`).
- AVX2 kernels are gated on `cpu_avx2_ok()` as well as the env switches; reaching `iq256_gu_rows` on a pre-Haswell CPU
  was a real crash (`native_expert.cpp:92-99`, `:148-150`).
- `iq_avx2.cpp` builds its sign table at compile time (`constexpr`, `iq_avx2.cpp:56-82`) so the AVX2 translation unit
  has no static constructor that could execute AVX2/BMI2 before `main()` on an older CPU (#391).

## Callers and dependencies

- [pool.md](pool.md): `ExpertPool::run_split_multi_native` (modes 5/6 in `drain`, `src/kernels/cpu/pool.cpp:561-601`)
  calls `native_gu_rows`, `native_quant_h`, `native_down_rows`.
- [expert-layout.md](expert-layout.md): `expert_layout_load` calls `native_fmt` for every layer line.
- `src/core/expert_source.cpp:2193` quantizes each verify-window token with `native_quant_act`.
- `src/core/expert_source.cpp:1252-1258` runs the routing-aware prefetch's router with `bf16_rows_dot_multi`
  (AVX2), `bf16_rows_dot_multi_avx1` (AVX) or a scalar loop.
- ggml-cpu (`ggml.h`, `ggml-cpu.h`, `ggml-common.h`); `ggml_cpu_init()` is called once (`native_expert.cpp:23-26`).

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `STRATA_IQ_MT_MIN` | 2 | Smallest group size that uses the multi-token i-quant / IQ4_NL kernels |
| `STRATA_NO_IQ512` | unset | AVX-512 CPU uses the AVX2 i-quant kernel instead |
| `STRATA_NO_IQ256` | unset | Drop the AVX2 i-quant kernel (with `STRATA_NO_IQ512` too, ggml-cpu only) |
| `STRATA_NO_IQ4NL` | unset | IQ4_NL down rows on ggml-cpu |
| `STRATA_KQ256` | 0 | `=1`: K-quant multi-token kernels (opt-in; "measured no faster in the engine", `native_expert.cpp:105-108`) |
| `STRATA_IQ_PREFETCH` | 2048 | Software prefetch distance in bytes for both i-quant kernels (0 = off) |
| `STRATA_IQ_GATHER` | unset | AVX-512: IQ3 grids via a gather (measured 3-5% slower on Zen 4) |
| `STRATA_IQ256_GATHER` | 0 | AVX2: IQ3 grids via a gather (~1.3x on an i7-12850HX for two formats) |
| `STRATA_FORCE_AVX2=1`, `STRATA_FORCE_ISA` | unset | Cap the CPU probes ([expert-layout.md](expert-layout.md)) |

CMake: `STRATA_NATIVE_EXPERTS` (ON), `STRATA_GGML_DIR`, `STRATA_PORTABLE` (ggml-cpu built AVX2 baseline),
`STRATA_ISA_FLOOR` (`avx` / `none`: ggml-cpu built for the floor, defines `STRATA_ISA_FLOOR_AVX` / `_NONE`).

## Model-specific behaviour

- **Qwen3.6-35B-A3B** (2048 x 512, 256 experts top-8): UD-IQ3_XXS packs are IQ2_S gate/up with IQ3_XXS down (IQ4_XS
  down on layers 34, 38, 39); UD-Q4_K_XL is Q4_K / Q5_K, with Q6_K down on some layers
  (`CMakeLists.txt` comments on the `native_expert_parity_qwen36_*` tests; `tests/data/native_experts/qwen36-*.txt`).
  IQ2_S gate/up rows get the multi-token i-quant kernels; Q4_K / Q5_K gate/up rows (unless `STRATA_KQ256=1`) and
  all IQ3_XXS / IQ4_XS / Q5_K / Q6_K down rows go through ggml-cpu's `vec_dot`.
- **Flash-Next** (2560 x 640, 512 experts top-10), per the layer tables copied into `tests/data/native_experts/`:
  the IQ2_XS pack is IQ2_XXS / IQ2_S / IQ1_M (type 29, three layers) gate/up with Q2_0 down; the IQ3_XXS pack is
  IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S gate/up with Q2_0 or IQ4_NL down; the IQ3_S and Coder IQ1_M tables are
  IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS gate/up with IQ4_NL or Q2_0 down. Q2_0 down rows take the Q2_0 kernels, IQ4_NL
  down rows `iq4nl256_down_rows`, IQ1_M gate/up rows ggml-cpu.
- A 256-value down type at Flash-Next's `n_ff = 640` is refused by `native_fmt` (640 is not a multiple of 256); the
  ctest `native_expert_parity_refuses_q6_K_down_at_640` is marked `WILL_FAIL`.

## Tests

`native_expert_parity` (CPU path vs ggml `to_float` reference, plus the GPU path) and its 15 ctests;
`router_dot_parity` (AVX2 and AVX1 router dot vs double); `tests/core/pool_native_widths_test.cpp` (the pool's
native path at 2048 x 512). See [tests.md](tests.md) and [../parity/README.md](../parity/README.md).

## Gotchas

- `native_fmt`'s width guard (`n_embd <= 2560`, `n_ff <= 640`) is a pool-buffer limit, not a ggml limit.
- `kq_avx2.cpp` has its own `MAXT = 8` (`kq_avx2.cpp:28`) and loops over groups of 8 tokens; the i-quant
  `dot_rows_nt` split above 4 tokens into groups of 4 (`iq_avx512.cpp:229-240`, `iq_avx2.cpp:394-405`).
- `bf16_rows_dot` (single-token) has no caller in `src/`.

## See also

- [q2-expert.md](q2-expert.md), [pool.md](pool.md), [expert-layout.md](expert-layout.md)
- [docs/qwen36-port/step3-experts.md](../../../qwen36-port/step3-experts.md) (native expert format coverage)
- [docs/QWEN36-35B-A3B.md](../../../QWEN36-35B-A3B.md)
