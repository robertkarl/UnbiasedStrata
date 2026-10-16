[Codebase docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Q2_0 expert kernel

# The Q2_0 expert kernel (AVX-512 VNNI and AVX2)

## Purpose

Computes one routed expert of a **canonical Strata Q2_0 pack** (or the Q2_0 rows of a native pack) on the CPU:
`out(H) = down(swiglu(gate(x), up(x)))`, all three projections 2-bit Q2_0, against an int8-quantized activation.
It is the kernel the [expert pool](pool.md) runs for every expert the GPU does not hold. The arithmetic was
promoted from upstream's `bench/micro/cpu_s2.cpp` (which is not in this checkout) and is DRAM-bandwidth bound by
design (`include/strata/kernels/cpu/expert.hpp:1-25`).

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/expert.hpp](../../../../include/strata/kernels/cpu/expert.hpp) | 179 | Geometry constants, `ActQ`, scratch structs, every entry point, the long design notes |
| [src/kernels/cpu/expert.cpp](../../../../src/kernels/cpu/expert.cpp) | 683 | AVX-512 (F/BW/VL/VNNI/VBMI) implementation, the "oracle" Q8_0 contract, the scalar reference |
| [src/kernels/cpu/q2_avx2.cpp](../../../../src/kernels/cpu/q2_avx2.cpp) | 130 | The same Q2_0 GGUF-layout rows and activation quantizer for AVX2-only CPUs |

`expert.cpp` is compiled with `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma`
(MSVC `/arch:AVX512`) and `q2_avx2.cpp` with `-mavx2 -mfma -mf16c` (MSVC `/arch:AVX2`), set per source file in
`CMakeLists.txt` (the `strata_kernels_cpu` block) so no other translation unit gets the wider ISA.

## Geometry (compile-time, Flash-Next only)

`expert.hpp:34-51` fixes the canonical blob: `H = 2560`, `FF = 640`, `NE = 512`, `QK = 64` weights per fp16
weight scale, `QKA = 32` activation elements per activation scale, `BB = 18` bytes per 64-weight block,
`BLOB = 1,382,400` bytes per expert. The blob is planar: gate/up codes interleaved by row (`2r` = gate row r,
`2r+1` = up row r) at `O_GU_CODES`, down codes at `O_D_CODES`, then gate/up fp16 scales (`O_GU_SCALES`) and
down scales (`O_D_SCALES`).

**The contract that matters** (`expert.hpp:22-25`): one 64-weight block spans **two** 32-element activation
chunks with different scales. Using one scale per weight block is the "natural-looking mistake".

These constants are why a canonical (non-native) pack is refused for any geometry other than 2560 x 640
(`src/kernels/cpu/expert_layout.cpp:240-248`, see [expert-layout.md](expert-layout.md)). Qwen3.6-35B-A3B
(2048 x 512) never reaches `s2_expert_*`; only its native packs run on the CPU, through
[native-experts.md](native-experts.md). A native pack whose layer is Q2_0 (ggml type 42) reuses the GGUF-layout
row kernels described below.

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `ActQ` | `expert.hpp:64` | int8 activation `q[H]`, per-chunk `scale`, integer `sum`, and `hx = scale*sum` (the code-offset correction precomputed once per layer, not per row) |
| `ExpertScratch`, `ExpertScratchMulti` | `expert.hpp:74`, `:147` | Caller-owned per-worker scratch; the token path allocates nothing |
| `CpuFeatures`, `cpu_features()` | `expert.hpp:83-93`, `expert.cpp:352` | CPUID leaf 7 probe of AVX512F/BW/VL/VNNI/VBMI; `reason()` names each missing bit (`expert.cpp:341`) |
| `cpu_require_expert_support()` | `expert.cpp:375` | Exits with a message if the CPU cannot run the VNNI kernel. Called by `src/program/generate.cpp:2382` for canonical packs only |
| `expert_set_oracle_q8_0(bool)` | `expert.cpp:453` | Switches both quantization boundaries to the pinned ggml x86 Q8_0 contract and the generic CPU Q2_0 dot order (`--cpu-oracle-q8-0`, `generate.cpp:2046`) |
| `act_quant_q8_1(x, n, a)` | `expert.cpp:386` | Legacy activation quantizer: FP32 scale `amax/127`, round half away from zero, clamp to +-127; AVX-512 body. Name is historical (not a ggml block layout) |
| `s2_expert_vnni` / `s2_expert_vnni_q` | `expert.cpp:457`, `:462` | Whole expert from float `x` / from an already-quantized shared `ActQ` (the pool uses `_q` so ten experts share one quantization) |
| `s2_expert_gu_rows` / `s2_expert_down_rows` | `expert.cpp:484`, `:494` | Row ranges of the gate/up and down phases, so several threads can share one expert (`run_split`) |
| `s2_expert_gu_rows_multi` / `s2_expert_down_rows_multi` / `s2_expert_vnni_multi` | `expert.cpp:522`, `:536`, `:611` | Up to `MAXT = 8` tokens routed to the same expert: the codes are unpacked once per block for all tokens |
| `q2_0_gguf_rows_multi` | `expert.cpp:597` | Q2_0 rows in the GGUF block layout (2 B fp16 d + 16 code bytes per 64 weights, interleaved), used for native packs' Q2_0 layers |
| `q2_0_gguf_rows_multi_avx2`, `act_quant_q8_1_avx2` | `q2_avx2.cpp:75`, `:90` | AVX2 versions of the two functions above |
| `s2_expert_scalar(blob, x, out, quant_acts)` | `expert.cpp:629` | Scalar transcription of ggml's formula: the oracle `expert_parity` compares against; `quant_acts` quantizes at **both** stages |

Dispatch between the AVX-512 and AVX2 variants for native packs goes through `q2_rows_any` and `act_quant_any`
(`src/kernels/cpu/expert_layout.cpp:212-221`), keyed on `cpu_avx512_ok()`.

## How the row dot works

`row_dot` (`expert.cpp:232`) evaluates the identity
`sum (c-1) d_w xhat d_x = d_w * (d_x*sum(c*xhat) - d_x*sum(xhat))` per 32-element chunk:

- `unpack_q2_0` (`expert.cpp:125`) unpacks 2-bit codes with `vpmultishiftqb`; the argument order is
  `(control, data)`, determined empirically (the comment records that the Intel guide's order produced 43 of 64
  wrong codes).
- `vpdpbusd` (VNNI) does the code x int8 products; the accumulator stays a float vector and is reduced once per
  row.
- By default (`kZmm`, `expert.cpp:153`) the 512-bit variant `row_dot_z` (`expert.cpp:174`) is used: one unpack
  of 64 codes, one VNNI dot and one FMA per block, scales built eight blocks at a time. `STRATA_CPU_YMM` (any
  value) selects the older 256-bit loop. The two differ in the last bits; plain decode and the verify window use
  the same one, so speculative decoding still reproduces plain decode.
- The oracle contract uses `row_dot_oracle` (`expert.cpp:261`): integer chunk dots with the code offset removed
  before conversion, reduced per 64-block in ggml's generic order, and `quantize_oracle_q8_0` (`expert.cpp:77`,
  adapted from llama.cpp 3cf03257 under MIT): FP16-rounded scales, round to nearest even.

SwiGLU is `silu(gate) * up` (`expert.cpp:474`); the intermediate is re-quantized with `act_quant_q8_1` before the
down projection.

## Invariants

- `s2_expert_vnni_multi` and the `_rows_multi` functions produce, for each token, **bitwise** the single-token
  result (same per-token operation order), checked by `expert_multi_test`.
- `s2_expert_gu_rows` + `act_quant_q8_1(ff)` + `s2_expert_down_rows` is bitwise `s2_expert_vnni_q`
  (`expert.hpp:132-135`), which is what lets the pool split an expert by rows.
- The multi-token path is legacy-contract only; with the oracle contract on, `s2_expert_vnni_multi` falls back to
  one `s2_expert_vnni_q` per token (`expert.cpp:613-616`) and the pool's `run_split*` fall back to `run`.
- The router weight is never applied here; the GPU's `moe_combine` applies it.

## Callers and dependencies

- [pool.md](pool.md): `ExpertPool::drain` calls `s2_expert_vnni_q` (mode 0), `s2_expert_gu_rows` /
  `s2_expert_down_rows` (modes 1-2), the `_rows_multi` pair (modes 3-4) and `q2_rows_any` (native Q2_0 layers,
  modes 5-6) (`src/kernels/cpu/pool.cpp:542-624`).
- `src/core/expert_source.cpp:2191` quantizes the verify window's activations with `act_quant_any` (native Q2_0)
  or `act_quant_q8_1` (canonical).
- No dependency on ggml: only `<immintrin.h>` and CPUID.

## Configuration

| Knob | Effect |
|---|---|
| `--cpu-oracle-q8-0` (`generate.cpp:1388`) | `expert_set_oracle_q8_0(true)`: experimental pinned ggml contract; the header says GPU expert-cache parity is not established for it |
| `STRATA_CPU_YMM` (env, any value) | Use the 256-bit row dot instead of the 512-bit one |
| CMake: per-source `-mavx512*` / `-mavx2` flags | See [CPU README](README.md#build) |

## Model-specific behaviour

Flash-Next canonical packs (`tools/strata_pack.py`) use every function here. Flash-Next native packs use only the
GGUF-layout rows for their Q2_0 down projections (the shipped IQ2_XS / IQ3_XXS packs: down type 42 in
`tests/data/native_experts/iq2_xs.txt`). Qwen3.6-35B-A3B packs have no Q2_0 layers and do not use this file.

## Tests

| Test | What |
|---|---|
| `expert_parity` (`src/kernels/cpu/expert_parity.cpp`) | VNNI vs `s2_expert_scalar` on a real blob from `pack/full/experts.bin`: <= 3e-3 with quantized activations; the FP32-activation gap is reported |
| `expert_multi_test` (`src/kernels/cpu/expert_multi_test.cpp`) | `s2_expert_vnni_multi` bitwise vs `s2_expert_vnni_q` for n = 1..8, synthetic blobs |
| `native_expert_parity` (`src/kernels/native_expert_parity.cpp`) | Exercises `q2_rows_any` / `act_quant_any` on native Q2_0 down rows |

Details and run conditions: [tests.md](tests.md).

## Gotchas

- In `act_quant_q8_1` the AVX-512 block returns unconditionally (`expert.cpp:395-420`), so the scalar loop after
  it (`expert.cpp:421-450`) is unreachable; it documents the rule the vector code reproduces bit for bit.
- The scalar oracle's `quant_acts` quantizes both stages; the upstream original quantized only the intermediate,
  which showed up as a ~1.2% gap that "read as a kernel bug" (`expert.hpp:164-167`, `expert.cpp:630-637`).
- `h2f` is written out by hand rather than `_cvtsh_ss` because of subnormal handling (`expert.cpp:26-50`).
- The header still says `bench/micro/cpu_s2.cpp` is the benchmark twin; `bench/` is not in this checkout.

## See also

- [native-experts.md](native-experts.md), [pool.md](pool.md), [expert-layout.md](expert-layout.md), [tests.md](tests.md)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) ("How it works")
