# Qwen3.6 port, Phase 2 step 3, experts: the native GPU expert kernels' format coverage

Working notes (2026-10-11, branch `qwen36-step3-experts`, from `qwen36-step2` at 8b7525e). The gap: the study's
section 3.7 called the native pack path "otherwise runtime", but `native_expert_supported()` refused every layer of
both real files because the DOWN-projection list `STRATA_D_FMTS` (`src/kernels/cuda/iq_kernels.cu`) held only the
types Flash-Next's and the Coder's packs use (IQ4_NL, IQ4_XS, Q2_0, Q5_1, Q5_0, Q8_0). Qwen3.6's files put IQ3_XXS
(37 layers) and IQ4_XS (34, 38, 39) on `ffn_down_exps` in UD-IQ3_XXS, and Q5_K (37) and Q6_K (34, 38, 39) in
UD-Q4_K_XL, whose layer 39 gate/up is Q5_K (`tests/data/native_experts/qwen36-{iq3_xxs,q4_k_xl}.txt`, step2-notes.md
section 3). The engine's refusal at `generate.cpp:2100` named the pair; it still does for anything not below.

## 1. What changed, and why it was small

| Decision | Where | Why |
| --- | --- | --- |
| IQ3_XXS (18) and Q5_K (13) join `STRATA_D_FMTS` | `iq_kernels.cu`, the three `STRATA_*_FMTS` lists | `native_down_kernel<TD>` and `native_down_multi_kernel<TD>` were already generic in `Fmt<TD>` (one row per warp, `row_dot<TD>` lane-strided over the row's dot calls); the list was the only restriction. The 256-value block types need `n_ff % 256 == 0`, which `native_expert_supported` already demanded: Qwen3.6's 512 passes, Flash-Next's 640 is refused as before (`native_grouped_parity.cpp` had noted "IQ4_XS: n_ff % 256") |
| Q6_K (14): `vec_dot_q6_K_q8_1` and `dq_q6_k` transcribed from llama.cpp (`vecdotq.cuh`, `dequantize.cuh` at the pinned commit), `Fmt<14>` with `VDR_Q6_K = 1` (`ipb = QI6_K / 1 = 32`, `step = 1`), on all three lists, in `is_iq` and `iq_row_bytes` (210 B per block) | `iq_kernels.cu` (the K-quant section after Q5_K, the dequant section after `dq_q5_k`) | 16 int8 sub-scales per block, the value `(ql | qh << 4) - 32` by `__vsubss4`, no min term, `__dp4a` only: native on sm_61, nothing Pascal lacks. The HIP shim already defines `__vsubss4`. Q6_K takes the per-entry kernels (no `Split<14>`), as Q4_K / Q5_K do |
| `native_expert_supported` unchanged | | its block-divisibility rule was right; only the type tables were short |
| Q6_K in the MMQ prompt path under `STRATA_MMQ_KQUANTS` | `src/prefill/moe_mmq.cu`, the CMake instance list (`[shared]` commit 52b5dcd) | llama.cpp ships `mmq-instance-q6_k.cu`; the Q4_K_XL file's three Q6_K layers then take the int8 tiles rather than the fp16 GEMM over dequantized weights. The option defaults OFF: the step-5 build must turn it on for the Q4_K_XL file |

Not changed: `moe_fused_iq.cu` (the sm_80+ fused native prompt experts: gate/up IQ2_XXS..IQ4_XS, down Q2_0 / IQ4_NL
only; the new types need a load stage of their own and its geometry is the prefill agent's), the CPU native path
(`native_fmt` is ggml-generic and takes Q6_K through ggml-cpu's `vec_dot`; `kq256_supported` keeps its Q4_K / Q5_1 /
Q8_0 AVX-2 kernels, Q5_K and Q6_K rows go to ggml's), `generate.cpp`.

## 2. Interface other agents consume

- `bool strata::kernels::native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff)`: true for
  every (gate/up, down) pair of both files at 2048 x 512: (22, 18), (22, 23), (12, 13), (12, 14), (13, 14). Gate/up
  types: 16, 17, 18, 21, 22, 23, 29, 42, 12, 13, 14, 6, 8; down types: 20, 23, 42, 7, 6, 8, 18, 13, 14, the 256-value
  ones only where `n_ff % 256 == 0`.
- `native_expert_grouped(...)`, `native_expert_layout(...)`, `native_expert_scratch_bytes(...)`,
  `iq_dequant_f16 / _f32 / _gu_f16`, `iq_embed_rows`, `iq_mmvq`: signatures unchanged; type 14 accepted everywhere
  `is_iq` gates.
- `strata::prefill::mmq::supported(14)`: true when the build has `STRATA_MMQ_KQUANTS=ON`, as for 12, 13, 7.
- `native_expert_parity [--geom HxFF] ...`: the geometry flag; `native_grouped_parity --bench-down`.

## 3. How it is checked

`src/kernels/native_expert_parity.cpp` (a CTest per pair): (a) the float reference is ggml's own `to_float` of the
blob (the dequantization `oracle-crosscheck.md` cross-checked bit for bit against `ref/load.py` on the real file)
times a double-precision SwiGLU expert on 3 random tokens; (b) the CPU path (ggml-cpu `vec_dot`, Q8_K activations);
(c) the GPU path (`native_expert_grouped`, q8_1 activations), once per kernel family (decode-once vs per-entry, bitwise);
(d) the GPU dequantizers against `to_float`; (e) the negative control. Tolerances (relative L1): GPU against (a)
2e-2, the Q8-activation bound of `oracle-crosscheck.md` "Tolerance policy" (the Q8 oracle's `ffn_moe_out` max 5.3e-3
with the 4x margin), 3e-2 only for a pair with a Q5_0 role (32-value blocks without a min: q4_K/q5_0 measures 2.7e-2
at 2560 x 640, the pre-existing pair the CMake note describes); CPU against (a) 3e-2 (ggml-cpu rounds the activations
to Q8_K, one scale per 256 values, against the GPU's q8_1 per 32: measured up to 2.2e-2); CPU against GPU 4e-2 (both
roundings, measured up to 2.4e-2). Qwen3.6's five pairs measure 9.7e-3 to 1.3e-2 on the GPU (section 4).

The negative control (e), per the brief and `upstream-tooling.md` section 3: the high byte of one block's fp16 scale
(its sign) flipped in down row 5 must move that row's GPU output by more than the gate and no other row by any bit
(row locality: a flipped byte in row r's block reaches only row r); the same in gate row 3 must move the output by
more than 1e-4 (one of n_ff SwiGLU terms). The gate is 1e-2 for a 256-value block of a 512-wide row (Qwen3.6's down
types) and scales with the block's share of the row for the others (`1e-2 x qk/256 x 512/n_ff`: 1e-3 for a 32-value
block at Flash-Next's 640, where one of 20 blocks moved the row by 8e-3 to 1.4e-2 on the 4090). The Q6_K scale sits
at the block's last two bytes (`ggml_type_size(Q6_K) - 2`), the others at byte 0.

`src/kernels/native_grouped_parity.cpp`: every gate/up format with every down format at 512 x 256 (96 pairs, the
grid-stride launches bitwise equal to v1's over 84 call shapes each), the five Qwen3.6 pairs at 2048 x 512, plus
`--bench-down`. `src/kernels/iq_multi_parity.cpp`: the IQ3_XXS down pair at 1024 x 512 (decode-once vs per-column,
bitwise). `tests/cuda/prefill_mmq_kquant_test.cpp`: Q5_K and Q6_K down products at 2048 x 512 through MMQ against a
double-precision product of ggml's dequantized weights (its screen: rel L2 < 0.04).

Real file: `native_expert_parity --geom 2048x512 <file> 0 34 39` takes expert 7 of `blk.{0,34,39}` straight from the
GGUF (the three tensor slices into one blob) through (a) to (e). Layers 0 and 39 are the brief's; 34 adds the third
pair of each file.

## 4. Results, RTX 4090 (VM 221, sm_89, CUDA 12.4, gcc-13, `-DSTRATA_MMQ_KQUANTS=ON`)

Build: `/models/work/build-step3-experts.sh configure` then `targets` (116 s for the five test targets).

    cd /models/work/qwen36-step3-experts/build
    ctest -R "native_expert_parity|native_grouped_parity|iq_multi_parity|prefill_mmq_kquant|^iq_parity"
    ./native_expert_parity --geom 2048x512 /models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf 0 34 39
    ./native_expert_parity --geom 2048x512 /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf 0 34 39
    ./native_grouped_parity --bench-down

ctest: 19 of 19 pass (15 `native_expert_parity_*` including the five `qwen36_*` pairs, `q6_K_q8_0` at 2560 x 640 and
the `refuses_q6_K_down_at_640` WILL_FAIL; `native_grouped_parity`, `iq_multi_parity`, `iq_parity`,
`prefill_mmq_kquant_test`).

Synthetic pairs at 2048 x 512 (relative L1 against the float reference; the negative control's down-row move, with
0 other rows touched in every case, and the gate-row move):

| pair | cpu | gpu | cpu-gpu | GPU dequant vs `to_float` | down row flipped | gate row flipped |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| iq2_s / iq3_xxs | 1.7e-2 | 9.7e-3 | 1.8e-2 | bit-exact | 1.8e-1 | 3.5e-4 |
| iq2_s / iq4_xs | 1.8e-2 | 1.2e-2 | 2.0e-2 | bit-exact | 5.8e-1 | 1.2e-3 |
| q4_K / q5_K | 1.7e-2 | 1.0e-2 | 1.9e-2 | bit-exact | 1.5 | 2.5e-3 |
| q4_K / q6_K | 2.2e-2 | 1.3e-2 | 2.4e-2 | bit-exact | 1.5e-1 | 1.3e-3 |
| q5_K / q6_K | 2.2e-2 | 1.2e-2 | 2.3e-2 | bit-exact | 1.8e-1 | 3.3e-3 |

(The table's first three rows are the real-file layers 0 / 34 / 39 of UD-IQ3_XXS and the q4_K / q5_K row is layer 0 of
UD-Q4_K_XL; the synthetic runs of the same pairs are within 1e-3 of them. The decode-once and per-entry kernels are
bitwise equal on every pair; the CPU path's width invariance holds.)

Real files, both: 0 failures. UD-IQ3_XXS layers 0 (iq2_s/iq3_xxs), 34 and 39 (iq2_s/iq4_xs): gpu 9.7e-3, 1.2e-2,
1.1e-2. UD-Q4_K_XL layers 0 (q4_K/q5_K), 34 and 39 (q4_K/q6_K): gpu 1.0e-2, 1.2e-2, 1.1e-2. Every layer's GPU
dequantization of all three matrices equals ggml's `to_float` bit for bit.

The down kernel alone (`--bench-down`: phase 2, 48 calls in a graph, each layer its own 32 experts streamed from
DRAM; medians of 40 launches), microseconds per expert at 2048 x 512:

| down type | MB / expert | 32 groups x 1 entry | 8 groups x 4 entries |
| --- | ---: | ---: | ---: |
| Q5_1 (the comparator) | 0.786 | 0.92 | 1.73 |
| Q8_0 | 1.114 | 1.23 | 2.55 |
| IQ4_XS | 0.557 | 0.98 | 1.97 |
| IQ3_XXS | 0.401 | 0.96 | 1.88 |
| Q5_K | 0.721 | 0.86 | 1.99 |
| Q6_K | 0.860 | 0.96 | 2.53 |

On this card the 32 x 1 row is bandwidth-bound for every type (25 to 36 MB per 29 to 39 us call, 0.85 to 0.92 TB/s),
so the new types cost what their bytes cost; the 8 x 4 row shows the per-entry kernels' cost (Q6_K, Q5_K: no
`Split<>`, the dot redone per entry) against the decode-once ones (IQ3_XXS, IQ4_XS). The 1080 Ti numbers (section 5)
set the step-5 budget.

## 5. Results, GTX 1080 Ti (VM 211, sm_61, CUDA 12.4, gcc-13)

`/home/rk/qwen36-step3-experts/run-vm211.sh` under the lock (2026-10-11, 20:02 UTC): configure with
`-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DSTRATA_MMQ_KQUANTS=ON` (5 s), the five test targets
(149 s at `-j 8`), then the same three commands as section 4 with `/home/rk/models/` for the files; logs in its
`logs/`. No guard was needed: the new kernels use `__dp4a`, `__vsubss4`, `__byte_perm` and fp16 conversions only.

- ctest: 20 of 20 pass (the 15 `native_expert_parity_*`, `native_grouped_parity`, `iq_multi_parity`, `iq_parity`
  with its fixture, `prefill_mmq_kquant_test` through llama.cpp's dp4a MMQ tiles on cc 61).
- Real files, both 0 failures, the same numbers as the 4090 to the printed digits (the kernels are integer dp4a
  chains with one float expression per block; the q8_1 rounding is the same): UD-IQ3_XXS layers 0 / 34 / 39 gpu
  9.7e-3 / 1.2e-2 / 1.1e-2, UD-Q4_K_XL layers 0 / 34 / 39 (q4_K/q5_K, q4_K/q6_K, q5_K/q6_K) 1.1e-2 / 1.2e-2 /
  1.2e-2; GPU dequantization bit-exact for every matrix; the negative control's down-row move 1.8e-1 to 1.6 with no
  other row touched, the gate-row move 3.5e-4 to 3.4e-3.
- `native_grouped_parity`: all 96 pairs at 512 x 256 and the seven model shapes bitwise equal to v1.

The down kernel alone per expert at 2048 x 512 (`--bench-down`, SM 1873 MHz, memory 5508 MHz, 46 C), the step-5
budget:

| down type | MB / expert | 32 groups x 1 entry | 8 groups x 4 entries (per group) |
| --- | ---: | ---: | ---: |
| Q5_1 (the comparator) | 0.786 | 4.43 | 14.3 |
| Q8_0 | 1.114 | 7.58 | 25.3 |
| IQ4_XS | 0.557 | 6.58 | 12.1 |
| IQ3_XXS | 0.401 | 6.78 | 10.8 |
| Q5_K | 0.721 | 4.29 | 12.2 |
| Q6_K | 0.860 | 5.32 | 15.6 |

Against Q5_1 at one entry per expert: Q5_K 0.97x, Q6_K 1.20x, IQ3_XXS 1.53x, IQ4_XS 1.49x. Pascal is not
bandwidth-bound here (Q5_1: 25 MB per 142 us call, 178 GB/s of the card's 484): the i-quant down types pay their
grid lookups and sign unpacking per call, so IQ3_XXS costs 1.5x Q5_1's time for half its bytes, while the K-quants'
plain nibble chains run at Q5_1's speed. At four entries per expert the decode-once kernels (IQ3_XXS, IQ4_XS) amortise
and come out ahead of the per-entry Q5_K / Q6_K. For a window of 4 tokens x top-8 routed to 32 distinct experts the
down projection is 137 to 217 us per layer, 5.5 to 8.7 ms over 40 layers; the gate/up (twice the rows at the wider
n_embd) is the larger half and was not re-timed here (the Flash-Next numbers in `native_grouped_parity --bench` stand).

## 6. Surprises

- `prefill_mmq_kquant_test` had a latent race: its sentinel memset on the legacy default stream was not ordered
  before the product on its non-blocking stream; the second product at 2048 x 512 read as entirely unwritten on the
  4090 and passed under `CUDA_LAUNCH_BLOCKING=1`. Fixed in the test (`[shared]` commit); not an MMQ fault.
- The negative control's 1e-2 gate, stated for Qwen3.6's 256-value blocks, is above what a single 32-value block's
  sign flip moves a 640-wide Flash-Next row (8e-3 to 1.4e-2): hence the block-share scaling.
- The study's 3.7 was right that the path is runtime per layer and that `native_experts.txt` carries the types; it
  missed that the GPU kernels' *type lists* did not include the files' down types, which is why "otherwise runtime"
  did not hold for either file.
