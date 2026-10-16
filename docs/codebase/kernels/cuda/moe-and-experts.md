[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › MoE and experts

# MoE: routing, experts, shared expert, native MMVQ

## Purpose

Everything in the MoE block that runs on the GPU:

- the router's softmax / top-k / renormalisation (the logits come from a BF16 MMVF, see
  [gemv-and-quant.md](gemv-and-quant.md));
- the routed experts held in VRAM: the expert cache and the prompt ring. Misses are computed by the CPU pool in
  `src/kernels/cpu/` (documented under [CPU kernels](../cpu/README.md));
- the shared expert with its scalar sigmoid gate;
- the final weighted combination;
- the pinned llama.cpp quantized matrix-vector product (MMVQ). The engine uses it for every native GGUF weight:
  dense projections, the output head and the experts.

Files: `src/kernels/cuda/router_top10.cu`, `native_router.cu`, `native_moe.cu`, `shared_expert.cu`,
`s2_expert_grouped.cu`, `iq_kernels.cu`, `native_mmvq.cu`, and headers `router_top10.hpp`, `native_router.hpp`,
`native_moe.hpp`, `shared_expert.hpp`, `s2_expert_grouped.hpp`, `iq_kernels.hpp`, `native_mmvq.hpp`.

## Quantization formats

All GGUF types use ggml's block layouts (`third_party/ggml/ggml-common.h`, included by `iq_kernels.cu:14-16`) and
llama.cpp's type ids. The activation side is always **q8_1**: 32 int8 values plus a `half2 {d, sum}` = 36 bytes, with
`d = amax / 127` and `sum` = the sum of the original float inputs (`native_mmvq.hpp:35-37`).

| id | Type | Block | Bytes | MMVQ path | Grouped gate/up | Grouped down |
| ---: | --- | ---: | ---: | --- | :---: | :---: |
| 2 | Q4_0 | 32 | 18 | `native_mmvq.cu` | | |
| 6 | Q5_0 | 32 | 22 | `native_mmvq.cu` | yes | yes |
| 7 | Q5_1 | 32 | 24 | `iq_kernels.cu` | | yes |
| 8 | Q8_0 | 32 | 34 | `native_mmvq.cu` | yes | yes |
| 11 | Q3_K | 256 | 110 | `native_mmvq.cu` | | |
| 12 | Q4_K | 256 | 144 | `native_mmvq.cu` | yes | |
| 13 | Q5_K | 256 | 176 | `native_mmvq.cu` | yes | yes (n_ff % 256 == 0) |
| 14 | Q6_K | 256 | 210 | `native_mmvq.cu` | yes | yes (n_ff % 256 == 0) |
| 16 | IQ2_XXS | 256 | 66 | `iq_kernels.cu` | yes | |
| 17 | IQ2_XS | 256 | 74 | `iq_kernels.cu` | yes | |
| 18 | IQ3_XXS | 256 | 98 | `iq_kernels.cu` | yes | yes (n_ff % 256 == 0) |
| 20 | IQ4_NL | 32 | 18 | `native_mmvq.cu` | | yes |
| 21 | IQ3_S | 256 | 110 | `iq_kernels.cu` | yes | |
| 22 | IQ2_S | 256 | 82 | `iq_kernels.cu` | yes | |
| 23 | IQ4_XS | 256 | 136 | `native_mmvq.cu` | yes | yes (n_ff % 256 == 0) |
| 29 | IQ1_M | 256 | 56 | `iq_kernels.cu` | yes | |
| 42 | Q2_0 | 64 | 18 (fp16 d + 16 B of 2-bit codes, value `(q - 1) * d`) | `native_mmvq.cu` | yes | yes |

The grouped columns are `STRATA_GU_FMTS` / `STRATA_D_FMTS` (`iq_kernels.cu:24-26`). The MMVQ column is the dispatch in
`native_mmvq` (`native_mmvq.cu:1599-1617`). Type 30 (BF16) is accepted only for the token embedding
(`embed_type_supported`, `iq_row_bytes`).

The S2 canonical form used by upstream's packs (codes packed 4 per byte LSB-first, one scale per 64) is described in
[gemv-and-quant.md](gemv-and-quant.md). The S2 expert blob stores those scales as fp16.

## Main entry points

### Router

| Function | Contract |
| --- | --- |
| `router_top10` (`router_top10.cu:422`) | Any `n_expert` up to 512 x 64 and `k <= 64`. One block per token with min(n_expert, 512) threads rounded up to a warp. Semantics of `ref/moe.py::router` (`router_top10.cu:9-13`): softmax over all experts (max by tree reduction; the 512 exps computed once, in double; the double sum on one thread in ascending order to keep the reference's last bits), then selection by rank `#{p_f > p_e} + #{f < e : p_f == p_e}` (stable descending, ties to the lower index), then renormalisation with ggml's `2^-14` lower clamp. A null stream synchronises |
| `native_router_top10` / `_multi` (`native_router.cu:110`, `:119`) | Pinned `topk-moe`: exactly 512 experts, top 10, F32 softmax, no bias, clamp `2^-14`, scale 1. One block of (32, 8) threads per token, with only warp 0 active (pinned geometry). Equal probabilities select the lower index |

The layer uses the native router only when `native_router_enabled() && n_expert == 512 && k == 10`
(`src/core/layer.cpp:371`, `src/core/verify.cpp:998`). Qwen3.6 (256 experts, top-8) therefore always uses
`router_top10`. `router_top10_variant` exists for HIP tests only.

### Combine

| Function | Arithmetic |
| --- | --- |
| `moe_combine` (`shared_expert.cu:363`) | `y = sum_k w[k] * parts[k] + shared`, accumulated in double in expert order, one thread per element (128-thread blocks). Shared is added plain, not router-weighted, and the weights arrive already renormalised (`shared_expert.hpp:81-100`) |
| `native_moe_combine` / `_multi` (`native_moe.cu:60`, `:77`) | Pinned `moe-weighted-reduction`: the first product rounded to F32, then `sum += p * w` in expert order, shared added once; k in [1, 15]; 256-thread blocks, grid y = token |

### Shared expert

`shared_expert` (`shared_expert.cu:242`) computes

```
h = silu(x @ gate.T) * (x @ up.T);  h = h @ down.T;  g = sigmoid(x . gate_inp);  return h * g
```

SiLU goes on the gate, and `ffn_gate_inp_shexp` is an `n_embd` vector giving one scalar per token
(`shared_expert.cu:3-14`). Each projection is computed by one of two paths:

- Canonical S-forms, chosen per weight by `SForm`: code_bits 2 uses `s2_gemv_q8` (Q8_0 activation); `act_kind == 1`
  uses `s_gemv_q8k_split` (Q8_K); anything else uses `s_gemv_q8_0_split`. The scalar gate is `scalar_gate_kernel`
  (`<<<1, 256>>>`, double reduction over BF16 activations), then `scale_kernel`.
- Native: each projection whose `NativeSharedWeights` entry has a type `native_mmvq_supported` accepts goes through
  `native_quantize_q8_1` + `native_mmvq`. The others fall back independently. Down reads the unrounded F32 SwiGLU,
  fused with its q8_1 quantization by `native_swiglu_quantize_q8_1` unless `STRATA_FUSED_SWIGLU_Q81=0`. With
  `shared_expert_set_native_bf16(true)` (`--native-bf16-extra`) the gate is `bf16_gemv_fp32_mmvf` on the F32 input
  plus an FP32 sigmoid.

`shared_expert_multi` (`shared_expert.cu:188`) is the verify-window form: 1..8 tokens, all three projections native
with multi-column MMVQ (the weights are read once), bitwise `shared_expert` per token. With the native BF16 gate and
`STRATA_DEC_BATCH` (default on) the gate is one multi-row MMVF plus a fused sigmoid-and-scale.
`shared_expert_scratch_bytes` sizes the caller-owned scratch. The comment at `shared_expert.hpp:44-48` records that
the kernel once called `cudaMalloc` per call, which blocked graph capture.

### Native GGUF experts (`iq_kernels.cu`)

The expert path for native packs: GGUF model files whose experts keep their own ggml types. The layout of one expert
blob is `NativeExpertLayout` (`iq_kernels.hpp:42-49`, built by `native_expert_layout`, `iq_kernels.cu:1928`):
`[gate rows | up rows | down rows]`, raw GGUF blocks.

`native_expert_grouped` (`iq_kernels.cu:2507`) evaluates grouped experts. Group g's blob sits at device address
`grp_ptr[g]`, which can be a VRAM cache slot or mapped host memory read over PCIe. Its entries are
`[grp_start[g], grp_start[g+1])`; each reads token `ent_tok[e]`'s q8_1 row and writes row `ent_dst[e]` of `out`.
Every count is read on the device. The steps:

1. Gate/up: `launch_gu<T>`. With a `Split<T>` (the i-quants, IQ4_NL/XS, Q2_0) it uses `native_gu_multi_kernel`
   (`:1139`), which decodes each weight word once and applies it to every entry of the group (decode once, bitwise
   equal to per-entry). Otherwise it uses `native_gu_kernel` (`:1109`), one dot per entry. The block is 256 threads =
   `GU_ROWS` 8 rows; the grid is `(2 n_ff / 8, gy)`, where `gy = grid_groups` block rows stride over the groups. The
   results do not depend on `gy`.
2. SwiGLU and q8_1: `swiglu_q8_1_entries_kernel` (`:1400`) in one pass over the call's own entries. The SwiGLU
   product is rounded on its own (`__fmul_rn`) so it cannot contract into the block sum. `STRATA_GROUPED_V1=1` (or
   `native_grouped_set_v1(true)`) uses the older two kernels over all `cap_entries`.
3. Down: `launch_down<T>`, `native_down_multi_kernel` / `native_down_kernel`, 8 rows per block. At n_ff 640 it uses
   32 rows for IQ4_NL (`row_dot_40_sub8`, 8-lane sub-warps) and 16 for Q2_0.

Scratch (`native_expert_scratch_bytes`, `:1942`) is `3 x cap x n_ff` floats plus `cap x n_ff/32` q8_1 blocks.
`native_expert_supported` (`:1922`) is checked for every layer at start-up. It needs `n_embd % 256 == 0` and
`n_ff % qk(down)`, so a 256-value down type needs n_ff 512 (Qwen3.6), not 640 (Flash-Next).

Other entry points in this file:

| Function | Use |
| --- | --- |
| `quantize_q8_1_rows` (`:1861`) | q8_1 activations (`quantize_q8_1_kernel`, 256 threads, `q8_1_finite` clamp) |
| `iq_mmvq` (`:1868`) | single-matrix MMVQ for the `STRATA_MMVQ_FMTS` types; `mmvq_kernel` (128 threads = 4 rows, one warp per row) or `mmvq_multi_kernel` with NC = 1, 2, 4, 8 columns |
| `iq_dequant_f16` / `_f32` (`:1882`, `:1907`), `iq_dequant_gu_f16` (`:1913`) | flat dequantization (32 threads per 256-value block); gate/up interleaved row 2r = gate r, 2r+1 = up r for the prompt path's FP16 GEMMs |
| `iq_embed_rows` (`:1898`) | token-embedding rows (any `embed_type_supported` type, the table may be mapped host memory) to F32 |

`iq_set_old_kernels(true)` / `STRATA_OLD_IQ_MMVQ=1` selects the per-column kernels, which are bitwise the same.
`STRATA_IQ_STAGE_GRID=0` disables staging the 64-bit codebook grids (IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M) in shared memory.
`STRATA_IQ_STAGE_GRID_MMVQ=1` also stages them for the single-matrix MMVQ, which measured 10-20% slower on an RTX 5070
(`iq_kernels.cu:1772-1779`). `native_expert_set_mode` and `STRATA_EXP_MODE` are AMD (gfx906) bench layouts.

### Pinned MMVQ (`native_mmvq.cu`, `--use_fast_math`)

`native_mmvq(type, w, x_q8_1, y, n_in, n_out, ncols, stream)` (`native_mmvq.cu:1599`) is llama.cpp's generic MMVQ at
3cf03257 for Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS, with exact `vec_dot_*_q8_1`
expressions and integer dot order. It forwards Q5_1 and the 256-value i-quants to `iq_mmvq`. Per type there is
`native_<type>_mmvq` (q8_1 input) and `native_<type>_f32`, which quantizes into caller scratch first.

- Launch: 4 warps (128 threads) per output row, with the cross-warp reduction in shared memory. When `n_in` is too
  short to fill the warps, 2 rows per block (`SmallK`).
- Columns: `ncols` 1..8. `native_mmvq_set_multi_exact(true)` (the default, `native_mmvq.cu:1027`) runs ncols > 1 with
  the ncols = 1 layout, so every column is bitwise a single-column call (`native_mmvq_multi_kernel`, `:1031`). `false`
  uses llama.cpp's multi-column table (4 or 2 warps, 2 rows per block), which equals it only to rounding.
- `native_quantize_q8_1` (`:1255`) and `native_swiglu_quantize_q8_1` (`:1269`): one warp per 32-value block.
- `native_mmvq_weight_bytes` and `native_q8_1_bytes` validate sizes. Bad shapes throw. Weights must be 4-byte aligned
  and are read in place. Q2_0 blocks are only 2-byte aligned, so their codes are loaded as `int16_t`
  (`native_mmvq.cu:277-279`).

Callers: `src/core/native_dense.cpp` (dense GDN/QSA projections), `src/core/layer.cpp`, `shared_expert.cu`,
`ple.cu` (native PLE key), `src/core/native_head.cpp` (output head).

### S2 GPU expert tier (`s2_expert_grouped.cu`)

This is upstream's expert cache for S2 packs (Q2_0 experts in the canonical form). It is hard-coded to Flash-Next's
geometry: `H = 2560`, `FF = 640` (`s2_expert_grouped.cu:32-33`). The blob holds gate/up codes interleaved by row
(2r gate, 2r+1 up), then down codes, then fp16 gate/up scales, then fp16 down scales. One warp per output row computes

```
sum_j (c_j - 1) d_w xhat_j d_x  =  d_w d_x (sum_j c_j xhat_j - sum_j xhat_j)
```

with two `__dp4a` chains per 32-element chunk: codes against x, and ones against x
(`s2_expert_grouped.cu:5-10`). The Q8_0 activation comes from `quantize_q8_0`, or with fp32 scales from
`quantize_q8_0_scaled`. The fp32 scales (`x_scales`) exist so that a cache hit and a CPU miss produce the same number;
fp16 `d` disagreed by 4.761e-04 relative.

| Function | Use |
| --- | --- |
| `moe_hit_grouped_s2` (`:579`) | n_hits experts from a slot arena, `dst_index` placing each on its routed row; 4 launches per layer (gate/up, SwiGLU, quantize, down) |
| `moe_hit_select` / `_dev` / `moe_hit_add` (`:676`, `:683`, `:1138`) | device-built hit list from routed ids and a residency row; capacity-sized grid with a device count (the P4 token graph) |
| `moe_hit_select_multi`, `moe_hit_grouped_s2_multi` (`:713`, `:720`) | verify window: per-token activations, up to 128 routed entries |
| `moe_grouped_s2` (`:1099`), `moe_group_resident` (`:1090`) | grouped by expert (each blob row read once for all its entries); the MTP layer's 512 resident experts |
| `moe_hit_grouped_s2_cpu_order` (`:1146`) | experimental: reproduces the CPU VNNI accumulator order (`--expert-cache-cpu-order`) |

`STRATA_OLD_GROUPED=1` / `moe_grouped_select_old` selects the previous kernels (bitwise equal). `STRATA_GROUPED_PAIR_MIN_HITS=N`
keeps them below N hits. Misaligned activations, scratch, arena or slot sizes also fall back to them
(`new_grouped` / `new_hit`, `:518-529`). `moe_grouped_last_path` reports which path ran, for the parity test.

## Control and data flow

```
mixed --BF16 MMVF--> logits --router_top10 | native_router_top10--> ids, weights
   --> doorbell_publish (elementwise.cu): x, ids, weights to mapped host memory for the CPU pool
   --> VRAM hits: native_expert_grouped (native packs) | moe_grouped_s2 / moe_hit_grouped_s2_dev (S2 packs)
   --> shared_expert | shared_expert_multi
   --> moe_combine | native_moe_combine  -> block output
```

Callers: `src/core/layer.cpp:360-475` (decode), `src/core/verify.cpp:998-1163` (verify window), `src/core/mtp.cpp`
(MTP drafter), `src/core/session.cpp:866` (`moe_hit_grouped_s2_dev`), and `src/core/peer_experts.cpp` /
`remote_experts.cpp` (`native_expert_grouped` for experts held by another process or GPU). The prompt path dequantizes
experts with `iq_dequant_gu_f16` / `dequant_f16` for cuBLAS GEMMs (`src/prefill/prefill.cpp:3048-3052`).

## Configuration

- CLI: `--native-router`, `--native-moe-combine`, `--native-bf16-extra` (shared gate), `--native-dense-gguf`
  (native projections, including the shared expert's), `--expert-cache-cpu-order`; `--native SHARD1` sets all but
  the last.
- Environment: `STRATA_OLD_IQ_MMVQ`, `STRATA_IQ_STAGE_GRID`, `STRATA_IQ_STAGE_GRID_MMVQ`, `STRATA_GROUPED_V1`,
  `STRATA_OLD_GROUPED`, `STRATA_GROUPED_PAIR_MIN_HITS`, `STRATA_FUSED_SWIGLU_Q81`, `STRATA_DEC_BATCH`; HIP only:
  `STRATA_MMVQ_WAVE`, `STRATA_EXP_MODE`, `STRATA_HIP_SWIGLU_FUSED`, `STRATA_ROUTER_BLOCK`, `STRATA_HIP_ROUTER_FAST`,
  `STRATA_HIP_ROUTER_OLD`.
- All `*_set_*` switches must be set before graph capture, and captured graphs keep their kernels.

## Model-specific behaviour

| | Flash-Next | Qwen3.6 |
| --- | --- | --- |
| Experts / top-k | 512 / 10 (`native_router` eligible) | 256 / 8 (`router_top10` only) |
| n_ff | 640: down types with a 32- or 64-value block only (IQ4_NL, Q2_0, Q5_1, Q5_0, Q8_0) | 512: also the 256-value IQ3_XXS, IQ4_XS, Q5_K, Q6_K as down |
| Packs | S2 (`s2_expert_grouped`) or GGUF IQ2_XS / IQ3_XXS / IQ3_S / Q2_0 / UD-Q4_K_XL | GGUF UD-IQ3_XXS (IQ2_S gate/up, IQ3_XXS / IQ4_XS down), UD-Q4_K_XL (Q4_K / Q5_K / Q6_K) |

The Qwen3.6 port added Q6_K (`vec_dot_q6_K_q8_1`, VDR 1, and `dq_q6_k`) and put IQ3_XXS, Q5_K and Q6_K on the down
list (`iq_kernels.cu:20-26`).

## Tests

- `iq_parity` (i-quant MMVQ against ggml fixtures generated at configure time), `iq_multi_parity` (decode-once vs
  per-column, bitwise), `mmvq_multi_parity` (native MMVQ multi-column; a second run with `STRATA_MMVQ_WAVE=0`),
  `native_expert_parity` (one expert three ways, real or `--synthetic` rows), `native_grouped_parity` (grouped vs
  per-entry, v1 vs current, `grid_groups` invariance).
- `s2_expert_grouped_parity` (new vs previous S2 kernels, bitwise; `--bench`).
- `shared_expert_parity`, `router_top10_parity` (five logit distributions including exact ties, at 512/10 and 256/8; asserts the `2^-14` clamp cannot trigger at either geometry), `route_window_parity`
  (batched routing vs per-token, bitwise), `qwen36_seam_parity`, `q5_projection_parity` (Q5_1 shared projection vs
  ggml).
- Oracle-linked (sources not in this repository): `native_mmvq_parity`, `native_shared_parity`, `native_moe_parity`,
  `native_router_parity`.

## Gotchas

- `native_swiglu_quantize_q8_1` stores `make_half2(d, sum)` without the `q8_1_finite` clamp that
  `native_quantize_q8_1` and `quantize_q8_1_kernel` apply (`native_mmvq.cu:156-170`). An overflowing block there
  would still become inf.
- `router_top10` with a null stream synchronises the device. The engine always passes a stream.
- `s2_expert_grouped` literals must match `include/strata/kernels/cpu/expert.hpp`'s blob geometry. They are restated
  on purpose (`s2_expert_grouped.cu:29-31`), and the CPU-vs-GPU parity (`moe_hit_parity`) lives in `bench/micro/`,
  which is not in this repository.
- The grouped native kernels' launch grid is sized by capacity. A call with no groups still launches blocks that
  return at once (`grid_groups` reduces that).

## See also

- [gemv-and-quant.md](gemv-and-quant.md) (q8_0 / Q8_K quantizers, `dp4a.hpp`, `q8_1_finite.hpp`, BF16 MMVF),
  [sampling-and-verify.md](sampling-and-verify.md) (`resident_plan`, `fetch_blobs`, `ident_hits`)
- [CPU kernels](../cpu/README.md), [Core experts](../../core/experts/README.md)
- `docs/qwen36-port/step3-experts.md`, `docs/QWEN36-35B-A3B.md` section 3.4
