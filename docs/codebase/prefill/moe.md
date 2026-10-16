[Codebase docs](../README.md) › [Prefill](README.md) › MoE

# The MoE half of a prompt chunk

## Purpose

Half 1 of every layer: route each token to its top-k experts, run the shared expert, multiply every routed
`(token, k)` row by its expert's gate/up and down matrices, and combine. This page covers the selection among four
routed-expert implementations and the data each needs. The expert bytes come from a resident cache slot or a ring
slot, see [expert-streaming.md](expert-streaming.md); the MMQ kernels themselves are covered in
[ggml-bridge.md](ggml-bridge.md).

Files: [src/prefill/prefill.cpp](../../../src/prefill/prefill.cpp) (`:2498-3152`, selection and orchestration),
[src/prefill/kernels.cu](../../../src/prefill/kernels.cu) (router, SwiGLU, combine, Q2_0 blob dequant),
[src/prefill/moe_fused.cu](../../../src/prefill/moe_fused.cu),
[src/prefill/moe_fused_iq.cu](../../../src/prefill/moe_fused_iq.cu),
[src/prefill/moe_mmq.cu](../../../src/prefill/moe_mmq.cu), and the headers
[moe_fused.hpp](../../../include/strata/prefill/moe_fused.hpp),
[moe_fused_iq.hpp](../../../include/strata/prefill/moe_fused_iq.hpp),
[moe_mmq.hpp](../../../include/strata/prefill/moe_mmq.hpp).

## Router and shared expert (all paths)

1. `ffn_gate_inp` (BF16) times `mixed_bf` -> `logits [T, n_expert]` (`prefill.cpp:2507`).
2. `route` (`kernels.cu:1016`): softmax over all experts, top-k by rank with ties to the lower id, the k weights
   renormalised with a `2^-14` floor on the sum (the native router's semantics). 512 and 256 experts use the warp
   kernel `route_kernel<16|8>` (`kernels.cu:684`); any other count `strata::kernels::router_top10`. `k` is a runtime
   parameter (10 for Flash-Next, 8 for Qwen3.6). Writes `ids [T, k]` and `w [T, k]`.
3. Shared expert: `ffn_gate_shexp`, `ffn_up_shexp` (FP16 GEMMs over dequantized native weights), `swiglu_pair`,
   `ffn_down_shexp`; its scalar gate `ffn_gate_inp_shexp` (BF16, `N = 1`) -> `sg [T]` (`prefill.cpp:2511-2517`).
4. After the routed products: `moe_combine` (`kernels.cu:772`):
   `bo[t] = shared[t] * sigmoid(sg[t]) + sum_j w[t, j] * Dm[slot[t, j]]`, summed in rank order like the decode
   combine and llama.cpp.

The router multiplies BF16-rounded activations, the decode router FP32 ones; near-tie experts can swap ranks between
the two paths (the Qwen3.6 harness allows a swap only within 2e-3 in the oracle's softmax). `STRATA_PREFILL_BF16X2`
narrows this by adding the activation's BF16 remainder as a second product.

## Which routed-expert path a layer takes

From `prefill.cpp:2525-2533`:

```
use_mmq   = mmq_plan().any && mmq_plan().layer[l]            // built, STRATA_PREFILL_MMQ != 0, both types fit
fused_nat = use_mmq && stream_all && no_peer && lay.native && fused::native_supported(gu_type, d_type)
fused_l   = (use_mmq && stream_all && no_peer && !lay.native && fused::enabled()) || fused_nat
```

| Path | Taken when | Weight formats |
| --- | --- | --- |
| Fused Q2_0 int8 (`moe_fused.cu`) | Strata Q2_0 pack (not native), streamed walk (chunk >= 1024), no `--peer-device`, sm_80+, `STRATA_PF_FUSED` not `0` (default ON since 0.1.36) | the Strata Q2_0 blob at Flash-Next's widths only |
| Fused native int8 (`moe_fused_iq.cu`) | native pack, streamed walk, no peer, sm_80+, `STRATA_PF_FUSED=1` explicitly, `STRATA_PF_FUSED_NATIVE` not `0`, the layer's pair covered | gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0, IQ4_NL |
| MMQ (llama.cpp `mmq.cuh`) | otherwise, when `mmq_plan` covers the layer | Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0; Q4_K, Q5_K, Q5_1, Q6_K with `STRATA_MMQ_KQUANTS`; Q5_0 with `STRATA_ORCA_Q4KS_MMQ` |
| FP16 GEMM over dequantized experts | a layer MMQ does not cover (IQ1_M, a type the build lacks, a tile that does not fit the card's shared memory, #420), `STRATA_PREFILL_MMQ=0`, or a build without the ggml sources | anything `iq_dequant_*` / `blob_dequant_f16` decode |

`mmq_plan()` (`prefill.cpp:729`) is computed once per process from `expert_layout()`: per layer, MMQ if
`mmq::fits(gu_type, 2*n_ff)` and `mmq::fits(d_type, n_embd)`, using the native pack's own widths (Qwen3.6
2048 / 512) or the Q2_0 pack's 2560 / 640; `fallback` is set when any layer is not covered, which also carves the FP16
path's buffers (`Xs`, `Hh`). For the canonical Strata Q2_0 pack the type id used is 42, the ggml Q2_0 type its
blob is converted to on the gather (the llama.cpp submodule is not checked out here, so the enum value itself was not
verified against ggml.h; `include/strata/kernels/cpu/expert_layout.hpp:65` uses the same 42 for Q2_0).

**Qwen3.6.** Its packs are native (UD-IQ3_XXS: IQ3_XXS / IQ4_XS down projections; UD-Q4_K_XL: Q4_K / Q5_K gate/up,
Q5_K / Q6_K down, per `docs/qwen36-port/step3-experts.md`). None of those down types is in the fused native
kernels' list, so `fused_ring()` is false and every Qwen3.6 MoE layer runs MMQ; UD-Q4_K_XL needs a build with
`STRATA_MMQ_KQUANTS=ON` (the `Makefile` sets it), else those layers fall back to the FP16 GEMM. `moe_fused.cu` is the
Q2_0 blob format of Flash-Next and is never reached by Qwen3.6.

## Path 1: host grouping + MMQ (`prefill.cpp:2595-3121`)

1. Copy `ids` to the mapped host table with a copy kernel (`copy_i32`) and synchronize the compute stream: the one
   host sync per MoE layer. `STRATA_PREFILL_TIMING` folds its event marks here.
2. Count rows per expert, assign each `(token, k)` pair a row `slot[i]` grouped by expert (`off[e]` prefix sums),
   `src[row] = token`. With a peer GPU, peer-held and peer-streamed experts' rows are placed last as one block.
3. `mmq::quantize(mixed, src, Xq, gu_type, ...)`: the layer's activations as q8_1 rows in expert order.
4. Bounds: absolute per expert (gate/up reads the layer's rows) and relative per 16-expert group (down reads the
   group's own H), uploaded via the mapped table.
5. Experts in id order (`order`), 16 per group (`MMQ_GROUP`): each expert is gathered into the group buffer
   (`gather_native` / `gather_native_group` for native GGUF slices, `gather_strata_q2` converting the Strata Q2_0 blob
   to GGUF Q2_0 blocks); after the 16th (or last) the group's products run: zero a 4096-byte tail after the last
   expert (MMQ reads up to one 256-value tile past a row), gate/up `mmq::Context::run`, `mmq::swiglu` (interleaved
   for the Strata pack, split for GGUF), `mmq::quantize` of H, down into `Dm` rows (`prefill.cpp:2996-3042`).
6. A ring slot is released (its `used` event) once gathered, so the copy engine can refill it before the products
   finish.

## Path 2: FP16 GEMM fallback (`prefill.cpp:3044-3061`)

`gather_rows16(mixed_h, src, Xs)` once per layer, then per expert: dequantize gate/up and down to FP16 into one of two
buffers (`DQ = 2`) with `iq_dequant_gu_f16` / `iq_dequant_f16` (native) or `blob_dequant_f16` (Q2_0), then two
`Gemm::f16` products with `swiglu_interleaved` between them. The SwiGLU output is saturated to +-65504 so an extreme
activation cannot become inf / NaN in the FP16 down product (`hf_sat`, `kernels.cu:46-49`).

## Path 3: fused Q2_0 int8 (`moe_fused.cu`; `prefill.cpp:2536-2594`)

No host sync. Per layer:

- `fused::quantize_act(mixed, T, N, Xq)`: int8 per 32 values once per token (not per slot), 80 bytes per 64 values:
  64 codes in a permuted K order plus `{d0, c0, d1, c1}` where `c = -(1.5 * 2^23 + sum of codes)` (`moe_fused.cu:71`).
- `fused::group(ids, ...)`: counts (shared-memory histogram), an exclusive scan, and placement into
  `slot` / `src` plus a list of 64-row tiles, all on the GPU (`moe_fused.cu:431`). The row order inside an expert is
  not fixed; no output depends on it.
- Experts in id order in batches of at most `kMaxBatch = 128` and at most a third of the ring of streamed ones: each
  expert's blob pointer is its ring slot or its cache slot (no gather). One wait on the batch's last copy.
  `fused::experts` launches `expert_kernel<true>` (gate/up with SwiGLU and int8 H in the epilogue) then
  `expert_kernel<false>` (down into `Dm`) (`moe_fused.cu:449`).
- Arithmetic (`moe_fused.cu:1-18`): Q2_0 is `w = d * (q - 1)`, so `sum w x = d_w d_x (sum q a - sum a)`; the 2-bit
  codes are unpacked in registers and multiplied with `mma.sync m16n8k32` s8 x s8 -> s32; the int dot becomes a float
  exactly via the `0x4B400000` magic plus the stored `c`. 512 threads, 256 weight rows x 64 tile rows per work item, a
  4-stage cp.async pipeline.
- Availability (`dev_info`, `moe_fused.cu:360`): sm_80+, device code present (PTX >= 80), and at least one block per
  SM fits; otherwise MMQ. Never on gfx906.

`fused_layout` shrinks `GU`, `H`, `Xq`, `Hq` to the fused path's needs (`prefill.cpp:760-774`), which is what let the
ring grow to 1024 slots on the Q2_0 pack. The numbers differ from MMQ's by about as much as MMQ differs from FP32.

## Path 4: fused native int8 (`moe_fused_iq.cu`)

Same grouping, batches and epilogues as path 3; the load stage decodes i-quant blocks to int8 in shared memory
(codebooks and sign tables in shared memory, up to 8 KB for IQ2_S) with the block's scale per 32 values (per 16 for
IQ2_XS / IQ2_S, multiplied with `m16n8k16`). Activations and H are int8 per 32 in natural order
(`quantize_act_native`). The work item is 256 x 64 or 128 x 128 (weight rows x routed rows), chosen by the average rows
per expert (`pick_ww`, 128 when 56 to 112; `STRATA_PF_FUSED_TILE=64|128` forces). Widths come from `NativeGeom`
(`n_embd` a multiple of 256, `2 n_ff` and `n_embd` multiples of the item's weight rows); anything else exits with a
message (`moe_fused_iq.cu:612-645`). Opt-in because it changes numbers; upstream measured IQ2_XS prompts +12 % at 4K,
+3 % at 32K.

## Peer GPU share (MMQ only)

With `--peer-device` the experts the peer holds (and with peer streaming a fraction of the streamed pinned ones) are
computed there in compact groups of at most 16 experts / `G` rows, and their rows come back into `Dm` before the
combine (`prefill.cpp:2703-2894`, `:3123-3127`). The fused paths are disabled when a peer is configured
(`core::peer_portable()`), because the ring and buffers were sized for MMQ.

## Configuration

| Knob | Effect |
| --- | --- |
| `STRATA_PF_FUSED` | unset or not `0`: fused Q2_0 on; `0`: MMQ for Q2_0 (byte-identical to 0.1.35); `1`: also the native fused kernels |
| `STRATA_PF_FUSED_NATIVE=0` | native packs keep MMQ even under `STRATA_PF_FUSED=1` |
| `STRATA_PF_FUSED_TILE=64|128` | native fused work-item shape |
| `STRATA_PREFILL_MMQ=0` | the FP16 GEMM path everywhere |
| `STRATA_PREFILL_GROUP_GATHER=0` | one gather per expert |
| `STRATA_PREFILL_BF16X2` | router (and other BF16 GEMMs) with the activation remainder |
| `STRATA_DBG_NAN=1` | reports the first layer of a chunk whose MoE produced non-finite values |
| CMake `STRATA_MMQ_KQUANTS`, `STRATA_ORCA_Q4KS_MMQ` | extra MMQ instances, see [ggml-bridge.md](ggml-bridge.md) |

## Tests

`prefill_fused_moe_test`, `prefill_fused_iq_test` (fused vs a double-precision reference and vs the MMQ sequence),
`prefill_mmq_kquant_test` (K-quant MMQ), `prefill_qwen36_test` (`ffn_out` and `ffn_moe_topk` seams). See
[tests.md](tests.md).

## Gotchas

- A native pack where any layer is not covered by the fused native kernels keeps MMQ-sized buffers for every layer
  (`fused_ring`, `prefill.cpp:168-177`): shrinking them overflowed the MMQ layers (garbage, illegal access or a hang
  on gfx1151, per the comment).
- The fused paths need the stream-all walk, so a last chunk below 1024 tokens runs MMQ in the same (fused-sized)
  buffers; `moe_bufs` sizes them for MMQ at `stream_all_min() - 1` tokens.
- `mmq::fits` aborts nothing itself, but llama.cpp's own tile choice would abort the process (`J_best=0`) if called on a
  shape with no tile; `fits` mirrors that test so such a layer is routed to the FP16 path.

## See also

[expert-streaming.md](expert-streaming.md), [ggml-bridge.md](ggml-bridge.md), [gemm.md](gemm.md),
[docs/qwen36-port/step3-experts.md](../../qwen36-port/step3-experts.md),
[docs/upstream/DETAILS.md](../../upstream/DETAILS.md) (0.1.36 note on `STRATA_PF_FUSED`).
