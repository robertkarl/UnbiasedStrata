[Codebase docs](../README.md) › [Prefill](README.md) › GDN and attention

# The attention half: residual-block kernels, chunked GDN, prompt attention

## Purpose

Half 0 of every layer: read the residual into a normalized input, run the mixer (Gated DeltaNet on three of every
four layers, full attention on every fourth), and write back. The GDN mixer's conv and recurrence are recurrences in
time, so their kernels walk the chunk's tokens in order inside one launch, starting from and updating the session's
state; the attention mixer appends the chunk's K/V into the session's paged pools and attends to everything up to
each query. The same file also holds the hyper-connection and plain-block kernels used around both halves.

Files: [src/prefill/kernels.cu](../../../src/prefill/kernels.cu),
[include/strata/prefill/kernels.hpp](../../../include/strata/prefill/kernels.hpp), the orchestration in
[src/prefill/prefill.cpp](../../../src/prefill/prefill.cpp) (`:2212-2497`, `:3157-3183`). The attention kernels
themselves (`qsa_prompt_attn_batch`, `qsa_decode_attn_batch`, the indexer, `kv_append_q4`, KV staging) live in
`src/kernels/cuda/` and are documented with the kernels area ([Kernels › CUDA](../kernels/cuda/README.md)).

All buffers are row-major with the token as the slow index. Kernel launch errors exit the process (`check`,
`kernels.cu:64`); the prompt path has no error return below `Prefill::run`.

## Residual block: read and write

### Flash-Next: hyper-connection (hc = 4, 2560 wide, bottleneck 320)

Compile-time widths `N = 2560, HC = 4, D = 10240, LR = 320` (`kernels.cu:23`); these kernels are Flash-Next only.

| Function | Line | Math |
| --- | ---: | --- |
| `gr_norm` | `kernels.cu:921` | per stream RMS norm of `R[t, c, :]` times `w_norm`, FP32 and BF16 image (optional BF16 remainder) |
| `gr_norm_rs` | `:926` | F-1: only the row scales `rs[t, c]` and the BF16 image; the FP32 normalized rows are never stored |
| `gr_silu` | `:943` | `lo16 = bf16(silu(lo / hc))` |
| `gr_mix_r` / `gr_mix` | `:931`, `:947` | `mixed[t, d] = mean_c xn[t, c, d] * sigmoid(gated[t, c, d])` (xn recomputed from R and rs, same bits), FP32 / BF16 / FP16 images |
| `gr_write` | `:952` | `R[t, c, :] += bo[t, :] * 2 sigmoid(inj[t, c] / hc)` |
| `gr_write_norm_rs` | `:937` | F-2: the write fused with the next half's `gr_norm_rs`, same bits, `R` not re-read |
| `gr_broadcast` | `:956` | embedding to the four streams |

Read sequence (`prefill.cpp:2223-2240`): `gr_norm_rs` -> `hc_*_down` (BF16 GEMM, `D -> LR`) -> `gr_silu` ->
`hc_*_up` (`LR -> D`) and `hc_*_inject` (`D -> HC`) -> `gr_mix_r`.

### Qwen3.6: plain block (hc = 1)

| Function | Line | Math |
| --- | ---: | --- |
| `rms_norm_images` | `kernels.cu:1030` | `mixed = R * rsqrt(mean R^2 + eps) * w` with BF16 (`mixed_bf`) and FP16 (`mixed_h`) images, `gr_norm_kernel`'s arithmetic on one stream |
| `residual_add` | `:1036` | `R += bo` |

Half 0 reads with `attn_norm.weight`, half 1 with `post_attention_norm.weight`; both must be resident F32.

## Gated DeltaNet (linear attention)

`GdnGeom {hv, hk, c, gate}` (`kernels.hpp:54`) selects one of exactly two template instantiations through
`gdn_dispatch` (`kernels.cu:641`): `<48, 16, 10240>` (Flash-Next) and `<32, 16, 8192>` (Qwen3.6); any other geometry
exits (`Prefill::init` refuses it first by name via `gdn_geometry_supported`, `kernels.cu:960`). Head size `S = 128`,
conv 4 taps. Value head `h` pairs with key head `h % hk`. The output gate is `GdnGate`, taken from
`ModelGeometry::gdn_gate()` with no default, so the prompt and decode paths agree by construction: `sigmoid(z)` for
Flash-Next, `z * sigmoid(z)` (SiLU) for Qwen3.6 (`gdn_gate<SILU>`, `kernels.cu:214`).

Per GDN layer (`prefill.cpp:2243-2265`):

1. Projections: `attn_qkv` -> `qkv [T, C]` and `attn_gate` -> `z [T, value_dim]` (FP16 GEMMs over dequantized
   native weights from `mixed_h`); `ssm_alpha` and `ssm_beta` (BF16 GEMMs from `mixed_bf`) interleaved into
   `ab [T, 2 hv]`.
2. `gdn_gates` (`kernels.cu:963`): `gate = softplus(alpha + dt_bias) * ssm_a`, `beta = sigmoid(beta_raw)`.
3. `gdn_conv` (`kernels.cu:971`): the 4-tap causal conv + SiLU over the chunk starting from the layer's conv history
   `[C][3]`, then the history is set to the chunk's last three inputs, then L2 norm of every q and k head
   (`gdn_l2_kernel`). For `T > 64` the conv is tiled over 64-token tiles (C-3: each tile reads its three
   predecessors from the inputs, so tiles are independent and the bits are the same) and a separate kernel writes the
   history; `STRATA_GDN_CONV_SERIAL=1` keeps the one-thread-per-channel walk.
4. `gdn_recurrence` (`kernels.cu:987`): the delta rule over the chunk with the 128 x 128 state `M` of each value
   head in registers: per token, `g = exp(gate)`, `delta = (v - g * (M^T k)) * beta`, then `M = g M + k delta^T`
   (decay before the rank-1 update) and `o = (M^T q) / sqrt(128)`; then the gated RMS norm
   `y16 = rmsnorm(o) * ssm_norm * gate(z)` in FP16 for the out projection. Variants, all with
   the same per-column arithmetic order, so bit-identical (checked by `gdn_rec_parity`):

   | Kernel | Line | Used when |
   | --- | ---: | --- |
   | `gdn_rec_kernel<HV, HK, C, SILU>` | `kernels.cu:293` | one block per value head, norm inline; `STRATA_GDN_REC_HEADS=1` |
   | `gdn_rec_cols_kernel` | `:354` | value columns split over 4 blocks per head (D-2), norm in `gdn_out_norm_kernel`; `STRATA_GDN_PIPELINE=0` |
   | `gdn_rec_cols_pipe_kernel` | `:398` | the default fallback: next token's inputs loaded into registers while this one computes |
   | `gdn_rec_kh_kernel` | `:497` | default on CUDA sm_80+ when all `HK * 4` blocks are resident at once (`gdn_keyhead_ok`, `:600`): one thread serves the value heads sharing a key head, inputs staged in 8-token blocks by `cp.async`, 2 syncs per token instead of 5 (measured 1.41x on a 4080 Super); not on HIP; `STRATA_GDN_KEYHEAD=0` disables |
   | `gdn_out_norm_kernel<HV, SILU>` | `:622` | the output norm after the column kernels |

5. `ssm_out` projection from `y_h` -> `bo` (FP16 GEMM).

State layout: `ss.gdn_state + (gdn_index - ss.gdn_ord0) * gdn_floats`, where `gdn_floats = 128 * hv * 128 + C * 3`:
the recurrent state `[128 k][hv][128 v]` followed by the conv history (`prefill.cpp:1788-1789`, `:2252-2253`).
`gdn_ord0` is non-zero on a layer-split stage whose arrays start at a later GDN ordinal.

## Full attention

Per attention layer (`prefill.cpp:2266-2497`):

1. Projections from `mixed_h`: `attn_k` -> `Kc [T, n_kv * 256]`, `attn_v` -> `Vc`, `attn_q` -> `Qf [T, n_head, 512]`
   (query | gate per head, in both models). Flash-Next also projects the indexer key and query (BF16).
2. `rms_rows` with `attn_k_norm` over each 256-wide K head, then `rope` (`kernels.cu:1100`): NEOX rotary on the
   first 64 dims, pairs `(p, p + 32)`, positions `p0 + t` through the IMRoPE section table (`mrope_pos`), with the
   process rope scaling (`STRATA_ROPE_TABLE=1`, #280, reads angles from a float64 table; the default kernel is the
   table-free `<false>` instantiation).
3. KV append into the layer's `QsaState` at cells `[p0, p0 + T)` through its page table (`prefill.cpp:2315-2338`):
   - FP16 or INT8 (+ one FP16 scale per 64 values) via `kv_append` (`kernels.cu:896`; grid `(T, 2 KV heads, 8)`,
     one block per (token, KV head, 64-value group, K or V); init refuses any KV layout but 2 x 256);
   - Q4_0 via `kv_append_q4`; the K8V4 hybrid (INT8 K, rotated Q4_0 V); optional 256-point FWHT rotation of K and V
     (and of the queries and the output) when `kv_rot`.
   - KV streaming (`kv_mode == 1`): the layer's earlier cells are staged from the host copy into a one-layer staging
     pool first (`kv_stage_from_host`), and the append writes the pool page only if that block is resident, plus the
     host copy and the staging pool (identity layout). HIP-only opt-in `STRATA_KV_HOST_DMA=1` writes the host copy
     by one DMA instead of from the kernel.
4. `split_q`, `rms_rows` with `attn_q_norm`, `rope`, optional FWHT.
5. Selection:
   - Flash-Next (`G.indexer`): rope the indexer queries, append the chunk's indexer keys in three launches
     (`native_qsa_indexer_append_batch`, C-2; `STRATA_INDEXER_PER_TOKEN=1` per token), then per 256 queries block
     scores on tensor cores (`qsa_block_scores_tc`, 3xTF32; `STRATA_SELECT_OLD=1` the warp kernel) and
     `qsa_block_topk` into `sel_ids [T, cap]` (`cap` = the sparse width).
   - Qwen3.6 (dense): no indexer, no ids (a null `ids` is the identity selection: every cell up to the query's own),
     `cap` = the primary state's `max_cells`; `qsa_attn_group` must accept the head ratio (G = 2, 8 or 12) or `run`
     fails with its message (`prefill.cpp:2347-2350`).
6. Attention: `qsa_prompt_attn_batch` (tensor cores, one block per (query, KV head), online softmax, FP16 MMAs with
   hi/lo splits, FP32-level but not bitwise equal to the decode kernel). It declines (returns false) below sm_70 and on
   gfx906, and then `qsa_decode_attn_batch` runs 32 queries at a time with split-K scratch (bounded at 128 blocks
   per KV head for dense shapes); `STRATA_PROMPT_ATTN_OLD=1` forces the decode kernel. On Pascal the decode kernel is
   therefore the prompt attention.
7. Inverse FWHT of the output if rotated, `gate_attn` (`attn * sigmoid(gate)` to FP16), `attn_output` projection.

The QSA step records (`qsa_step_fill`, one per position, uploaded per chunk) carry each query's position, width and
block counts; for the dense shape the width is the query's own cell count.

## Model differences

| | Flash-Next | Qwen3.6 |
| --- | --- | --- |
| Residual block | hyper-connection (`gr_*`) | plain (`rms_norm_images`, `residual_add`) |
| GDN | 48 / 16 / 10240, 3 value heads per key head, sigmoid gate | 32 / 16 / 8192, 2 per key head, SiLU gate |
| Attention heads | 24 query / 2 KV (G = 12) | 16 / 2 (G = 8) |
| Selection | sparse indexer, `sel_ids` | dense, null ids |
| Indexer buffers carved | yes (`idx_raw`, `q_idx`, `sel_ids`, `sel_scores`) | none |

## Configuration

`STRATA_GDN_CONV_SERIAL`, `STRATA_GDN_REC_HEADS`, `STRATA_GDN_PIPELINE=0`, `STRATA_GDN_KEYHEAD=0`,
`STRATA_INDEXER_PER_TOKEN`, `STRATA_SELECT_OLD`, `STRATA_PROMPT_ATTN_OLD`, `STRATA_KV_HOST_DMA` (HIP),
`STRATA_GR_UNFUSED`, `STRATA_PREFILL_BF16X2`. Debug-only: `STRATA_SEL_OVERLAP` (selection overlap of 16-query tiles),
`STRATA_IDX_FP16_CHECK` (would FP16 indexer keys select the same cells), `STRATA_QSA_DUMP=<file>` /
`STRATA_QSA_DUMP_LAST` (selected cells for `tools/qsa_locality.py`), `STRATA_STATE_HASH_GDN` (hash of every GDN
state after the prompt). Compile-time: `STRATA_GDN_CP_ASYNC` (1 on sm_80+ device code, plain copies below).

## Tests

`gdn_rec_parity` (both geometries, bitwise variants, a double-precision reference, rival readings, refusals),
`prefill_qwen36_test` (`linear_attn_out`, `attn_output`, `attn_norm`, `attn_post_norm` seams). See
[tests.md](tests.md). The prompt attention kernel has its own parity tests in the kernels area.

## Gotchas

- `gdn_rec_kh_kernel`'s comments were written for Flash-Next (three value heads per key head, 192 vs 64 blocks); the
  kernel is templated with `VPK = HV / HK` and a `static_assert`, so at Qwen3.6's geometry it serves two.
- `gdn_keyhead_ok` is decided per device (a layer split can mix cards); below 64 resident blocks the column kernel is
  used, because a second wave would double the time.
- `rope` exits on an invalid rope-scaling config; the engine validates the same rule at startup in `generate.cpp`.

## See also

[pipeline.md](pipeline.md), [moe.md](moe.md), [tests.md](tests.md),
[docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md),
[docs/qwen36-port/step3-gdn.md](../../qwen36-port/step3-gdn.md), [docs/qwen36-port/step3-attn.md](../../qwen36-port/step3-attn.md),
[docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md) ("Gated DeltaNet").
