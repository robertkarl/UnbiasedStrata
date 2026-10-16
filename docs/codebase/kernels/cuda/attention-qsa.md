[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Attention (QSA and dense)

# Attention: QSA, dense attention and the pinned adapters

## Purpose

These kernels implement everything in a full-attention layer between its projections (`attn_q`, `attn_k`,
`attn_v`, `attn_output`, and Flash-Next's `indexer.q_proj` / `indexer.k_proj` are GEMVs from
[gemv-and-quant.md](gemv-and-quant.md) / [moe-and-experts.md](moe-and-experts.md)):

- the paged KV cache (FP16 here; INT8 / Q4_0 / K8V4 and streaming in [kv-cache.md](kv-cache.md));
- Flash-Next's QSA indexer: raw-key tail, pooled block keys, per-block scores, top-k cell selection;
- attention over the selected cells (or over every cell for Qwen3.6's dense layers), for decode, the verify window
  and the prompt;
- the output gate `attn * sigmoid(gate)`.

Files: `src/kernels/cuda/qsa.cu`, `qsa_select.cu`, `qsa_decode_attn.cu`, `qsa_prompt_attn.cu`, `native_qsa.cu`,
`native_qsa_indexer.cu`, `native_qsa_score.cu`, `native_flash_attn.cu`, and their headers
`include/strata/kernels/qsa.hpp`, `qsa_select.hpp`, `qsa_decode_attn.hpp`, `qsa_prompt_attn.hpp`,
`native_qsa.hpp`, `native_qsa_indexer.hpp`, `native_qsa_score.hpp`, `native_flash_attn.hpp`.

## Geometry and the step record

`QsaShapes` (`include/strata/kernels/qsa.hpp:97-107`) carries `n_head`, `n_head_kv`, `head_dim`, `n_rot`,
`idx_n_head`, `idx_dim`, `idx_block` (r = 4 cells per pooled key), `idx_top_k` and `page_size`.

| | Flash-Next QSA (`qsa_real_shapes`, `qsa.hpp:110`) | Qwen3.6 dense |
| --- | --- | --- |
| heads / KV heads / head_dim | 24 / 2 / 256 (G = 12) | 16 / 2 / 256 (G = 8) |
| n_rot | 64 (NEOX pairs; IMRoPE sections in the native path) | 64 |
| indexer | 4 heads x 128, block 4, top-k 2048 | none: `idx_n_head = idx_dim = idx_top_k = 0` |
| page_size | 4 cells (one indexer block, the KV-streaming granule) | 4 |

- `qsa_dense(s)` (`qsa.hpp:134`) is true when the three indexer widths are all zero. A shape with only some of them
  zero is refused by `qsa_shapes_valid` (`src/kernels/cuda/qsa.cu:687`).
- `qsa_selection_width(n_kv, s)` (`qsa.hpp:147`) is `min(n_kv, idx_top_k + idx_block - 1)` = min(n_kv, 2051),
  or `n_kv` when dense. The `+ r - 1` covers the incomplete tail block.
- `qsa_selection_cap(max_cells, s)` (`qsa.hpp:304`) is 2,051 for sparse shapes (`kTopkMaxCells` = 32,768) and
  `max_cells` for dense ones.
- `QsaStep` (`qsa.hpp:176-182`): a 4 x int32 device record `{pos, n_kv = pos+1, n_bid = n_kv / r, width}` written
  once per token (`qsa_step_fill`, `qsa.cu:719`). Every capturable kernel reads its per-token counts from it and
  sizes its grid from a constant capacity. The header explains why: a graph that captured a host-scalar `pos` would
  attend over one cell forever.

KV layout (FP16): `[page][kv_head][page_size][head_dim]`. The physical row of cell t and head h is
`(table[t / page_size] * n_head_kv + h) * page_size + t % page_size` (`qsa.hpp:63-77`). The page table is never
assumed to be the identity. With KV streaming it is a residency map, where -1 means the page is not resident.
The query-to-KV head map is integer division `q / (n_head / n_head_kv)`, not modulo (`qsa.hpp:317`).

## Main entry points

### Reference QSA path (`qsa.cu`)

| Function | Kernel / launch | What it does |
| --- | --- | --- |
| `kv_append_step` (`qsa.cu:583`) | `kv_append_kernel` (`:110`), 128 threads per block | K and V of the new cell to FP16 with `f16_from_f32` (not `__float2half`); writes the host copy too when streaming |
| `indexer_key_append` (`qsa.cu:742`) | `indexer_key_append_kernel` (`:143`), one block, one thread per indexer dim | Stores the raw key in the 3-row tail. On a block completion it pools `mean(raw[4 cells])`, applies RMSNorm x `w_k_norm`, rotates at the block's first cell, and writes `pooled[b]`; the spare row `pooled[n_bid]` = rotated `rms_norm(raw[0])`. Arithmetic is in double with `__dadd_rn`/`__dmul_rn` so the pooled key is bit-exact against the reference |
| `qsa_index_step` (`qsa.cu:596`) | `qsa_index_kernel` (`:237`), one block per pooled row, one warp per indexer head, FP64 dots | `score = bias + sum_h relu(dot(pooled[b], q_idx[h]))`, mapped to the block's cells; incomplete-tail cells get `+1e9f` |
| `topk_512_step` (`qsa.cu:610`) | `topk_kernel` (`:295`), 1 block x 256 threads | Exact top-`width` cells, ties to the lower index, output ascending. The width-th largest score is found as a 32-bit order key by 32 monotone passes, then one ascending walk emits the selection. Limited to n_kv <= 32,768 |
| `kv_gather_step` (`qsa.cu:629`) | `kv_gather_kernel` (`:394`), 256 threads | Selected cells through the page table into contiguous FP16 scratch |
| `qsa_attend_step` (`qsa.cu:645`) | `qsa_attend_kernel` (`:427`), grid `n_head`, block `head_dim`, dynamic smem `(max_ids + 32) * 4` B | Softmax GQA over the gathered cells; optional per-head weights out |
| `qsa_gate_apply` / `_f32` (`qsa.cu:832`, `:823`) | 256-thread elementwise | `attn * sigmoid(q_full[h][head_dim + d])` (the gate is the second half of each head's `2*head_dim` block, not interleaved); the FP16 form rounds once at the end |

The host-scalar wrappers (`kv_append`, `qsa_index`, `topk_512`, `kv_gather`, `qsa_attend`, `qsa.cu:736-797`) fill a
module-level step buffer and are for tests and direct launches only. The header states that the layer must use the
`_step` forms (`qsa.hpp:350-364`).

### Block scores and selection at scale (`qsa_select.cu`)

These replace `qsa_index_step` + `topk_512_step` for decode (`--no-fast-select` keeps the old pair), the verify
window and the prompt. They score once per 4-cell block instead of once per cell.

| Function | Kernels | Notes |
| --- | --- | --- |
| `qsa_block_scores` (`qsa_select.cu:991`) | `block_scores_kernel` (`:28`, one warp per (query, block), 8 warps per block); `block_scores_multi_kernel` (`:725`, a fixed 256-block grid that reads each key once for up to `MQ` = 8 queries) | FP32. Block `n_bid` (the tail) scores the `dead` key and gets +1e9 when it has cells. The multi kernel is used when no active-block count is given (decode and prefill's pooled16 call) unless `STRATA_SCORES_MULTI=0` |
| `qsa_block_scores_tc` (`:1015`) | `block_scores_tc_kernel` (`:195`, 128 threads, `mma.sync m16n8k8` TF32, 3xTF32 split) + `block_scores_tail_kernel` | sm_80+ only, prompt path. FP32-level accuracy, another summation order. Returns false (nothing launched) on another geometry or below sm_80; `STRATA_QSA_WARP=1` or `select` forces the warp kernel |
| `qsa_block_topk` (`:1174`) | cluster / register / wide / ref kernels | 4-pass radix select over `n_bid + 1` blocks weighted by their cell counts, then cells emitted ascending with ties to the lowest index. The ids are identical to `topk_kernel`'s |
| `qsa_block_topk_cluster` (`:1108`) | `block_topk_cluster_kernel` (`:847`), clusters of `CL_N` = 8 CTAs x 1,024 threads per query via `cudaLaunchKernelEx`, DSMEM (`mapa`, `barrier.cluster`) | sm_90+ with code built for sm_90 (checks `ptxVersion`/`binaryVersion` >= 90 and `cudaOccupancyMaxActiveClusters`); calls of <= 16 queries; `STRATA_QSA_CLUSTER=0` disables |
| `qsa_block_topk_ref` (`:1078`) | `block_topk_kernel` (`:57`), 256 threads, one shared histogram | the original kernel; `STRATA_TOPK_OLD=1` |

`qsa_block_topk` dispatch: cluster if eligible; otherwise `block_topk_reg_kernel<TK_PER = 33>` (1,024 threads,
keys in registers, contexts up to 4 x 1024 x 33 cells); past that reach, `block_topk_wide_kernel` (CUDA, keys reread
per radix pass). On sm_75 only (or with `STRATA_TOPK_ACTIVE_ANY=1`), the prompt path's `active_blocks` bound picks
the kernel instead of the capacity; `STRATA_TOPK_CAPACITY_GUARD` turns that off. Small prompt batches (< 4,608 active
blocks past the register reach) use the ref kernel (`qsa_select.cu:1210-1218`).

### Split-K decode attention (`qsa_decode_attn.cu`)

`qsa_decode_attn_step` (`:773`) and `qsa_decode_attn_batch` (`:767`) read the KV pools directly through the page
table and the selection (no gather copy).

- Kernel 1: `attn_chunk_kernel<KV_MODE, G>` (`:104`), grid `(n_chunks, n_head_kv, n_q)`, 256 threads. Each block
  serves all G query heads of one KV head. It walks its span of the selection in 64-cell sub-chunks with a running
  (max, sum, acc) rescale and writes one partial per block.
- Kernel 2: `attn_merge_kernel<G>` (`:616`), grid `(n_head, n_q)`, 256 threads. A log-sum-exp merge of the
  partials.
- `KV_MODE`: 0 FP16, 1 INT8 (+ fp16 scale per 64), 2 Q4_0 (rotated; the caller rotates q and the output with
  FWHT), 3 hybrid K8V4 (INT8 K, rotated Q4_0 V). `kv_mode_of` (`:669`) picks it from which pool pointers are set.
- `G` is a template parameter instantiated for 2, 8 and 12 (`run`, `:707`). `qsa_attn_group` (`:736`) refuses
  any other ratio by name and exits.
- `chunks_for(cap, dense)` (`:37`): `ceil(cap / 64)` blocks per KV head for sparse shapes; capped at
  `kMaxChunks` = 128 for dense shapes, so the scratch stays at `chunks x n_head x 258 + 64` floats per query row
  (`qsa_decode_attn_scratch_floats`, `:763`; 2.02 MiB at 16 heads) instead of growing with the context.
- `ids == nullptr` is the identity selection (cell j for j < `step[kStepWidth]`): dense attention needs no ids
  buffer. `qsa_identity_ids` (`:778`) writes the explicit form for consumers that need a buffer (the KV-streaming
  resolver, tests).
- A page of -1 (a block KV streaming could not make resident) is masked: score `-FLT_MAX`, weight 0.

`STRATA_EXPERIMENTAL_SM60` builds add `attn_chunk_kernel_pre75` / `_pre75_single` (PR #540, `:225-613`). They are
bit-identical to the standard kernel with fewer shuffles and run only on cards below sm_75. `STRATA_ATTN_PRE75=0`
turns them off (`pre75_attn`, `:644`).

### Tensor-core prompt attention (`qsa_prompt_attn.cu`)

`qsa_prompt_attn_batch` (`:1459`) takes the same arguments as `qsa_decode_attn_batch` minus the scratch. It returns
false (nothing launched) when it cannot run, and `src/prefill/prefill.cpp` then falls back to the decode kernel.

- One block of 128 threads per (query, KV head), grid `(n_q, n_head_kv)` in slices of 65,535 queries. It walks the
  selection in chunks of `CH` = 32 cells with an online softmax; there is no split-K scratch and no merge kernel.
- q.k and p.v are `mma.sync` FP16 MMAs with FP32 accumulation. q and p are split into FP16 hi + lo halves (two MMAs
  each), which keeps about 22 bits. INT8 codes enter exactly and their per-64 scales are applied in FP32. The result is
  not bitwise equal to the decode kernel; `qsa_prompt_attn_parity` bounds the difference.
- Dispatch (`dispatch<G>`, `:1413`) by KV format and card:

| KV | sm_80+ | sm_75 (Turing) | sm_70 (Volta, SM60 build only) |
| --- | --- | --- | --- |
| FP16 (mode 0) | `launch<0>` | `launch<0>` (m16n8k16 done as two m16n8k8) | `launch70<0>` (m8n8k4 kernel) |
| INT8 (mode 1) | `launch_i8` (cp.async gather, `prompt_attn_i8_kernel`, `:466`) | `launch<1>` | `launch70<1>` |
| K8V4 (mode 3) | `launch<3>` | `launch<3>` | `launch70<3>` |
| Q4_0 (mode 4) | `launch<4>` | refused (no parity run yet) | refused |

  Below sm_70 the function returns false. `STRATA_QSA_WARP=1` or `attn` makes it act as sm_75. `STRATA_PROMPT_ATTN_V1=1`
  forces the first INT8 version on sm_80+. `STRATA_PROMPT_ATTN_Q4=0` refuses Q4_0. HIP builds compile the tensor-core
  kernels out; RDNA4 has an opt-in WMMA kernel (`STRATA_HIP_WMMA=1`).
- The geometry must be head_dim 256 with `n_head == G * n_head_kv` for G in {2, 8, 12}.

### Pinned llama.cpp adapters ("native")

| Function | Enabled by | Contract |
| --- | --- | --- |
| `native_qsa_rms_norm_weighted` (`native_qsa.cu:108`) | `--native-qsa` (`native_qsa_set_enabled`) for q/k norms; always for Qwen3.6's `attn_norm` / `post_attention_norm` (`src/core/layer.cpp:1231-1236`) and the MTP layer | ggml `rms_norm` then `mul`: 256 threads below 1,024 columns, else 1,024, with the pinned XOR tree; `(scale * x) * gamma`. A row of n rows is bitwise a one-row call |
| `native_qsa_gate_apply` (`native_qsa.cu:120`) | `--native-qsa` | `attn * sigmoid(q_full second half)` in F32 |
| `native_qsa_indexer_append` / `_append_batch` (`native_qsa_indexer.cu:228`, `:255`) | `--native-qsa-indexer` | Pinned F16 raw-key cache (the tail holds half-rounded raws expanded to F32), F32 pooling, weighted RMSNorm, text IMRoPE, with `rope_scaling` arguments. The relative position is a cell index from `kStepPos`. The batch form is for the prompt path (host-side positions, not capturable) |
| `native_qsa_score` (`native_qsa_score.cu:191`) | `native_qsa_score_set_enabled` | Pinned MMF dispatch (TF32 `mma.sync` + `ldmatrix`, raw F32 bits with no `cvt.rn.tf32`) for budget 2048 with n_kv padded to 256. Below sm_80 an FP32-FMA path rounds differently. No caller in `src/core`, `src/prefill` or `src/program` uses it; only the oracle parity test (not in this repository) did |
| `native_flash_attn_short_step` (`native_flash_attn.cu:192`) | `--native-flash-attn-short` (diagnostic, requires `--max-context <= 256`) | FlashAttention-vec for exactly Q24x256 / KV2x256, scale 1/16, 24 blocks x (32, 4) threads. An invalid step writes `status = kNativeFlashAttnUnsupportedStep` and NaN output. The caller must check `status` |

## Control and data flow

Decode (`src/core/layer.cpp:995-1055`, per QSA layer):

```
q, k, v projections -> norm (rms_norm_weighted | native_qsa_rms_norm_weighted) -> rope (rope_neox_apply | native_rope_apply)
kv_append_step | kv_append_q8_step | kv_append_q4_step            (kv-cache.md)
Flash-Next: indexer_key_append | native_qsa_indexer_append
            qsa_block_scores -> qsa_block_topk        (or qsa_index_step -> topk_512_step)
            qsa_kv_resolve -> kv_stream_resolve        (only with --kv-resident)
            [fwht256 on q when the KV is rotated]
            qsa_decode_attn_step(ids)                  (or gather + qsa_attend: dumps, short flash)
Qwen3.6:    qsa_decode_attn_step(ids = nullptr)        (dense: the only supported path, layer.cpp:997-1007)
native_qsa_gate_apply | qsa_gate_apply_f32 -> attn_output projection
```

The verify window (`src/core/verify.cpp:919-965`) and the MTP drafter (`src/core/mtp.cpp:691`, `:934`) use
`qsa_decode_attn_batch`. The drafter's ring uses a sliding window of ids from `window_ids`
([sampling-and-verify.md](sampling-and-verify.md)). The prompt path (`src/prefill/prefill.cpp:2383-2485`) uses
`qsa_block_scores(_tc)`, `qsa_block_topk` with `active_blocks`, and `qsa_prompt_attn_batch`.

## Configuration

- CLI: `--native-qsa`, `--native-qsa-indexer`, `--native-flash-attn-short`, `--no-fast-select` /
  `--no-fast-attn` (engine-level switches in `src/program/generate.cpp:2620-2623`), `--kv`, `--kv-resident`,
  `--max-context` (sets the capacities).
- Environment: `STRATA_SCORES_MULTI`, `STRATA_QSA_WARP`, `STRATA_QSA_CLUSTER`, `STRATA_TOPK_OLD`,
  `STRATA_TOPK_CAPACITY_GUARD`, `STRATA_TOPK_ACTIVE_ANY`, `STRATA_ATTN_PRE75`, `STRATA_PROMPT_ATTN_V1`,
  `STRATA_PROMPT_ATTN_Q4`, `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA` (HIP), `STRATA_EMULATE_CC`.
- Compile time: `STRATA_EXPERIMENTAL_SM60` (pre-sm_75 decode kernel, Volta prompt kernel); `D1_CH`, `D1_NO_QLO`,
  `D1_NO_PLO` are experiment macros in `qsa_prompt_attn.cu` (chunk size, dropping the lo halves).

## Model-specific behaviour

- Flash-Next uses sparse QSA with G = 12 and the indexer. Its selection is bounded at 2,051 cells, so the decode
  kernel uses one block per 64 cells at any capacity. The MTP drafter's ring (window 32,768 by default) runs the same
  sparse grid (`qsa_decode_attn.cu:33-35`).
- Qwen3.6 uses dense attention with G = 8 and no indexer. The indexer entry points refuse a dense shape
  (`validate_indexer`, `qsa.cu:98`). Only the split-K decode kernel and the prompt kernel serve it, and
  `layer.cpp:997-1002` refuses the gather path and the short flash adapter. KV streaming saves no VRAM for dense
  attention because every block is selected every token (`docs/QWEN36-35B-A3B.md` section 3.3 item 6).

## Tests

- `qsa_parity` (cache, indexer, scores, top-k, attention, gate, the `native_qsa_indexer` append; page sizes 1/4/512
  with a permuted table must give identical attention).
- `qsa_dense_parity` (Qwen3.6: G 2/8/12, null ids vs explicit identity, bounded scratch, every KV format, the
  prompt kernel; oracle seams with `--fixtures`).
- `qsa_prompt_attn_parity` (tensor-core vs FP32 kernel vs FP64), `qsa_topk_parity`, `qsa_topk_active_parity`,
  `decode_cluster_parity` (cluster top-k, bitwise and under graph capture; skips below sm_90), `kv_hybrid_parity`,
  `kv_stream_parity`.
- `qsa_attn_bench` is a benchmark target, not a test.
- Oracle-linked (`CMakeLists.txt:424`, sources not in this repository): `native_qsa_parity`,
  `native_qsa_indexer_parity`, `native_qsa_score_parity`, `native_flash_attn_parity`.

## Gotchas

- Nothing outside `qsa.hpp` may assume the KV page mapping (`qsa.hpp:74-77`).
- The tail bias is `+1e9f` on incomplete-tail cells even when `bias` is null, matching llama.cpp's causal decode
  input (`qsa.hpp:279-281`).
- `topk_512` refuses more than `kTopkMaxCells` = 32,768 cells; longer contexts must use `qsa_block_topk`.
- `qsa_attend` sizes its shared memory from `max_ids` and raises the opt-in limit once
  (`cudaFuncSetAttribute`, `qsa.cu:672`). It exits if the card cannot hold it.
- The prompt kernel and the decode kernel do not agree bit for bit. Only the decode kernel's batch form is bitwise
  equal to its single-query form.
- On Turing, Q4_0 KV prompt attention falls back to the decode kernel, because the pair-of-m16n8k8 form has no parity
  run (`qsa_prompt_attn.cu:1428-1434`).

## See also

- [kv-cache.md](kv-cache.md), [norms-rope-elementwise.md](norms-rope-elementwise.md) (RoPE, scaling, M-RoPE),
  [sampling-and-verify.md](sampling-and-verify.md) (`dense_steps`, `window_ids`)
- `docs/qwen36-port/step3-attn.md` (dense attention), `step3-prefill.md` (prompt path), `pascal-build.md`
- `docs/QWEN36-35B-A3B.md` section 3.3
