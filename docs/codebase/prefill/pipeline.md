[Codebase docs](../README.md) › [Prefill](README.md) › Pipeline

# The `Prefill` pipeline

## Purpose

`strata::prefill::Prefill` owns everything a batched prompt needs on one GPU: the chunk buffers, a cuBLAS handle,
the expert staging ring and its events, a copy stream, host staging threads, and optional multi-GPU helpers. It
runs the prompt's positions `[pos0, pos0 + n)` in chunks through all layers (or a layer range on a layer split),
writes the recurrent / attention state into the shared `core::SessionState`, and reports per-chunk residual rows to
the caller for the MTP draft layer.

Files: [include/strata/prefill/prefill.hpp](../../../include/strata/prefill/prefill.hpp),
[src/prefill/prefill.cpp](../../../src/prefill/prefill.cpp).

## Main types

| Type / function | Where | What it does |
| --- | --- | --- |
| `PrefillStats` | `include/strata/prefill/prefill.hpp:29` | tokens, chunks, total ms, host staging ms, experts streamed / by DMA / resident, PLE ms; read by `generate.cpp` for its log line |
| `Prefill` | `include/strata/prefill/prefill.hpp:44` | the public object; non-copyable, pimpl (`Impl`) |
| `Geom` | `src/prefill/prefill.cpp:95` | the model widths (`n`, `hc`, `d = hc*n`, `lr`, `k`, `ne`, `ff`, `ff2`, `ffs`, GDN `c/zv/hv/hk`, attention `nh/nkv/hd/qf/av/kvw`, `indexer`, `gate`) from `ModelGeometry` via `Geom::of`; `gdn()` makes a `GdnGeom` |
| `Alloc` | `src/prefill/prefill.cpp:255` | bump allocator: counts only (`bytes_needed`), carves from a borrowed region, or `cudaMalloc`s; each take is 256-byte aligned plus 256 bytes; overflow fails (`STRATA_PREFILL_ALLOC_TRACE=1` prints which) |
| `Stager` | `src/prefill/prefill.cpp:296` | host threads copying unpinned expert blobs into a pinned ring; see [expert-streaming.md](expert-streaming.md) |
| `PeerPrefill` | `src/prefill/prefill.cpp:428` | the second GPU's share of a chunk's experts (`--peer-device`) or a layer-split stage's helper |
| `Prefill::Impl` | `src/prefill/prefill.cpp:518` | all device pointers of a chunk layout, the ring (`stage_dev`, `copied`, `used`, `used_of`, `stage_live`), the mapped grouping tables, PLE buffers, KV staging pool, hand-off buffers |
| `PfTimer` / `PeTimer` | `src/prefill/prefill.cpp:1696`, `:1732` | `STRATA_PREFILL_TIMING=1`: CUDA events per phase on the compute stream (and the peer's) |

### Public API (`prefill.hpp`)

| Member | Line | Notes |
| --- | ---: | --- |
| `init(wt, g, ss, src, cache, host_res, chunk, stream, err, borrow, borrow_bytes)` | 66 | checks geometry, creates the copy stream, events, the `Stager`, host buffers, the mapped grouping tables, the KV-streaming identity page table, then the GEMM scratch (32M FP16 elements) and workspace (32 MiB) and `carve(T)` |
| `relayout(chunk, borrow, borrow_bytes, err)` | 72 | with borrowed buffers only: lay the buffers out again for a smaller chunk (at most `init`'s) in a new region; the serve path calls it per request |
| `reset()` | 61 | frees everything `init` made; used to retry `init` with a smaller chunk |
| `bytes_needed(g, ss, chunk)` / `bytes_needed_no_ring` | 87, 91 | the same allocation sequence counted (`prefill.cpp:1581`), plus 8 MiB slack; what a borrowed region must hold |
| `set_pinned_share`, `set_ring_override`, `set_ring_budget`, `ring_*` | 77-111 | static ring-sizing knobs; see [expert-streaming.md](expert-streaming.md) |
| `run(tokens, n, pos0, err)` | 115 | `run_impl` then `drain_pipeline` (`prefill.cpp:3345`) |
| `stats()` | 117 | `PrefillStats` |
| `set_peer(peer, cap_rows, err)` | 122 | the `--peer-device` GPU computes the rows of experts it holds (needs P2P and the MMQ path) |
| `on_chunk` | 127 | after each chunk (stream synchronized): the chunk's final residual rows, `T`, `pos0` |
| `on_stage_chunk` | 133 | layer split: every stage calls it after a chunk, for mid-prompt checkpoints |
| `should_stop` | 136 | checked before every chunk; true makes `run` fail with `"cancelled"` |
| `on_seam` | 146 | test hook: device rows at llama.cpp's seam names (`l_in`, `attn_norm`, `attn_post_norm`, `linear_attn_out`, `attn_output`, `ffn_moe_topk`, `ffn_out`, `l_out`); writing `l_in` teacher-forces a layer |
| `embd_rows` | 151 | vision: host rows by absolute position used in place of the token embedding |
| `set_stage(lb, le, next)` | 158 | layer split: this object runs layers `[lb, le)` and hands rows to `next` |
| `set_stage_helper(helper, err)` | 169 | layer split, opt-in `STRATA_PREFILL_HELP=1`: the next stage's idle GPU streams and computes a share of this stage's experts for a one-chunk prompt |
| `draft_kv(mtp, R_rows, next_tokens, n, cell0, err)` | 51 | the MTP draft layer's K/V for prompt cells through this path's GEMMs and idle scratch |

## Initialization and memory layout

`Prefill::init` (`prefill.cpp:796`):

1. Geometry refusals (`:809-834`), each naming what it saw: `qwen4exp` must be 2560 / hc 4 / hc_lr 320 / top-10;
   otherwise `hc` must be 1; `k` must equal `n_expert_used`, lie in 1..32 and not exceed `n_expert`; the GDN geometry
   must be one the kernels are built for (`gdn_geometry_supported`, state 128, conv 4); head_dim 256, 2 KV heads,
   n_rot 64, n_embd a multiple of 256, n_ff a multiple of 64.
2. Layer-split hand-off buffers (two pinned `chunk x D` floats) when `next_` is set (`:841-845`).
3. Copy stream, `ring_cap()` pairs of `copied` / `used` events, the `Stager` (threads: `STRATA_STAGER_THREADS`,
   default 2..4, or 32 with a 128-deep ring when the expert source reads a GGUF in place: `src->transient`),
   (`:854-877`).
4. Mapped pinned grouping tables `[ids | slot | src | bounds]` (`:880-899`; `STRATA_GROUP_COPY=1` uses plain
   copies instead). Kernels read them over PCIe so they never queue behind expert blobs on the copy engine.
5. PLE double buffers (pinned, pageable fallback) and the KV-streaming identity page table (`:900-920`).
6. GEMM scratch and cuBLAS workspace taken first from the same `Alloc`, then `carve(T)`.

`Prefill::carve` (`prefill.cpp:941`) takes, in order: `emb`, `R`, the hyper-connection buffers (only when
`hc > 1`), `mixed` (FP32) and its BF16 / FP16 images, `bo`, optional BF16 low parts (`STRATA_PREFILL_BF16X2`), the
QSA step records, then ONE region of `max(gdn_set_bytes, qsa_set_bytes, moe_set_bytes)` holding the three
phase-local sets (three `Alloc`s over the same base), the two dequantized-expert buffers (`DQ = 2`, FP16 path), the
MMQ identity row map / bounds / two 16-expert group buffers with a 4096-byte zero tail (`MMQ_TAIL`), the ring's
`ring_slots(T)` blob slots, PLE buffers, and the KV staging pool (`take_stage`, KV streaming only). The order must
match `bytes_needed`; a mismatch makes `init` fail with "do not fit" rather than overlap.

`moe_bufs` (`prefill.cpp:766`) shrinks the MoE buffers when the layout runs the fused Q2_0 kernels
(`fused_layout`): ~100 KB per token less, sized so a final chunk below `stream_all_min()` can still run MMQ in them.

## The chunk loop (`run_impl`, `prefill.cpp:1772`)

Per chunk of `T = min(m.T, n - c0)` tokens at `p0 = pos0 + c0`:

1. `should_stop`, progress beat (`core::progress_at`, the serve watchdog names layer and chunk on a stall).
2. **Embeddings** (`:1855-1911`): a later layer-split stage uploads the previous stage's rows; otherwise one
   batched device gather (`NativeEmbed::gather_dev` or `embedding_gather_dev` for the pack's quantized table) unless
   a row is an image row or out of range, in which case per-token `core::embed_row`. Then `R = emb` (plain) or
   `gr_broadcast` to the four streams.
3. **PLE rows** (Flash-Next, when `ss.ple.ready()` and the stage covers layer 1): gathered for the whole chunk on a
   `std::async` thread from the PLE table (an SSD read per missing row), double-buffered so chunk `c+1`'s gather runs
   during chunk `c`; uploaded just before layer 1 (`ple_land`, `:1921`).
4. **QSA step records** for every position (`qsa_step_fill`), uploaded to `steps_dev` (`:1952`).
5. **Expert stream plan** for chunks of `stream_all_min()` (1024) tokens or more: see
   [expert-streaming.md](expert-streaming.md).
6. **Layer loop** `l` in `[LB, LE)` (`:2148`), two halves per layer:
   - Read: plain block `rms_norm_images(R, attn_norm | post_attention_norm)` producing FP32 `mixed`, BF16
     `mixed_bf`, FP16 `mixed_h` (`:2213-2221`); or the hyper-connection read: `gr_norm_rs`, BF16 `hc_*_down`,
     `gr_silu`, `hc_*_up`, `hc_*_inject`, `gr_mix_r` (`:2223-2240`). F-2: the previous write may already have
     normed `R` for this half (`normed`).
   - Half 0, GDN layer (`!is_qsa_layer`): see [gdn-and-attention.md](gdn-and-attention.md) (`:2243-2265`).
   - Half 0, attention layer: see [gdn-and-attention.md](gdn-and-attention.md) (`:2266-2497`).
   - Half 1, MoE: see [moe.md](moe.md) (`:2498-3152`).
   - Write: plain `residual_add(R, bo)` (`:3159`); hyper-connection `gr_write` or the fused
     `gr_write_norm_rs` that also norms `R` for the next half when nothing touches `R` in between (not the stage's
     last half, not before layer 1's PLE block, not under a control vector) (`:3161-3179`). Then
     `--control-vector` (`cvec_apply`) after half 1 where it covers the layer.
   - Layer 1 (Flash-Next with PLE): the PLE block runs before the halves, batched over the chunk in sub-batches carved
     from the idle scratch region (`native_ple_postops_batch`, `STRATA_PLE_BATCH=0` for the per-token block)
     (`:2155-2208`).
7. After the last layer: join the issuer thread, then either hand the rows to the next layer-split stage
   (async `next_->run_impl`, `:3196-3216`) or synchronize and call `on_stage_chunk` / `on_chunk` (`:3230-3241`).

After all chunks (`:3245-3321`): `ss.ple_prev` is advanced to the last two tokens; both streams are synchronized
(a failed expert copy surfaces here, PR #121); `STRATA_PREFILL_TIMING` prints the per-phase GPU timeline and the
host shares.

## What prefill leaves for decode

| State | Written by | Where |
| --- | --- | --- |
| GDN recurrent state `[128 k][hv][128 v]` and conv history `[C][3]` per GDN layer | `gdn_conv`, `gdn_recurrence` | `ss.gdn_state + (gdn_index - ss.gdn_ord0) * gdn_floats` (`prefill.cpp:2252-2253`) |
| K/V of every attention layer's prompt cells | `kv_append` / `kv_append_q4` into the `QsaState` pools via its page table (FP16, INT8 + scales, Q4_0, K8V4 hybrid; optional FWHT rotation); with KV streaming also the host copy | `prefill.cpp:2315-2338` |
| Flash-Next indexer state (tail, dead, pooled keys, block positions) | `native_qsa_indexer_append_batch` | `prefill.cpp:2356-2373` |
| PLE history (Flash-Next) | `native_ple_postops_batch` / `ple_history_advance` | `ss.ple.hist` |
| `ss.ple_prev` | end of `run_impl` | `prefill.cpp:3245` |
| MTP draft layer K/V | `on_chunk` -> `draft_kv` or `MtpDrafter::prefill` (`src/core/mtp.cpp:1131`) | the drafter's `QsaState` |

`generate.cpp` batches positions `[0, n_prompt - 1)` (or `--prefill-until P`) and starts the token loop at the last
prompt position, whose prediction is the first generated token; for a native pack that position is the first verify
window (`src/program/generate.cpp:7909-8010`). `ss.ple_prev` is then reset from the token list for the token path.

## `draft_kv` (`prefill.cpp:1057`)

Computes the MTP draft layer's K/V for prompt cells from each chunk's final residual rows, in batches carved from the
idle scratch region (at least 64 rows; a ring KV caps a batch below the ring's span). It returns false with `err`
empty when not applicable (KV streaming mode 1, a hybrid K/V, another device, too little scratch,
`STRATA_MTP_BATCH=0`), and the caller then runs `MtpDrafter::prefill`. Projections go through Q8_1 x Q8_0 MMQ when
available (`STRATA_MTP_BATCH_F16=1`: FP16 GEMMs). Not bit-identical to the drafter's own pass; only drafts may differ,
never the target's logits (`prefill.hpp:46-50`).

- Qwen3.6 (`:1078-1200`): the `nextn` block: `e = rms_norm(embed(next_token)) * enorm`,
  `h = rms_norm(rms_norm(R) * output_norm) * hnorm`, `cur = eh_proj . concat(e, h)`, `x = rms_norm(cur) * attn_norm`,
  `K = rope(k_norm(attn_k . x))`, `V = attn_v . x`, appended to the drafter's cache.
- Flash-Next (`:1201-1338`): `fc_embedding` and `fc_hidden` per stream, `add_streams_broadcast`, the attention
  hyper-connection read, then K / V the same way.

## Multi-GPU

- **Layer split** (`set_stage`, `prefill.hpp:158`): each stage's `Prefill` runs its layer range on its own device;
  a non-final stage copies its chunk rows to one of two pinned hand-off buffers and starts the next stage's
  `run_impl` on a `std::async` thread (`prefill.cpp:3196-3216`), so stage `i+1` reads chunk `c` while stage `i`
  reads chunk `c+1`. `run` drains the chain (`drain_pipeline`, `:3324`). Only the last stage calls `on_chunk`.
- **Peer GPU** (`set_peer`, `prefill.cpp:1472`): with `--peer-device`, per MoE layer the primary sends its normed
  activations over P2P, the peer quantizes and multiplies the rows routed to the experts it holds and sends result
  rows back into the primary's `Dm` (laid out local-first). Compact group buffers by default
  (`STRATA_PF_PEER_COMPACT`), per-group result pipelining (`STRATA_PF_PEER_OUT_PIPE`), and peer streaming: a share
  (`STRATA_PF_PEER_STREAM`, default 0.35) of the primary's streamed pinned experts go through a 48-slot peer ring
  (`STRATA_PF_PEER_RING`). MMQ path only.
- **Layer-split helper** (`set_stage_helper`, `prefill.cpp:1366`; `bind_stage_helper`, `:1408`): opt-in
  `STRATA_PREFILL_HELP=1`. For a one-chunk prompt of at least 1024 tokens, the next stage's GPU (idle until the hand
  over) streams a share of this stage's experts (`split_help_frac`: about 0.45 down to 0.3, off from about 3.3K
  tokens; `STRATA_PREFILL_HELP_FRAC` fixes it) through mapped host memory, reusing the helper stage's own buffers.

## How `generate.cpp` drives it

| Site | What happens |
| --- | --- |
| `src/program/generate.cpp:1440-1454` | `--prefill CHUNK|auto|auto:N`, `--no-prefill-borrow`, `--prefill-until P` parsed (`STRATA_PREFILL_AUTO_MAX` also raises the auto ceiling to 16384 / 32768) |
| `:4727-4737` | `set_pinned_share` from the arena's pinned bytes |
| `:4759-4880` | `lend_slots`, `request_chunk` (round to 256), `plan_lend` (`:4805`) and the auto chunk scan with `bytes_needed` / `bytes_needed_no_ring` / `ring_cap_for` / `set_ring_budget` |
| `:7909-8010` | `generate`: lend slots, `init`, `on_chunk` -> `draft_kv` / `mtp.prefill`, `run`, refill lent slots (queued, one wait; `STRATA_REFILL_BLOCKING=1` each), print `PrefillStats` |
| `:5237-5262`, `:5337-5360` | `serve`: per stage `set_stage` + `init` (stepping the chunk down 6144 .. 512 on "do not fit"), `set_peer`, `set_stage_helper` |
| `:7174-7216` | `serve` per request: `lend` the slots this prompt needs and `relayout` each participant |
| `:7243-7251` | `serve`: `sp.run` whole, or one chunk at a time between decode slots with `--batch` (`STRATA_BATCH_DECODE_SHARE`) |
| `:5723-5764` | `serve`'s `on_chunk`: draft K/V (batched `draft_kv` only on one GPU) and the `PP` progress line for the server window; on a layer split it is moved to the last stage (`:5763`) |

## Configuration

| Knob | Effect |
| --- | --- |
| `--prefill CHUNK`, `--prefill auto[:16384|:32768]` | chunk size; auto = the largest chunk whose buffers the cache can lend (up to 8192 by default) |
| `--no-prefill-borrow` | own buffers reserved for the session instead of borrowed cache slots |
| `--prefill-until P` | validation: batch only `[0, P)` and run the rest through the token path |
| `--peer-device`, `--peer-prefill-rows N` | peer GPU share of prompt rows (0 keeps the prompt on the primary) |
| `STRATA_PREFILL_TIMING=1` | per-phase GPU timeline and host shares |
| `STRATA_PF_STEP_SYNC=1` | sync after each step and log any over 250 ms (stall diagnosis) |
| `STRATA_PREFILL_BF16X2=1|2` | adds each BF16 activation's remainder as a second GEMM (about 16 mantissa bits): 1 includes the hyper-connection, 2 all but it; changes numbers, off by default |
| `STRATA_GR_UNFUSED=1` | keep the FP32 normalized rows and separate norm/mix/write (A/B of F-1 / F-2) |
| `STRATA_PLE_BATCH=0` | per-token PLE block |
| `STRATA_MTP_BATCH=0`, `STRATA_MTP_BATCH_RING=0`, `STRATA_MTP_BATCH_F16=1`, `STRATA_DRAFT_TIMING` | `draft_kv` controls |
| `STRATA_PREFILL_HELP`, `STRATA_PREFILL_HELP_FRAC` | layer-split helper |
| `STRATA_PF_PEER_COMPACT`, `STRATA_PF_PEER_OUT_PIPE`, `STRATA_PF_PEER_STREAM`, `STRATA_PF_PEER_RING` | peer GPU |
| `STRATA_PREFILL_ALLOC_TRACE`, `STRATA_TRACE`, `STRATA_PREFILL_DUMP_R=<file>`, `STRATA_DBG_NAN`, `STRATA_STATE_HASH_GDN` | debugging |

## Gotchas

- `bytes_needed` once over-counted (`xn` always, never `grs`); the corrected count is under the default
  `STRATA_RING_BYTES` and `STRATA_RING_BYTES=0` restores the 0.1.39 count (`prefill.cpp:1593-1605`).
- The MMQ identity row table lives in borrowed slots that the post-prompt refill overwrites, so it is rewritten at
  every `run` (`prefill.cpp:1815`).
- A model with no attention layer (the `gdn3` test fixture) has `qsa_alloc == 0`; every read of the primary attention
  state is guarded (`primary_cells`, `take_stage`), a bug the Qwen3.6 harness found.
- `draft_kv` for Qwen3.6 requires the MTP drafter's `qwen36()` tensors and an F32 `output_norm.weight`.
- `init` requires native GGUF blocks on every quantized dense projection (`--native`): `native_proj` fails with
  "has no native GGUF blocks (run with --native)" otherwise (`prefill.cpp:1668`).

## See also

[expert-streaming.md](expert-streaming.md), [moe.md](moe.md), [gdn-and-attention.md](gdn-and-attention.md),
[gemm.md](gemm.md), [tests.md](tests.md),
[docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md),
[docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md) (the Qwen3.6 `draft_kv`).
