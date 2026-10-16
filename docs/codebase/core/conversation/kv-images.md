[Codebase docs](../../README.md) › [Core](../README.md) › [Conversation state](README.md) › K/V images

# K/V images: capturing and restoring an attention layer's cache

## Purpose

A parked conversation must carry the K/V cache, because the next conversation will overwrite the session's cells.
`conversation_snapshot.cpp` copies one attention layer's K/V into a `ConversationKv` and back. A layer here is one
owned QSA or attention layer, or the MTP draft layer. It handles:

- every KV storage format the engine has;
- the three residency modes (all in VRAM, streamed from host, the draft ring);
- **incremental** re-capture, which reuses pages unchanged since the image was restored;
- a read-back verifier.

Files: [src/core/conversation_snapshot.cpp](../../../../src/core/conversation_snapshot.cpp) (269 lines),
[include/strata/core/conversation_snapshot.hpp](../../../../include/strata/core/conversation_snapshot.hpp) (`:11-31`),
[include/strata/core/conversation_cache.hpp](../../../../include/strata/core/conversation_cache.hpp) (`ConversationKv`,
`:43-50`; `ConversationKvReuse`, `:52-64`).

## `ConversationKv`

| Field | Meaning |
|---|---|
| `format` | `qsa_kv_format(st)` (`kKvF16 = 0`, `kKvInt8 = 1`, `kKvQ4 = 2`; `include/strata/kernels/kv_stream.hpp:46`). **3** = hybrid K8V4 (snapshot-only id, never passed to the block movers). **+16** when INT8 K/V is rotated (`kv_rot`, #293, `STRATA_KV_ROT=1`) |
| `cells` | `upto` rounded up to whole pages (`page_size` from `qsa_real_shapes()`) |
| `heads`, `head_dim`, `page_size`, `pooled_rows`, `idx_dim` | Geometry the image was captured with |
| `k`, `v`, `k_scale`, `v_scale`, `pooled` | `ConversationBuffer`s (segmented host storage; see [conversation-cache.md](conversation-cache.md)) |

## Layout (`layout`, `conversation_snapshot.cpp:27-72`)

- Bytes per head per cell:

  | Format | Bytes |
  |---|---|
  | F16 | `head_dim * 2` |
  | INT8 (and the K of hybrid) | `head_dim` |
  | Q4_0 | `kv_q4_bytes_per_head(head_dim)` |

  `data = cells * n_head_kv * per`.
- INT8 scales: `cells * n_head_kv * (head_dim / 64) * 2` bytes. None for Q4_0 or F16.
- Hybrid K8V4: V is Q4 (`value_data`) and has no V scales.
- Pooled indexer rows: `upto / idx_block + 1`, only when the layer has an indexer. The `+1` is the indexer's moving
  spare row, included so a checkpoint restore can rebuild it when rewinding. The draft layer passes `index = false`.
- Refusals: hybrid with any other format flag; `head_dim` not a multiple of 32 (Q4) or 64 (INT8); extents past
  `max_cells`; any byte-count overflow.

## Which buffers are authoritative (`pools`, `:74-83`)

| `kv_mode` (`include/strata/core/layer.hpp:230-234`) | Meaning | Captured from / restored to |
|---|---|---|
| 0 | Every page in VRAM, identity page table | The device pools (`k_pool`/`v_pool`, `k_q`/`v_q` + scales, `k_q4`/`v_q4`) |
| 1 | Streamed: authoritative K/V in host pools, `n_slots` pages cached in VRAM | **The host pools** (`st.host.*`), not the replaceable VRAM slots |
| 2 | Ring (the MTP drafter's window) | The host copy. After restore, the resident ring pages are rebuilt |

Hybrid K8V4 states and `idx_pooled` always use the device buffers. `valid` (`:85-108`) checks the page counts, the
slot counts, the streaming map's pointers (mode 1) and the ring's pools (mode 2) before anything moves.

## Functions

| Function | Lines | Behaviour |
|---|---|---|
| `conversation_kv_capture_bytes(image, st, g, upto, index, bytes, err)` | `:121-136` | Sum of each buffer's `allocation_peak(size)`: what growing *this* image to `upto` would allocate, including the transient segment directory |
| `conversation_kv_bytes(st, g, upto, index)` | `:138-142` | The same for a fresh image (0 on error) |
| `conversation_kv_save(image, st, g, upto, index, err, unchanged_tokens, reused_bytes)` | `:144-178` | Sets the header fields, resizes each buffer, and copies with `cudaMemcpy(cudaMemcpyDefault)`, one segment at a time. With `unchanged_tokens > 0`, the whole pages before it (and the indexer rows before `unchanged / idx_block`) are **kept** and only the rest is copied. `reused_bytes` accumulates what was not transferred |
| `conversation_kv_validate(image, st, g, upto, index, err)` | `:180-199` | Format, cells, heads, head_dim, page size, pooled rows and idx_dim must match the target's layout. Each payload size must match, and each target buffer must exist. **No CUDA calls** |
| `conversation_kv_restore(image, st, g, upto, index, err)` | `:201-224` | Validate, then copy each buffer back. Mode 1: `kv_stream_reset(st.map)`, because the VRAM slots still hold the outgoing conversation and must be refilled from the restored host pools before any attention read. Mode 2: `kv_ring_restore` of the last `n_slots` pages. Then `cudaGetLastError` |
| `conversation_kv_verify(image, st, g, upto, index, fingerprint, err)` | `:226-267` | Diagnostic. Reads every authoritative byte back through a 64 KiB stack buffer, compares, and FNV-1a hashes it. For a ring, also compares the resident pages at `page % n_slots`. Never changes state |

## Incremental capture (retained K/V)

When a parked image is restored, the serve loop does not free its K/V buffers. `ConversationCache::retain(kv, tokens)`
keeps them (`generate.cpp:6962-6966`). Every rewrite then lowers `unchanged_tokens` (`limit_reuse(first_dirty)`,
`generate.cpp:7034`).

At the next park, `take_reuse()` hands the buffers to `conversation_snapshot_save`, which re-captures into them. Pages
wholly before the first rewritten token are kept, and the partial page and everything after it are copied again. The
draft layer uses `unchanged - 1`, so its last page is always refreshed. `ConversationBuffer::resize` never relocates
existing segments, so the kept payload keeps its address (tested).

The admission estimate (`conversation_snapshot_capture_bytes`) counts each retained buffer's growth peak in place of a
fresh image. `STRATA_SNAPSHOT_FULL_CAPTURE` disables the retention.

## Configuration

| Env / build | Effect |
|---|---|
| `--kv fp16\|int8\|q4_0\|k8v4` (`k8v4` = hybrid) | Determines `format`. An image restores only into a state of the same format |
| `STRATA_KV_ROT=1` | Rotated INT8 K/V: a distinct format (+16), so rotated and unrotated images never cross |
| `STRATA_SNAPSHOT_VERIFY` | After a parked restore, `conversation_kv_verify` on the draft layer, logging `SNAPSHOT_VERIFY draft=<hash>` (`generate.cpp:6930-6946`) |
| `STRATA_SNAPSHOT_FULL_CAPTURE` | Do not retain restored K/V; every park is a full capture |

## Model-specific behaviour

Qwen3.6's attention has no indexer, so `pooled_rows` is 0 and there is no `pooled` payload (`:45-46`). Flash-Next's QSA
layers carry the pooled indexer rows. Both have the draft layer's K/V when parked on one GPU.

## Tests

- `conversation_snapshot_test` (CUDA). `full_session(fmt, mode, experts)` runs each KV format in each residency mode. It
  covers:
  - captures, refused restores and byte-identical round trips;
  - growth and rewind fixtures where "only complete unchanged pages or rows are retained" and the incremental image
    stays within the admitted peak;
  - restoring a segmented incremental snapshot;
  - `conversation_kv_verify` on the draft ring.
- `conversation_validation_test`: "KV byte overflow rejected", a malformed retained draft rejected before admission
  with no CUDA calls, and read-back detecting a corrupted byte or a transfer failure (transfer variant).

## Gotchas

- Equal token ids do not prove that the K/V bytes are unchanged. `unchanged_tokens` is valid only for storage retained
  from an image that was actually restored into this session (header comment, `conversation_snapshot.hpp:14-16`).
- The mode-1 restore must reset the streaming map. Otherwise attention would read the previous conversation's VRAM
  slots.
- A layer-split stage's image has no draft layer (`draft == nullptr`). Validate and restore it the same way it was
  saved.

## See also

[running-state.md](running-state.md) · [conversation-cache.md](conversation-cache.md) ·
[docs/qwen36-port/step3-attn.md](../../../qwen36-port/step3-attn.md)
