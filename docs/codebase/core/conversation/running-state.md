[Codebase docs](../../README.md) › [Core](../README.md) › [Conversation state](README.md) › Running state and checkpoints

# Running state, checkpoints and whole-session snapshots

## Purpose

A model with recurrent layers cannot resume from the K/V cache alone: the GDN layers carry a state that summarises
every token so far. This page covers:

- what that running state is;
- how a `ConversationCheckpoint` copies it out and back;
- how a whole-session snapshot (`SavedConversation`) is assembled, validated and restored around the per-layer K/V
  images of [kv-images.md](kv-images.md);
- how the serve loop builds, prunes and uses its chain of checkpoints.

Files: [src/core/conversation_state.cpp](../../../../src/core/conversation_state.cpp),
[include/strata/core/conversation_snapshot.hpp](../../../../include/strata/core/conversation_snapshot.hpp),
[include/strata/core/conversation_cache.hpp](../../../../include/strata/core/conversation_cache.hpp) (types),
[src/core/conversation_checked.hpp](../../../../src/core/conversation_checked.hpp).

## What is in a checkpoint

`ConversationCheckpoint` (`conversation_cache.hpp:23-39`):

| Field | Source buffer (device) | Size (`conversation_state_sizes`, `conversation_state.cpp:110-136`) |
|---|---|---|
| `ids` | the token prefix | `n` int32 |
| `imgs` | `ConversationImageKey{start, hash}` per image below `n` | strictly increasing `start` within `[0, n)` (`image_keys`, `:46-53`) |
| `gdn` | `SessionState::gdn_state` | per GDN layer `(S * v_heads * S + conv_channels * (d_conv - 1))` floats: the delta-rule recurrent state **and** the causal-conv tail, one block per owned GDN layer (`gdn_alloc`) |
| `ple` | `SessionState::ple_hist` | `NG_HIST * NG_HC_DIM` floats when the model has a PLE (Flash-Next), else 0 |
| `tails` | `QsaState::idx_tail` per owned QSA layer | `(idx_block - 1) * idx_key_dim` floats: the indexer's incomplete block |
| `dead` | `QsaState::idx_dead` | `idx_key_dim` floats: the indexer's moving spare (pooled) row |
| `block_pos` | `QsaState::idx_block_pos` | one int32 when the model has an indexer |
| `used` | | LRU stamp for the in-session chain |
| `stage_parts` | | Layer split: one part per later stage, each with the same ids (see [conversation-cache.md](conversation-cache.md)) |

The K/V cells are **not** in a checkpoint. In the session they are positional and persist in place. A checkpoint is
therefore valid only while the session's cells `[0, n)` still hold that prefix.

### The geometry key and the layer carve

`geometry_key(g)` (`:13-18`) is the 18 integers:

```
n_embd, n_layers, qsa_interval, ssm_state_size, ssm_k_heads, ssm_v_heads, ssm_d_conv, ssm_conv_channels,
ssm_value_dim, n_head, n_head_kv, head_dim, idx_q_heads, idx_key_dim, hc, hc_lr, n_expert, n_ff
```

All must be positive, with three exceptions (`:113-118`):

- `n_layers` (key index 1) may be 0;
- `idx_q_heads` and `idx_key_dim` may be 0 when the model has no indexer;
- `hc_lr` may be 0 when `hc == 1`.

A snapshot carries this key and must match it exactly. It is "runtime compatibility only; NOT a model/weights identity
or disk schema" (`conversation_cache.hpp:123`). Parked images live only in this process's RAM.

`conversation_session_sizes(g, ss, sizes, err)` (`:138-148`) applies the session's **carve** (#216). A layer-split stage
owns `gdn_alloc` GDN rows and QSA states `[qsa_ord0, qsa_ord0 + qsa_alloc)`, and every save, validate and restore
touches only those. A whole-model checkpoint is refused by a carved session and vice versa (tested).

## Functions

| Function | Lines | Behaviour |
|---|---|---|
| `conversation_state_sizes(g, sizes, err)` | `:110-136` | Whole-model sizes with overflow-checked arithmetic (`conversation_detail::add` / `product`, `conversation_checked.hpp:9-22`) |
| `conversation_session_sizes(g, ss, sizes, err)` | `:138-148` | The same, scaled to the carve. Rejects an inconsistent carve |
| `conversation_checkpoint_validate(c, ss, g, err)` | `:150-160` | Targets exist (`checkpoint_targets`, `:55-75`: buffers present, `n <= max_cells`, indexer pooled rows in range); payload sizes equal the carve's; image keys well formed |
| `conversation_checkpoint_save(c, ss, g, err)` | `:162-178` | Resizes and `cudaMemcpy(Default)`s each buffer out. The caller has synchronised the device |
| `conversation_checkpoint_restore(c, ss, g, err)` | `:180-201` | Validates, copies back, and **rewrites the indexer's pooled row** at `n / idx_block` from `dead` (the moving spare row the restored prefix expects). Reconstructs `ss.ple_prev` from the last two ids, then `cudaDeviceSynchronize` |
| `conversation_snapshot_bytes(view, ss, g, draft, bytes, err)` | `:203-228` | Validates the live view (`view_validate`, `:77-98`: non-empty non-negative ids, every checkpoint a prefix of the live ids with matching images) and sums metadata, running state, checkpoints and each layer's K/V estimate |
| `conversation_snapshot_capture_bytes(reuse, ...)` | `:230-253` | The same, but for layers whose retained K/V from a restored image is reused: the fresh estimate is swapped for the retained buffers' growth peak |
| `conversation_snapshot_save(image, view, ss, g, draft, err, reuse, reused_bytes)` | `:255-282` | Sync, then build into a **new** object (a failure never publishes a partial image): geometry key, carve, live ids/images/checkpoints, `cvec`, the live running state, one `conversation_kv_save` per owned QSA layer, and the draft layer's with `unchanged - 1`. The draft's final cell may not have been computed when the output cap was reached |
| `conversation_snapshot_validate(image, ss, g, draft, err)` | `:284-300` | No stage parts in `live`; same geometry key and carve; the view; the live checkpoint; K/V layer count `qsa_alloc + (draft ? 1 : 0)`; each K/V image (`conversation_kv_validate`). **No CUDA call** |
| `conversation_snapshot_restore(image, ss, g, draft, err)` | `:302-313` | Validate (returns `invalid`), sync, each layer's K/V, the draft's, then the running state. Any copy failure returns `transfer_failed` |
| `&draft` / `nullptr` overloads | `:315-339` | With a draft `QsaState&` (whole session, or the stage that owns the draft), or a pointer that may be null (a later layer-split stage). The whole-session `validate` refuses an image that carries `stage_images` |

## How the serve loop uses checkpoints

All in `src/program/generate.cpp`, `--serve`:

- **Creating** (`checkpoint_at`, `:5685-5725`). Only when `--prompt-cache > 0`. If a checkpoint of that length exists,
  its stamp is refreshed. Otherwise the device is synchronised (a failed sync is reported as a GPU hang, #613) and
  `conversation_checkpoint_save` runs for the main session and every later stage. A layer split's mid-prompt
  checkpoints are saved by each stage as it reaches the position (`part_at`). Points:
  - the system prompt's end (`root_at`): the first `--turn-token` boundary when the prompt is read from token 0 and that
    boundary is at least `--prompt-cache-root` (2048) tokens in;
  - the last turn boundary before the new turn (`turn_at`). `--tail-role-token` can pull it before a short trailing
    turn of that role (#458);
  - every `--prompt-cache-every` (16384) fresh tokens (`:5753-5755`, `:7411`).
- **Retention.** Above `--prompt-cache` entries, `conv_cache::eviction_victim` removes the least recently used
  checkpoint and **never the root** (see [conversation-cache.md](conversation-cache.md)).
- **Matching** (`:6812-6846`). The live session counts as a source when the prompt starts with it. So does any
  checkpoint in `checks` that is a prefix, as do idle batch slots and parked conversations. The longest one wins.
- **Pruning** (`:6979-6982`). Before reading, checkpoints longer than `resume` or not on this prompt's path are erased:
  their cells will be overwritten.
- **Restoring** (`:6994-7027`). From a checkpoint (`from_live == false`): `conversation_checkpoint_restore` for the main
  session and each stage part, plus `mtp.kv_restore(resume)` for the draft ring. `STRATA_CKPT_REREAD=1` instead
  re-reads the prefix from token 0, to check that the checkpoint missed nothing (with `--adapt-swaps 0` the answers
  must match). At `resume == 0` the session is zeroed (`session_zero`).
- **Batch slots** (`--batch`). `conversation_checkpoint_save` followed by `..._restore` copies running state between
  the main session and a slot's session (`:6303-6336`).

## Configuration

| Flag / env | Default | Effect |
|---|---|---|
| `--prompt-cache N` | 6 | Checkpoints kept (0 = read every prompt from the start) |
| `--prompt-cache-every N` | 16384 | Extra checkpoint every N fresh prompt tokens (0 = off) |
| `--prompt-cache-root N` | 2048 | Minimum system-prompt length for the root checkpoint |
| `--turn-token ID` | 248045 (`<\|im_start\|>`) | Turn boundary token |
| `--tail-role-token ID` | -1 | Keep a trailing turn of this role out of the checkpoint |
| `STRATA_CKPT_REREAD=1` | off | Validation: re-read instead of restoring |

## Model-specific behaviour

- **Flash-Next:** QSA layers with the lightning indexer (`has_indexer()`: `ModelArch::Qwen4Exp`, `layout.hpp:86`),
  a PLE history, and hyper-connections (`hc > 1`).
- **Qwen3.6:** dense attention (no indexer, so `tails`, `dead`, `block_pos` and pooled rows are 0), no PLE (`ple`
  empty, `ple_hist == nullptr`), and `hc == 1`. Before the Qwen3.6 serve port the checkpoint rejected these zeros as
  "invalid model geometry" ([step5-serve.md](../../../qwen36-port/step5-serve.md), row 2, commit 3a806b5). The allowed
  zeros at `:113-118` and `:64-67` are the fix.

## Tests

- `conversation_validation_test` (`src/core/conversation_validation_test.cpp`, no CUDA initialised). Fixtures in
  canonical, pruned and zero-QSA geometries, with and without PLE. It checks:
  - a complete image validates;
  - the byte estimate equals `image.bytes()`;
  - each corruption returns `invalid` with state unchanged;
  - missing QSA targets are rejected;
  - a whole-model checkpoint or snapshot is refused by a carved session, and a carve-sized one validates;
  - a carve past the last QSA ordinal or the GDN rows is rejected;
  - overflow and a zero layer interval are rejected.
- `conversation_transfer_test` (the same file with `CONVERSATION_TEST_TRANSFERS`). Wraps `cudaMemcpy`,
  `cudaDeviceSynchronize` and `cudaGetLastError` to inject failures:
  - a pre-sync failure and a first-copy failure leave state unchanged;
  - a later failure produces partial state and `transfer_failed`;
  - a successful restore reproduces `gdn`, `ple` and `ple_prev`;
  - incremental capture copies exactly `full - reused` bytes;
  - a failed incremental capture publishes nothing.
- `conversation_snapshot_test` (CUDA, `full_session`, `:107`). Captures A and B, refuses a bad image without touching
  the session, restores A and re-captures it byte-identical (`gdn`, `ple`, `tails`, …), and checks
  `ple_prev == {64, 65}`. Covers growth and rewind with retained pages, and stage images (no draft layer, smaller by
  the draft ring).

## Gotchas

- The caller synchronises before a save. Restore synchronises itself.
- `transfer_failed` means a half-restored session. The serve loop answers `ERR restoring parked conversation` and exits
  (`generate.cpp:6918-6922`), so the server restarts the engine.
- The checkpoint restores only the running state. Restoring from a checkpoint whose K/V cells have been overwritten
  would be silently wrong, which is why `checks` is pruned to prefixes of each new prompt.

## See also

[kv-images.md](kv-images.md) · [conversation-cache.md](conversation-cache.md) ·
[docs/qwen36-port/step5-serve.md](../../../qwen36-port/step5-serve.md) · [docs/qwen36-port/step3-gdn.md](../../../qwen36-port/step3-gdn.md)
