[Codebase docs](../README.md) › [Program](README.md) › Prompt cache

# Prompt cache: conversation checkpoints, parking and slot reuse

**Purpose:** a chat client resends the whole conversation every turn. `--serve` avoids re-reading it by resuming
from the longest saved prefix of the new prompt. **Covers:** `include/strata/program/conv_cache.hpp`,
`src/program/conv_cache_test.cpp`, and in `src/program/generate.cpp` the helpers `:1079-1125`, the serve state and
lambdas `:5517-5781`, resume selection `:6793-7046`, segment boundaries `:7342-7415`, and the end-of-request update
`:7644-7650`. The checkpoint/snapshot data structures and copy routines (`ConversationCheckpoint`,
`ConversationCache`, `SavedConversation`, `conversation_checkpoint_save/restore`, `conversation_snapshot_*`) live in
[core](../core/README.md) (`src/core/conversation_state.cpp`, `conversation_snapshot.hpp`, `conversation_memory.hpp`).

## Positional vs running state

The comment at `:1079-1090` is the key idea:

- **Positional** state - the KV cache of the attention layers, their pooled indexer keys, the draft layer's KV - is
  written once per position and read only by later positions, so rewinding to position `L` is simply writing from
  `L` again.
- **Running** state - the GDN recurrences and conv histories, each attention layer's indexer tail, the PLE history -
  summarises "everything so far" and cannot be rewound. A **checkpoint** is a copy of exactly this set (the comment
  says ~118 MB; it is the set the verifier snapshots to roll back drafts). `conv_state_sizes` /
  `checkpoint_save` / `checkpoint_restore` wrap the core routines for one session's carve.

A checkpoint is only valid while the positional cells below it still hold its tokens. Since the KV cache holds one
branch of history at a time, the serve loop keeps only checkpoints that are prefixes of the live tokens: they form a
**chain** sorted by length.

## Retention policy (`conv_cache.hpp`)

`strata::program::conv_cache::eviction_victim(stamps, n, cap)` returns the index to erase when the chain holds more
than `cap` (`--prompt-cache`) items:

- `cap < 2` or `n < 2`: index 0 (the oldest; with one slot only the newest point stays - the pre-pin behaviour).
- otherwise the root (index 0) is **pinned** and the victim is the least recently used of the rest (earliest index on
  ties).

The root is in practice the end of the system prompt shared by every chat of a client; FIFO used to evict it after
`--prompt-cache` newer checkpoints, costing a full re-read (the header cites ~30 s for a 30K system prompt at ~1,000
tok/s). Stamps come from `check_clock`, advanced when a checkpoint is created, when a new one lands on an existing
length (`checkpoint_at`, `:5688-5689`), and when a request mounts through it (`:6998`).

## Where checkpoints are taken

`checkpoint_at(L, parts)` (`:5685-5722`) copies the running state after `cur[0, L)` (after `cudaDeviceSynchronize`;
a failed sync is reported as a GPU hang, #613), for CUDA0 and every split stage, records the images below `L`,
stamps it, appends, and evicts with `eviction_victim`. It is called:

| Point | Rule |
| --- | --- |
| Turn boundary | the last `--turn-token` (`<|im_start|>`, 248045) after `resume`: the conversation so far, before the new turn's header. With `--tail-role-token R`, if the turn before it opens with `R` (a trailing reasoning-effort turn), the checkpoint moves in front of that turn (#458) |
| Root | for a prompt read from 0 (or a resumed partial read that started at 0): the **first** turn boundary, if at least `--prompt-cache-root` (2048) tokens in (PR #65) |
| Periodic | every `--prompt-cache-every` (16384) freshly read tokens, from `sp.on_chunk` (`:5741-5759`). Under a layer split each stage saves its own part when it reaches the position (`on_stage_chunk`, `:5765-5780`) and the last stage assembles them |

The prompt is read in segments ending at these points (`for (to : {reread_to, root_at, turn_at, n - 1})`,
`:7373-7415`); checkpoints are taken at `turn_at` and `root_at`.

## Choosing where a request resumes (`:6795-7046`)

`starts_with(prefix, imgs)` requires the prefix's tokens **and pictures** to match the start of the prompt, with length
`1 .. n - 1` (the last token always goes through the first window). Candidates, longest wins:

1. `live` - what the session holds after the previous request (only if `live_ok` and the control-vector setting
   matches `cvec_cached`);
2. a checkpoint in `checks`;
3. an idle batch slot's tokens or its turn checkpoint (`slot_source`, text only);
4. a **parked conversation** (`conversations.best(ids, imgs, want_cvec)`) when `--conversation-cache-mib > 0`.

Then: if the outgoing session is not simply continued, `park_current` saves it (below); a slot source is copied back
(`copy_from_slot`); a parked conversation is validated per stage and restored (`conversation_snapshot_restore`,
fatal if it fails after validation), with its retained K/V handed back to the cache for its next park; checkpoints
not on this prompt's path are erased; `resume == 0` zeroes the sessions; a non-live resume restores the checkpoint on
every stage; `mtp.kv_restore(resume)` trims the drafter ring. `STRATA_CKPT_REREAD=1` instead re-reads the
checkpoint's tokens from 0 (and disables short window reads) to verify a restore token for token.

After decode, unless cancelled, `live = consumed` (prompt + committed tokens) and `live_ok = --prompt-cache > 0`.

## Parked conversations (`--conversation-cache-mib`)

`ConversationCache conversations(budget, slots)` holds whole conversation images in host RAM so switching between
independent chats does not lose either. `park_current(held)` (`:5536-5655`) drops superseded copies of the same
conversation (#342), sizes the snapshot (reusing retained K/V buffers when possible), `make_room`s within the budget,
checks a physical-RAM floor (`--conversation-cache-min-free-mib`) before and after capture with
`conversation_memory_admit`, then `conversation_snapshot_save`s CUDA0's image and one per later stage (the draft
layer's K/V goes with the last stage) and `put`s it. Any failure to fit is logged and skipped, never fatal;
`std::bad_alloc` is caught. Parking is refused for a single-GPU split (`--split-device 0`) at start.
`STRATA_SNAPSHOT_FULL_CAPTURE` disables retaining K/V buffers; `STRATA_SNAPSHOT_VERIFY` logs drafter-KV paths and
verifies a restored draft KV hash.

## Batch-slot caches

A batch slot that finished keeps `ids` and its best turn checkpoint (`BSlot::checks`) while idle (`cached`); a
pipelined slot or one with pictures is not reused. `BYIELD` copies a partially read prompt into a slot
(`partial`), and the resent request resumes there with its checkpoints (`:7243-7301`, `:6893-6897`).

## Configuration

`--prompt-cache` (6; 0 = off), `--prompt-cache-every` (16384), `--prompt-cache-root` (2048), `--turn-token` (248045),
`--tail-role-token` (-1), `--conversation-cache-mib` (0), `--conversation-cache-slots` (4),
`--conversation-cache-min-free-mib` (2560), `--short-read`. Env: `STRATA_CKPT_REREAD`, `STRATA_SNAPSHOT_VERIFY`,
`STRATA_SNAPSHOT_FULL_CAPTURE`, `STRATA_STATE_HASH`, `STRATA_STATE_HASH_GDN`.

## Model-specific behaviour

Qwen3.6 has no PLE and no indexer, so its checkpoints hold only GDN state; step 5 fixed checkpoint/snapshot code that
had demanded indexer buffers ([step5-serve.md](../../qwen36-port/step5-serve.md), fault 2).

## Tests

`conv_cache_test` (`src/program/conv_cache_test.cpp`, CTest `conv_cache_test`) walks the policy by hand: over budget
with fresh stamps the oldest leaf leaves (the old FIFO); the root's stale stamp never makes it the victim; a mount moves
the rotation to the next LRU leaf; a two-slot budget keeps root + newest; a one-slot budget unpins; ties pick the
earlier index deterministically. Build with `-DSTRATA_BUILD_TESTS=ON` (or `STRATA_BUILD_CONVERSATION_TESTS`) and run
`ctest -R conv_cache_test`. `STRATA_STATE_HASH=1` (serve) prints fingerprints of every part of the session
(`STATE_HASH L=... gdn=... kv=... mtp=...`) for comparing a restored session with a re-read one.

## See also

- [serve-mode.md](serve-mode.md), [layer-split.md](layer-split.md), [../core/README.md](../core/README.md)
