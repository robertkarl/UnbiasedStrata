[Codebase docs](../../README.md) › [Core](../README.md) › [Conversation state](README.md) › Conversation cache

# The conversation cache: parking, matching, budgets

## Purpose

`ConversationCache` holds whole-session images (`SavedConversation`) of conversations that are not the session's
current one. It is opt-in, and only active when `--conversation-cache-mib` is above 0 and `--prompt-cache` is above 0.
When a request matches a parked conversation's prefix better than anything in the live session, the serve loop:

1. parks the outgoing conversation;
2. restores the parked one;
3. reads only the new tail.

This page covers the cache's policy (matching, byte and slot budget, eviction, superseded copies, retained K/V), its
storage type (`ConversationBuffer`), the physical-RAM admission check, the layer-split split and merge, and the
separate retention policy for the in-session checkpoint chain (`conv_cache.hpp`).

Files: [include/strata/core/conversation_cache.hpp](../../../../include/strata/core/conversation_cache.hpp),
[include/strata/core/conversation_buffer.hpp](../../../../include/strata/core/conversation_buffer.hpp),
[include/strata/core/conversation_memory.hpp](../../../../include/strata/core/conversation_memory.hpp),
[src/core/conversation_memory.cpp](../../../../src/core/conversation_memory.cpp),
[include/strata/program/conv_cache.hpp](../../../../include/strata/program/conv_cache.hpp).

## `SavedConversation` (`conversation_cache.hpp:122-142`)

| Field | Meaning |
|---|---|
| `geometry` | The 18-value runtime geometry key (see [running-state.md](running-state.md)) |
| `layer_lo`, `layer_hi` | The session carve it came from. A restore needs the same carve |
| `live` | The live state as a checkpoint: ids, images, running state |
| `checkpoints` | The in-session checkpoint chain at park time, restored with it |
| `kv` | One `ConversationKv` per owned attention layer, then the draft layer's |
| `cvec` | The control-vector (steering) mode it was read with |
| `stage_images` | Layer split: the later stages' own images, in stage order |

`bytes()` sums capacities, not sizes, so the budget charges what is really allocated.

## `ConversationCache` (`conversation_cache.hpp:159-281`)

Built as `ConversationCache(budget_bytes, slots)` (`generate.cpp:5531-5533`). `enabled()` requires both values to be
non-zero.

| Member | Lines | Behaviour |
|---|---|---|
| `best(prompt, images, cvec)` | `:196-211` | Scans entries newest first. For each it tries `live` and every checkpoint with `conversation_prefix`, skipping entries of the other `cvec` mode. Returns the longest match `{index, tokens, live}`. Ties go to the most recently parked entry |
| `take(index)` | `:213-218` | Removes the entry and returns it (for restore) |
| `put(image, held)` | `:267-275` | Refuses an image larger than `budget - held`. Then `drop_superseded`, then `make_room`, then appends as newest |
| `make_room(incoming, held)` | `:222-231` | First drops retained K/V if needed. Then evicts the **oldest** entries (front of the deque) while the slot count is reached or bytes exceed `budget - held - incoming`. `held` is an incoming image already `take`n but still alive during the exchange, so it counts against the budget too |
| `can_fit(incoming, held)` | `:190-193` | The same test, without evicting |
| `drop_superseded(ids, images, checkpoints, cvec)` | `:240-264` | #342. Drops parked copies of *this* conversation from a turn back: entries whose deepest checkpoint the outgoing chain still holds. Without it, a subagent's successive turns each parked one copy and pushed the parent conversation out. Entries that only share the system-prompt root are kept |
| `retain(kv, tokens, stage_kv)` / `limit_reuse(first_dirty)` / `take_reuse()` | `:176-188` | Keeps the K/V buffers of the image just restored, for an incremental re-capture at the next park (see [kv-images.md](kv-images.md)). Retained bytes count toward the budget but take no slot. `retain` never evicts a parked entry to fit; under pressure `make_room` drops the retained buffers first. A rewrite from position 0 drops all reuse |
| `bytes()`, `size()`, `evictions()`, `superseded()`, `retained_bytes()` | | Accounting, logged on every park |

### The park sequence (`park_current`, `generate.cpp:5536-5655`)

Parking happens only when switching, rewinding or resuming from another source, never on every request. The steps:

1. `drop_superseded`.
2. Estimate the image size. `conversation_snapshot_bytes` gives the fresh size; `conversation_snapshot_capture_bytes`
   gives the size with retained K/V. Under a layer split, each later stage adds its own estimate.
3. `make_room(estimate, held)`. If that fails, log "skip parking" and continue: parking is best effort.
4. **Physical RAM admission:** `conversation_memory_admit(conversation_available_memory(), additional, floor)`, where
   `additional` = estimate minus retained bytes and `floor` = `--conversation-cache-min-free-mib`.
5. `conversation_snapshot_save`, plus each stage's image.
6. A second admission check with zero allocation, after the capture (the floor must still hold).
7. `put`. On `std::bad_alloc`, log and skip without touching the active state.

## `ConversationBuffer` (`conversation_buffer.hpp:16-152`)

The byte container for snapshot payloads. It holds a vector of segments of at most `segment_bytes = 16 MiB`.

- **Appending never relocates existing payload.** Only the small segment directory moves. Retained pages therefore
  keep their addresses during an incremental re-capture.
- `allocation_peak(n)` (`:47-61`) predicts the bytes that `resize(n)` would hold, including the old directory while
  `reserve` allocates its replacement. Any overflow saturates to `SIZE_MAX`, so admission fails before the resize.
- Segment capacity grows geometrically from 64 KiB up to 16 MiB when an existing buffer grows (`segment_capacity`,
  `:126-129`). Thousands of small chat turns therefore add a few segments, not thousands of restore transfers (tested:
  at most 4 segments after 4096 appends).
- `visit(offset, count, fn)` walks segments for the copies, and `fn` returning false aborts. `read` is built on it.
  `operator==` compares content regardless of segmentation.

## Physical RAM admission (`conversation_memory.hpp:11-17`, `conversation_memory.cpp`)

- `conversation_mem_available(istream)` (`.cpp:18-34`) parses exactly one well-formed `MemAvailable: N kB` line. A
  missing, duplicated, signed, oversized or trailing-text value, or an I/O error, means **unknown** (`nullopt`). A
  measured zero is a valid answer.
- `conversation_available_memory()` (`.cpp:36-49`) reads `/proc/meminfo` on Linux and `ullAvailPhys` on Windows.
  Elsewhere it returns unknown.
- `conversation_memory_admit(available, allocation, floor)` returns `available && *available >= floor &&
  allocation <= *available - floor`. Unknown fails closed, and the arithmetic cannot underflow or overflow.

This probe is separate from the expert tier's `host_available_memory`, and it does not look at cgroup limits.

## Layer split: split and merge (`conversation_cache.hpp:66-120`)

Under a layer split, each running checkpoint holds stage 0's state, plus one `stage_parts` entry per later stage.

- `conversation_checkpoints_split(checks, stages)` **moves** them apart:
  - complete checkpoints go into `stage0` and `parts[k]`, each part carrying the checkpoint's ids, images and stamp;
  - checkpoints without a full set of parts go into `rest`.
- `conversation_checkpoints_merge` puts them back in the original order. It refuses (returning an empty output) when a
  stage's part count or ids do not line up. `merge(split(x)) == x`.

`park_current` uses these so each stage's image carries its own checkpoint parts, with no running state copied
(`generate.cpp:5555-5560`). A restore merges them back (`:6949-6956`).

## In-session retention: `conv_cache.hpp`

`strata::program::conv_cache::eviction_victim(stamps, n, cap)` (`conv_cache.hpp:35-41`) picks which checkpoint of the
session's chain leaves when more than `--prompt-cache` are held. Because the K/V arena holds one branch of history,
the retained checkpoints always form a chain: each is a prefix of the next. Index 0 is the root, usually the end of the
system prompt.

- With `cap >= 2`, the root is **pinned** and the least recently used of the others leaves. A stamp advances on
  creation, when a new checkpoint lands on the same length, and when a request mounts through it.
- With `cap < 2`, the oldest leaves.

The header gives the reasoning: dropping a 30K-token root costs every later chat about 30 s of re-reading.

## Configuration

| Flag | Default | Effect |
|---|---|---|
| `--conversation-cache-mib N` | 0 (off) | Byte budget for parked conversations |
| `--conversation-cache-slots N` | 4 | Maximum parked conversations |
| `--conversation-cache-min-free-mib N` | 2560 | Physical RAM floor for parking |
| `--prompt-cache N` | 6 | In-session checkpoints. With 0, parking is also off (warning at `generate.cpp:1596-1598`) |

A layer split with `--split-device 0` (both stages on one GPU) and parking enabled is refused at start-up (exit 2,
`generate.cpp:1635-1638`). The front end passes none of the parking flags itself. They can be given after `--`, which
`frontend/main.cpp` forwards.

## Model-specific behaviour

The policy is model-independent. Image sizes differ by model: Flash-Next images include the indexer's pooled rows and
the PLE history, and Qwen3.6 images have neither (see [running-state.md](running-state.md)). Image keys
(`ConversationImageKey{start, hash}`) only matter with `--vision`. The front end refuses `--vision` as an engine flag;
image input there goes through `--mmproj`.

## Tests

- `conversation_cache_test` (CPU, `src/core/conversation_cache_test.cpp`). Covers:
  - buffer segmentation: payload addresses preserved, peak prediction, cross-segment reads, equality after rewind,
    overflow;
  - small appends staying within the predicted peak, with at most 4 segments;
  - retained storage counted but taking no slot, `limit_reuse` monotonic, pressure dropping reuse before evicting;
  - A/B/A recovery, eviction by slot limit, byte accounting after `take`;
  - an edited suffix falling back to a parked checkpoint;
  - steering-mode isolation, and the last-token rule (equal-length, short and empty prompts do not match);
  - ties going to the most recently parked entry;
  - image identity (same, different hash, missing, extra, after the prefix);
  - in-flight (`held`) accounting and oversized images refused without evictions;
  - superseded copies (8 dropped, the parent kept, a sibling sharing only the root kept);
  - disabled caches;
  - split and merge round trip by move, with torn or mixed parts refused;
  - per-stage retained K/V.
- `conversation_memory_test` (CPU): malformed telemetry is unknown; only `MemAvailable` counts; zero is known; the
  admission edge cases and overflow. On Windows it also checks the provider returns a physically bounded value.
- `conv_cache_test` (CPU, `src/program/conv_cache_test.cpp`): FIFO with fresh stamps, the root never chosen, LRU after
  a mount, one- and two-slot budgets, purity.

## Gotchas

- Matching requires the image keys too. The same pad tokens with different pixels do not match.
- A restored parked image's own checkpoints become the session's chain. Under a layer split, a merge failure clears
  the chain (`checks.clear()`) rather than keeping torn parts.
- Parking is best effort. Every refusal path logs `conversation cache: skip parking (...)` and continues.

## See also

[running-state.md](running-state.md) · [kv-images.md](kv-images.md) ·
[docs/qwen36-port/step5-serve.md](../../../qwen36-port/step5-serve.md)
