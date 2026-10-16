[Codebase docs](../../README.md) › [Core](../README.md) › [Experts](README.md) › VRAM expert cache

# The VRAM expert cache and the expert profile

## Purpose

`ExpertCache` is the VRAM tier. It allocates slot storage on the current device, keeps a `(layer, expert) -> slot`
residency table, fills slots from host blobs, and can read a slot back to prove it holds the expert it claims. It
computes nothing itself: the verify window's grouped kernels, the token path's hit kernel and the prompt path read
the slots. The same class backs the extra GPU tiers (`PeerExperts`, `RemoteExperts`) on their devices.

This page also covers the expert profile: `read_expert_profile`, `rank_learned_profile` and `write_expert_profile`.
The profile decides which experts start resident.

Files: [include/strata/core/expert_cache.hpp](../../../../include/strata/core/expert_cache.hpp),
[src/core/expert_cache.cpp](../../../../src/core/expert_cache.cpp).

## `ExpertCache`

### Allocation

| Function | Lines | What it does |
|---|---|---|
| `open(n_slots, n_layers, n_expert, blob_bytes, err)` | `expert_cache.cpp:403-480` | Reads `cudaMemGetInfo` **before** allocating and refuses, naming both numbers, if free VRAM is below `n_slots * blob_bytes` ("Lower --expert-cache"). Then makes one `cudaMalloc`, or the segmented arena (below). The arena is zeroed so that an unfilled slot is a deterministic wrong answer. Builds the residency table (`kNotResident = -1`) and seeds per-layer cursors. On HIP it also allocates a one-blob pinned staging buffer for blocking fills. |
| `open_sized(slot_bytes, n_layers, n_expert, err)` | `:482-513` | Native packs: slots of different sizes, back to back, each rounded up to 256 bytes. Allocates through `open` with 1-byte "slots" (so the same VRAM check applies), then sets `off_` (slots + 1 offsets) and re-seeds the per-layer cursors (#369). |
| `close()` | `:515-538` | Frees the arena (or releases its segments) and clears the table. |

Uniform-slot address: `base + slot * blob`. Sized-slot address: `base + off_[slot]` (`device_slot`, `:576-586`).
`slot_offsets()` exposes `off_` so the verify plan can compute slot pointers on the host
(`ExpertDispatch::cache_slot_off`).

### Residency and admission

- `slot_of(layer, expert)` (`:551`) is bounds-checked. An out-of-range query returns `kNotResident` and never reads a
  neighbouring table entry.
- `admit(layer, expert)` (`:556-574`) returns the existing slot, or claims the next free one. **It never evicts.** The
  header (`expert_cache.hpp:133-136`) says eviction policy was left as a measured question. Replacement happens only
  through `replace` (adaptive swaps) and the callers' own table (`host_res`).
  - Default policy: one shared counter `next_free_`, in arrival order.
  - **Per-layer admission** (`set_per_layer_admission(true)`, `--expert-cache-per-layer`): layer `l` may only take
    slots `[l*q, (l+1)*q)` with `q = slots / n_layers`, and the last layer takes the remainder (`layer_slot_range`,
    `:542-549`). The header comment (`:145-160`) records why: with the shared counter, 256 slots all went to roughly 26
    layers of position 0 and measured a 2.97% hit rate.
- `replace(layer, old_expert, new_expert)` (`expert_cache.hpp:139-143`): after a swap's slot copy has landed, moves the
  slot from the old expert to the new one (used by `RemoteExpertOpt::adapt`).
- `resident()` is `admitted_` under per-layer admission, else `next_free_`. `fills()` counts slot fills.
  `pin_cache_complement` requires `fills() == resident()` (the cache is fully filled) before it builds the RAM copy.

### Filling and verifying

| Function | Lines | Use |
|---|---|---|
| `fill_slot(slot, host_blob, stream, err, bytes)` | `:588-608` | Asynchronous H2D on the caller's stream. The token-path hit admission and the peer tier use it. |
| `fill_slot_blocking(...)` | `:610-639` | Synchronous `cudaMemcpy`, used for the start-up profile fill. The header (`:175-181`) explains why: `verify_slot` reads back on the legacy stream, which is not ordered against a non-blocking stream. On HIP it stages through the pinned buffer instead of registering parts of the mmap. |
| `fill_slot_queued(...)` / `sync_queued(err)` | `:641-665` | The same legacy-stream copy, queued; one sync after many refills. |
| `verify_slot(slot, host_blob, err, bytes)` | `:667-694` | Reads the slot back and compares it byte for byte. On a mismatch it reports the first differing byte. Every start-up fill verifies one slot (`generate.cpp:3834`; the peer and remote tiers do the same). |

`bytes` is the blob's own size when it is smaller than the slot (native packs); 0 means the whole slot.

### Segmented arena: `--vram-elastic` (#533)

`set_segment_bytes(n)` before `open` makes the arena one reserved virtual address range backed by physical segments,
using CUDA virtual memory management (`cuMemCreate` / `cuMemMap` / `cuMemSetAccess`). The driver entry points are
looked up through `cudaGetDriverEntryPoint(ByVersion)` (`:157-233`), so there is no link against the driver library.

- `open_segmented` (`:235-286`) rounds the arena and the segment size to the driver's recommended granularity and maps
  every segment. If one fails, the error says "cudaMalloc failed" on purpose, so the auto-size retry treats it as an
  allocation failure.
- `shrink(keep_bytes)` (`:322-354`): synchronises, unmaps and releases the segments past `keep_bytes` (rounded up to a
  segment), and sets `slots()` to the slots wholly inside what stays (`slots_within`). The caller must evict those
  slots first.
- `grow(want_bytes)` (`:356-385`): maps segments again at the same addresses, up to `want_bytes` rounded down. It stops
  at the first segment the driver cannot back.
- Every slot keeps its address across shrink and grow, which the captured graphs and verify plan rely on.
- Not available on HIP (`STRATA_EC_NO_VMM`): `open` refuses it.
- In serve mode the `VRAM <reserve_mib>` command drives it (`generate.cpp:6159` onwards, `vram_command`). The server
  exposes that as `POST /v1/vram` (help text at `generate.cpp:655-657`).

## The expert profile

Format (`expert_cache.hpp:37-40`, `:65-68`; `tools/make_profile.py`), little-endian:

```
"STRP" | u32 version, n_layers, n_expert, slots, n_ranked | n_ranked x (u16 layer, u16 expert) | n_layers*n_expert x i32
```

The ranked pairs are in descending routing frequency. The trailing table is the profile's own working: a frequency
table in files written by `make_profile.py`, a rank table (-1 = not ranked) in files written by `write_expert_profile`.
The engine does not read the trailing table.

- `read_expert_profile(path, n_layers, n_expert, ranked, slots, err)` (`expert_cache.cpp:23-79`): checks the magic and
  that the header's `n_layers x n_expert` matches the model ("a profile for a different artifact"). Refuses
  `n_ranked > slots` and any out-of-range pair. The version is read but not checked.
- `rank_learned_profile(n_layers, n_expert, resident, heat, prior)` (`:81-108`, #477): ranks **every** pair with a
  stable sort on these keys, in order:
  1. resident in VRAM now first;
  2. then by `heat` (the routing the adaptive tier counted), descending;
  3. then by the rank in `prior` (the profile the run started from);
  4. then by index.

  A restart from the result starts where this run ended.
- `write_expert_profile(path, n_layers, n_expert, ranked, err)` (`:110-150`): writes the format above with
  `slots = n_ranked = ranked.size()`. It writes `<path>.tmp` and then renames it over `path`, so a reader never sees half
  a file. Both dimensions must be at most 65535 (u16 pairs).

### How the engine uses the profile

- `--expert-profile P` is read at `generate.cpp:2986` (and at `:2858` for `--split-skip-if-fits`). The front end always
  passes `<pack>/expert-profile.bin` and refuses to start without it (`frontend/main.cpp:665-678`).
- **Fill** (`generate.cpp:3802-3843`): pairs are admitted and filled in rank order with `fill_slot_blocking` until the
  cache is full. Under per-layer admission, a full layer skips only its own pairs (#369). Slot 0 is verified. With
  `slots == pairs`, decode-time admission then finds no room, so the profile *is* the policy.
- **`--expert-cache auto`** (`generate.cpp:3551-3600`): `slots = (free VRAM - reserve) / max_blob`, capped at the
  profile's length. The reserve is `--vram-reserve-mib` (default 700), plus the prompt buffers when they are not
  borrowed, plus the draft head's bind bytes. On a small card the reserve can shrink to 300 MiB so a working cache
  still fits (#496). After opening, the arena is zeroed and free VRAM re-read. While it is short of the reserve, the
  cache is reopened smaller (`:3725-3760`; under WDDM the free figure reads high). A failed `cudaMalloc` retries at
  three quarters of the size, up to 8 times (#60).
- The RAM budget mode and the other GPU tiers take the next ranks (see [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md)
  and [multi-gpu-tiers.md](multi-gpu-tiers.md)).
- `--dump-routing PATH` writes the routing trace that `tools/make_profile.py` turns into a profile (`drive_pool`,
  `generate.cpp:792-819`).

## Configuration

| Flag / macro | Default | Effect |
|---|---|---|
| `--expert-cache N\|auto` | 0 in `generate`, `auto` from the front end | Slots in VRAM |
| `--expert-cache-per-layer` | off | Per-layer admission ranges |
| `--expert-profile P` | none (front end: `<pack>/expert-profile.bin`) | Static residency plan |
| `--expert-profile-save P`, `--expert-profile-save-every MIN` | off, 10 min | Write the learned profile (serve) |
| `--vram-reserve-mib N` | 700 | VRAM left after an auto cache |
| `--vram-cap MIB` | 0 | Plan as a smaller card |
| `--vram-elastic`, `--vram-segment-mib N` | off, 512 | Segmented arena |
| `STRATA_TEST_CACHE_FAIL=N` | | The first N opens fail (tests the retry) |
| `STRATA_USE_HIP`, `STRATA_HIP_GFX906` | | No VMM; HIP pinned staging for blocking fills |

## Model-specific behaviour

The cache only sees sizes. A Flash-Next canonical pack uses uniform `BLOB` slots. Native packs (all Qwen3.6 packs)
use `open_sized`, because blobs differ per layer, and a slot keeps its first size: swaps stay within a layer.

## Tests

- `expert_cache_per_layer_test` (`tests/core/expert_cache_per_layer_test.cpp`, CUDA): 6 slots, 3 layers, uniform and
  sized. Each layer admits into its own range, and admissions past it return `kNotResident`.
- `expert_cache_segmented_test` (`tests/core/expert_cache_segmented_test.cpp`, CUDA VMM): a shrink returns VRAM and
  keeps the surviving slots' addresses and bytes; a grow maps them again at the same addresses. Uniform and sized.
- `expert_profile_save_test` (`tests/core/expert_profile_save_test.cpp`, CPU): the rank order (resident by heat, then
  the rest by heat, then the prior, then the index), the no-heat case, and the write / read round trip in
  `make_profile.py`'s format.
- `tools/test_make_profile.py` and `tools/test_iq_pack.py` cover the Python side of the format.

## Gotchas

- `slots()` can be smaller than `full_slots()` after `shrink`. Reports and the prompt path's loan follow `slots()`.
- Without a profile and without per-layer admission, the compulsory-miss policy fills slots with whatever the run
  routes first (`generate.cpp:3792`), and the cache never changes after that.
- The GPU and CPU compute the same quantized expert in a different float order. Near-ties can flip, so output with the
  cache on can differ slightly from output with it off (the message printed at `generate.cpp:3777-3786`).

## See also

[dispatch.md](dispatch.md) · [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md) ·
[multi-gpu-tiers.md](multi-gpu-tiers.md) · [pack-format.md](pack-format.md) ·
[docs/qwen36-port/geometry-inventory.md](../../../qwen36-port/geometry-inventory.md) (row 118)
