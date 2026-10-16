[Codebase docs](../../README.md) › [Core](../README.md) › [Experts](README.md) › Resident RAM and adaptive swapping

# The resident RAM mode and adaptive swapping

## Purpose

This page covers two mechanisms that keep the right experts in the faster tiers.

- **The resident RAM mode.** With `--mmap-experts`, the experts the GPU does not hold would otherwise come from the
  mapped files through the OS page cache. `FileExpertSource::pin_cache_complement` copies them, or the hottest of them
  under a budget, into one compact RAM allocation. The copy is page-locked when the driver allows it, so the CPU never
  page-faults on them and the GPU can read them over PCIe.
- **Adaptive swapping.** The VRAM tier starts from the profile and then follows the conversation. Between verify
  rounds, the most-routed non-resident experts replace the least-routed resident ones of the same layer. In the
  resident RAM mode the swap is an exchange: the evicted expert takes the swapped-in expert's place in the RAM copy, so
  no file is read.

`--expert-profile-save` persists what the adaptive tier learned.

Files: [src/core/expert_source.cpp](../../../../src/core/expert_source.cpp) (`detail::` helpers `:77-179`,
`pin_cache_complement` `:1355-1754`, exchanges `:1756-1824`),
[include/strata/core/expert_source.hpp](../../../../include/strata/core/expert_source.hpp) (`:88-111`, `:440-494`).
The adaptive loop lives in `src/program/generate.cpp` (cited below).

## The compact RAM copy ("cache complement")

### Pure planning helpers (CPU-only, unit-tested)

| Function | Lines | What it computes |
|---|---|---|
| `detail::make_cache_complement_plan(n_layers, n_expert, layer_blob_bytes, primary_gpu_pairs, additional_gpu_pairs, offsets, bytes, err)` | `expert_source.cpp:77-147` | One offset per `(layer, expert)` into a compact buffer for every pair that neither the primary GPU cache nor an additional GPU tier holds. Held pairs get `kNoCacheComplement`. Refuses duplicates, out-of-range pairs, overlap between the two tiers, zero-size blobs and overflow. |
| `detail::cache_complement_blob_or_fallback(index, offsets, host, fallback)` | `:149-155` | The blob in the copy, or the mapped-file fallback |
| `detail::choose_resident_keep_from(slot_bytes, base_bytes, budget, lend_from)` | `:157-171` | The resident mode's lend region: the base (experts no slot holds) must fit `budget`, then slots are added from the last one down to `lend_from` while they fit. Returns the first slot kept in RAM, or -1 when the base alone does not fit. |
| `detail::exchange_cache_complement(offsets, in, out)` | `:173-179` | `out` takes `in`'s place in the copy. Only valid when `in` is held and `out` is not. |

### `pin_cache_complement(cache, err, pin, additional_gpu_pairs, lend_from_slot, headroom_bytes, budget_bytes, rank)`

`expert_source.cpp:1355-1754`. It can only run after the GPU cache is fully filled (`fills() == resident()`).

1. **Collect the GPU cache's pairs** and each slot's blob size, then plan the complement.
2. **Windows (#467).** First trims the process working set (`SetProcessWorkingSetSize(-1, -1)`), so the mapped pages
   that the GPU fill touched count as available again.
3. **RAM budget** (`budget_bytes > 0`, `--resident-budget-gib N`, `:1411-1463`):
   - The budget is clamped to the available RAM minus `headroom_bytes` minus a 256 MiB margin (#403).
   - Complement pairs are then taken in **`rank` order** (the expert profile, hottest first) while they fit. The rest
     stay on the files.
   - No lend region in this mode.
   - `kResidentWhatFits` (`~0`) is the soft `--resident-experts` mode's second try (#467). It sizes the budget by RAM
     alone and fails only if not even one expert fits.
4. **Safety check** (`:1465-1486`). Without a budget, the whole complement must fit in the available RAM minus
   `headroom_bytes`, or the call fails and nothing is allocated.
5. **Lend region** (`:1487-1502`). The prompt path borrows the cache's last slots for its buffers during a prompt. It
   streams those slots' experts from RAM while lending, then copies them back. With `lend_from_slot >= 0`,
   `choose_resident_keep_from` decides how many of the lent slots also keep their experts in the copy, and the plan is
   rebuilt without them as "held".
6. **Allocation** (`:1504-1615`):
   - `pin = true`: `cudaHostAlloc(Mapped | Portable)` plus a device alias. If the driver refuses, the same bytes go in
     `malloc` memory that is locked in the working set (`platform::lock_resident`).
   - `pin = false`: pageable `malloc` (the ROCm arm).
   - With a budget and `STRATA_PARTIAL_PIN=1`, the largest expert-aligned prefix up to `STRATA_PARTIAL_PIN_GIB`
     (default 24) is `cudaHostRegister`ed, trying in 2 GiB steps. Only that prefix is pinned, and the rest is locked.
     Because the bytes are in profile order, the pinned prefix holds the hottest experts. This is off by default because
     the PCIe share measured no faster on the RTX 5070 PC (`:1562-1564`).
7. **Copy** (`:1617-1711`). Up to 6 threads, one layer at a time, run `copy_from_files`. Unbuffered sources batch 32
   blobs per `read_direct`. For `experts.bin` on POSIX, each finished layer's pages are released from the mapping and
   the page cache (`MADV_DONTNEED`, `POSIX_FADV_DONTNEED`). A line is printed every 8 layers.
8. **Publish** (`:1725-1753`). Sets `complement_ready_` and counts `complement_count_` (the `ram_copies` in `/health`).
   Prints whether the copy is "mapped pinned", "locked resident" or "pageable resident", and how many lent slots kept
   their experts.

After this, `blob`, `copy_blob`, `pinned`, `device_alias`, `prefetch` and `warm` consult the copy first (see
[expert-sources.md](expert-sources.md)).

### How `generate.cpp` drives it

`generate.cpp:4888-4950`:

- `lend_from` is `slots - plan_lend(chunk)` when the prompt path borrows slots.
- The first call uses the whole complement, or the `--resident-budget-gib` budget.
- If that fails in the soft mode (`--resident-experts`), a second call with `kResidentWhatFits` keeps what fits. If
  that also fails, the run continues in plain mmap mode with a warning.
- A `--resident-budget-gib` that cannot be kept is a warning. A failure under plain `--resident-cpu-experts` ends the
  start.
- With adaptive swapping on, `reserve_exchanges(min(adapt_swaps, 96))` allocates the exchange buffers.
- On Windows with a budget, `recheck_unbuffered` decides again (#577).

`--resident-experts` sets `--mmap-experts --resident-cpu-experts`, pinning, soft mode and a 4 GiB headroom
(`generate.cpp:1555-1563`). `STRATA_RESIDENT_PIN=0` keeps the copy pageable, and `STRATA_RESIDENT_HEADROOM_GIB=N`
changes the headroom.

## Adaptive swapping

### Counting

`ExpertDispatch::usage` (`n_layers x n_expert` floats) is allocated when `--adapt-every > 0 && --adapt-swaps > 0` and
not every expert is resident (`generate.cpp:5783-5787`). `expert_pool_dispatch_multi` adds 1 per routed entry
(`expert_source.cpp:2059-2061`). After each adaptation, every count is multiplied by `--adapt-decay` (default 0.7).

### The swap rule (`adapt`, `generate.cpp:5843-5920`)

For each layer:

- candidates are non-resident experts with `usage >= 2`, not held by the peer or a remote tier;
- victims are resident experts.

Candidates are sorted hottest first and victims coldest first. The i-th candidate replaces the i-th victim while
`cand >= victim + 1.5`. All layers' swaps are sorted by gain and the top `--adapt-swaps` (default 96) are kept. Then:

1. **Resident RAM exchange** (`resident_stage_swaps`, `generate.cpp:233-264`). When `in` is in the RAM copy and `out`
   is not, `out`'s bytes are copied D2H from its slot into exchange buffer `q`, and `stage_exchange(layer, in, out, q)`
   makes `blob(out)` read that buffer. Swaps that need no exchange go ahead as normal. Swaps beyond the buffers' room
   wait for a later round.
2. **Refill.** `cudaMemcpyAsync(slot_of(out) <- blob(in))` on `adapt_stream` (or the layer's stage stream under a layer
   split). `host_res[out]` becomes -1 **immediately**, so the CPU computes `out` meanwhile. `(in, slot)` goes on
   `pending`, and `srcp->prefetch(out)` brings back pages a file-backed arena released.
3. **Event.** `cudaEventRecord(adapt_ev)`, then the peer tier's own `adapt` (see [multi-gpu-tiers.md](multi-gpu-tiers.md)),
   then decay.

`adapt` runs on its own thread every `adapt_every` verify rounds, while the verifier commits and drafts
(`generate.cpp:7565-7568`).

### Landing (`apply_pending`, `generate.cpp:5818-5840`)

`apply_pending` waits for or polls `adapt_ev` (and the stages' events). It then:

- unpins any `pin_blobs` pages;
- calls `src.commit_exchanges()`, which copies each staged `out` blob into `in`'s old place in the RAM copy and
  updates the offsets (`expert_source.cpp:1809-1824`);
- sets `host_res[in] = slot`;
- calls `srcp->release(in)`, since the expert is in VRAM now and its RAM pages are no longer needed;
- uploads the residency table to the device(s).

Since #463 it blocks before each verify window (`apply_pending(!adapt_nowait())`, `generate.cpp:7497`), so whether a
swapped-in expert ran on the GPU or the CPU does not depend on copy timing. `STRATA_ADAPT_NOWAIT=1` restores the old
non-blocking query.

The non-serve speculative loop has the same logic (`generate.cpp:8294-8403`).

### Learned profile (`--expert-profile-save`, #477)

With `--expert-profile-save P`, `heat += usage` at each adaptation, so heat is proportional to the routing itself.
`save_profile` (`generate.cpp:5922-5938`) ranks with `rank_learned_profile(resident, heat, profile_loaded)` and writes
the file atomically. Pending swaps count as resident. It runs between requests every `--expert-profile-save-every`
minutes (default 10, `:6610`) and at QUIT (`:7897`). The adaptive tier and a starting `--expert-profile` must both be
on, otherwise a message says nothing will be saved.

## Configuration

| Flag / env | Default | Effect |
|---|---|---|
| `--resident-experts` | off | The low-RAM PC's mode (soft, pinned, 4 GiB headroom) |
| `--resident-cpu-experts` | off | The RAM copy, pageable, strict |
| `--resident-budget-gib N` | 0 (whole complement) | Keep only the N GiB hottest by profile |
| `--ram-cap GIB` | 0 | Clamps the RAM probe (and a larger explicit budget) |
| `--adapt-every N`, `--adapt-swaps N`, `--adapt-decay F` | 4, 96, 0.7 | Adaptive tier cadence, size, decay. 0 = static residency |
| `--expert-profile-save P`, `--expert-profile-save-every MIN` | off, 10 | Persist the learned ranking |
| `STRATA_PARTIAL_PIN=1`, `STRATA_PARTIAL_PIN_GIB` | off, 24 | Register a hot prefix of a budgeted copy |
| `STRATA_RESIDENT_PIN=0`, `STRATA_RESIDENT_HEADROOM_GIB` | | A/B arms of `--resident-experts` |
| `STRATA_ADAPT_NOWAIT=1` | off | Do not wait for swaps before a window |
| `STRATA_REFILL_BLOCKING` | off | The refill of lent slots after a prompt uses `fill_slot_blocking` instead of the queued copy (`generate.cpp:220`, `:7127`, `:7978`) |

## Model-specific behaviour

The logic is model-independent. Blob sizes come from `ExpertLayout`, and swaps never cross layers, so a sized slot
keeps its size. On Qwen3.6 there are 40 x 256 pairs.

## Tests

`file_expert_source_test` (`tests/core/file_expert_source_test.cpp`):

- `test_complement_plan` (`:188`): plan offsets, exclusions and refusals.
- `test_resident_lend_region` (`:210`): four slots of 5, 3, 3 and 5 bytes with a 10-byte base. `choose_resident_keep_from`
  under various budgets.
- `test_resident_exchange` (`:225`): two layers by three experts with the GPU holding (0,1) and (1,2). An exchange
  moves (1,0) into the GPU and (1,2) into its RAM place. The size and the set of places are unchanged, and exchanging
  back restores the original plan.
- The cgroup budget arithmetic (`:256`).

The adaptive loop itself has no unit test. It runs in the serve and speculative paths.

## Gotchas

- `pin_cache_complement` refuses to run twice and refuses a partly filled cache.
- An expert evicted from VRAM in a swap is computed by the CPU from its exchange buffer until the swap lands. Only then
  does it move into the RAM copy.
- In budget mode, the experts outside the budget, including evicted ones that had no RAM place, are read from the files.

## See also

[expert-sources.md](expert-sources.md) · [vram-cache.md](vram-cache.md) · [dispatch.md](dispatch.md) ·
[multi-gpu-tiers.md](multi-gpu-tiers.md) · [../conversation/README.md](../conversation/README.md)
