[Codebase docs](../../README.md) › [Core](../README.md) › [Experts](README.md) › Expert sources

# Expert sources: the RAM arena and the file tier

## Purpose

An `ExpertSource` turns `(layer, expert)` into a pointer to that expert's blob in host memory. The CPU pool computes
from it. The VRAM tiers copy from it when they fill or swap slots. The GPU reads from it directly over PCIe when the
memory is pinned and mapped. There are two implementations:

- **`ArenaExpertSource`** (the default). It reads every expert into one resident `PinnedArena` at start-up.
- **`FileExpertSource`** (`--mmap-experts`, and the low-RAM resident modes). It maps `experts.bin`, or the model's
  GGUF shards in place, and can hold a compact RAM copy of the experts the GPU does not hold.

This page also covers the routing-aware prefetcher (`RouterLookahead`), the GGUF expert check and loader, the
retrying reads in `robust_read.hpp`, and the host-RAM probe.

Files: [include/strata/core/expert_source.hpp](../../../../include/strata/core/expert_source.hpp),
[src/core/expert_source.cpp](../../../../src/core/expert_source.cpp),
[include/strata/platform/robust_read.hpp](../../../../include/strata/platform/robust_read.hpp).

## The interface: `ExpertSource`

`include/strata/core/expert_source.hpp:122-171`. Every method except `blob` has a default implementation.

| Method | Contract |
|---|---|
| `blob(layer, expert)` | The blob pointer, or null. The pointer has to stay valid only until the next `blob` call. A caching source must point into its cache, never into a reused staging buffer. |
| `reads()` | Blobs handed out, for reports. |
| `pinned(l, e)` / `device_alias(l, e)` | Whether the blob lies in page-locked, CUDA-registered memory, and its device address if it does. Both are needed for the GPU's PCIe share. |
| `pcie_layer(l)` | Whether the verify window may give this layer any PCIe share. The default asks `device_alias(l, 0)`. |
| `begin_layer(l, ids, k)` | Called before a layer's first expert. `FileExpertSource` uses it to age its staging buffers. |
| `prefetch(l, e)` / `prefetch(l, experts, n)` | Start reading pages that the CPU will need soon. The batch form lets the GGUF-in-place tier assemble a layer's misses on several threads. |
| `warm(l, experts, n)` / `warms()` | A predictor's hint: start reading pages. Never changes what `blob` returns. |
| `release(l, e)` | The expert now lives in VRAM, so its RAM pages may go back to the kernel. Returns the bytes released. |
| `transient(l, e)` / `copy_blob(l, e, dst)` | Whether `blob` would return a short-lived staging buffer (GGUF in place). Consumers that keep a blob longer, such as the prompt path's stager, copy it with `copy_blob` instead. The default `copy_blob` (`expert_source.cpp:633`) is a `memcpy` of `blob_bytes(layer)`. |

## `ArenaExpertSource` (the default)

Declared at `expert_source.hpp:660-716`. `open` is at `expert_source.cpp:2752-2970`.

1. **Layout and size check before allocating.** The process-wide `ExpertLayout` must match `n_layers x n_expert`.
   If the pack has `experts.bin`, its size must equal `lay.total` (or `total + one blob`, the padded
   `STRATA_ARENA_MMAP` file). A native pack without `experts.bin` takes its experts from the GGUF. First,
   `check_experts_gguf` proves that every span in `native_experts.txt` is the tensor it claims to be (see below).
2. **`STRATA_ARENA_MMAP=1`** (Linux only, `:2791-2828`). The arena is the pack's `experts.bin` mapped read-only, not
   locked and not pinned. The first start writes that file (padded by one blob) when the drive has room for it plus
   2 GiB (`:2919-2942`). Later starts map it. The GPUs get no mapped alias in this mode, so the comment says to run
   with `--pcie-frac 0`.
3. **RAM check (#633, `:2830-2859`).** `detail::host_available_memory` runs first. If the cgroup limit is below
   `total + blob`, the load is refused with a message, because the process would be OOM-killed part-way through. If
   less RAM is available than needed, `ram_warning()` is set but the load still proceeds. A
   `--shared-expert-arena FILE` is not checked.
4. **Allocation.** `PinnedArena(want + blob, bounds, max_pinned_bytes, shared_arena_file, pack_hash)`
   (`src/core/pinned.cu`, outside this area). It has one registration slice per layer, so no expert straddles two
   `cudaHostRegister` calls. It is one blob longer than the data, so a whole-slot copy can start at any expert. With
   `--shared-expert-arena`, `shared_arena_pack_hash` (`:2469`) fingerprints the pack's `manifest.json`, `index.txt`,
   `native_experts.txt` and three 64 KiB samples of each expert source file. A backing file is then reused only for the
   same pack.
5. **Load.** From the GGUF: `load_experts_gguf`. From `experts.bin`: `load_experts_direct` (unbuffered) or
   `load_experts_ranges`. Both of those live in `pinned.cu`. `experts_unbuffered` decides between them from the RAM the
   file cache would need.
6. **Device aliases** (`:2946-2960`). One `cudaHostGetDevicePointer` per registered slice. `pinned(l, e)` is true when
   the blob ends inside `registered_bytes` (`:3020`). `device_alias` returns the slice's alias plus the offset (`:3026`).

`blob` (`:3035`) is pointer arithmetic into resident memory. `prefetch` / `release` (`:2987`, `:3005`) only act when the
arena is the mapped file: `MADV_WILLNEED`, and `MADV_DONTNEED` on the pages wholly inside the blob. The comment at
`:2999-3004` says why `POSIX_FADV_DONTNEED` is not used: it made every eviction a disk re-read.

How many bytes are pinned is decided by the caller. `generate.cpp:3497-3507` caps pinning at 8 GiB on a multi-GPU run
under WDDM. `STRATA_ARENA_PIN_GIB` overrides that, and `--ram-cap` bounds it.

## `FileExpertSource` (`--mmap-experts`)

Declared at `expert_source.hpp:418-642`.

### Opening

`open(pack_dir, n_layers, n_expert, err)` (`expert_source.cpp:365-532`) validates the layout. A canonical layout must
total `n_layers * n_expert * BLOB`. A native layout must have contiguous per-layer offsets whose blob sizes match
`fmt[i].bytes`. Then it does one of two things:

- **`experts.bin` present:** maps it whole (`mmap(PROT_READ, MAP_SHARED)`, or `CreateFileMappingW` / `MapViewOfFile` on
  Windows). The size must equal the layout's total, otherwise the open is refused with both numbers. The Windows
  comment at `:454-464` explains why no `FILE_FLAG_RANDOM_ACCESS` is used: with it, the page cache dropped pages as fast
  as they were faulted in (1.93 GB/s).
- **Native pack, no `experts.bin`, after `set_gguf(shard)`:** `open_gguf` (`:647-723`). It runs `check_experts_gguf`
  first, then maps every file `native_experts.txt` names (per layer in v3, per role in v4). It records, per layer and
  role, `role_ptr_` (the role's expert 0) and `role_bytes_` (bytes per expert of that role). A blob is never contiguous
  in this mode, so `mapped_blob` returns null (`:1341`).

### Reading a blob

`blob` (`:1826-1853`) tries, in order:

1. **The RAM copy** (once `pin_cache_complement` has built it). `detail::cache_complement_blob_or_fallback` looks the
   offset up in `complement_offsets_` (`kNoCacheComplement` = not held). This increments `ram_reads_`.
2. **A staged exchange.** `override_[index]` points at an exchange buffer while an adaptive swap is in flight (see
   [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md)).
3. **The files.** With staging (GGUF in place, or any unbuffered source) this is `staged_blob`. Otherwise it is
   `mapped_blob`, a pointer into the mapping whose page faults are the read. Once the RAM copy exists, every blob taken
   from the files also increments `file_reads_`. That counter is what `/health` reports as `file_reads`.

### Staging buffers (GGUF in place)

A blob assembled from three role slices goes into a small pool of buffers, one per recent `(layer, expert)`
(`:747-840`). `claim_stage` finds or reserves the buffer. A buffer is reused only when it is not being filled and at
least one of these holds:

- `kStageAge = 3` layer changes have passed (`begin_layer` advances `epoch_`), or
- 256 assemblies have passed.

This rule is why a pointer stays valid through its layer and the next ones. Concurrent requests for the same key wait
on `stage_cv_`. A failed fill unpublishes the key. If 512 buffers are ever in use at once, a one-time message is
printed.

`prefetch(layer, experts, n)` (`:842-866`) claims buffers for the misses that are not in the RAM copy, then `fill_many`
fills them. The fill uses `fetch_threads_` threads (`STRATA_FETCH_THREADS`, default 8), because each mapped page fault
is one outstanding request and several threads keep the SSD queue full. On Windows, one `PrefetchVirtualMemory` call
covers all slices first (`STRATA_FETCH_PVM=0` disables it).

### Unbuffered reads (Windows, #286 / #577)

`set_unbuffered(ram_bytes, why)` (`:982`) calls `experts_unbuffered` (from `pinned.hpp`) on the expert bytes outside
the RAM budget. If the file cache could not keep them, every path is reopened with
`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` (`open_direct`, `:954`). `recheck_unbuffered` (`:1001`) repeats the
decision after the RAM copy is built, using what the copy really holds; `STRATA_UNBUFFERED_LOAD` forces the choice.

`read_direct` (`:1034-1152`) builds 4 KiB-aligned windows per role, merges windows closer than 1 MiB into requests of
up to 32 MiB (NTFS serialises unbuffered reads of a mapped file), issues every request before the first wait, then
copies each role's bytes into place. A failed batch falls back to `copy_from_files` blob by blob. `prefetch_pairs`
(`:868`) batches the profile fill's reads 64 pairs at a time. These functions are no-ops elsewhere, where reads always
go through the file cache.

### Routing-aware prefetch: `RouterLookahead`

`expert_source.hpp:178-209`, `expert_source.cpp:1190-1297`. When the CPU pool starts layer `l`
(`expert_pool_dispatch_multi`, `:2021`), `submit` copies up to 8 tokens of that layer's MoE input. A worker thread
then applies layer `l+1`'s BF16 router to it and takes each token's top `k`. It drops experts the VRAM table holds
(`host_res >= 0`) and calls `src->warm(l+1, ...)`.

- The dot product is AVX2 (`bf16_rows_dot_multi`), AVX1, or plain C++ with a separate multiply and add, never
  `std::fma` (`:1248-1276`).
- `submit` never waits. A prediction still running makes it count `skipped`.
- `FileExpertSource::warm` (`:1154`) skips experts in the RAM copy, stamps `warm_stamp_`, and issues
  `madvise(MADV_WILLNEED)` or `PrefetchVirtualMemory` on the three role slices.
- `warms()` is true only for GGUF in place without unbuffered reads.
- Turned on at `generate.cpp:4023-4045`. `STRATA_LOOKAHEAD=0` disables it, and `STRATA_LOOKAHEAD_K` sets `k`
  (default 10). `warmed_hits()` counts file-tier blobs that had been warmed for their layer (in the start-up report).

## The GGUF experts: `check_experts_gguf` and `load_experts_gguf`

`check_experts_gguf(native, lay, err)` (`expert_source.cpp:2520-2571`) runs before anything is read. For every layer
and role (`gate`, `up`, `down`) it opens the GGUF file that `native_experts.txt` names (`expert_gguf_file`, `:2512`:
a name beside the `--native` shard, or the shard itself). It checks that `blk.L.ffn_<role>_exps.weight`:

- exists;
- has the layout's type;
- has shape `[cols, rows, n_expert]`;
- has payload `per_role_bytes * n_expert`;
- starts at the absolute offset the pack recorded;
- lies inside the file.

Any mismatch is refused with the tensor, file and reason, plus "repack with tools/iq_pack.py". A pack older than v2 (no
offsets) is refused.

`load_experts_gguf(native, dst, lay, threads, unbuffered)` (`:2580-2748`) fills `dst` in the `experts.bin` layout. It
works one layer per worker, reads 16 experts per chunk, and scatters each expert's slice to
`blob_offset(l, e) + role_offset`. On Windows with `unbuffered`, each chunk's sector-aligned window is read with
`FILE_FLAG_NO_BUFFERING`. Elsewhere it uses the retrying reads below. Every 4 GiB it prints progress
(`strata generate: experts: X of Y GiB read`). It returns `LoadStats` (`ok`, `error`, `bytes`, `seconds`).

### Retrying reads: `robust_read.hpp`

This header is new in UnbiasedStrata, for models on network stores (S3 FUSE, NFS). It is POSIX only.

- `ReadGate(readers)` (`robust_read.hpp:35-81`): a counting gate. A refused read halves the limit, once per burst (only
  readers that saw the current limit lower it). After `kRecover = 32` consecutive good reads the limit climbs back by one.
- `read_error_is_transient(e)` (`:84`): treats as retryable ENOMEM, EAGAIN, EIO, ETIMEDOUT, ENOBUFS, EBUSY, ESTALE and
  the connection errors.
- `read_at(fd, buf, n, off, gate, pread_fn, max_wait_s = 120)` (`:101-128`): reads exactly `n` bytes. It retries
  transient errors with a pause that doubles from 0.05 s up to 5 s, and gives up after `max_wait_s` seconds without
  progress. The pause resets when bytes arrive. EINTR loops. It returns `ENODATA` for a short file. `read_error`
  formats "cannot read PATH at offset N: ...".
- `read_range(path, off, dst, bytes, readers, err, what, ...)` (`:132-177`): splits the range into 16 MiB chunks. Each
  reader has its own fd (a network store's speed is per stream).

In this area only `load_experts_gguf` uses `read_at` (`expert_source.cpp:2679-2719`), on the arena load from GGUF
shards. The other user is `src/kernels/ngram.cpp`. The `PreadFn` parameter exists so tests can inject failures, but no
test in `tests/core` exercises this header (not determined whether one exists elsewhere).

## Host RAM probe and `--ram-cap`

`detail::host_available_memory(m, meminfo, self_cgroup, cgroup_root)` (`expert_source.hpp:71-86`,
`expert_source.cpp:235-346`) returns `HostMemory{available, cgroup_limit}`:

- Linux:
  - `MemAvailable` from `/proc/meminfo`.
  - cgroup v2: walks `memory.max` / `memory.current` up to the root. For a finite limit, the room is the limit minus
    usage after reclaiming clean inactive file cache (`cgroup_available_bytes`, `:62`, which needs `memory.stat`'s
    `inactive_file`, `file_dirty` and `file_writeback`).
  - cgroup v1 (#633): `memory.limit_in_bytes - memory.usage_in_bytes` for the memory controller's group and its
    ancestors. A group whose files are not visible is skipped.
- Windows: `ullAvailPhys`.

`set_host_ram_cap(bytes)` (`--ram-cap`, set at `generate.cpp:2176`) clamps both fields, so the RAM tier is sized as on
a smaller machine. The page cache is not capped.

## Configuration

| Knob | Effect |
|---|---|
| `--mmap-experts` | Use `FileExpertSource` instead of the arena (`generate.cpp:3463-3488`) |
| `--native SHARD` | The GGUF shard. A native pack without `experts.bin` reads its experts from the shards (arena load or in place) |
| `--shared-expert-arena FILE` | Linux: back the arena with one `MAP_SHARED` file (put it on `/dev/shm`) |
| `--ram-cap GIB` | Clamp the RAM probe and the pinned arena |
| `STRATA_ARENA_MMAP=1` | Map the arena file read-only (Linux) |
| `STRATA_ARENA_PIN_GIB` | Cap or force arena pinning |
| `STRATA_FETCH_THREADS` | GGUF-in-place fill threads (default 8) |
| `STRATA_FETCH_PVM=0` | Windows: no `PrefetchVirtualMemory` batch |
| `STRATA_UNBUFFERED_LOAD` | Force the unbuffered decision |
| `STRATA_LOOKAHEAD=0`, `STRATA_LOOKAHEAD_K` | Routing-aware prefetch off, or its top-k |
| `STRATA_NATIVE_EXPERTS` (CMake) | Without it, a native pack is refused (`expert_layout.cpp:249-251`) |

## Model-specific behaviour

- **Flash-Next.** A canonical Q2_0 pack (fixed `BLOB`), or native IQ packs whose blob size varies per layer.
- **Qwen3.6-35B-A3B.** Native packs only. `expert_layout_load` refuses a canonical pack for a model whose widths are
  not 2560 x 640 (`expert_layout.cpp:240-245`). The pack can name per-role shards (v4) for UD-Q4_K_XL-style splits.

## Tests

- `file_expert_source_test` (`tests/core/file_expert_source_test.cpp`):
  - a wrong `experts.bin` size is refused (`check_file_size_rejection`, `:68`);
  - canonical and native variable layouts map the right bytes (`:82`, `:125`);
  - the cgroup budget arithmetic (`:256`);
  - `host_available_memory` against fake `/proc` and cgroup v1/v2 trees (`:293`).
- `expert_layout_test` part 3 (`tests/core/expert_layout_test.cpp:204`): `check_experts_gguf` and `load_experts_gguf`
  on synthetic shards at 2560 / 640 with Unsloth's format pairs. It compares every byte, then checks the refusals:
  wrong offset, truncated file, role in the wrong file, wrong type or shape, missing file.

## Gotchas

- A pointer from `FileExpertSource::blob` in GGUF mode is only good for about three layer changes. Keep blobs longer
  only through `copy_blob`.
- A zero-copy GPU read of the RAM copy needs `pinned()`. A partially registered copy (`STRATA_PARTIAL_PIN`) answers
  true only for blobs inside the registered prefix (`expert_source.cpp:1857-1865`).
- `ArenaExpertSource` does not check RAM for a shared arena. Under `STRATA_ARENA_MMAP` it gives the GPU no alias.

## See also

[vram-cache.md](vram-cache.md) · [dispatch.md](dispatch.md) · [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md) ·
[pack-format.md](pack-format.md) · [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) ("How it works")
