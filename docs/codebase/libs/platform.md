[Codebase docs](../README.md) › [Libraries](README.md) › Platform I/O and memory

# Platform: direct file I/O, robust reads, host memory

## Purpose

`strata::platform` holds the OS-specific pieces that the engine's memory tiers need, kept behind small portable
interfaces:

- **`DirectFile`**: unbuffered (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`), asynchronous 4 KiB-aligned reads from a
  thread pool. Used for the n-gram (PLE) table so it never enters RAM or the OS file cache.
- **`robust_read.hpp`**: positional reads that retry transient errors with back-off and adaptively limit the number of
  concurrent readers. Used when loading experts or the PLE table from a slow or network file system.
- **`memory.hpp`**: lock a large host region resident (`mlock` / `VirtualLock`), query physical RAM and (Windows)
  the GPU's shared-memory budget, and decide whether the OS file cache can hold the expert-file reads.

The headers in `include/strata/platform/hip_compat/` (CUDA-name shims for the HIP build) live in the same folder but
belong to the build system; see [Build](../build/README.md).

Host-available-memory and **cgroup** limits are *not* in this folder: `HostMemory` / `host_available_memory()` and
the cgroup v1/v2 parsing live in `include/strata/core/expert_source.hpp:71-85` and `src/core/expert_source.cpp:62-345`
(see the [Core](../core/README.md) docs). This folder only provides `total_physical_memory()` and the
`file_cache_keeps()` arithmetic.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/platform/direct_file.hpp](../../../include/strata/platform/direct_file.hpp) | 60 | `Completion`, `DirectFile`, `now_us()` |
| [src/platform/direct_file.cpp](../../../src/platform/direct_file.cpp) | 365 | Windows (IOCP + overlapped) and POSIX (`O_DIRECT` + `pread` pool) implementations |
| [include/strata/platform/robust_read.hpp](../../../include/strata/platform/robust_read.hpp) | 181 | Header-only, POSIX only: `ReadGate`, `read_at`, `read_range`, transient-error classification |
| [include/strata/platform/memory.hpp](../../../include/strata/platform/memory.hpp) | 46 | `LockResult`, `lock_resident`, `unlock_resident`, `gpu_shared_memory_budget`, `total_physical_memory`, inline `file_cache_keeps` |
| [src/platform/memory.cpp](../../../src/platform/memory.cpp) | 138 | Windows and POSIX implementations |
| [src/platform/memory_test.cpp](../../../src/platform/memory_test.cpp) | 72 | `file_cache_keeps` cases and a 256 MiB lock test |
| [tests/platform/direct_file_async.cpp](../../../tests/platform/direct_file_async.cpp) | 60 | `DirectFile` queue / EOF / wake / reopen test |

Build placement: `memory.cpp` is compiled into `strata_core` (`CMakeLists.txt:282`); `direct_file.cpp` into
`strata_kernels` (`CMakeLists.txt:368`), next to its only engine user `src/ngram/ple_reader.cpp`. `robust_read.hpp`
is header-only. Both libraries are only built with a GPU backend on (`STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP`).

## DirectFile

`class DirectFile` (`include/strata/platform/direct_file.hpp:23`).

| Member | What it does |
|---|---|
| `open(path, err)` | Opens read-only and unbuffered, records the size, starts the issuing thread pool. POSIX: `open(O_RDONLY \| O_DIRECT)` (`src/platform/direct_file.cpp:308`). Windows: `CreateFileW` with `FILE_FLAG_NO_BUFFERING \| FILE_FLAG_OVERLAPPED \| FILE_FLAG_RANDOM_ACCESS` and an I/O completion port (`:165-178`). |
| `submit(offset, buffer, length, tag, err)` | Validates alignment (offset, length and buffer address multiples of `alignment() = 4096`, length non-zero) and **only queues** the request; a pool thread issues it. A queued request always produces exactly one completion. |
| `wait(out, max, timeout_ms)` | Collects up to `max` completions; `timeout_ms < 0` waits forever, `0` polls. |
| `wake()` | Thread-safe; makes a blocked `wait` return a completion with `tag == WAKE_TAG` (`~0ull`). |
| `alloc_aligned` / `free_aligned` | Page-aligned buffers (`posix_memalign` / `VirtualAlloc`). |
| `close()` | Stops and joins the pool (draining queued work first), closes the handle, clears pending state. |
| `now_us()` | Monotonic microseconds (steady clock), used for latency accounting. |

`Completion { tag, bytes, ok }`: `bytes` is short only at end of file. A read past EOF returns the bytes that exist;
Windows maps `ERROR_HANDLE_EOF` and `STATUS_END_OF_FILE` to a successful completion (`:120`, `:249-250`).

**Thread pool.** `STRATA_IO_THREADS` sets the number of issuing threads, clamped to 1..64; defaults are 4 on Windows
(overlapped submits return quickly) and 16 on Linux (each thread does one blocking `pread`, so the thread count is the
queue depth) (`io_threads`, `:62-66`). The comments credit this design to "perf-review C-4 / F-2": a single issuing
thread had limited the n-gram table's prompt reads to about 80K reads/s.

POSIX data flow (`:266-362`):

```
submit() --push--> queue --(cv_work)--> worker thread: pread(fd, O_DIRECT) --push--> done --(cv_done)--> wait()
```

Windows data flow: workers call `ReadFile` with an `OVERLAPPED` taken from a free list (`Req`, `:74-77`); results
arrive on the completion port and `wait` uses `GetQueuedCompletionStatusEx`. Requests that fail immediately go to an
`immediate` queue and post a null packet so a waiter collects them (`:114-122`).

## robust_read.hpp (POSIX only)

The whole header is wrapped in `#if !defined(_WIN32)` (`include/strata/platform/robust_read.hpp:10`). It exists for
models on network stores (S3-backed FUSE, NFS), which can refuse reads transiently. The top-level README reports a
real case: an S3-backed RunPod volume refused reads twice with "cannot allocate memory" and the loader retried.

| Symbol | Line | What it does |
|---|---|---|
| `PreadFn` | `:31` | `pread`'s signature, injectable so tests can fail reads on purpose. |
| `class ReadGate` | `:35` | Concurrency limiter shared by a group of readers. Starts at the reader count; `refused(seen)` halves the limit once per burst (only the first refusal that saw the current limit lowers it); `leave(ok)` raises it by one after `kRecover = 32` consecutive good reads, up to the initial count. |
| `read_error_is_transient(e)` | `:84` | `ENOMEM, EAGAIN, EIO, ETIMEDOUT, ENOBUFS, EBUSY, ESTALE, ECONNRESET, ECONNABORTED, ENETDOWN, ENETUNREACH, EHOSTUNREACH, ENOTCONN`. |
| `read_error(path, off, e)` | `:94` | Message naming file, offset and error (`ENODATA` = "the file ends before that"). |
| `read_at(fd, buf, n, off, gate, pread_fn = ::pread, max_wait_s = 120)` | `:101` | Reads exactly `n` bytes. Retries `EINTR` immediately; on a transient error, tells the gate, sleeps (50 ms doubling to 5 s) and retries until `max_wait_s` of waiting without progress. Any progress resets the back-off. Returns 0, `ENODATA` on early EOF, or the errno. Prints a stderr line when the gate lowers the limit. |
| `read_range(path, off, dst, bytes, readers, err, what, pread_fn, max_wait_s)` | `:132` | Loads a contiguous range in 16 MiB chunks with up to `readers` threads, **each on its own file descriptor** (a network store's speed is per stream). Progress to stderr every 4 GiB when `what` is given. |

Callers:

- `src/core/expert_source.cpp:2676-2718`: the expert loader's positional reads use a shared `ReadGate` and `read_at`
  (comment "UnbiasedStrata: ..." marks this as a fork addition).
- `src/kernels/ngram.cpp:279`: `--ple-io ram` loads the whole PLE table with `read_range(..., 16, err, "n-gram table")`.

## memory.hpp

| Function | Where | What it does |
|---|---|---|
| `lock_resident(p, bytes)` | `src/platform/memory.cpp:18` (Windows), `:112` (POSIX) | POSIX: one `mlock`; on failure the note says "raise ulimit -l". Windows: raises the process's minimum working set by `bytes + 512 MiB` (soft limits), then `VirtualLock`s in 1 GiB chunks; partial success is reported in `locked_bytes` and `note`, with `ok` true if at least one chunk locked. |
| `unlock_resident(p, bytes)` | `:57`, `:122` | `munlock` / chunked `VirtualUnlock`, best effort. |
| `gpu_shared_memory_budget(luid, budget, usage, why)` | `:64`, `:126` | Windows only (#243): loads `dxgi.dll` on demand, finds the adapter whose LUID matches the CUDA device's, and reads the non-local (shared) segment's budget and current usage via `IDXGIAdapter3::QueryVideoMemoryInfo`. Page-locked host memory the GPU maps is charged there. Elsewhere returns false with "DXGI is Windows-only". |
| `total_physical_memory()` | `:106`, `:132` | `GlobalMemoryStatusEx` / `sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)`; 0 when unknown. |
| `file_cache_keeps(avail, arena_bytes, read_bytes, margin = 4 GiB)` | `include/strata/platform/memory.hpp:40` | True when `avail - arena_bytes - margin >= read_bytes`, i.e. the OS file cache could keep the expert-file reads beside the engine's own RAM copy. Callers must pass the bytes really read from files and the RAM really held (#357/#577). |

Callers: `src/core/pinned.cu` (shared-memory budget `:299`, physical RAM `:307`, locking the pinned expert arena
`:394`, `:419`, `:431`, and `file_cache_keeps` `:573` to choose buffered vs unbuffered file-tier reads);
`src/core/expert_source.cpp` (lock/unlock of the RAM arena, `:540`, `:1517`, `:1609`).

## Configuration

| Knob | Effect |
|---|---|
| `STRATA_IO_THREADS` | `DirectFile` pool size, 1..64 (default 16 Linux, 4 Windows). |
| `ulimit -l` (Linux) | Must allow the locked size for `lock_resident` to succeed. |
| Windows privilege | `lock_resident` needs `SeIncreaseWorkingSetPrivilege`, which ordinary accounts hold (header comment). |

## Model-specific behaviour

None in this code. In practice `DirectFile` serves only the PLE table, which only Qwen3.8-Flash-Next has; the robust
reads and memory locking serve both models' expert loading.

## Tests

| Test | Registered | Covers |
|---|---|---|
| `platform_memory_test` (`src/platform/memory_test.cpp`) | `CMakeLists.txt:986-988`, GPU-backend builds with `STRATA_BUILD_TESTS` | 12 `file_cache_keeps` cases (the #577 96 GB PC, #357/#362 low-RAM PCs, edge cases), then locks a 256 MiB `malloc` region and checks `locked_bytes == bytes` (and the working set on Windows). Fails if `mlock` is not permitted for 256 MiB. |
| `direct_file_async_test` (`tests/platform/direct_file_async.cpp`) | `CMakeLists.txt:1400-1405`, Unix but not Apple, `STRATA_BUILD_TESTS` | Unaligned request refused; 17 queued reads incl. a 17-byte EOF tail; `wake()`; `close()` with a read outstanding; reopen without stale completions. Sets `STRATA_IO_THREADS=4`. |
| `ple_reader_test --selftest` | see [ngram-ple.md](ngram-ple.md) | Exercises `DirectFile` end to end through the PLE reader. |

There is no dedicated unit test for `robust_read.hpp` in this checkout (the `PreadFn` hook exists for one).

## Gotchas

- `DirectFile` alignment: offset, length and buffer address must all be multiples of 4096; `submit` refuses otherwise.
  On POSIX, `submit` does not check `is_open()` (the Windows build does): after `close()` there are no pool threads,
  so a request queued then is never issued and never completes.
- `O_DIRECT` is Linux-specific; on file systems that do not support it, `open` fails with "cannot open".
- `robust_read.hpp` compiles to nothing on Windows, so Windows callers need their own path.
- `lock_resident` on Linux is all or nothing; on Windows it can lock part of the region.

## See also

- [ngram-ple.md](ngram-ple.md): the main `DirectFile` user
- [../kernels/cpu/ngram.md](../kernels/cpu/ngram.md): `PleTable`, `--ple-io` modes
- [../core/README.md](../core/README.md): expert source, pinned arena, host memory and cgroups
- [../build/README.md](../build/README.md): `hip_compat/` headers
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works" and "Troubleshooting" (`--ple-io direct` vs
  `ram`, #605)
