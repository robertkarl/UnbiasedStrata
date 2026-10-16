[Codebase docs](../README.md) › [Libraries](README.md) › PLE / n-gram table reader

# PLE / n-gram table reader (`strata::ngram::PleReader`)

## Purpose

Qwen3.8-Flash-Next has a per-layer embedding (PLE) driven by a hashed n-gram table, `per_layer_token_embd.weight`,
stored in the model's second GGUF shard. The table is huge: 320,001,536 rows of 90 bytes (IQ4_NL, 26.8 GiB), or 160
bytes per row in the FP8 E4M3 variant (51.2 GB), or 110 bytes for Q5_0 rows. Each token needs 16 rows on 16
different pages, and which rows depends on the token itself.

`PleReader` reads those rows **straight from the SSD with unbuffered 4 KiB reads** (`platform::DirectFile`), so the
table never occupies RAM or the OS file cache, and lets the caller overlap the reads with compute:

```
Ticket t = reader.issue(rows, n, out_raw);   // as soon as the token id is known
...                                          // embedding, layer 0
reader.collect(t);                           // before layer 1 reads the rows
```

It is the I/O layer only. Hashing tokens to rows, decoding rows (IQ4_NL / FP8 / Q5_0) and the `--ple-io` mode switch
live in `src/kernels/ngram.cpp` (`PleTable`), documented at [../kernels/cpu/ngram.md](../kernels/cpu/ngram.md).

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/ngram/ple_reader.hpp](../../../include/strata/ngram/ple_reader.hpp) | 100 | `ROW_BYTES`, `PAGE`, `ReaderStats`, `PleReader` |
| [src/ngram/ple_reader.cpp](../../../src/ngram/ple_reader.cpp) | 547 | Row cache, job queue, I/O worker, keep-alive, fault injection |
| [src/ngram/ple_reader_test.cpp](../../../src/ngram/ple_reader_test.cpp) | 298 | `--selftest` on a synthetic table; `--gguf` direct-vs-mmap check on the real table |

`ple_reader.cpp` is compiled into `strata_kernels` together with `src/platform/direct_file.cpp`
(`CMakeLists.txt:367-368`), so it is only built with a GPU backend on.

## Public API (`include/strata/ngram/ple_reader.hpp`)

| Symbol | Line | What it does |
|---|---|---|
| `ROW_BYTES = 90`, `PAGE = 4096` | `:27-28` | Default row size (IQ4_NL) and read granularity. |
| `struct ReaderStats` | `:30` | Rows requested, cache hits, rows deduplicated onto a page already being read, SSD reads and bytes, time blocked in `collect`, time inside submit, per-read latency sum and a ring of the last 65,536 latencies (`percentile(q)`), injected-late reads, keep-alive reads and their worst latency. |
| `open(path, table_offset, n_rows, max_inflight, cache_rows, err, io_thread = true, row_bytes = ROW_BYTES)` | `:62` | `table_offset` and `n_rows` must come from a validated GGUF parse. `max_inflight` 1..1024; `row_bytes` 1..4096. Refuses a table that extends past EOF. Allocates `max_inflight` read slots of 2 pages each, sizes the row cache, and (by default) starts the I/O worker thread. |
| `issue(rows, n, out_raw)` | `:70` | Starts fetching `n` rows; row i lands at `out_raw + row_bytes * i`. `out_raw` must stay valid until `collect` returns. Out-of-range rows are zero-filled (matching the mmap path). |
| `collect(ticket, err)` | `:73` | Blocks until every row of the ticket is in place; false with a message on an I/O error. Tickets may be collected in any order. |
| `set_keepalive(period_ms, window_s)` | `:81` | io-thread mode only: when no read has gone out for `period_ms`, read one random table page, until `window_s` after the last `issue`. |
| `set_injected_delay_us(us)` | `:85` | Fault injection: every read completes no earlier than `us` after issue. |
| `stats()`, `snapshot()`, `reset_stats()`, `cache_capacity()`, `cache_size()`, `row_bytes()`, `close()`, `is_open()` | `:64-93` | `snapshot` takes a consistent copy under the lock; `stats()` is only safe with no ticket in flight. |

## How `issue` builds reads (`src/ngram/ple_reader.cpp:421`)

For each requested row:

1. Out of range: zero-fill, done.
2. Row-cache hit: copy, count `cache_hits`, done.
3. Otherwise compute the aligned page `first = (table_offset + row * rb) / PAGE * PAGE` and the length (`PAGE`, or
   `2 * PAGE` when the row straddles a page boundary). Rows that fall on a page already in this ticket are attached
   to that job (`dedup_rows`).

Jobs are then **sorted by offset** (prefill chunks read the SSD in near-sequential order) and appended to the queue.
`TicketState::pending` counts the ticket's jobs. In io-thread mode the caller only builds jobs, then notifies the
worker and calls `DirectFile::wake()` in case it is blocked in `wait`; in caller-thread mode it calls `pump()` itself.

## Worker, completion and errors

```
issue() --jobs--> queue --pump()--> DirectFile::submit (<= max_inflight slots)
                                         |
worker_loop: DirectFile::wait --> process() --> finish(): memcpy rows to dst, insert into row cache,
                                                         --pending; free slot; pump() more
collect(): wait on cv_done until pending == 0
```

- `pump()` (`:147`) moves queued jobs into free slots and submits them.
- `finish()` (`:199`) checks every use fits inside the bytes returned (a short read inside the table is an error),
  copies rows out, fills the cache, and frees the slot.
- `worker_loop()` (`:298`) owns all `DirectFile` calls in io-thread mode. It blocks in `DirectFile::wait` **without
  holding the mutex**, so `issue` can queue work concurrently. When idle it waits on `cv_work`, with a timeout when a
  keep-alive read is due.
- Any failure sets a sticky `error`, cancels queued jobs (decrementing their tickets), and wakes collectors, which
  return the error. The reader must be reopened after an error.
- `close()` (`:380`) stops the worker, drops unsubmitted work, and **drains in-flight reads before freeing their
  buffers**.

Threading comment (`:94-97`): with `io_thread` one worker owns every `DirectFile` call, so roughly 11 us per
`ReadFile` no longer lands on the host loop.

## Row cache

`RowCache` (`:35`): set-associative, 8 ways per set, round-robin replacement within a set, `cache_rows / 8` sets,
keyed by row index. Bounded by construction; `cache_rows = 0` disables it. The header cites measurements
(bench/results/2026-09-23-ngram-io, not in this checkout): about 1M rows (~95 MB) would serve up to ~82% of reads,
and 20-34% of rows recur within one long prompt.

## Keep-alive

Some SSDs enter a power state after about 250 ms without a command and stall the next reads 50-150 ms (observed on a
WD_BLACK SN7100 under Windows 11, header `:75-80`). That happens in decode after a few rounds served entirely from the
row cache. The keep-alive (`keepalive_due`, `queue_keepalive`, `:181-197`) reads one pseudo-random table page
whenever `period` has passed without a read, for `window` seconds after the last `issue`; the next `issue` re-arms it.
Keep-alive reads are not counted in `reads`, `bytes` or latencies. A keep-alive read that fails, or cannot be
submitted, turns the keep-alive off but does not fail the reader.

## How the engine uses it

Only through `strata::kernels::PleTable` in `src/kernels/ngram.cpp`:

- `ngram.cpp:249-262`: with `PleIo::Direct`, the table's offset comes from the `GgufFile` parse
  (`data_start + tensor offset`), the mapping is then dropped so no page enters the process, and
  `reader.open(...)` and `reader.set_keepalive(...)` are called.
- Defaults in `PleIoOptions` (`include/strata/kernels/ngram.hpp:115-119`): `max_inflight = 256`,
  `cache_rows = 1 << 20`, `io_thread = true`.
- Row size is chosen there: 90 (IQ4_NL), 160 (FP8, `PLE_ROW_BYTES_FP8`), or 110 (Q5_0) (`ngram.cpp:203-214`).

## Configuration

Set in `src/program/generate.cpp` and passed through `PleIoOptions`:

| Flag / env | Default | Effect |
|---|---|---|
| `--ple-gguf PATH` | (shard holding `per_layer_token_embd.weight` with `--native`) | The table's file. |
| `--ple-io direct\|mmap\|ram` | `direct` | `direct` uses this reader; `mmap` and `ram` do not (see [../kernels/cpu/ngram.md](../kernels/cpu/ngram.md)). `ram` is refused on Windows. |
| `--ple-row-cache N` | 1,048,576 | `cache_rows`; 0 = off. |
| `--ple-inflight N` | 256 | `max_inflight`. |
| `--ple-delay-us U` | 0 | `set_injected_delay_us`. |
| `--ple-sync-submit` | off | `io_thread = false` (A/B arm; disables keep-alive). |
| `STRATA_SSD_KEEPALIVE` | 100 ms (clamped 0..10000; 0 = off) | Keep-alive period (`generate.cpp:2734-2745`). |
| `STRATA_SSD_KEEPALIVE_WINDOW` | 60 s (clamped 1..86400) | Keep-alive window. |
| `STRATA_IO_THREADS` | 16 Linux / 4 Windows | `DirectFile` pool size (see [platform.md](platform.md)). |

`generate.cpp` also warns at startup when the table sits on a rotational disk with `--ple-io direct` (#605).

## Model-specific behaviour

Flash-Next only. Qwen3.6-35B-A3B has no PLE table; the port plan makes the PLE optional at the CLI so its absence is
the normal case for that architecture ([docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) section 3.9,
[docs/qwen36-port/geometry-inventory.md](../../qwen36-port/geometry-inventory.md)). Without a PLE table there is
nothing for this reader to open.

## Tests

`ple_reader_selftest` = `ple_reader_test --selftest` (`CMakeLists.txt:997-1000`, GPU-backend builds with
`STRATA_BUILD_TESTS`; the test itself needs only CPU and disk). It writes a 500,000-row synthetic table after a
192-byte header (the real shard's data offset, so rows are misaligned the same way), where each row encodes its own
index, and runs for row sizes 90 and 110, both threading modes, cache 0 and 4096, in-flight 1/8/64:

- decode-shaped tickets of 16 random rows; rows straddling page boundaries; duplicates, neighbours, first/last and
  out-of-range rows; a prefill-shaped ticket much larger than the in-flight window; two tickets collected in reverse
  order; cache bound respected;
- a 3 ms injected delay is observed and does not change bytes;
- keep-alive: about 8-20 reads in 300 ms at a 20 ms period, none counted as row reads, stops after the window,
  restarts on the next issue (even a fully cached one), stops when turned off, cleared by `reset_stats`;
- no keep-alive without the worker thread.

`--dir D` chooses where the temporary table goes (default: the system temp directory).

Manual, on the real table: `ple_reader_test --gguf SHARD2 [--rows N] [--tokens FILE] [--inflight N]
[--direct-first] [--direct-only] [--sync]` compares `PleTable` in mmap and direct modes for random rows and for the
16 rows of every token in `--tokens`, requires bit-identical results, and prints per-token timings plus the reader's
I/O report.

## Gotchas

- The header says the row cache uses "clock eviction"; the implementation is set-associative with round-robin
  replacement inside each 8-way set.
- `stats()` returns a reference the worker may be updating; use `snapshot()` while reads or keep-alives can be in
  flight.
- Errors are sticky until `close()`/`open()`.
- `row_bytes` must not exceed one page; a row can straddle at most one page boundary, which the 2-page slots cover.
- The 110-byte (Q5_0) run in the selftest is the stated evidence for allowing Q5_0 PLE with `--ple-io direct`
  (`ple_reader_test.cpp:78-80`).

## See also

- [../kernels/cpu/ngram.md](../kernels/cpu/ngram.md): `PleTable`, hashing, decoding, `--ple-io` modes
- [platform.md](platform.md): `DirectFile` and `read_range` (used by `--ple-io ram`)
- [artifact.md](artifact.md): the `GgufFile` parse that supplies the table offset
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works" (SSD tier) and "Troubleshooting" (#605)
