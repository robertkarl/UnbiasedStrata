[Codebase docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › N-gram hash and PLE table

# The PLE n-gram hash and the per-layer token table (`PleTable`)

## Purpose

Qwen3.8-Flash-Next (`qwen4exp`) has a per-layer embedding (PLE) on layer 1 whose input is 16 rows gathered from a
hashed n-gram table, `per_layer_token_embd.weight`: 320,001,536 rows of 160 values (IQ4_NL, 90 B per row, 28.8 GB
of elements in the model's **second GGUF shard**). The table is not in the pack; it is the only tensor the engine
reads straight from the GGUF. This file is the host side of that module:

- `ngram_rows` - the **host hash**: 16 row indices per token from the token and its two predecessors (64-bit multiply
  and XOR, `qwen4exp.cpp` L1092-1124 upstream).
- `PleTable` - the **row gather**: opens the shard, validates the tensor, and returns the 16 rows as 2,560 floats
  (head-slowest), through one of three I/O modes (`--ple-io direct|mmap|ram`).

The PLE block that consumes the rows on the GPU is `src/kernels/cuda/ple.cu`
([../cuda/norms-rope-elementwise.md](../cuda/norms-rope-elementwise.md)). The unbuffered SSD reader used by the default
mode is `strata::ngram::PleReader` ([../../libs/ngram-ple.md](../../libs/ngram-ple.md)).

Qwen3.6-35B-A3B has no PLE: nothing here runs for it (`docs/qwen36-port/step2-notes.md`).

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/ngram.hpp](../../../../include/strata/kernels/ngram.hpp) | 216 | Geometry constants, `PleConsts`, hash, dequantizers, `PleIo`, `PleIoOptions`, `PleTable`, the prefetch switch |
| [src/kernels/ngram.cpp](../../../../src/kernels/ngram.cpp) | 521 | Implementation |

`ngram.cpp` is host-only but is compiled into the GPU library `strata_kernels` (with `src/ngram/ple_reader.cpp` and
`src/platform/direct_file.cpp`) because `ple.cu` shares its geometry constants (`CMakeLists.txt`, comment above
`ple_parity`). It is therefore only built when a GPU backend is enabled.

## Geometry constants (`ngram.hpp:31-49`)

`NG_N_EMBD = 2560`, `PLE_HEAD_DIM = 160`, `NGRAM_SIZE = 3`, `HEADS_PER_NGRAM = 8`, `PLE_N_HEADS = 16`,
`PLE_CONV_KERNEL = 4`, `PLE_EOS_TOKEN_ID = 248044`, `NG_HC = 4`, `NG_HC_DIM = 10240`, `NG_HIST = 9`,
`TOKEN_NULL = -1`, `PLE_TABLE_ROWS = 320001536`, `PLE_ROW_BYTES = 90` (IQ4_NL), `PLE_ROW_BYTES_FP8 = 160`.
These are Flash-Next's and are compile-time.

## Main functions and types

| Symbol | Where | What it does |
|---|---|---|
| `ple_artifact_consts()` | `ngram.cpp:38` | The artifact's three layer multipliers, 16 head vocab sizes (20000003, 20000023, ... - not an arithmetic sequence) and their running-sum offsets, written out verbatim |
| `ngram_mixed(ctx, mult, n)` | `ngram.cpp:59` | `ctx[0]*m[0] ^ ctx[1]*m[1] ^ ...` in wrapping uint64 |
| `ngram_rows(tokens, prev, n_tokens, c, out)` | `ngram.cpp:68` | Per token: context = token + predecessors (`prev` oldest first); an EOS or missing (`< 0`) predecessor cuts and propagates to every older position; heads 0-7 hash the bigram, 8-15 the trigram; `row = mixed % vocab[h] + offset[h]` |
| `iq4nl_dequant_row(row, out160)` | `ngram.cpp:102` | 5 blocks of 32: fp16 `d`, then 16 bytes; byte j holds elements j and j+16 (split halves, not interleaved) |
| `fp8_e4m3_dequant_row(row, scale, out160)` | `ngram.cpp:134` | E4M3 "fn" table (0x7F/0xFF NaN read as 0) times the table's single scale |
| `PleIo`, `PleIoOptions` | `ngram.hpp:113-128` | `Direct` (default) or `Mmap`; `max_inflight` 256, `cache_rows` 1<<20, `io_thread`, `lock` (ram mode), keep-alive settings |
| `PleTable::open(path, err, io)` | `ngram.cpp:179` | Parses the GGUF with `GgufFile`, finds `per_layer_token_embd.weight`, accepts IQ4_NL, Q5_0 (110 B rows) or I8 tagged `strata.ple.format = f8_e4m3` with a positive `strata.ple.scale`; checks the tensor fills the file (alone in its shard) or fits at its offset; then sets up the I/O mode |
| `issue(rows16)` / `collect(out2560, err)` | `ngram.cpp:373`, `:399` | The split API: issue as soon as the token is known, collect before layer 1. One token in flight per table |
| `prefetch_rows(rows16)` / `wait_prefetches()` | `ngram.cpp:300`, `:293` | Up to 16 non-blocking prefetch tickets (Direct), or `PrefetchVirtualMemory` (Windows mapped mode) |
| `gather_batch(rows, n_tokens, out, err)` | `ngram.cpp:413` | Many tokens as one reader request (page dedupe and sort); reuses matching prefetch tickets directly |
| `gather(rows16, out2560)` | `ngram.cpp:477` | Synchronous `issue` + `collect` (Direct) or 16 `read_row`s (mapped) |
| `read_row`, `rows()`, `format()`, `locked()`, `bytes_read()`, `io_report()` | `ngram.cpp:345-475` | Diagnostics; `io_report` prints the reader's hit rate, SSD reads and latency percentiles |
| `ple_prefetch_enable(bool)` | `ngram.cpp:35` | A/B switch for the Windows `PrefetchVirtualMemory` hint (`--no-ple-prefetch`) |

## The three `--ple-io` modes

| Mode | `PleIoOptions` | What happens at `open` | Row reads |
|---|---|---|---|
| `direct` (default) | `mode = Direct` | The GGUF mapping is dropped after the header parse; `PleReader::open` at `data_start + tensor offset` | Unbuffered 4 KiB reads on an I/O thread, bounded row cache; never in RAM or the page cache |
| `mmap` | `mode = Mmap` | `GgufFile`'s read-only mapping is kept | Page faults into the mapping (Windows: `PrefetchVirtualMemory` hint first) |
| `ram` | `mode = Mmap, lock = true` | POSIX only. **UnbiasedStrata change**: the table is copied into an anonymous private mapping with `platform::read_range(..., 16 readers)`, the GGUF is released, then `mlock` is attempted | Plain memory reads; no later page fault can go back to the file |

Selected in `src/program/generate.cpp:2722-2760` from `--ple-io`, `--ple-inflight`, `--ple-row-cache`,
`--ple-sync-submit`, `--ple-delay-us`; `--ple-io ram` is refused on Windows (`generate.cpp:1807-1812`). The C++ front end
passes `--ple-gguf <...-00002-of-00002.gguf> --ple-io ram` whenever the model is a two-shard file
(`frontend/main.cpp:660-680`), so a Flash-Next server always runs the RAM mode. In `ram` mode `locked()` reports
whether `mlock` succeeded; the memory is resident either way (`ngram.cpp:286`), so a failed lock (for example an
8 MB `ulimit -l` in a container) is not an error.

## Data flow

```
token t, prev[2] ──ngram_rows──> rows16 ──PleTable::issue──> (Direct: PleReader tickets on the I/O thread)
                                                   ...embedding, layer 0 on the GPU...
                         PleTable::collect ──decode (IQ4_NL / Q5_0 / FP8)──> out2560 (host)
                         ──cudaMemcpyAsync──> emb_dev ──> ple.cu block on layer 1
```

Callers:
- Decode: `ple_issue_token` / `ple_finish_token` in `src/core/layer.cpp:1456-1478` (`ple_stage_token` does both) (hash, `issue`, `collect`, H2D copy).
- Verify window: `src/core/verify.cpp:1518-1543` (hash per token, `prefetch_rows`, then `gather_batch`).
- MTP drafter: `src/core/mtp.cpp:1269-1272`.
- Prompt path: `src/prefill/prefill.cpp:1836-1841` (`gather_batch` per chunk).
- The table is opened once in `src/program/generate.cpp:2722`.

## Invariants and rival readings (`ngram.hpp:12-21`)

The hash is "the one place in this model where a wrong bit is silently plausible"; each property has a rival that
still yields an in-range index, and each is asserted observable by `ple_parity`:

- XOR, not sum; `% vocab_size`, not a mask (the vocab sizes are not powers of two).
- The EOS cut is evaluated before storing and propagates to every older position; a missing predecessor is
  `TOKEN_NULL = -1` (`t < 0`); token id 0 is a real token; a token's own EOS does not cut its own context.
- `prev` is oldest first: predecessor s positions back is `prev[i*2 + (2 - s)]` (`ngram.cpp:75-78`; the read is at `ngram.cpp:78`).
- The IQ4_NL codebook has no `-8` at the pinned ggml (3cf03257) (`ngram.hpp:84-89`).
- Rows are flattened head-slowest (`out[h*160 .. (h+1)*160)`), as `ggml_get_rows` does.
- The table offset comes from a real GGUF parse: the data section starts after the header (192 B in the original
  shard), and reading at file offset 0 would decode shifted but plausible rows (`ngram.hpp:157-161`).

## Configuration

| Knob | Effect |
|---|---|
| `--ple-gguf PATH` | The shard holding `per_layer_token_embd.weight` |
| `--ple-io direct\|mmap\|ram` | I/O mode (above) |
| `--ple-row-cache N` | Row cache capacity in rows (default 1,048,576; 0 = off) |
| `--ple-inflight N` | Outstanding SSD reads (default 256, 1..1024) |
| `--ple-delay-us U` | Fault injection: every row read completes no earlier than U us |
| `--ple-sync-submit` | Submit reads on the token thread (A/B arm) |
| `--no-ple-prefetch` | Disable the Windows `PrefetchVirtualMemory` hint |
| `STRATA_SSD_KEEPALIVE`, `STRATA_SSD_KEEPALIVE_WINDOW` | Direct mode only: ms without a read before a keep-alive read (default 100, 0 = off) and seconds after the last request (default 60) (`generate.cpp:2734-2744`) |

## Tests

| Test | What |
|---|---|
| `ple_parity --selftest` (ctest `ple_parity`) | Hash vs `ref/ngram.py` via `ple_oracle_vectors.inc`; table rows vs numpy over the original GGUF; the PLE block vs a ggml capture. Needs the real shard, `pack/full/dense.bin` and `bench/micro/ple_{in,out}.bin` |
| `ple_fp8_parity` (no ctest) | FP8 table through `PleTable` in Direct and Mmap modes and `gather_batch`, bit for bit vs the checkpoint's decoded values |
| `ple_q5_parity` (no ctest) | Q5_0 rows vs ggml's dequantizer on a real file |
| `ple_reader_selftest` (`src/ngram/ple_reader_test.cpp`) | The reader, and `PleTable` mmap vs direct, on a temporary file |

See [../parity/README.md](../parity/README.md).

## Gotchas

- The note on `madvise(MADV_RANDOM)` in `ngram.hpp:204-214` is upstream's (Windows-centric) and says no advice is
  applied; that is still true of this file.
- `ngram.hpp:120-122` still describes `lock` as locking "the whole mapped table"; since the UnbiasedStrata change it
  copies the table into anonymous memory instead (`ngram.cpp:265-287`).
- `issue` returns false if the previous token's rows were never collected; `gather_batch` refuses while a token is in
  flight.
- In Direct mode `gather` is `const` but mutates the reader through a `const_cast` (`ngram.cpp:480`).
- `ple_parity`'s default GGUF path is `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` relative to the
  source tree unless `STRATA_PLE_GGUF` or `--gguf` is given.

## See also

- [../../libs/ngram-ple.md](../../libs/ngram-ple.md) (`PleReader`), [../cuda/README.md](../cuda/README.md)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) ("How it works": the SSD tier)
- [docs/qwen36-port/geometry-inventory.md](../../../qwen36-port/geometry-inventory.md) (PLE optional for Qwen3.6)
