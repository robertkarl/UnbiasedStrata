[Codebase docs](../README.md) › Libraries

# Support libraries

The support libraries are the small, mostly GPU-free building blocks underneath the engine. `src/program/generate.cpp`
uses them through the core and kernel layers. **Artifact** (`strata_artifact`) parses GGUF model files and their
split shards, guards the architecture (`qwen4exp` for Flash-Next, `qwen35moe` for Qwen3.6), and provides scalar
reference dequantizers. **Platform** wraps OS-specific I/O and memory: unbuffered asynchronous `O_DIRECT` reads,
retrying reads with an adaptive reader limit for network stores, and memory locking and RAM queries for the expert
arena. **ngram** (`PleReader`) streams Flash-Next's n-gram (PLE) table rows straight from the SSD for the PLE kernel
in `src/kernels/ngram.cpp`. **Spec** (`strata_spec`) holds the suffix-lookup drafter and the online policy that picks
the MTP window or a lookup window for each verify round. **Plan** holds the older `strata-plan` VRAM arithmetic; the
engine itself uses `vram_plan.hpp`, documented under Core. Apart from `ngram/` and parts of `platform/`, which are
compiled into the GPU libraries, everything here builds and tests without CUDA or a model.

## Contents

| Page | Description |
|---|---|
| [spec.md](spec.md) | Speculative-decoding policy: `SuffixDrafter` (prompt lookup), `DraftPolicy` (MTP vs lookup window), the unused `Controller` |
| [platform.md](platform.md) | `DirectFile` (O_DIRECT / IOCP async reads), `robust_read.hpp` (retrying reads for network stores), `memory.hpp` (lock resident, RAM, DXGI budget, file-cache arithmetic) |
| [plan.md](plan.md) | `plan.hpp` memory planner and the `strata-plan` CLI; pointer to the engine's `vram_plan.hpp` |
| [artifact.md](artifact.md) | GGUF v3 reader, split-shard model view, architecture guard, scalar dequantizers, `strata-gguf` / `strata-dequant` |
| [ngram-ple.md](ngram-ple.md) | `PleReader`: ticketed, deduplicated, cached direct reads of the PLE / n-gram table, SSD keep-alive |

Related pages written elsewhere: [../core/vram-plan.md](../core/vram-plan.md) (the engine's VRAM planner),
[../kernels/cpu/ngram.md](../kernels/cpu/ngram.md) (`PleTable`, which drives `PleReader`),
[../build/README.md](../build/README.md) (including `include/strata/platform/hip_compat/`).

## Source map

| Path | Lines | Role |
|---|---|---|
| [include/strata/spec/suffix_drafter.hpp](../../../include/strata/spec/suffix_drafter.hpp) | 53 | Suffix-lookup drafter API |
| [src/spec/suffix_drafter.cpp](../../../src/spec/suffix_drafter.cpp) | 88 | Trigram hash index, longest-match proposal |
| [src/spec/suffix_drafter_test.cpp](../../../src/spec/suffix_drafter_test.cpp) | 141 | Unit tests and `--simulate` replay |
| [include/strata/spec/draft_policy.hpp](../../../include/strata/spec/draft_policy.hpp) | 52 | MTP-vs-lookup window policy API |
| [src/spec/draft_policy.cpp](../../../src/spec/draft_policy.cpp) | 101 | Learned costs and acceptance, choice rule |
| [src/spec/draft_policy_test.cpp](../../../src/spec/draft_policy_test.cpp) | 63 | Policy tests on simulated rounds |
| [include/strata/spec/controller.hpp](../../../include/strata/spec/controller.hpp) | 73 | Cost-model speculation controller (test-only) |
| [src/spec/controller.cpp](../../../src/spec/controller.cpp) | 73 | Controller implementation |
| [src/spec/controller_test.cpp](../../../src/spec/controller_test.cpp) | 74 | Controller tests |
| [include/strata/platform/direct_file.hpp](../../../include/strata/platform/direct_file.hpp) | 60 | Unbuffered async file reads API |
| [src/platform/direct_file.cpp](../../../src/platform/direct_file.cpp) | 365 | Windows IOCP and POSIX O_DIRECT thread-pool implementations |
| [include/strata/platform/robust_read.hpp](../../../include/strata/platform/robust_read.hpp) | 181 | Retrying positional reads, `ReadGate`, parallel `read_range` (POSIX) |
| [include/strata/platform/memory.hpp](../../../include/strata/platform/memory.hpp) | 46 | Lock-resident, RAM, DXGI budget, `file_cache_keeps` |
| [src/platform/memory.cpp](../../../src/platform/memory.cpp) | 138 | Windows / POSIX implementations |
| [src/platform/memory_test.cpp](../../../src/platform/memory_test.cpp) | 72 | `file_cache_keeps` cases and a lock test |
| [include/strata/plan/plan.hpp](../../../include/strata/plan/plan.hpp) | 168 | P1.S9 Flash-Next memory planner (header-only) |
| [src/plan/plan_main.cpp](../../../src/plan/plan_main.cpp) | 73 | `strata-plan` CLI |
| [include/strata/plan/vram_plan.hpp](../../../include/strata/plan/vram_plan.hpp) | 178 | Engine VRAM planner (documented in [../core/vram-plan.md](../core/vram-plan.md)) |
| [include/strata/artifact/gguf_reader.hpp](../../../include/strata/artifact/gguf_reader.hpp) | 627 | GGUF v3 reader, `GgufModel`, architecture guard |
| [include/strata/artifact/gguf_split.hpp](../../../include/strata/artifact/gguf_split.hpp) | 52 | Split-shard path resolution |
| [include/strata/artifact/dequant.hpp](../../../include/strata/artifact/dequant.hpp) | 292 | Scalar reference dequantizers |
| [src/artifact/gguf_reader.cpp](../../../src/artifact/gguf_reader.cpp) | 67 | `strata-gguf` CLI |
| [src/artifact/dequant.cpp](../../../src/artifact/dequant.cpp) | 123 | `strata-dequant` CLI |
| [src/artifact/gguf_reader_test.cpp](../../../src/artifact/gguf_reader_test.cpp) | 93 | Duplicate-name refusal test |
| [include/strata/ngram/ple_reader.hpp](../../../include/strata/ngram/ple_reader.hpp) | 100 | `PleReader` API |
| [src/ngram/ple_reader.cpp](../../../src/ngram/ple_reader.cpp) | 547 | PLE table direct reader |
| [src/ngram/ple_reader_test.cpp](../../../src/ngram/ple_reader_test.cpp) | 298 | Synthetic selftest and real-table check |

Related tests outside these folders: [tests/platform/direct_file_async.cpp](../../../tests/platform/direct_file_async.cpp)
(60 lines) and [tests/core/gguf_split_test.cpp](../../../tests/core/gguf_split_test.cpp) (220 lines).

## Build targets

| Target | Kind | Sources | Needs GPU backend |
|---|---|---|---|
| `strata_artifact` | INTERFACE (header-only) | `include/strata/artifact/` | no |
| `strata-gguf`, `strata-dequant` | executables | `src/artifact/*.cpp` | no |
| `strata_plan` | INTERFACE (header-only) | `include/strata/plan/` | no |
| `strata-plan` | executable | `src/plan/plan_main.cpp` | no |
| `strata_spec` | STATIC | `src/spec/{suffix_drafter,controller,draft_policy}.cpp` | no |
| (in `strata_core`) | | `src/platform/memory.cpp` | yes |
| (in `strata_kernels`) | | `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp` | yes |

Tests are registered only with `-DSTRATA_BUILD_TESTS=ON`, which defaults to OFF in this checkout because
`tests/CMakeLists.txt` and `bench/micro` are absent (`CMakeLists.txt:88-92`). Each leaf page lists its tests.

## Key concepts and data flow

```
                         model files (.gguf, maybe split -00001-of-0000N)
                                         |
                    gguf_split_paths --> GgufModel / GgufFile  (artifact: mmap, parse, guard)
                       |                       |                         |
          expert loader (core)         PLE table offset            dense/head loaders (core)
          read_at + ReadGate             (data_start+off)          dequant refs for parity tests
          lock_resident (platform)              |
                                        PleReader (ngram)  --issue/collect-->  PleTable (kernels/ngram.cpp)
                                                |
                                        DirectFile (platform: O_DIRECT / IOCP, STRATA_IO_THREADS)

  verify loop (generate.cpp):  SuffixDrafter.propose --> DraftPolicy.choose(T_mtp, k, match)
                               --> verify window --> DraftPolicy.observe(accepted, round_ms); SuffixDrafter.append
```

- **Refuse early.** The GGUF reader refuses duplicate names, inconsistent split keys and missing shards at open;
  `plan.hpp` throws `DoesNotClose` instead of returning a plan it cannot honour.
- **Keep the PLE table out of RAM.** `GgufFile` only supplies the offset; the mapping is dropped and `PleReader` reads
  4 KiB pages unbuffered, with a small bounded row cache.
- **Output-neutral speculation.** The spec library only decides which drafts to verify; verification keeps the output
  identical to plain decoding.
- **Network stores.** `robust_read.hpp` (a fork addition) retries transient errors and halves concurrency on refusal.

## See also

- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works"
- [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) and [docs/qwen36-port/](../../qwen36-port/)
- [../core/README.md](../core/README.md), [../kernels/cpu/ngram.md](../kernels/cpu/ngram.md)
