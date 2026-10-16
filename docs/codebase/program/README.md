[Codebase docs](../README.md) › Program

# Program: the engine driver (`strata generate` / `--serve`)

`src/program/generate.cpp` is the engine's `main`: the one file that composes every component (weights, sessions,
CPU expert pool, VRAM expert cache, batched prompt path, verify windows, MTP drafter, conversation cache) into tokens.
It builds two binaries: the stand-alone `strata` target (`CMakeLists.txt:576`) and, through
`frontend/engine_entry.cpp` (which `#define main strata_engine_main` and `#include`s this file unchanged), the engine
half of `unbiased-strata`, which the HTTP front end starts as a child with `--engine --serve ...`
(`frontend/main.cpp:676-679`). It has **no tokenizer**: prompts arrive as token ids (`--tokens`, `--tokens-file`, or the
`GEN` lines of serve mode). It runs in one of three shapes: a one-shot generation (prefill, token loop, speculative
loop, report), a persistent `--serve` process that reads requests on stdin and writes tokens on stdout, or a
measurement mode that prints a GPU timing table and exits. Downward it calls into
[core](../core/README.md) (sessions, verifier, MTP, expert sources/cache, conversation state),
[kernels](../kernels/README.md) (sampler, PLE, KV formats, rope scaling, CPU expert pool),
[prefill](../prefill/README.md) (`strata::prefill::Prefill`) and the [libraries](../libs/README.md)
(`strata::spec` drafters, `strata::plan` VRAM planner, the GGUF reader, platform memory probes). Upward its only
client is the [front end](../frontend/README.md).

## Contents

| Page | What it covers |
| --- | --- |
| [cli-flags.md](cli-flags.md) | Every command-line flag `main` parses (151 `--` spellings plus `-h`), with default and effect, grouped by subsystem |
| [startup.md](startup.md) | The start-up sequence from argv to "session is up": validation, model detection (Flash-Next vs Qwen3.6), VRAM plan, weight/PLE/MTP/head loading, KV mode, expert source and expert-cache sizing |
| [expert-tiers.md](expert-tiers.md) | The expert tiers as the driver wires them: VRAM cache sizing and profile fill, the hit path, resident RAM mode, prompt-path slot lending, adaptive expert swaps (`--adapt-every`), remote/peer GPUs |
| [layer-split.md](layer-split.md) | Multi-GPU `--layer-split`: placement search, per-stage weights/sessions/caches, hand-offs, own vs borrowed prompt buffers |
| [generation-loop.md](generation-loop.md) | The one-shot path: batched prefill, the per-token loop, sampling, the final report and `--stats` |
| [speculative-decoding.md](speculative-decoding.md) | Verify windows, MTP drafts, suffix (prompt-lookup) drafts and the draft policy, `--spec-min-p`, acceptance and commit |
| [serve-mode.md](serve-mode.md) | The `--serve` loop and its line protocol with the front end (`GEN`, `T`, `DONE`, `STATS`, `VRAM`, batch slots, watchdog) |
| [prompt-cache.md](prompt-cache.md) | Conversation checkpoints, the retention policy in `conv_cache.hpp`, parked conversations, batch-slot caches; `conv_cache_test` |
| [diagnostics.md](diagnostics.md) | Dumps and measurement modes: logits (`logits_selection.hpp`), residual/layer/half/routing dumps, `--gpu-only-full`, `--gpu-stages`, `--teacher-force`, serve-side traces |
| [env-vars.md](env-vars.md) | Every environment variable read in `generate.cpp` (47 `STRATA_*` names plus `CUDA_MODULE_LOADING`) and what it changes |

## Source map

| File | Lines | Role |
| --- | --- | --- |
| [src/program/generate.cpp](../../../src/program/generate.cpp) | 8,843 | The driver: options, start-up, measurement modes, serve loop, one-shot generation and reporting. `main` alone is lines 1281-8843 |
| [src/program/conv_cache_test.cpp](../../../src/program/conv_cache_test.cpp) | 67 | Unit test of the checkpoint retention policy (`conv_cache_test`, `CMakeLists.txt:1067-1069`) |
| [include/strata/program/conv_cache.hpp](../../../include/strata/program/conv_cache.hpp) | 43 | `conv_cache::eviction_victim`: pinned root, LRU leaves |
| [include/strata/program/logits_selection.hpp](../../../include/strata/program/logits_selection.hpp) | 30 | `--logits-stride` parsing and row selection for `--dump-logits` |

## Layout of `generate.cpp`

The file is one anonymous namespace of helpers followed by one very long `main`. Line ranges are approximate section
boundaries.

| Lines | Section |
| --- | --- |
| 1-107 | Header comment ("THE DRIVER"), includes |
| 108-263 | WDDM detection, `--vram-cap` clamp (`vram_info`), AMD page pinning for swaps (`pin_blobs`), `rss_probe`, `resident_stage_swaps` |
| 265-567 | `struct Options`: every flag's storage and default |
| 569-757 | `usage()` |
| 759-777 | `parse_i64_list` (token lists) |
| 780-892 | `Drive` / `drive_pool` / `drive_pool_multi` (CPU-pool adapters + routing trace), `SplitDrive`, `GpuStage` |
| 894-1077 | Watchdog support: `mem_sample`, `stage_text`, disk-wait probes, `stall_report`, `mem_mark`, `adapt_nowait`, `argmax` |
| 1079-1125 | Conversation-cache helpers (`ConvCheckpoint` alias, `fnv1a`, `checkpoint_save` / `checkpoint_restore`) |
| 1127-1199 | `load_control_vectors` (llama.cpp control-vector GGUFs) |
| 1201-1277 | `probe_pcie_h2d_gbps`, `pcie_frac_for_gbps` |
| 1281-1593 | `main`: environment defaults, argv parsing (split into two `if` chains because of MSVC C1061) |
| 1594-2487 | Option validation, layer-split parsing, `--native` expansion, CPU ISA checks, model geometry, VRAM cap, Qwen3.6 VRAM plan, PCIe probe, rope resolution |
| 2488-3397 | Loading: native embedding, weight arena, native dense, PLE, layer-split stages, profile, split search, sessions, batch slot sessions, MTP drafter, head, logits |
| 3399-4153 | Expert source (arena or mmap), CPU pool, expert cache sizing and profile fill, stage caches, remote/peer tiers, router lookahead, R4 hit path, routing dump |
| 4169-4322 | Graph capture, sampler setup, dump files |
| 4354-4582 | Measurement modes that print and exit: `--gpu-stages`, `--graph-only`, `--gpu-only-full` |
| 4584-4954 | Token graph, device residency table, prompt-path lend planning, resident RAM mode |
| 4955-7899 | `--serve` (returns at 7898) |
| 7901-8006 | One-shot batched prefill |
| 8008-8228 | One-shot token loop |
| 8230-8660 | One-shot speculative loop |
| 8662-8843 | Report and cleanup |

## Key concepts and data flow

```
argv ──► Options ──► validation ──► model geometry (GGUF metadata: qwen4exp = Flash-Next, qwen35moe = Qwen3.6)
                                         │
             ┌───────────────────────────┴──────── Qwen3.6 only: strata::plan VRAM plan (may lower --max-context)
             ▼
   weights (pack arena + native GGUF projections) ─ PLE table (Flash-Next) ─ session(s) ─ MTP drafter ─ head
             │
   expert source (pinned arena | mmap/GGUF + RAM copies) ─► CPU ExpertPool
   expert cache (VRAM slots, filled from --expert-profile) ─► host_res / d_res residency table
             │
   ┌─────────┴───────────────┬──────────────────────────────────┐
   ▼                         ▼                                  ▼
 measurement modes        --serve loop                     one-shot generation
 (print, exit)            stdin GEN ─► resume source       Prefill.run (chunks)
                          ─► Prefill / short windows        ─► token loop (graphs)
                          ─► verify windows + MTP/suffix    ─► verify windows + MTP
                          ─► T / DONE on stdout             ─► report on stdout
```

Ideas that recur across the pages:

- **Model detection.** There is no `--model-type` flag. With `--native SHARD1` the geometry is read from the GGUF
  metadata (`read_model_geometry`, `src/program/generate.cpp:2091`); `g.arch` is `ModelArch::Qwen4Exp` (Flash-Next)
  or `ModelArch::Qwen35Moe` (Qwen3.6-35B-A3B). Without `--native` the canonical Flash-Next defaults stand. The arch
  gates the VRAM plan, `check_all`, the drafter's geometry and whether a PLE exists. See [startup.md](startup.md).
- **Native pack vs canonical pack.** `expert_layout().native` (an IQ/GGUF-form pack from `tools/iq_pack.py`) decodes
  only through verify windows and needs `--native`, `--spec >= 2` and `--prefill`; the canonical Q2_0 pack runs the
  captured per-layer graphs. See [generation-loop.md](generation-loop.md).
- **Residency table.** `host_res[layer * n_expert + expert]` holds the VRAM slot of a resident expert or
  `kNotResident`; it is mirrored to the device (`d_res`) and to each layer-split stage. Lending slots to the prompt
  path, adaptive swaps and `VRAM` shrinks all work by editing it and re-uploading. See [expert-tiers.md](expert-tiers.md).
- **Running vs positional state.** A conversation checkpoint copies only running state (GDN recurrence, PLE history,
  indexer tails); the KV cache is positional and is rewound simply by writing from a position again. See
  [prompt-cache.md](prompt-cache.md).

## Tests

- `conv_cache_test` (CTest, built with `STRATA_BUILD_TESTS` or `STRATA_BUILD_CONVERSATION_TESTS`): the retention policy.
  Run with `ctest -R conv_cache_test` from the build directory.
- The driver itself has no unit test; it is exercised end to end by the port notes' runs
  ([step4-runner.md](../../qwen36-port/step4-runner.md), [step5-serve.md](../../qwen36-port/step5-serve.md)) and the
  front end's fake-engine tests (`frontend/engine_test.cpp`, `frontend/engine_backpressure_test.cpp`) that mimic the
  serve protocol.

## See also

- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) - upstream's user-facing notes (speed, flags as the setup
  passes them, troubleshooting).
- [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) and [docs/PLAN-qwen36-35b-a3b.md](../../PLAN-qwen36-35b-a3b.md) - the
  Qwen3.6 port plan the VRAM-plan and geometry code cites.
- Code comments also cite `docs/BATCHING.md`, `docs/SECOND_GPU.md`, `docs/INSTALL.md` and `docs/DETAILS.md`; these
  files are not in this fork (only `docs/upstream/DETAILS.md` is).
