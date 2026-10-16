[Codebase docs](../README.md) › [Program](README.md) › Environment variables

# Environment variables read by the driver

**Purpose:** every `getenv` in `src/program/generate.cpp` (47 `STRATA_*` names plus `CUDA_MODULE_LOADING`), with
its default and effect. Variables read by the components (prefill, verifier, kernels, expert sources) are documented
with those areas; a few that interact closely are listed at the end. Line numbers are where the variable is read.

## Variables the driver sets

| Variable | Line | Effect |
| --- | --- | --- |
| `CUDA_MODULE_LOADING` | 1298-1304 | Set to `EAGER` unless already set, so all kernels load before the expert cache takes VRAM |
| `GPU_PINNED_MIN_XFER_SIZE` | 1292-1293 | AMD + `STRATA_ARENA_MMAP=1` (Linux): set to `1000000` unless set, so ROCclr stages big copies instead of pinning mapped pages |

## Start-up, memory and expert tiers

| Variable | Line(s) | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_ARENA_MMAP` | 167, 1292 | unset | AMD/HIP Linux: a file-backed arena; enables `pin_blobs` page-locking for adaptive-swap sources and the ROCclr setting above |
| `STRATA_ARENA_RELEASE` | 4639 | release | `0` keeps VRAM-held experts' arena pages instead of handing them back to the OS |
| `STRATA_ARENA_PIN_GIB` | via `arena_pin_cap_gib()` (3496) | unset | Cap (GiB) on pinning the resident arena; 0 = pin all; overrides the 8 GiB WDDM multi-GPU cap |
| `STRATA_EARLY_REMOTE_CONTEXTS` | 2348 | off | `1`: create every helper GPU context before the weight arena (WSL2 pinned-budget workaround) |
| `STRATA_FETCH_THREADS` | 3471 | source default | Fetch threads of the mmap/GGUF expert source |
| `STRATA_UNBUFFERED_LOAD` | 3482 | unset | Make the file tier read unbuffered even without `--resident-budget-gib` |
| `STRATA_RESIDENT_PIN` | 1560, 1571 | pin | `0`: keep the resident RAM copy pageable (`--resident-experts`, `--resident-budget-gib`) |
| `STRATA_RESIDENT_HEADROOM_GIB` | 1562, 1573 | 4 | RAM left free by the resident copy |
| `STRATA_LOOKAHEAD` | 4024 | on | `0` disables the routing-aware prefetch of the file tier |
| `STRATA_LOOKAHEAD_K` | 4035 | 10 | Experts per token the lookahead warms |
| `STRATA_PEER_HOT` | 3000 | 0.45 | `--peer-device`: share of the hottest profile ranks moved to the peer (0 = off) |
| `STRATA_PEER_HOT_AT` | 3002 | 8700 | How many leading ranks that share is drawn from |
| `STRATA_TEST_CACHE_FAIL` | 3709 | 0 | Test: the first N expert-cache opens fail like an out-of-commit `cudaMalloc` |
| `STRATA_KV_ROT` | 1822 | off | `1`: INT8 K/V through the Hadamard rotation (`--kv int8`) |
| `STRATA_NO_IQ256` | 2387 | unset | Only changes the log line naming the CPU kernel family on a CPU without AVX-512 (the kernel choice itself reads it in `src/kernels/cpu/native_expert.cpp`) |
| `STRATA_SSD_KEEPALIVE` | 2739 | 100 ms | `--ple-io direct`: read one page after this long without a read (0 = off; clamped 0..10000) |
| `STRATA_SSD_KEEPALIVE_WINDOW` | 2740 | 60 s | How long after the last row request the keep-alive continues (1..86400) |
| `STRATA_RSS_TRACE` | 207 | off | Print `RssFile` at start-up steps |
| `STRATA_TRACE` | 1059, 5995 | off | VRAM free after each start-up step (`mem_mark`); per-request step trace in serve |

## Prompt path

| Variable | Line(s) | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_PREFILL_AUTO_MAX` | 1444 | 8192 | With bare `--prefill auto`: allow chunks up to 16384 or 32768 |
| `STRATA_PREFILL_LEND_PCT` | 4781 | 90 or 85 | Max % of cache slots the prompt path may borrow under `--prefill auto` (90 when >= 90% of expert bytes are pinned) |
| `STRATA_REFILL_BLOCKING` | 221 | off | Refill lent slots one blocking copy at a time instead of queued copies + one wait |
| `STRATA_REFILL_SERIAL` | 7148 | off | Layer split: refill stage after stage instead of issuing all stages' copies first |
| `STRATA_BATCH_DECODE_SHARE` | 7231 | 0.5 | `--batch`: slots decode for this share of each prompt chunk's time between chunks |

## Layer split

| Variable | Line(s) | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_SPLIT_OWN` | 1716 | borrow | `1`: every stage keeps its own prompt buffers; `auto`: only when they are <= 12% of each card |
| `STRATA_SPLIT_OWN_PLACE` | 3076 | unset | `reserve`: the `auto` placement search counts own buffers as a reserve |
| `STRATA_SPLIT_OWN_BUFFERS` | 5178 | on | `0`: every stage borrows even if its card has room for its own buffers |
| `STRATA_SPLIT_RING` | 5016 | 96 when >= 75% resident | Streamed-ring slots on a split; 0 = the pinned-share rule |
| `STRATA_SPLIT_SMALL_OWN` | 5366 | 0 | Keep S-token prompt buffers per stage for the whole session |
| `STRATA_SPLIT_SMALL_MAX` | 5368 | S | Requests up to M tokens read in S-token chunks without lending |
| `STRATA_SPLIT_MISS_MS` | 3108 | 190 | Cost of a non-cached unit of routed mass in the `auto` placement model |
| `STRATA_STAGE_TRIM` | 2533 | off | Same as `--trim-stage-weights` |
| `STRATA_SPLIT_TIMING` | 7860 | off | Per-stage verify timing after each request |

## Decode and adaptive tier

| Variable | Line(s) | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_ADAPT_NOWAIT` | 1068 | off | `1`: a verify window may start before adaptive-swap copies land (pre-#463 behaviour) |
| `STRATA_TRACE_ADAPT` | 8311, 8340 | off | One-shot: log adapt rounds and whether copies had landed |
| `STRATA_DECODE_TIMING` | 7447 | off | Serve: per-request decode time breakdown |
| `STRATA_DUMP_FIRST_LOGITS` | 8531 | unset | One-shot: write the first verify window's logits to this path |

## Serve diagnostics and safety

| Variable | Line(s) | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_WATCHDOG_S` | 6068 | 60 | Seconds without progress during a request before the engine reports and aborts; 0 = off |
| `STRATA_CKPT_REREAD` | 6999, 7055 | off | Re-read checkpoints from token 0 instead of restoring; all reads batched |
| `STRATA_SNAPSHOT_VERIFY` | 5729, 6935 | off | Log drafter prefill path; verify restored draft KV |
| `STRATA_SNAPSHOT_FULL_CAPTURE` | 6962 | off | Do not retain a restored conversation's K/V buffers for its next park |
| `STRATA_STATE_HASH` | 7651 | off | Session fingerprints after each request |
| `STRATA_STATE_HASH_GDN` | 7693 | off | Per-GDN-layer fingerprints |
| `STRATA_SERVE_LOGIT_TRACE` | 7525 | unset | Per-token top-5 logit trace file |
| `STRATA_SERVE_LOGIT_DUMP` | 7536 | unset | Full logit rows per request (with the trace) |
| `STRATA_LOGPOS` | 7092 | unset | Teacher-forced log-probs of window-read prompt tokens |
| `STRATA_LOGPOS_EXTRA` | 7096 | 248046 | The extra id whose log-prob is appended |

## Read elsewhere but relevant

| Variable | Read in | Effect |
| --- | --- | --- |
| `STRATA_SPEC_COUPLED` | `include/strata/core/coupled_draft.hpp` | Default of `--coupled-draft` |
| `STRATA_RING_BYTES`, `STRATA_PREFILL_RING` | `src/prefill/prefill.cpp:197`, `:207` | Prompt-path ring sizing used by the lend planner |
| `STRATA_VERIFY_PROFILE` | `src/core/verify.cpp` | Verifier GPU stage profile (named in the `--gpu-stages` error) |
| `STRATA_ISA_FLOOR` | CMake cache option (`CMakeLists.txt:39`), not an environment variable | The older-CPU build the start-up ISA check reports |
| `STRATA_EMULATE_CC` | kernel code (e.g. `src/kernels/cuda/sampler.cu`) | Emulated compute capability, noted in the GPU line |

## See also

- [cli-flags.md](cli-flags.md), [diagnostics.md](diagnostics.md)
