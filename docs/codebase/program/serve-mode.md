[Codebase docs](../README.md) › [Program](README.md) › Serve mode

# `--serve`: the persistent engine and its line protocol

**Purpose:** load everything once, then take requests on stdin and stream tokens on stdout. This is how the
[front end](../frontend/README.md) uses the engine: `frontend/engine.hpp` starts `unbiased-strata --engine --serve
...`, waits for `READY`, writes `GEN`/`GENI` lines and parses `T`/`DONE`/`ERR` (`frontend/engine.hpp:55-143`).
**Covers:** `src/program/generate.cpp:4955-7899`. Prompt-cache logic is on [prompt-cache.md](prompt-cache.md),
decoding on [speculative-decoding.md](speculative-decoding.md), layer-split plumbing on [layer-split.md](layer-split.md).

## Requirements (`:4956-4961`)

`--spec >= 2`, `--mtp DIR`, `--prefill CHUNK|auto`, and - when the graphed hit path is in use - a filled residency
table. Otherwise `strata serve: needs --spec T, --mtp DIR and --prefill CHUNK ...` and exit 2.

## Set-up inside the serve block

| Lines | Step |
| --- | --- |
| 4962-5230 | Prompt-path loan planning per participant (`PfPart`): chunk and ring choice, own-buffer exceptions, log lines |
| 5232-5356 | `init_prompt_paths` with step-down to smaller chunks; peer prompt rows; stage helpers |
| 5358-5394 | `STRATA_SPLIT_SMALL_OWN` / `_MAX` |
| 5395-5407 | Penalty history buffer `d_hist` (`4096 x kVerifyMaxT` ids) on the head's device |
| 5408-5516 | `Verifier ver` init (`VerifyHits` from the residency table), layer-split hand-offs, `mtp.bind`, batch-slot sessions, pool callback (`drive_pool_multi` or `drive_pool_split`), `set_split`, `set_pcie_mode` |
| 5517-5781 | Conversation cache state, `park_current`, `checkpoint_at`, `sp.on_chunk` (drafter KV + `PP` lines + periodic checkpoints), per-stage checkpoint parts |
| 5782-5937 | Adaptive tier (`apply_pending`, `adapt`, `save_profile`) - see [expert-tiers.md](expert-tiers.md) |
| 5938-5993 | stdin reader thread (`read(2)`, not `std::cin`, so `exit()` cannot hang on stdin's lock); `STOP` sets `stop_req` immediately; other lines go to a queue |
| 6001-6023 | Free-VRAM check: >= 256 MiB fine; below: a LOW warning suggesting `--vram-reserve-mib` (or that the reserve was already lowered) |
| 6024-6062 | `INFO` line |
| 6063-6090 | Watchdog thread |
| 6091 | `READY <max_context> stop` |
| 6093-6107 | `S`, `S_mtp`, `SuffixDrafter sfx`, `DraftPolicy policy`, vision constants (`kImagePad = 248056`) |
| 6108-6253 | `VRAM` command, `tier_stats` (`STATS`) |
| 6254-6558 | Batch slots (`BSlot`), `copy_to_slot`, `copy_from_slot`, `batch_step`, pipelined groups (`pump`, `pipe_drain`) |
| 6559-7896 | The request loop |
| 7897 | `save_profile("exit")`, return 0 |

## Protocol

### Engine to front end (stdout)

| Line | When | Fields |
| --- | --- | --- |
| `INFO key=value ...` | once, before `READY` | `context kv kv_resident expert_slots expert_cache_mib expert_slots_primary expert_cache_primary_mib spec mtp_max lookup vram_free_mib cvec arena_mib pool_workers pcie_frac spec_min_p conversation_cache_mib conversation_cache_slots conversation_cache_min_free_mib tail_role_token vram_elastic [batch_slots slot_cache] engine` (`:6044-6061`). Expert slots count every tier (stages and helpers add) |
| `READY <max_context> stop` | once | the (possibly plan-lowered) context; `stop` = this engine honours `STOP`. The front end takes the context from here |
| `RESUME <n>` | per request, before reading | prompt tokens reused from the live session / a checkpoint / a slot / a parked conversation |
| `PP <pos> <prompt_tokens> <ms> <tok/s>` | after each batched chunk and each window-read part | progress; tok/s over fresh tokens |
| `REUSED <n>` | prompt read, first window next | same `n` as `RESUME` |
| `T <id>` | each generated token, as its window commits | |
| `YIELDED <slot> <tokens>` | the read gave way (`BYIELD`) | before `DONE ... cancel` |
| `DONE ...` | end of request | see below |
| `BADM <slot> <0\|1>` | after a `BGEN`'s first token | 1 = continues in the batch slot |
| `BT <slot> <id>` | batch windows | one token of a slot |
| `BDONE <slot> <produced> <stop\|length\|cancel> <ms>` | a slot finishes | |
| `STATS {json}` | reply to `STATS` | see below |
| `VRAM reserve_mib=.. expert_slots=.. expert_slots_full=.. expert_cache_mib=.. expert_cache_full_mib=.. vram_free_mib=.. prompt_chunk=..` | reply to `VRAM` | |
| `ERR <message>` | a request or command cannot run | many paths `return 1` after it (fatal) |

`DONE` (`:7775-7781`), space-separated, appended over time so older parsers read a prefix:

| # | Field |
| --- | --- |
| 1 | tokens generated |
| 2 | prompt tokens `n` |
| 3 | prompt ms |
| 4 | decode ms |
| 5 | finish: `stop` (EOS), `length`, `cancel` |
| 6, 7 | drafts accepted, drafts offered |
| 8 | prompt tokens reused (`resume`) |
| 9, 10 | VRAM cache hits, lookups (decode, this request) |
| 11, 12, 13 | RAM-copy blob reads, file blob reads, file MB (this request) |
| 14 | prompt tokens actually read (fewer when cancelled mid-read, #471) |
| 15 | experts offloaded (read by the GPU over PCIe or computed by another GPU; not in lookups, #588) |

The front end parses the first seven (`frontend/engine.hpp:102`).

### Front end to engine (stdin)

| Line | Effect |
| --- | --- |
| `GEN <max_new> [key=value ...] <id,id,...>` | a text request |
| `GENI <max_new> [key=value ...] <embeddings-file> <id,...>` | with images; needs `--vision` |
| `BGEN <slot> <max_new> ...` / `BGENI ...` | `--batch`: read as a `GEN 1`, then continue in slot `<slot>` |
| `BSTOP <slot>` | end that slot at its next batch window |
| `BYIELD <slot>` | a prompt read in progress should give way at the next chunk boundary and park in `<slot>` |
| `STOP` | cancel the running request at the next prompt chunk or window (handled on the reader thread) |
| `STATS` | reply with the tier census |
| `VRAM [reserve_mib]` | `--vram-elastic` resize between requests |
| `QUIT` | leave the loop (saves the learned profile), exit 0. EOF on stdin does the same |

Request keys (`:6626-6663`; unknown keys skipped, ids start at the first token without `=`): `temperature` (absent or
`<= 0` = greedy), `top_p` (1.0), `top_k` (20; the sampled path needs 1..64), `min_p`, `penalty_last_n`,
`penalty_repeat`, `penalty_freq`, `penalty_present`, `seed` (0 = time-based), `cvec=0|1` (a loaded control vector on or
off for this request), and tuning keys `pcie_frac` and `spec_min_p` (clamped to 0..1, this request only). Validation:
`max_new >= 1`, ids in vocabulary, `n + max_new + 8 <= max_context`.

### `STATS` (the server's `/health`)

`tier_stats` (`:6138-6157`): `{"experts","vram_slots","vram_held","ram_copies","mapped","mean_blob","tokens",
"vram_hits","lookups","ram_reads","file_reads","file_mb","warm_hints","major_faults","max_context"}` - the census of
expert tiers (VRAM slots/held, RAM copies, mapped = the rest) and cumulative counters since start (tokens served, VRAM
hits, lookups = hits + admitted + refused, RAM-copy and file reads, warm hints from the router lookahead, major page
faults from `/proc/self/stat` relative to start). The same line is logged at start (`tiers at start:`).

## Request lifecycle (`:6559-7896`)

1. **Dispatch** (`:6559-6611`): while batch slots are active, pending lines are polled and batch windows run between
   them (`batch_step` or `pump`); otherwise block on the next line. `BSTOP`, `BYIELD`, `STATS`, `VRAM` are handled
   inline; `BGEN` is rewritten to `GEN 1`; the periodic profile save happens before a request.
2. `BusyScope` marks the watchdog busy; stale `STOP`s are cleared.
3. **Parse** the line and keys; for `GENI`/after an image request, rebuild the M-RoPE position table (`t = p`,
   `h = p + y`, `w = p + x` for image cells; text continues at `p + max(nx, ny)`) from strata-vision records
   (`'SVE1'`, n, nx, ny, n_embd, then floats) and upload it to every device; image rows feed the prompt path
   (`sp.embd_rows`). Each image is keyed by an FNV-1a hash of its grid and embeddings for cache matching.
4. **Choose where to resume** ([prompt-cache.md](prompt-cache.md)): the live session, the longest checkpoint, an idle
   batch slot, or a parked conversation; park the outgoing conversation when switching; restore; drop checkpoints
   that are no longer prefixes; `session_zero` when nothing matches. Print `RESUME`.
5. **Read the prompt** (`:7342-7427`) in segments ending at `STRATA_CKPT_REREAD`'s point, the system-prompt root,
   the last turn boundary, and `n - 1`. A segment of at most `--short-read` text tokens goes through verify windows
   (`read_windows`: every row committed, head sampling off, drafter fed); longer ones through the batched path after
   `lend` (`read_part`, which with batch slots interleaves slot decoding between chunks for
   `STRATA_BATCH_DECODE_SHARE` of the chunk's time and honours `BYIELD`). Checkpoints at the root and turn boundary.
   `refill` returns lent slots. Print `REUSED`.
6. **Decode** (`:7428-7622`): the verify-window loop with per-request sampling and penalties, `T` lines as tokens
   commit, finish on EOS (`stop`), `max_new`/context (`length`) or `STOP` (`cancel`).
7. **Finish** (`:7623-7895`): `ver.wait_commit`; the session's tokens become `live` (unless cancelled); optional
   `STRATA_STATE_HASH`; `YIELDED`; `DONE`; for `BGEN`, copy the session into the slot and `BADM`; flush the routing
   trace; stderr summaries (prompt/decode rates, cache hit rate with offload share, resident RAM, tiers, split timing,
   KV-streaming hit rate, suffix drafts, helper GPUs).

Fatal errors (`ERR` + `return 1`) end the process; the front end restarts the engine. A CUDA fault seen while reading
the prompt exits with `std::_Exit(1)` at once so destructors do not hang on a poisoned context (#224).

## Batch slots (`--batch N`)

Each slot has its own session per stage (carved at start, [startup.md](startup.md)). A `BGEN` reads its prompt and
first token on the main session like `GEN 1`; if more tokens are wanted the main session is copied into the slot
(`copy_to_slot`: running-state checkpoint + KV up to the length) and the slot decodes one token per batch window
(`Verifier::run_slot_rows` / `commit_slots`), emitting `BT`/`BDONE`. Slots keep their conversation when idle
(`cached`) and the turn checkpoint, so the next turn of that chat resumes from the slot (`copy_from_slot`). With a
layer split and `--batch-groups G`, groups of slots are pipelined through the stages (`batch_launch`/`batch_poll`).
Timing summaries are printed when all slots go idle (`strata batch: ...`).

## Watchdog (`:6063-6090`)

A detached thread watches `progress().beats`; while a request is busy and no heartbeat (token, chunk, window) arrives
for `STRATA_WATCHDOG_S` seconds (default 60, 0 = off), it prints `no progress for N s`, a `stall_report` (twice, 2 s
apart: pool and verifier diagnostics, disk-waiting threads and wait channels on Linux, memory and page faults, and a
minidump on Windows), calls `release_gpu_waits`, and `abort()`s so the server restarts the engine (issues #29, #31,
#267, #605).

## Gotchas

- The comment above the serve block says "Decoding is greedy" (`:4723`); that predates per-request sampling, which
  the code now does.
- `VRAM` is refused while batch slots decode, with a loan out, a peer, or the resident RAM mode.
- `ERR` on the protocol is per request for parse/validation problems (`continue`), but most runtime failures return
  from `main`.

## See also

- [prompt-cache.md](prompt-cache.md), [speculative-decoding.md](speculative-decoding.md), [expert-tiers.md](expert-tiers.md)
- [../frontend/README.md](../frontend/README.md) - the other side of this protocol
- [docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md) - serving Qwen3.6, faults found and fixed
