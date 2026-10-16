[Codebase docs](../../README.md) › [Core](../README.md) › [Experts](README.md) › Dispatch

# Dispatch: who computes each routed expert

## Purpose

This page covers the adapter between the engine's layer loop and the expert tiers. For each MoE layer it receives
the activation and the router's ids, and it decides for every routed entry whether that entry is:

- a VRAM hit (the GPU computes it from a cache slot);
- read by the GPU over PCIe from pinned host memory;
- computed on another GPU;
- computed by the CPU pool.

Then it runs the CPU's share. It also keeps the hit counters that the serve loop reports and `/health` shows.

Files: [include/strata/core/expert_source.hpp](../../../../include/strata/core/expert_source.hpp) (`ExpertDispatch`,
`GpuPlanSink`, the entry points), [src/core/expert_source.cpp](../../../../src/core/expert_source.cpp)
(`:1879-2400`), [include/strata/core/hit_hook.hpp](../../../../include/strata/core/hit_hook.hpp),
[include/strata/core/progress.hpp](../../../../include/strata/core/progress.hpp).

## `ExpertDispatch` (one per session)

`expert_source.hpp:239-383`. Per-session state, reused every layer so the token path allocates nothing.

| Group | Fields | Meaning |
|---|---|---|
| Wiring | `pool`, `src`, `lookahead`, `remote[3]` / `remote_count`, `peer`, `n_expert` | The CPU pool, the expert source, the file-tier prefetcher, the extra GPU tiers |
| VRAM tier | `cache`, `host_res`, `cache_base`, `cache_blob`, `cache_slot_off` | The cache, the host copy of the residency table (`n_layers x n_expert`, slot or -1) and the slot addressing |
| Hit counters | `cache_hits`, `cache_admitted`, `cache_refused` | Lookups served from VRAM, lookups that took a free slot (token-path admission only), and lookups the CPU served |
| Window counters | `multi_misses`, `multi_entries`, `pcie_experts`, `offload_entries`, `peer_entries` | Distinct experts and entries the CPU computed in verify windows; distinct experts read over PCIe; entries the GPU computed from outside its cache (PCIe or another GPU, #588); entries the peer served |
| PCIe share | `plan` (`GpuPlanSink*`), `pcie_num` | The verify window's GPU plan, and the share in 1/256 units |
| Adaptive tier | `usage` | Decayed routing counts per `(layer, expert)`; empty means off |
| R4.2c hit path | `hit_out`, `parts_out`, `x_q8_0_hit`, `x_q8_0_hit_scale`, `d_slot`, `d_dst`, `is_hit`, `decided`, `hit_done`, `hit_poke`, `hit_ready` / `hit_late` | See "The token-path hit hook" below |
| Failure | `failed` (latches), `fail`, `fail_layer`, `fail_expert` | The loop has no error channel. The driver checks these after the loop. Once set, every later call returns at once (`:375-381`), so the engine never continues from a half-filled `parts` |
| Timing | `ms_plan`, `ms_actq`, `ms_jobs`, `ms_run` | Verify-window dispatch sections |

The header's contract (`expert_source.hpp:8-21`) lists three rules:

1. `x_f` is the same pinned buffer every layer, so no conversion may be cached by pointer.
2. The pool does **not** apply the router weight. `moe_combine` does, on the device.
3. The canonical kernel's geometry is fixed, so a different width is refused rather than mis-indexed.

## Single-token pool: `expert_pool_dispatch`

`expert_source.cpp:1881-2001`. It has `PoolFn`'s signature. `generate.cpp`'s `drive_pool` wraps it, times it and
writes the `--dump-routing` trace.

- Refuses `n_embd != H` (2560) and any native pack (`:1891-1903`). Native packs run their experts in verify windows
  only.
- Quantizes `x_f` to `ActQ` on every call (rule 1).
- An entry is a hit either when `host_res` says so (the token graph's static table) or when the R4.2c `Launch` phase
  marked it in `is_hit`. A hit's output row is zeroed (`y_miss` is a reused pinned buffer) and skipped. Rows that a
  remote tier owns are zeroed too.
- Each remaining expert becomes an `ExpertJob` with `weight = 1.0` (rule 2). Then `pool->run_split` (rows of every
  expert split across all threads; the default, `--no-split-rows` disables it) or `pool->run`.
- `blob == nullptr` sets `failed` ("the expert source could not produce a blob").

## Verify-window pool: `expert_pool_dispatch_multi`

`expert_source.cpp:2011-2293`. Called by `drive_pool_multi` (`generate.cpp:824-830`) for `n_tok <= MAXT` tokens.
`n_tok * k` must fit `kMaxWindowEntries = 128` (`:2007`).

1. **Prefetch hint and counts.** `lookahead->submit` warms the next layer's predicted file-tier experts. Each routed
   entry adds 1 to `usage[layer][e]`.
2. **The GPU plan** (only with `plan`, `:2066-2148`). Distinct experts are taken in routing order (`first_of`). A
   miss is an expert with no VRAM slot and not held by the peer. With `pcie_ok = pcie_num > 0 && src->pcie_layer(layer)`,
   `m = (nmiss * pcie_num) >> 8`. The **last `m` misses in routing order** become kind 1 if `src->pinned(l, e)` and
   staging has room (at most `staging_cap` and 64). Kinds per entry:
   - `0` VRAM: the group pointer is the slot address;
   - `1` PCIe;
   - `2` peer;
   - `-1` CPU.

   VRAM groups go into `GpuPlanSink::ptr/start/dst/tok` first, then PCIe groups into `ptr2/start2`. `counts[0..2]` hold
   the VRAM group count, the entry count and the PCIe group count. Then `publish(ctx)` runs, and `fetch(ctx, src, n, bytes)`
   starts the copy-engine DMA of the PCIe blobs alongside the CPU's work. `pcie_mode` decides how a PCIe group reaches
   the GPU: `0` DMA into staging, `1` the grouped kernel reads the mapped host alias directly, `2` a copy kernel stages
   it in the graph. `--pcie-mode dma|direct|kernel|auto` maps dma to 0, direct to 1, and anything else to 2
   (`generate.cpp:5516`). The plan sink belongs to the verifier (`Verifier::plan_sink`, `src/core/verify.cpp`).
3. **Other GPUs.** `peer->launch` (rows of kind 2), then `remote[r]->begin`. Entries a remote tier owns become kind 2.
4. **CPU share.** If any entry is kind -1:
   - Activations are quantized per token. The Q2_0 kernels use `act_quant_any`; other native packs use
     `native_quant_act` at the layer's width `W = fmt.n_embd`; the canonical pack uses `act_quant_q8_1`. With
     `--remote-expert-opt`, tokens whose entries all run on GPUs are skipped.
   - `src->prefetch(layer, misses)` fetches the CPU's experts together.
   - One `ExpertJobMulti` per distinct missed expert (`job_of`) carries every token routed to it.
   - `pool->run_split_multi_native(fmt, ...)` or `run_split_multi(...)`.
5. **Rows the GPU owns** are zeroed on the host, except when the GPU's `copy_rows_from_mapped` already zeroes them
   (`STRATA_DEC_BATCH` on, the default, and `STRATA_VERIFY_DEVICE_PLAN` off), or a direct peer launch is writing them.
6. `remote[r]->finish`, `peer->finish`. Counters: kind 0 adds to `cache_hits`. Kinds 1 and 2 add to `offload_entries`
   (kind 2 with a peer also adds to `peer_entries`). Kind -1 adds to `cache_refused` and `multi_entries`. `multi_misses`
   adds the distinct CPU experts.

`STRATA_POOL_TRACE` prints each step per layer.

## The PCIe fraction (`--pcie-frac`)

The share of each layer's distinct missed experts that the GPU reads from pinned host memory instead of the CPU
computing them. It applies only in verify windows, and only for pinned, aliased blobs.

| Where | Rule |
|---|---|
| Default (`generate.cpp:2361-2378`) | `--pcie-frac` < 0 means the model's default: 0.2 for the canonical Q2_0 pack, 0.55 for native packs. For native packs the H2D bandwidth is probed once and `pcie_frac_for_gbps(bw, 0.55)` (`:1275`) scales the share linearly below 20 GB/s (#485, PR #44) |
| Per layer split stage | Each stage probes its own link (`:2972`) |
| Per request | The serve request key `pcie_frac` (`:6660`) overrides it for one request (`:7335`) |
| Off | `--pcie-frac 0`; also needed with `STRATA_ARENA_MMAP` (no device alias) |

The resident RAM mode's copy answers `pcie_layer` for the whole copy (`FileExpertSource::pcie_layer`,
`expert_source.cpp:1873`). Under a partial pin only the registered prefix qualifies.

## The token-path hit hook (R4.2c): `hit_hook.hpp` and `expert_hit_run`

`HitFn` / `HitPhase` (`hit_hook.hpp:28-39`) is the seam that lets `session_loop` order the work without knowing what
an expert cache is. It is wired when `--expert-cache > 0` (`generate.cpp:4148-4149`).

```
ring -> hits(Launch)  GPU: quantize + grouped hit kernel into hit_out
     -> pool()        CPU: the misses, concurrent with the GPU
     -> H2D misses into parts
     -> hits(Combine) GPU: parts += hit_out
     -> post[l]
```

`expert_hit_run` (`expert_source.cpp:2295-2400`):

- **Launch.** Makes the hit/miss decision once, on this layer's ids. A resident expert is a hit. A non-resident one is
  `admit`ted and filled asynchronously on the same stream if a slot is free (so it is a hit in the same call), or
  counted in `cache_refused`. Then it:
  - zeroes `hit_out`;
  - uploads the slot and dst lists;
  - quantizes `mixed` with `quantize_q8_0_scaled`, using the CPU's fp32 scales so a hit reproduces a miss (R4.2h);
  - launches `moe_hit_grouped_s2` (or the experimental `_cpu_order` variant, `--expert-cache-cpu-order`);
  - records `hit_done`.

  `hit_poke` (on by default; `--no-hit-poke` disables it) queries the event once, so the driver starts the work before
  the pool runs.
- **Combine.** `add_inplace(parts_out, hit_out)`. `hit_ready` / `hit_late` count whether the GPU had finished by the
  time the pool returned.
- `hits_ready()` (`expert_source.hpp:339-343`) requires every buffer. A half-wired hit path would compute some experts
  twice.

When the token graph holds a static `host_res` (`generate.cpp:4677`), the pool's half uses `host_res` and no `Launch`
decision is needed (`graph_hits` at `expert_source.cpp:1920`).

## Hit statistics and `/health`

Counters live in `ExpertDispatch` (above) and in `FileExpertSource` (`ram_reads`, `file_reads`, `file_read_bytes`,
`warmed`, `exchanges`).

- **Per request** (serve, `generate.cpp:7831-7857`) the log shows:
  - "decode expert cache hit rate: X% (hits / lookups)", where lookups = `cache_hits + cache_admitted + cache_refused`;
  - the entries offloaded over PCIe or to another GPU, separately (#588). Raising `--pcie-frac` raises the hit share of
    all routed entries without being a VRAM hit;
  - the resident RAM copy's exchanges and file reads;
  - the per-tier totals.
- **`STATS`** (serve stdin command, `generate.cpp:6578`) prints `tier_stats()` (`:6131-6158`), one JSON line:

  | Field | Source |
  |---|---|
  | `experts` | `n_layers * n_expert` |
  | `vram_slots`, `vram_held` | `xcache.slots()`, `xcache.resident()` |
  | `ram_copies` | `src.complement_count()` (resident mode only) |
  | `mapped` | the rest |
  | `mean_blob` | average blob bytes |
  | `tokens` | tokens served |
  | `vram_hits`, `lookups` | `cache_hits`, hits + admitted + refused |
  | `ram_reads`, `file_reads`, `file_mb` | `FileExpertSource` counters (0 with the arena) |
  | `warm_hints` | `src.warmed()` |
  | `major_faults` | from `/proc/self/stat` since start |
  | `max_context` | `--max-context` |

- **`GET /health`** (`frontend/main.cpp:733-772`) sends `STATS` to the engine when its lock is free. It returns
  `{"status":"ok","tiers":{...}}`, or the last reading with `"tiers_stale":true` (or `"busy":true`) during a request.

## Watchdog progress: `progress.hpp`

`Progress` (`progress.hpp:17-25`) is a process-wide heartbeat. It holds `beats`, `busy`, `where` (a stage string),
`detail`, `chunk`, `since_ms` and `ticks`. `progress_at` sets the stage and `progress_beat` / `progress_tick` count.

The prompt path (`src/prefill/prefill.cpp`) and the serve loop call them. `--serve` runs a watchdog: a request whose
heartbeat stops for `STRATA_WATCHDOG_S` seconds (default 60; 0 = off) ends the engine, naming its stage
(`generate.cpp:6065-6070`). `diag_pool_fn` is registered by the CPU pool (`src/kernels/cpu/pool.cpp:388`). It and
`diag_verify_fn` print diagnostics when the watchdog fires. `release_gpu_fn` releases the verify window's GPU spin
waits before the process exits.

## Configuration

| Flag / env | Default | Effect |
|---|---|---|
| `--pcie-frac F` | model default (see above) | PCIe share of missed experts |
| `--pcie-mode auto\|dma\|kernel\|direct` | auto (= kernel) | How PCIe groups reach the GPU |
| `--no-split-rows` | off | One whole expert per pool thread |
| `--no-hit-poke` | off | A/B arm of the R4.2d driver poke |
| `--expert-cache-cpu-order` | off | Experimental CPU-order GPU reduction |
| `--pool-workers N`, `--pool-affinity`, `--no-host-worker` | | CPU pool shape (see the CPU kernel docs) |
| `STRATA_DEC_BATCH=0`, `STRATA_VERIFY_DEVICE_PLAN=1` | | Who zeroes GPU-owned rows |
| `STRATA_POOL_TRACE` | off | Per-step trace |
| `STRATA_WATCHDOG_S` | 60 | Serve watchdog |

## Model-specific behaviour

- **Flash-Next canonical Q2_0 pack.** May use the single-token path (2560-wide). The default PCIe share is 0.2.
- **Native packs (all of Qwen3.6, Flash-Next IQ).** Verify windows only. The activation width is the layer's own
  (`fmt.n_embd`, 2048 for Qwen3.6, `:2032-2042`), and the default PCIe share is 0.55, scaled by the link probe.

## Tests

The dispatch functions have no unit test in `tests/core`. `pool_native_widths_test` checks the CPU pool's native rows
at Qwen3.6 widths (see [pack-format.md](pack-format.md)). End-to-end coverage comes from the verify and runner tests
(`tests/cuda/qwen36_verify_test.cpp`, `tests/python/qwen36_runner_check.py`, registered in `CMakeLists.txt` under
`STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA`). These are documented with the verify and program areas.

## Gotchas

- `lookups` excludes PCIe and peer entries, so the reported VRAM hit rate is a share of non-offloaded lookups.
- The PCIe share takes the *last* misses in routing order, not the coldest or hottest.
- `ExpertDispatch::failed` latches for the session. Clearing `fail` does not re-arm it.

## See also

[expert-sources.md](expert-sources.md) · [vram-cache.md](vram-cache.md) ·
[resident-ram-and-adaptive.md](resident-ram-and-adaptive.md) · [multi-gpu-tiers.md](multi-gpu-tiers.md) ·
[docs/upstream/DETAILS.md](../../../upstream/DETAILS.md)
