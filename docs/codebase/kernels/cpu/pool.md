[Codebase docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Expert pool

# The CPU expert pool (`ExpertPool`)

## Purpose

Every routed expert that is not resident in VRAM is computed on the CPU, from the expert bytes in host RAM, while the
GPU computes the resident ones. The pool's job "is not to be clever - it is to keep every physical core reading expert
bytes for the whole layer, and to be cheap enough that ten dispatches per layer cost less than one expert"
(`include/strata/kernels/cpu/pool.hpp:3-6`). It owns pinned worker threads (one per physical core except the one
the host loop spins on), a lock-free claim protocol, and four ways of splitting a layer's experts across the threads.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/pool.hpp](../../../../include/strata/kernels/cpu/pool.hpp) | 286 | `ExpertJob`, `ExpertJobMulti`, `PoolAffinity`, `CpuTopology`, `ThreadAffinity`, `ExpertPool`; protocol notes |
| [src/kernels/cpu/pool.cpp](../../../../src/kernels/cpu/pool.cpp) | 783 | Topology detection, pinning, the worker loop, claim / publish / wait, the run modes |
| [src/kernels/cpu/pool_affinity_linux.hpp](../../../../src/kernels/cpu/pool_affinity_linux.hpp) | 69 | `detail::` helpers over `pthread_{get,set}affinity_np` with dynamically sized `CPU_ALLOC` masks (CPU ids above 1023 included) |
| [src/kernels/cpu/pool_affinity_win.hpp](../../../../src/kernels/cpu/pool_affinity_win.hpp) | 64 | `detail::` helpers: hard group affinity for pool threads, CPU Sets for the host thread |

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `ExpertJob` | `pool.hpp:47` | One expert: `blob`, shared read-only `act` (`ActQ`), `out` (H floats), `weight` (diagnostic only, never applied), `slot` |
| `ExpertJobMulti` | `pool.hpp:57` | One expert for `nt <= MAXT` tokens of a verify window: per-token `act`, `out`, and native activations `nact` |
| `PoolAffinity` | `pool.hpp:68` | `All` (default: one worker per physical core), `Auto` (hybrid: P-cores, then their SMT siblings, then E-cores), `PCores` (P-cores and siblings only) |
| `detect_cpu_topology(skip_first, affinity)` | `pool.cpp:32` | Windows: `GetLogicalProcessorInformationEx` with `EfficiencyClass`; Linux: the allowed CPU mask plus sysfs `topology/{physical_package_id,core_id}` and `cpu_capacity`. Keeps the first allowed CPU of each physical core; on a hybrid CPU sorts P-cores first (#642). `skip_first` reserves the first core as `host_core` |
| `physical_cores(skip_first, affinity)` | `pool.cpp:276` | `detect_cpu_topology(...).worker_cores` |
| `pin_current_thread(core)` / `restore_thread_affinity(prev)` | `pool.cpp:297`, `:323` | Pin the calling (host) thread and restore it later. Windows uses reversible CPU Set selection; Linux saves the full mask |
| `ExpertPool(n_workers, pin, host_works, affinity)` | `pool.cpp:371` | `n_workers <= 0`: every physical core except the first (hybrid with `Auto`/`PCores`: P-cores - 1). Workers are pinned to `topo_.worker_cores[i]`; each owns an `ExpertScratch` |
| `run(jobs, n)` | `pool.cpp:724` | Whole experts, one per claim (mode 0) |
| `run_split(jobs, n)` | `pool.cpp:638` | `n <= kMaxSplit (16)` experts split by rows across all threads: gate/up rows (mode 1), quantize each intermediate on the host, down rows (mode 2). Bitwise the same as `run` |
| `run_split_multi(jobs, n)` | `pool.cpp:654` | Canonical Q2_0, multi-token: `3 * threads` equal row ranges over all `n * FF` gate/up rows (mode 3), quantize, then over `n * H` down rows (mode 4); at most `kMaxSplitMulti (96)` experts |
| `run_split_multi_native(fmt, jobs, n)` | `pool.cpp:693` | Native packs: modes 5 and 6 with the layer's own `n_ff` / `n_embd`, in batches of 96 experts |
| `diag(FILE*)` | `pool.cpp:350` | Watchdog dump: epoch, claims, done, parked/sleeping counts, each thread's state |
| `phase_ms(...)` | `pool.hpp:188` | Accumulated host time waiting for park / drain / re-park; `ms_multi_gu/q/down`, `multi_bytes` for the multi paths |

Constants: `kSpinBeforeSleep = 20 ms` (`pool.hpp:201`), `kStall = 60 s` (`pool.hpp:204`), `kMaxSplit = 16`,
`kMaxSplitMulti = 96`.

## The claim / park protocol

```
host (run / run_phase)                            worker i
------------------------------------              ------------------------------------------
wait_parked(): parked_ == n_                      parked_++ (once, at start)
write jobs_/mode_/mrows_/...                      spin on epoch_ (_mm_pause); after 20 ms sleep on sleep_cv_
begin_batch(n):                                     (sleepers_++ then re-check epoch_)
  done_ = 0
  head_ = epoch+1 << 32 | n << 16 | 0  (release)
  publish(): epoch_++ ; if sleepers_ notify_all   sees new epoch -> seen = epoch_ ; parked_--
host_works: claim(e) and run jobs too              drain(seen): claim(seen) = CAS on head_ that only
wait_done(n): done_ >= n                             succeeds while head_'s epoch == seen and index < n;
wait_parked(): parked_ == n_                         run job ; done_++ (release)
                                                  parked_++ and spin again
```

- **Epoch-tagged claims (issue #29).** `head_` is one 64-bit word `epoch | njobs | index` (`pool.hpp:250`,
  `pool.cpp:27-29`). A worker that wakes late from the sleep can only claim from the batch it woke for, so it can never
  run a job of the next batch twice or bump `done_` after the host reset it (`pool.hpp:19-27`, `pool.cpp:473-482`).
- **Waiting for park as well as done.** `run` waits for every worker to be parked before writing a batch and again
  after `done_ == n`, because a worker can still be inside its drain loop after its last increment
  (`pool.hpp:14-17`).
- **Lost wake-up.** `publish` and the sleeper both use seq_cst on `epoch_` / `sleepers_`, so one of them always sees
  the other (`pool.cpp:410-421`).
- **Each hot atomic on its own cache line** (`alignas(64)`), and no counter is incremented in the spin loop (review
  finding C3, `pool.hpp:231-242`).
- **Stall watchdog.** A wait that sees no progress for `kStall` prints the state, calls
  `strata::core::release_gpu_waits` (so a GPU spinning on this layer's flag is released, #267) and aborts, so the
  server can restart the engine (`pool.cpp:498-540`).
- **The host drains too** (`host_works`, default on): during `run` the host claims jobs through the same `head_`
  instead of spinning on `done_`. Measured 33.7 GB/s with five workers and an idle host core before the change
  (`pool.cpp:743-756`). `--no-host-worker` is the A/B arm.
- With `n_ == 1` (one worker), `run` computes every job inline on the host (`pool.cpp:726-729`) and `run_split` falls back to `run`; the multi-token paths have no such check and still dispatch to the one worker (plus the host when `host_works`).
- `begin_batch` aborts on more than 65,535 jobs (16-bit index field, `pool.cpp:484-488`).

## Row splitting

```
run_split_multi_native(fmt, jobs[0..n))           drain(), mode 5 / 6, per claimed task i
  mtasks = 3 * (workers + host)                    rows [mrows*i/mtasks, mrows*(i+1)/mtasks) over all experts
  mrows = n * fmt.n_ff   -> run_phase(5)          mode 5: native_gu_rows (or q2_rows_any for Q2_0 + SwiGLU)
  host: native_quant_h / act_quant_any per token           -> SplitBufMulti[e].ff[t]
  mrows = n * fmt.n_embd -> run_phase(6)          mode 6: native_down_rows (or q2_rows_any for Q2_0 down)
                                                           -> jobs[e].out[t]
```

A task's row range may span two experts. Modes 5/6 use the **layer's** widths (`NativeFmt::n_ff`, `n_embd`); using
the canonical `FF`/`H` wrote 512 rows past every output row on Qwen3.6 and decoded every verify window with a CPU
miss to token 0 (`pool.cpp:561-567`; regression test `tests/core/pool_native_widths_test.cpp`).

## How the CPU and the GPU overlap

The pool is driven from `src/core/expert_source.cpp`, which is called by the session's layer loop
(`src/core/session.cpp`). Per layer of a single token (`session_loop`, `session.cpp:659-760`; `session_run_token` calls the pool the same way at `session.cpp:946`):

```
GPU stream                               host thread (pinned to the reserved core)
pre[l] graph: mixer, router, top-k   -->  doorbell ring (mapped pinned memory: x_f, ids, weights)
                                          spin on the ring (cudaEventQuery + volatile read)
hits(Launch): VRAM-resident experts  <--  enqueue the GPU's share (does not wait)
   computing on the GPU ...               pool(...) = expert_pool_dispatch: one ActQ for the layer,
                                            an ExpertJob per routed expert that is NOT a hit,
                                            ExpertPool::run / run_split  (all cores, host included)
H2D copy of y_miss  <------------------  the CPU's expert outputs (hit rows written as zeros)
hits(Combine), post[l] graph: moe_combine applies the router weights to GPU + CPU parts
pre[l+1] ...
```

- `expert_pool_dispatch` (`src/core/expert_source.cpp:1881-1990`) builds one job per missed expert sharing one
  `ActQ`, zero-fills rows owned by the GPU or a remote helper, and calls `run_split` (default) or `run`
  (`--no-split-rows`).
- In a speculative verify window, `expert_pool_dispatch_multi` (`expert_source.cpp:2011-2260`) groups the window's
  tokens by distinct missed expert (one `ExpertJobMulti` per expert), quantizes each token's activation once
  (`act_quant_any`, `native_quant_act` or `act_quant_q8_1`) and calls `run_split_multi_native` or `run_split_multi`.
  A share of the misses can instead be read by the GPU over PCIe (`pcie_num`, see `include/strata/core/expert_source.hpp`).
- The host thread is pinned once per session to `physical_cores(false)[0]` - the core the pool keeps free - by
  `SessionLoopScratch` (`src/core/session.cpp:538-548`); measured 36.32 GB/s pinned vs 26.9 GB/s unpinned
  (`pool.hpp:93-101`).
- The pool never applies the router weight: `moe_combine` on the device does (`expert_source.hpp:15-17`).

## Callers and dependencies

- Constructed once in `src/program/generate.cpp:3537` as
  `ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker, o.pool_affinity)`.
- Called from `src/core/expert_source.cpp:1990-1991` and `:2259-2260`.
- Kernels: [q2-expert.md](q2-expert.md) (modes 0-4, Q2_0 in 5/6) and [native-experts.md](native-experts.md) (5/6).
- `strata/core/progress.hpp`: `diag_pool_fn()` registration and `release_gpu_waits()` (header-only).
- `Threads::Threads`.

## Configuration

| Knob | Effect |
|---|---|
| `--pool-workers N` | Worker count (0 = default); the front end passes `--threads N` or a quota-derived default as this (`frontend/main.cpp:685-691`) |
| `--pool-affinity all\|auto\|p-cores` | `PoolAffinity` (`generate.cpp:1400-1409`) |
| `--no-host-worker` | `host_works = false` (A/B arm) |
| `--no-split-rows` | `expert_pool_dispatch` uses `run` instead of `run_split` (`generate.cpp:4015`) |
| `STRATA_POOL_SPIN_US` | Spin time before a parked worker sleeps (test knob; default 20 ms) |
| `--cpu-oracle-q8-0` | Disables the split / multi paths (they fall back to `run`, one token at a time) |

## Tests

| Test | What |
|---|---|
| `pool_test --selftest` | Bitwise vs serial, every job exactly once (sentinel outputs), 200 repeated batches, batch sizes around the worker count; needs `pack/full/experts.bin` |
| `pool_stress 20` | 20 s of tiny batches (0-4 jobs, `run` and `run_split`) with sleeps long enough for the workers to sleep; a 10 s watchdog turns a hang into exit 3 (issue #29) |
| `pool_affinity_test` | Linux: affinity round trips (full, sparse/nested, high CPU ids, invalid targets, failed pin keeps the mask); Windows: CPU Set round trips |
| `tests/core/pool_native_widths_test.cpp` | `run_split_multi_native` at 2048 x 512 against direct kernel calls, with guard rows |

Run conditions and targets: [tests.md](tests.md).

## Gotchas

- `pauses()` always returns 0: the counter was deleted because incrementing it in the spin loop caused the contention
  it measured (`pool.hpp:176-179`).
- The pool reserves the first physical core only by convention: if the caller does not pin the host thread there, the
  reservation "is a fiction" (`pool.hpp:93-97`).
- On Linux without sysfs topology every allowed CPU counts as a core, including SMT siblings (`pool.cpp:155-158`).
- `run_split_multi` with more than 96 experts, or with the oracle contract on, falls back to single-token
  `run_split` in chunks of 16 (`pool.cpp:656-670`).
- A worker that went to sleep counts as parked; this is exactly why claims carry the epoch.

## See also

- [q2-expert.md](q2-expert.md), [native-experts.md](native-experts.md), [tests.md](tests.md)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) ("How it works": "The CPU computes the experts that are not on the GPU in place, at the same time as the GPU works on the cached ones")
