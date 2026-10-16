[Codebase docs](../README.md) › [Core](README.md) › Verify window

# The speculative verify window (`Verifier`)

## Purpose

Speculative decoding checks several drafted tokens in one pass. A `Verifier` runs **T tokens at consecutive positions**
(the last accepted token and T-1 drafts; T = 2..8, `kVerifyMaxT = 8` in `include/strata/kernels/verify_kernels.hpp`)
through every layer in **one captured graph** and produces, for each row t, what plain greedy decode would produce
after token t - bit for bit. Every kernel in the window is either the single-token kernel applied per row or a
multi-token kernel whose per-row arithmetic is the single-token kernel's (multi-column MMVQ in exact mode, T-token GDN
kernels, the per-token hit activation, multi-token CPU expert rows). A draft is therefore accepted exactly when greedy
decode would have produced it, and the output does not depend on the drafts.

The window reads the dense weights once for T tokens; the CPU experts cost the union of the T tokens' misses (the
header quotes 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).

Files covered: [include/strata/core/verify.hpp](../../../include/strata/core/verify.hpp),
[src/core/verify.cpp](../../../src/core/verify.cpp).

## Lifecycle

| Call | Line | What it does |
| --- | --- | --- |
| `verify_geometry_supported(g, k, why)` | `verify.hpp:50`, `verify.cpp:122` | Refuses geometries the window cannot reproduce: at `hc > 1` the fused GR read's shape (2560 / 4 / 320), `k == g.n_expert_used`, `1 <= k <= n_expert`, GDN state 128 and conv 4 |
| `init(wt, g, ss, hits, head, max_t, err)` | `verify.cpp:319` | Binds the weight table, geometry and session; requires `2 <= max_t <= 8` and a **profile-filled VRAM expert tier** (`VerifyHits`: device residency table, cache base, blob size). Allocates the window's device arena, mapped staging, streams and events. Detects whether every expert of its layer range is resident (`all_resident_`: a "zero-doorbell" window) |
| `run(T, tokens, pos0, pool, user, out, err)` | `verify.cpp:1498` | Captures `exec_[T]` and `commit_exec_` on first use, stages tokens/steps/positions into mapped memory, issues PLE row reads, launches the window graph and serves every layer's CPU experts on the host (below), synchronises, optionally samples on the host side, and writes `out[t]` |
| `commit(n_keep, err)` | `verify.cpp:1807` | Makes the first `n_keep` rows permanent by launching the commit graph; advances `ss.ple_prev`. Asynchronous on a single GPU (`set_commit_async`) |
| `wait_commit(err)` | `verify.cpp:1838` | Waits for a pending async commit (event, not device sync) before anyone else touches the session |

### State discipline

The window appends K/V and indexer keys for all T positions but **leaves the GDN state untouched**. `commit(n_keep)`
then: advances the GDN conv history and recurrent state by replaying the accepted tokens from inputs the window
stored (`qkv_L_`, `h_L_`, `gate_L_`, `beta_L_` per GDN layer), restores the indexer's key tail from a snapshot and
re-appends the accepted keys, and sets the PLE history to its snapshot after token `n_keep - 1`. K/V cells beyond the
accepted prefix are simply overwritten later, before any query can read them.

## What the window graph does per layer (`record_window`, `verify.cpp:563`)

```
inputs: tokens/steps/positions from mapped staging; embeddings (NativeEmbed, canonical gather, or the
        previous stage's hand-off) broadcast to hc streams
for l in [lb, le):
  pre(l, group):  [l==1: PLE rows]  read/norm (GR fused read, or plain norm over T rows)
                  GDN: T-token conv/recurrence kernels  |  attention: per-row RoPE, K/V append, selection, attention
                  residual write; read/norm; router over the T rows (moe_route_window); DOORBELL (x, ids, weights)
                  shared expert (on a forked stream, STRATA_SH_STREAM)
  host:           pool(x_f, ids, n_tok, k, y_miss, layer) + the GPU plan the pool publishes (flag A)
  post(l, group): VRAM-resident experts (grouped kernels: native or S2), PCIe share once its DMA landed (flag B),
                  wait for the CPU misses (flag), add, combine with the shared expert, residual write
head: lm_head_mix over T rows -> NativeHead multi-column MMVQ (or canonical) -> argmax per row
```

* `set_split(true)` (`--spec-split`) splits the window into two token groups so the CPU experts of one group overlap
  the GPU work of the other (exact, documented as slower in the overlap study).
* `set_pcie_mode(0|1|2)` (`--pcie-mode auto|dma|kernel|direct`; generate.cpp maps `dma` to 0, `direct` to 1, anything else to 2): how the PCIe share of the misses reaches the GPU - DMA into
  VRAM staging, the grouped kernel reading the mapped arena directly, or a copy kernel inside the graph.
* `plan_sink()` returns the `GpuPlanSink` the expert dispatch writes each layer's GPU plan into; the window copies it
  from mapped memory after flag A. (The dispatch itself is in [experts/](experts/README.md).)
* With `all_resident_` every layer plans on the device and writes straight into `parts_` with no host round trip;
  `STRATA_VERIFY_DEVICE_PLAN=1` does the same per layer when only some layers are fully resident.

### Host side of `run`

For each (layer, group) the host spins on the ring in `h_seq_` (with a `cudaStreamQuery` every 2 ms to notice a
finished or faulted graph; 20 s is a timeout), calls the pool for that group's rows, publishes an empty GPU plan if the
pool did not, gathers the PLE rows at layer 0, and raises the flag. After the last layer it synchronises the stream
(a blocking sync on purpose: polling cost IQ3_S ~3% decode). On a timeout, `release_gpu_waits` raises all flags to
`UINT32_MAX` so no spin kernel outlives the engine, and the verifier then refuses further windows (#267).

### Sampling

The graph records the greedy argmax. A sampled or penalised request is sampled again **outside** the graph
(`sample_tokens` with `counter = pos0`, so row t draws `Philox(seed, pos0 + t)`), because a captured sampler would
replay the same parameters forever (`set_sampling`, `set_history` for penalty rows, `set_head_sampling(false)` for
prompt windows whose picks are discarded).

## Batch windows (several sequences)

`init_slots(slots)` binds S independent sessions (one per slot, carved like the main one). `run_slots` /
`run_slot_rows` run one token for each of S slots in one window - each row reads and writes its own slot's GDN,
KV, indexer and PLE state while the dense weights are read once. Greedy only, no drafts. `commit_slots` keeps
every row. Graphs are cached per row set (`exec_bm_`, `commit_bm_`, `batch_key`). `set_slot_sampling` gives a slot a
sampled draw on the last stage. CLI: `--batch N` / `--slots N` (serve, 2..8).

## Layer split (multi-GPU)

`set_stage(lb, le, handoff_in, handoff_out)` makes this verifier run layers `[lb, le)`; an earlier stage writes per
row the residual, the pending write and the injection (`handoff_floats` = `hc*n_embd + n_embd + hc`) to mapped
memory, and the next stage folds that write into its first read, so the split is bit-exact. `set_next` chains
stages; `run` / `commit` continue into the next stage. `batch_launch` / `batch_poll` pipeline batch windows across
stages (stage k runs group g while stage k+1 runs group g-1, `--batch-groups`). The verifier switches to its own
device with `OnDevice` ([device-and-memory.md](device-and-memory.md)).

## Callers

`src/program/generate.cpp`: `ver.init` (~5482, ~8267), `ver.run` + `mtp.draft` in the speculative loops (~7083,
~7511, ~8481), `set_stage` / `set_next` / `init_slots` for splits and batches (~5240-5516),
`Verifier::set_commit_async(!multi_gpu)` (~2894). A prompt can also be read through windows, with
`set_head_sampling(false)` since every row is committed.

## Configuration

| Flag / variable | Effect |
| --- | --- |
| `--spec T` | window size (T >= 2 enables speculation; MTP needs it) |
| `--spec-split` / `--no-spec-split` | two token groups per window |
| `--pcie-mode`, `--pcie-frac` | PCIe share of the misses and how it reaches the GPU |
| `--batch N`, `--batch-groups G` | batch windows; pipelined groups across a split |
| `STRATA_COMMIT_SYNC=1` | keep commit synchronous |
| `STRATA_DEC_BATCH=0` | per-token launches inside the window instead of row-batched ones |
| `STRATA_PLE_BATCH=0`, `STRATA_SH_STREAM=0`, `STRATA_FUSE_HEAD_GR=1` | A/B switches for the PLE projections, the forked shared-expert stream, and fusing the head's GR read |
| `STRATA_VERIFY_ALL_RESIDENT=0`, `STRATA_VERIFY_DEVICE_PLAN=1` | zero-doorbell window off; device planning on |
| `STRATA_VERIFY_PROFILE`, `STRATA_VERIFY_TRACE`, `STRATA_VERIFY_NODES`, `STRATA_VERIFY_DEBUG`, `STRATA_DBG_NAN`, `STRATA_LOGPOS_TOPK` | diagnostics (`profile_report`, `trace_dump`, node listing, `window_logprobs`) |
| `STRATA_VERIFY_COHERENT`, `STRATA_DOORBELL_STORE` | HIP-only A/Bs for gfx1030 timeouts (#649) |
| `STRATA_TEST_VERIFY_STALL=N` | test hook: withhold the last flag of window N |

## Model-specific behaviour

* `hc = 1` (Qwen3.6): the plain norms over T rows (`plain_norm_rows`), no pending write, `output_norm` head; a wired
  PLE is refused. Dense attention per row. `k` comes from the file (8).
* `hc = 4` (Flash-Next): fused GR read with the folded write, PLE at layer 1, sparse selection with indexer tail
  snapshots in commit.
* `layer_verify_compatible(g, why)` ([layer-forward.md](layer-forward.md)) must hold for the window to equal the
  single-token path.

## Tests

* `qwen36_verify_test` (`tests/cuda/qwen36_verify_test.cpp`; CTest with the `qwen36_runner_fixtures` fixture from
  `tests/python/qwen36_runner_fixture.py`, needs CUDA and numpy, skip 77): an 8-row window against the oracle's
  per-layer residuals and logits, the same rows from eight one-token windows bit for bit, two windows with a commit
  between them, on a native pack of the synthetic model.
* `qwen36_runner_check` (`tests/python/qwen36_runner_check.py`): the `strata` binary's streaming runner teacher-forced
  through windows.

## Gotchas

* `run` refuses a window that would pass `max_cells`.
* After an async commit, anything that reads the session from another stream or the host must call `wait_commit`
  first (a new request, a snapshot, the prompt path).
* A verifier that released its GPU waits is unusable until the engine restarts.

## See also

* [mtp-draft-head.md](mtp-draft-head.md), [session.md](session.md), [cuda-graphs.md](cuda-graphs.md), [experts/](experts/README.md).
* [docs/qwen36-port/step4-runner.md](../../qwen36-port/step4-runner.md), [step3-integration.md](../../qwen36-port/step3-integration.md).
