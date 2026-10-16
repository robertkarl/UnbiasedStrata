[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Sampling and verify helpers

# Sampling, and the speculative-verify / MTP helper kernels

## Purpose

- The sampler: logits to one token id per row, with llama.cpp's default chain and order
  `penalties -> top_k -> top_p -> min_p -> temperature -> pick`, greedy argmax, and the coupled draft sampling used
  by the MTP drafter.
- The small kernels the speculative verify window and the MTP draft layer need so that a whole window can be one
  captured graph with no host round-trips: flag waits, resident expert plans, gathers, id maps, step records and
  timestamps. The GDN verify/commit kernels from the same file are on [gdn.md](gdn.md).

Files: `src/kernels/cuda/sampler.cu`, `include/strata/kernels/sampler.hpp`; `src/kernels/cuda/verify_kernels.cu`
(non-GDN part), `include/strata/kernels/verify_kernels.hpp`.

## Sampler (`sampler.cu`)

### Parameters

`SamplerParams` (`sampler.hpp:15-28`): `top_k` (default 20; the sampled path keeps 1..64 as given, and 0 or more
than 64 keep 64, per `sampled_k`, `sampler.cu:423`), `top_p` 0.95, `min_p` 0, `temperature` 1 (<= 0 means greedy),
`min_keep` 1, `penalty_last_n`, `penalty_repeat`, `penalty_freq`, `penalty_present`, `seed`, `counter` (the absolute
draw index of row 0), `greedy`.

### Semantics

- Penalties are transcribed from `llama_sampler_penalties_apply` (`sampler.cu:62-79`). The repeat penalty multiplies
  non-positive logits and divides positive ones. Frequency is `count * freq`. Presence is boolean. Only the last
  `penalty_last_n` ids of each row's history count, through a vocabulary bitmap in shared memory.
- Order: one penalties stage, first; temperature after the truncation filters; top_p before min_p (issue #53,
  `sampler.cu:5-11`).
- The draw uses Philox 4x32-10, counter-based on (seed, counter + row), so a batch can be sampled in any order and a
  run is reproducible (`sampler.cu:38-60`). The pick walks the final distribution's cumulative sum in double.
- Ties: the selection order is (value desc, id asc). NaN and -inf are never picked.

### Kernels and dispatch (`sample_tokens`, `sampler.cu:977`)

| Path | Kernel | When |
| --- | --- | --- |
| Greedy, cluster | `sampler_greedy_cluster_kernel` (`:202`), clusters of `kAmCtas` = 8 CTAs x 1,024 threads per row, `cudaLaunchKernelEx`, DSMEM | greedy and no penalty window, on sm_90+ with code built for it (`sample_greedy_cluster`, `:928`, checks `ptxVersion >= 90` and occupancy); `STRATA_ARGMAX_MULTI=0` disables |
| Greedy | `sampler_greedy_kernel` (`:102`), one block of 1,024 threads per row | otherwise for greedy / temperature <= 0. The lowest index of the largest value; 0 when everything is -inf / NaN |
| Sampled, split (default) | `sampler_split_part_kernel` (`:661`), grid `(ceil(V / 4096), rows)`, 128 threads: each warp holds 1,024 logits in registers and keeps its own top-k; then `sampler_split_merge_kernel` (`:744`), one warp per row, merges and runs `sampled_tail_warp` | not under stream capture, vocabulary <= 64 x 4,096 = 262,144, rows <= 64, scratch available (`split_scratch`, `:881`, a per-(device, stream) buffer grown with `cudaMalloc`, never on a captured stream) |
| Sampled, one block | `sampler_one_block_kernel` (`:526`), 1,024 threads per row, `top_k` block-argmax rounds | `STRATA_SAMPLER_ONE_BLOCK=1`, and the fallback when the split cannot run |
| Sampled, old | `sampler_kernel` (`:279`) | `STRATA_OLD_SAMPLER=1`; engine 0.1.20's kernel, kept as the reference |

The three sampled paths pick the same token bit for bit (`sampler.cu:13-21`). The tail (`sampled_tail_warp`,
`:456`) shares the double `exp`s across the warp's lanes and keeps lane 0's ordered sums, which reproduces the old
chain's doubles exactly. With a null stream `sample_tokens` synchronises.

### Coupled draft sampling (`STRATA_SPEC_COUPLED=1`, `include/strata/core/coupled_draft.hpp`)

`coupled_draft_scratch_bytes`, `coupled_draft_stage` (copy a round's params and the penalty-history ring from mapped
memory), `coupled_draft_sample` (draft j's token from one row of logits with the target's chain, Philox counter
coupled to the target's, optional draft-head subset maps `sub_to_id` / `id_to_sub`; writes the id, its probability
under the final distribution, and the ring entry), `sampler.cu:1048-1081`. All per-round values come from device
memory, so the calls are capturable.

### `penalty_rows` (`sampler.hpp:73-84`)

A host helper. Row t of a verify window's penalty history is the last h tokens of the consumed tail followed by
`window[0..t]`, so drafting cannot change which tokens are penalised.

## Verify-window and MTP helpers (`verify_kernels.cu`)

Window size `kVerifyMaxT` = 8 (`verify_kernels.hpp:23`).

| Function | Launch | What |
| --- | --- | --- |
| `wait_flag_ge` (`:643`) | 1 thread | spin (`strata_spin_pause`) until a mapped host flag >= a capture-time value; several rings can be outstanding (the split verify window keeps two) |
| `wait_flag_ge_or`, `copy_i32_from_mapped_unless`, `copy_or_zero_from_mapped` (`:623`, `:627`, `:633`) | 1 / 128 / 256 threads | the same, short-circuited when a device `*skip` equals the ring value: a layer whose routed experts are all resident needs nothing from the host (perf-review E-6) |
| `resident_plan` (`:616`) | one block of `kResidentPlanMax` threads | builds the host pool's grouping plan on the device when every routed expert of the window is VRAM-resident, and sets `*skip` |
| `fetch_blobs` (`:365`), `rebase_ptrs` (`:372`) | 384 x 256, 1 x 128 | stage `*n` expert blobs from mapped host memory into VRAM with 16-byte loads (a layer's PCIe share), then point the group table at the staged copies |
| `gpu_stamp` (`:681`) | 1 thread | `%globaltimer` (ns) into a buffer: the verify window's stage profiler |
| `embedding_gather_dev` (`:648`) | grid `(n / 256, T)` | S2/S4/S8 embedding rows for token ids read from device memory; bitwise `embedding_gather` |
| `broadcast_streams` (`:657`), `add_streams_broadcast` (`:377`) | | `R[t][c] = x[t]` across the hc streams; `R = h + e` (the MTP layer's embedding branch) |
| `copy_indexed` (`:663`), `gather_rows` (`:396`), `map_ids` (`:407`) | | device-indexed copies and gathers (`uint4` / `uint32` / byte by row alignment); subset id back to token id |
| `ident_hits` (`:383`) | 1 x 1,024 | every routed expert resident (the MTP layer): `slot = id`, `dst = i` |
| `mtp_select` (`:389`) | 1 x 256 | the draft chain's next input: the chosen row's residual and token (`out` may be mapped memory), optional probability |
| `row_top_prob` (`:412`) | 1,024 threads per row | `softmax(logits)[id]` for each row's argmax |
| `dense_steps` (`:436`) | 1 x 64 | `QsaStep` records `[cell, cell+1, (cell+1)/4, cell+1]` for dense attention |
| `window_ids` (`:431`) | grid `(8, n)` | a sliding window over the last `window` cells (the MTP drafter's ring): ids and width rewritten in each step record |

All of them read per-window values from device memory, so they can be captured.

## Control and data flow

```
verify window (src/core/verify.cpp): per layer ... wait_flag_ge[_or] (CPU experts ready) -> copy_from_mapped / copy_or_zero
                                      resident_plan -> fetch_blobs -> rebase_ptrs -> native_expert_grouped
                                      ... final logits -> sample_tokens (T rows, penalty_rows histories)
MTP drafter (src/core/mtp.cpp):      add_streams_broadcast -> layer -> row_top_prob / sample_tokens / coupled_draft_sample
                                      -> mtp_select -> next draft
```

Callers: `src/core/verify.cpp` (`sample_tokens` at `:1274`, `:1662`, `:2167`), `src/core/mtp.cpp` (`:781`, `:994`),
`src/program/generate.cpp:8173` (plain decode).

## Configuration

- Request parameters map to `SamplerParams` (temperature, top_k, top_p, min_p, penalties, seed); see the
  [front end](../../frontend/README.md).
- Environment: `STRATA_OLD_SAMPLER`, `STRATA_SAMPLER_ONE_BLOCK`, `STRATA_ARGMAX_MULTI` (default on),
  `STRATA_SPEC_COUPLED` (read in `include/strata/core/coupled_draft.hpp:54`), `STRATA_EMULATE_CC`.

## Model-specific behaviour

The sampler is model-agnostic. The split path covers vocabularies up to 262,144; the comment at `sampler.cu:1006` sizes
it for 248,320 logits (61 split blocks), Qwen3.6's vocabulary (`docs/QWEN36-35B-A3B.md` section 1). `broadcast_streams` / `add_streams_broadcast` take `hc` (4 for Flash-Next, 1 for Qwen3.6).
`dense_steps` exists for dense attention (Qwen3.6, and dense-shaped rings).

## Tests

- `sampler_parity` in three CTest entries: `sampler_parity` (split), `sampler_parity_one_block`,
  `sampler_parity_old`, each with the matching `STRATA_OLD_SAMPLER` / `STRATA_SAMPLER_ONE_BLOCK` environment
  (`CMakeLists.txt:823-831`). They compare at the distribution level against an independently computed
  distribution.
- `decode_cluster_parity`: the cluster argmax against the one-block kernel, bitwise, also inside a captured graph;
  skips below sm_90; `--bench` for timings.
- The verify helpers are exercised by `gdn_parity`, `shared_expert_parity`, `qwen36_seam_parity` and the
  `tests/cuda/qwen36_verify_test.cpp` / `qwen36_mtp_test.cpp` integration tests.

## Gotchas

- `logits` and `out` are device pointers. Passing host memory faults later at an unrelated synchronising call
  (`sampler.hpp:32-37`).
- `penalty_last_n > 0` without a history exits (`sampler.cu:979-984`).
- The split path allocates scratch on first use per (device, stream). On a capturing stream the code always takes
  the one-block kernel instead (`stream_capturing`, `sampler.cu:863`).

## See also

- [gdn.md](gdn.md) (verify/commit of the recurrent state), [moe-and-experts.md](moe-and-experts.md)
  (`native_expert_grouped`, `moe_group_resident`), [attention-qsa.md](attention-qsa.md) (`QsaStep`)
- `docs/qwen36-port/step4-drafter.md`, `step4-runner.md`
