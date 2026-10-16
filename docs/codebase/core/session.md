[Codebase docs](../README.md) › [Core](README.md) › Session

# The session: per-sequence state and the decode drivers

## Purpose

A `SessionState` is everything one sequence needs that is not a weight: the residual stack, every GDN layer's
recurrent and conv state, every attention layer's KV cache and indexer state, the shared scratch buffers, the
doorbell pointer and the PLE history. All of it is carved from **one** device allocation sized up front, so the token
path never allocates. `session.cpp` also contains the three ways a decode token is driven through the layers.

Files covered: [include/strata/core/session.hpp](../../../include/strata/core/session.hpp),
[src/core/session.cpp](../../../src/core/session.cpp); the KV-state parts of
[include/strata/core/layer.hpp](../../../include/strata/core/layer.hpp) / [src/core/layer.cpp](../../../src/core/layer.cpp)
(`QsaState`, `qsa_state_*`, `qsa_set_kv_*`).

## `struct SessionState` (`session.hpp:31`)

| Field | Meaning |
| --- | --- |
| `max_cells` | context capacity the session was sized for (`--max-context`, possibly lowered by the VRAM plan) |
| `gdn`, `gdn_state` | shared GDN scratch; `gdn_state` holds `gdn_alloc` rows of `S*h_v*S + conv_channels*(d_conv-1)` floats |
| `qsa_states`, `qsa_state_arena` | one `QsaState` per attention layer (indexed by global attention ordinal) |
| `qsa_bufs` | attention scratch shared by all attention layers |
| `moe`, `block` (and their arenas) | MoE and block buffers; `R` aliases `block.R` (`hc x n_embd`) |
| `k` | experts per token (the file's `expert_used_count` in practice) |
| `layer_lo`, `layer_hi`, `qsa_ord0`, `qsa_alloc`, `gdn_ord0`, `gdn_alloc` | the layer range this session owns (multi-GPU split). Arrays keep global ordinal indexing; a range with no attention layer still allocates one "primary" state because the RoPE table, the prefill staging and the MTP drafter borrow it (`qsa_primary()`) |
| `db` | the `Doorbell` the host loop polls (set by the caller) |
| `ple_prev`, `ple_token`, `ple_hist`, `ple` | the PLE's token window, history rows and the whole `PleRun` (Flash-Next only) |

## Sizing and lifecycle

| Function | Line | What it does |
| --- | --- | --- |
| `session_bytes(g, max_cells, k, lo, hi)` | `session.cpp:56` | Pure arithmetic: GDN scratch + `gdn_n` states + one attention state with the RoPE table + `(q_n - 1)` without it + attention scratch + MoE + block + PLE history. The layer-split search uses it to price a placement |
| `session_init(...)` | `session.cpp:77` | Carves the arena in the same order; the first allocated attention state builds the RoPE table and the others borrow it (`share_rope`). Returns 0 if a state could not be built (e.g. no pinned RAM for KV streaming's host copy) |
| `session_release(s)` | `session.cpp:133` | Releases the RoPE table registration before the memory is freed (#280) |
| `session_zero(s, g, R_init, stream)` | `session.cpp:141` | `R` = `hc` copies of `R_init` (the token embedding) or zero; GDN state, KV and indexer state, PLE history zeroed; PLE window reset to -1 |

Callers: `src/program/generate.cpp` sizes and inits the main session (`session_bytes` / `session_init` around lines
2868-3306, including per-stage sessions for a layer split and per-slot sessions for batch windows).

## KV cache formats (`QsaState`, `layer.hpp:206`)

The format is a process-wide setting chosen before sizing (`qsa_set_kv_int8`, `qsa_set_kv_q4`,
`qsa_set_kv_hybrid`, `qsa_set_kv_int8_rotate`; `layer.cpp:177-184`) and copied into each state by
`qsa_state_init` (`layer.cpp:591`).

| `--kv` | State fields | Bytes per cell (2 KV heads x 256, per attention layer) | Notes |
| --- | --- | --- | --- |
| `fp16` (default) | `k_pool`, `v_pool` | 2,048 | |
| `int8` | `k_q`, `v_q`, `k_scale`, `v_scale` (fp16 scale per 64 values) | 1,056 | `STRATA_KV_ROT=1` adds the Hadamard rotation (`kv_rot`) |
| `q4_0` (alias `q4`) | `k_q4`, `v_q4` | 576 | always rotated (Walsh-Hadamard, `kv_q4.hpp`) |
| `k8v4` | `k_q` / `k_scale` + `v_q4` | 816 | INT8 K (unrotated, exact scores) + rotated Q4_0 V; not with `--kv-resident`; the MTP drafter's state stays plain INT8 |

Rotated formats rotate K and V before storing, rotate the query before attention and rotate the output back
(the transform is self-inverse). The byte figures are those of the `--kv` help text and the VRAM plan's
`kv_elem` (`generate.cpp`: 18/32, 1+2/64, 816/1024 or 2 bytes per element).

Other state: `page_table` (logical page -> physical slot), indexer `idx_tail` / `idx_dead` / `idx_pooled` /
`idx_block_pos`, the RoPE `cos_tab` / `sin_tab` (`max_cells x n_rot/2`, built on the host in float64; 64 MiB at
262K, shared), the per-token `step` / `pos_dev` device buffers and their pinned host sources `host_step` /
`host_pos`.

## KV streaming (`kv_mode`)

`kv_plan` (`layer.cpp:529`) decides per state:

| `kv_mode` | When | Behaviour |
| --- | --- | --- |
| 0 | default, or context fits | every page in VRAM, identity page table |
| 1 | `--kv-resident N` with `N` (at least `qsa_kv_resident_min()` = 20,480) below the context | authoritative K/V in pinned host memory (`host`), `n_slots` pages in VRAM; `qsa_kv_resolve` makes the blocks the selection names resident before attention reads them |
| 2 | the MTP drafter's state when its window is shorter than the context | a ring of `n_slots` pages at `block % n_slots`, with a host copy for resume (`MtpDrafter::kv_restore`) |

KV streaming applies only with an indexer: `qsa_kv_streaming_applies` refuses it for Qwen3.6 because dense
attention reads every cell every token (generate.cpp checks before loading; `qsa_state_init` refuses as a backstop).
`STRATA_KV_RING_OFF` / `STRATA_KV_MAIN_OFF` disable the ring / main streaming for A/B.

## The three decode drivers

### 1. Direct launches: `session_token` (`session.hpp:121`, `session.cpp:778`)

Layers in order, each a `block_layer` with `parts` supplied by the caller before the call; no graphs.
`sync_every_layer` (`--sync-every-layer`) synchronises after each layer to localise asynchronous errors. Used under
`--no-capture`.

### 2. Per-layer graphs + host loop: `session_capture` + `session_loop`

`session_capture` (`session.cpp:188`) captures two graphs per layer, `execs[l]` (= `block_layer_pre`) and
`posts[l]` (= `block_layer_post`), each on a fresh stream, baking in the `parts` device address. With `split`
(`--gpu-stages`) it also captures `preA`, `preB` and five prefix graphs for measurement.

`session_loop` (`session.cpp:565`) per token:

```
stage_token (host_step/host_pos for every attention layer)   doorbell_reset
H2D zero parts; launch pre[0]; record probe event
for l in layers:
    spin: cudaEventQuery(probe) + read h_seq until the ring for l is seen   (counts rings_mid_graph, ms_to_ring)
    hits(Launch, ids, k)       - VRAM-resident experts start on the GPU
    pool(x_f, ids, weights)    - CPU experts into pinned y_miss
    H2D y_miss -> parts;  hits(Combine)
    launch post[l];  [dump R];  [!overlap: sync]
    launch pre[l+1]; record probe                                            (ms_host)
final cudaStreamSynchronize
```

The per-iteration `cudaEventQuery` is required: the code comment records three experiments showing the device's
write to mapped memory became host-visible only when the driver was entered. `SessionLoopScratch`
(`session.hpp:369`) holds the mapped-pinned `y_miss`, the probe event and the host-thread pin, allocated once per
session (passing `nullptr` makes the loop allocate per call, which also costs a device sync).

`PoolFn` (`session.hpp:314`) and `HitFn` (`include/strata/core/hit_hook.hpp`) are the expert side's callbacks; the
`Launch` / `Combine` ordering is documented in `hit_hook.hpp` (see [experts/](experts/README.md)).

### 3. Whole-token graph: `session_capture_token` + `session_run_token`

`session_capture_token` (`session.cpp:826`) captures **all layers into one graph**. Between `pre[l]` and `post[l]`
it inserts, on the device: optionally the VRAM-hit path from `TokenHits` (`moe_hit_select` over a static
`n_layers x n_expert` residency table, `quantize_q8_0_scaled`, `moe_hit_grouped_s2_dev`), then `doorbell_wait` (a
one-thread kernel spinning until `d_flag` reaches the ring), `copy_from_mapped` of `y_miss` into `parts` (a kernel,
not a memcpy node, to avoid splitting WDDM submissions), `moe_hit_add`, and `post[l]`.

`session_run_token` (`session.cpp:900`) stages the token, launches the graph once, flushes once with
`cudaStreamQuery`, and then for each layer spins on `h_seq`, runs the pool into `y_miss`, fences and writes
`h_flag`. A slow ring triggers another `cudaStreamQuery` every `STRATA_TG_FLUSH_US` microseconds (default 2000) to
notice a fault or a finished graph; 20 s without a ring is an error.

generate.cpp (around lines 4602-4690 and 8068-8085) captures the token graph when none of these holds:
`--no-capture`, `--no-token-graph`, layer or half dumps, a VRAM expert tier whose residency is not static
(a cache filled on demand rather than from an expert profile, so `TokenHits` is incomplete), a native pack, or a
multi-GPU layer split ("a split's token graph cannot span stages"). Otherwise the plain decode loop uses
`session_loop`, and `--no-capture` selects `session_token`. (Speculative decode uses neither; see
[verify.md](verify.md).)

### Measurement replays

`session_replay` (pre graphs only), `session_replay_full` (pre + post, no pool: the GPU floor),
`session_replay_stages*`, `session_replay_stage_prefixes`, `session_replay_stage_sweep` exist for the stage
breakdowns described in the header; they are not on the serving path.

## Configuration summary

| Flag / variable | Effect |
| --- | --- |
| `--max-context N` | `max_cells` (default 4096; may be lowered by the VRAM plan) |
| `--kv fp16|int8|q4_0|k8v4` | KV format |
| `--kv-resident N` | KV streaming (Flash-Next only) |
| `STRATA_KV_ROT=1` | rotated INT8 |
| `--no-capture`, `--sync-every-layer`, `--no-token-graph`, `--gpu-stages`, `--dump-layers` | driver choice and debug |
| `STRATA_TG_FLUSH_US` | token-graph flush interval |

## Model-specific behaviour

* Qwen3.6: no PLE history is carved (`ple_hist_bytes` returns 0 when `!has_ple`); `R` is one stream; dense attention
  carves no cell scores, ids or gather scratch (`qsa_buffers_bytes`), and KV streaming is refused.
* Flash-Next: PLE run and history, 4-stream `R`, sparse QSA buffers.

## Tests

* `qwen36_verify_test` and `qwen36_runner_check` drive sessions at `hc = 1` (see [verify.md](verify.md)).
* `CMakeLists.txt` registers `graph_registry_test` and `pinned_capture` (doorbell capture) from `bench/micro/`, but
  only `if(EXISTS ...)`: the `bench/` directory is not present in this checkout, so they are not built here.

## Gotchas

* A captured graph re-reads `host_step` / `host_pos` at replay but does not recompute them: `stage_token` must run
  before every replay. The header records a bug where `session_loop` skipped it and every token after the first
  was rotated at position 0.
* `session_capture` is idempotent per `SessionGraphs`; graphs bake the session's pointers, so a new session needs a
  new capture.
* The non-primary attention states borrow the primary's RoPE table: release order matters (`session_release`).

## See also

* [layer-forward.md](layer-forward.md), [cuda-graphs.md](cuda-graphs.md), [verify.md](verify.md), [vram-plan.md](vram-plan.md).
* [conversation/](conversation/README.md) - snapshots and caches of this state.
