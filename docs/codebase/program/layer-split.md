[Codebase docs](../README.md) › [Program](README.md) › Multi-GPU layer split

# Multi-GPU layer split (`--layer-split`)

**Purpose:** run consecutive layer ranges on different GPUs, each with its own dense weights, session, expert cache,
verifier and prompt path, handing the residual from stage to stage once per verify window.
**Covers:** `src/program/generate.cpp` - `SplitDrive` / `drive_pool_split` (`:846-869`), `GpuStage` (`:871-892`),
option parsing (`:1600-1768`), `--split-skip-if-fits` (`:2848-2893`), per-stage weights (`:2894-2981`), the `auto`
placement search (`:3031-3204`), per-stage sessions (`:3244-3267`), stage caches (`:3845-3899`), and the serve-side
wiring (`:5008-5023`, `:5232-5394`, `:5416-5509`). **Serve mode only** (`--layer-split` without `--serve` is refused).

## Concepts

- `split_at` holds the **first layer of each later stage** (`"24,36,42"` for four GPUs - not a per-card count). Stage
  0 is CUDA0 and runs `[0, split_at[0])`; stage `i` runs `[split_at[i-1], split_at[i])`; the last runs to `n_layers`
  and also holds the head and the MTP drafter.
- `split_devs` are the later stages' devices (`--split-device`, default the next visible GPUs). `--split-device 0` with
  one K puts both stages on CUDA0 sharing weights, session and cache (`split_same`): the bit-exact A/B of the
  hand-off, which uses a second `Verifier ver_same` (`:5419-5452`). `multi_gpu` is true only for a real multi-device
  split.
- `GpuStage` owns per device: `WeightTable wt`, `NativeDense dense`, `NativeHead head` (last stage), `SessionState ss`
  carved to its range, streams, `ExpertCache cache`, its share of the profile, `d_res`, `Verifier ver`, `Prefill sp`,
  adaptive-swap stream/event, and its M-RoPE table.
- **The carve:** `session_bytes(g, max_context, K, lb, le)` prices only a stage's own layers' GDN rows and QSA pools;
  sessions are allocated after the placement search so the search can price them (`:3117-3125`, `:3206-3267`).

## Validation (`:1611-1768`)

Rising K from 2, one distinct device per K in range 1..n-1 (or `0` with one K), fewer than `SplitDrive::kMax` (8)
stages. `auto` with one visible GPU clears the split. Conversation parking with `split_same` is refused. A layer split
across GPUs needs `--expert-profile`; helper caches (`--expert-cache-device*`) use GPUs no stage runs on; HIP checks
each device's architecture. `--resident-cpu-experts` is not supported (setup's soft `--resident-experts` is
downgraded to plain mmap with a warning). The comment at `:1600-1604` still lists control vectors, helper caches and
prompt-path lending as "not yet" across GPUs, but later code replicates control vectors per stage (`cvec_replicate`,
`:2951`), places helper caches on GPUs no stage uses (`:1753-1763`) and lends per stage from each stage's own cache;
treat that comment as outdated.

## Own prompt buffers vs borrowing (`STRATA_SPLIT_OWN`, `:1695-1748`)

Default: each stage's prompt path borrows the tail of **its own** cache (see [expert-tiers.md](expert-tiers.md)).
`STRATA_SPLIT_OWN=1` keeps 0.1.29-style own buffers on every stage; `STRATA_SPLIT_OWN=auto` keeps them only when
`160 + chunk x 680 / 1024 + 96 x 4` MiB is at most 12% of every card's VRAM. Choosing own buffers sets
`no_prefill_borrow` and `split_own_auto`. During the search those buffers are ignored unless
`STRATA_SPLIT_OWN_PLACE=reserve`.

## Per-stage weights (`:2894-2981`)

For each later stage: `RemoteExperts::preflight` creates its context, then `cudaMalloc` + `WeightTable::load` (of the
whole pack, or with `--trim-stage-weights` / `STRATA_STAGE_TRIM=1` only its layers' dense tensors -
`add_foreign` + `NativeDense::set_layer_range`), `NativeDense::load`, two streams and an event,
`cvec_replicate`, its M-RoPE table under `--vision`, and its own PCIe probe unless `--pcie-frac` was given. CUDA0
trims too when `stage_trim` (`:2555-2560`), keeping all routers when `--mmap-experts` (the router lookahead copies them).

## `--split-skip-if-fits` (`:2848-2893`)

With `auto`: sum the profiled pairs' bytes; if they fit CUDA0's free VRAM minus the whole-model session and
`reserve + 1000 + 96` MiB (drafter/head, windows), the split is dropped and the run is one GPU. Not with helper caches.

## The `auto` placement search (`:3079-3180`)

`stage_room(dev, later, drafter)` = free VRAM minus reserve, prompt buffers (when not borrowing; with a streamed
ring of 96 blobs at chunks >= 1024, `Prefill::set_ring_override(96)`), 96 MiB of windows on later stages, and
1000 MiB of drafter + head on the last stage. Per device, `layer_ms = 0.33 x (84 x 2.617) / (SMs x GHz)` (an RTX 5080
reference). A placement's predicted window time is `miss_ms x (1 - held routed mass) + sum(layers x layer_ms)` where
the routed mass of profile rank `r` is `(r+1)^-1.2`, each stage's cache holds its layers' pairs hottest-first until
its room minus its carved session is used, and `miss_ms` = 190 (`STRATA_SPLIT_MISS_MS`). Two or three stages try
every placement; more share layers in proportion to speed. The chosen K and predicted ms are printed.

## Stage caches and residency (`:3193-3204`, `:3845-3899`, `:4659-4667`)

The profile is split by stage (`stage_of(layer)`); each stage opens its cache (sized slots for a native pack) in its
room, fills it from its pairs, and verifies slot 0. The one `host_res` table holds, for each layer, a slot index into
**that layer's stage's** cache, and is uploaded to every device.

## Serve-side wiring

- **Ring size** (`:5012-5023`): when >= 75% of pairs are resident the prompt path's streamed ring is 96 slots
  (`STRATA_SPLIT_RING=N` overrides; 0 = pinned-share rule).
- **Loans** (`PfPart`, `:4976-5230`): one participant per stage, each lending only rows of its own layers from its own
  cache; the chunk is the largest every participant can lend. A stage whose card still has room for the chunk's
  buffers plus 1.5 GiB keeps them instead (`STRATA_SPLIT_OWN_BUFFERS=0` disables). If a small card caps the chunk,
  a warning names it (#448) and suggests using it as a helper cache instead.
- **Prompt paths** (`:5237-5356`): each stage's `Prefill` gets `set_stage(lb, le, next)`; `init` is retried with
  smaller chunks (`kStepChunks` 6144..512) if buffers "do not fit", after a pre-check leaving 512 MiB headroom. Each
  stage's prompt path also gets the next stage's GPU as a helper (`set_stage_helper`).
- `STRATA_SPLIT_SMALL_OWN=S` / `STRATA_SPLIT_SMALL_MAX=M` (`:5358-5394`): keep S-token buffers per stage for the
  session so short prompts lend nothing.
- **Verifiers** (`:5416-5509`): mapped, portable pinned hand-off buffers of `kVerifyMaxT x handoff_floats(g)` floats
  between consecutive stages; `Verifier::set_stage(lb, le, in, out)`, `set_next(&next, &split_drive)`; later
  stages get `set_split`/`set_pcie_mode`. `Verifier::set_commit_async(!multi_gpu)` (`:2894`). The pool callback is
  `drive_pool_split`, which picks the stage by layer and swaps in that stage's plan sink, cache base, slot offsets and
  PCIe share before calling `drive_pool_multi`.
- **Checkpoints** across stages: each stage saves its part when it reaches a checkpoint position (`on_stage_chunk`),
  the last stage assembles them ([prompt-cache.md](prompt-cache.md)). Conversation parking saves one image per stage,
  with the draft layer's K/V on the last stage.
- **Batch groups** (`--batch-groups G`): the batch slots in G groups pipelined through the stages (`pump`,
  `:6457-6553`).
- Adaptive swaps stay in each layer's own cache, on that stage's `adapt_stream` (`:5880-5886`).
- `STRATA_SPLIT_TIMING=1` prints per-stage wait/pool/host/commit per window after each request.

## Gotchas

- `--gpu-only-full` and `--gpu-stages` are refused across a split (CUDA0's session holds only its range).
- The token graph is not captured under a split (`:4682`): the split always decodes through verify windows.
- A stage that cannot allocate weights reports the MiB needed and free on that card (#486).

## See also

- [expert-tiers.md](expert-tiers.md), [serve-mode.md](serve-mode.md)
- [../core/README.md](../core/README.md) (Verifier stages, session carve), [../prefill/README.md](../prefill/README.md)
