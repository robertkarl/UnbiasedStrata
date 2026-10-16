[Codebase docs](../README.md) › [Program](README.md) › Expert tiers

# Expert tiers as the driver wires them

**Purpose:** how `generate.cpp` decides where each routed expert lives (VRAM slot, RAM copy, mapped file, helper
GPU), how that placement changes at run time, and how the batched prompt path borrows VRAM from it.
**Covers:** `src/program/generate.cpp:227-263` (`resident_stage_swaps`), `:3399-4094` (sources, cache, tiers, hit
path), `:4609-4689` (device residency table), `:4727-4954` (lend planning, resident RAM mode), the serve-side
adaptive tier `:5782-5937` and the one-shot one `:8294-8407`. The classes themselves
(`ExpertCache`, `FileExpertSource`, `ArenaExpertSource`, `RemoteExperts`, `PeerExperts`, `ExpertDispatch`) are in
[core](../core/README.md); the CPU pool is in [kernels](../kernels/README.md).

## The tiers

| Tier | Object | Chosen by |
| --- | --- | --- |
| VRAM cache (CUDA0) | `strata::core::ExpertCache xcache` | `--expert-cache N\|auto`, `--expert-profile` |
| Stage caches | `GpuStage::cache` | `--layer-split` ([layer-split.md](layer-split.md)) |
| Helper GPUs | `std::array<RemoteExperts, 3> remote_experts` | `--expert-cache-device1..3` |
| Peer GPU | `PeerExperts peer` | `--peer-device` |
| RAM | `ArenaExpertSource arena_src` (every expert, pinned where allowed) | default |
| RAM copies + file | `FileExpertSource src` with `pin_cache_complement` | `--resident-experts`, `--resident-budget-gib`, or chosen automatically for Qwen3.6 when the arena does not fit RAM |
| File only | `FileExpertSource src` (mmap of `experts.bin` or the GGUF shards in place) | `--mmap-experts` |

`srcp` points at whichever source is active and is what every fill, refill and swap reads blobs from
(`srcp->blob(layer, expert)`).

## VRAM cache sizing (`:3545-3796`)

1. **`auto`** (`:3553-3622`): `slots = (free - reserve) / max_blob`, capped at the profile's length, where `reserve =
   --vram-reserve-mib + prefill buffers (only if not borrowing: 160 + chunk x 680 / 1024 MiB) + the draft head's bind
   bytes`. If that leaves fewer than a working minimum (`min_slots`: enough to lend a 256-token chunk's buffers plus
   128, or 1 without `--prefill`), and the reserve was not given explicitly, the reserve shrinks toward 300 MiB
   (`kSmallReserveMib`) and `reserve_adapted` is set (#496). Zero slots prints what is short and what frees VRAM.
2. Explicit size under a layer split is clamped to leave the reserve (`:3623-3640`).
3. **Sized slots** (`:3641-3661`): for a native pack with a profile (and not `--expert-cache-per-layer`), slots are cut
   to each profiled pair's own blob size (rounded to 256 B) instead of `max_blob`, fitting ~30% more experts.
4. `--vram-elastic` sets a segment size (`set_segment_bytes`) when allowed (serve, one GPU, no helpers, NVIDIA).
5. **Open loop** (`:3680-3770`): `xcache.open` / `open_sized`. On failure an auto cache retries at three quarters, up
   to 8 times (Windows commit limits, issue #60; `STRATA_TEST_CACHE_FAIL=N` fakes failures). On success an auto cache
   zeroes its slots, re-reads free VRAM and shrinks while below the reserve (WDDM reports free memory optimistically
   until pages are touched).
6. Per-layer admission (`set_per_layer_admission`) and the policy line: per-layer, compulsory-miss, or profile.

## Profile fill (`:3797-3843`)

With a profile, the ranked pairs are admitted in order until the cache is full (`xcache.admit` +
`fill_slot_blocking`); per-layer mode skips only a full layer's pairs (#369). An unbuffered file tier reads ahead in
batches of 64 on a `std::async`. **One slot is read back and compared** (`verify_slot`) - a residency table right
about indices but wrong about bytes would produce plausible wrong tokens. The fill is the policy: the cache is full,
so later admission finds no room and non-profiled experts stay CPU misses until adaptive swaps move them.

Helper caches (`:3901-3987`) take the ranking after CUDA0's share (plus, with `auto` or several helpers, every
unranked pair in expert-then-layer order), striped or by layer. The peer (`:3989-4006`) takes what the primary does
not hold; `STRATA_PEER_HOT` / `STRATA_PEER_HOT_AT` first move a share of the hottest ranks behind the primary's fill so
both GPUs compute experts every layer (`:2994-3023`).

## The hit path and the residency table

- `Drive drive` (`:4009-4019`) wraps an `ExpertDispatch` handed to the CPU pool: pool, source, remote/peer tiers,
  `split_rows`, CPU-order flag.
- `RouterLookahead` (`:4020-4045`): with a file tier that warms pages, the next layer's router (BF16 copies of every
  `ffn_gate_inp`) predicts its experts so their pages are warmed. `STRATA_LOOKAHEAD=0` disables,
  `STRATA_LOOKAHEAD_K` sets experts per token (default 10).
- R4 hit buffers (`:4046-4094`): scratch for `moe_hit_grouped`, slot/dst lists, q8 activations with fp32 scales (so
  GPU hits and CPU misses quantize identically), the `hit_done` event; `hit_poke` unless `--no-hit-poke`.
- `host_res` / `d_res` (`:4611-4680`): built when the cache is profile-filled (`graph_hits`), the token graph is on,
  and no per-layer dump is armed. Every `(layer, expert)` maps to its slot in the cache that owns that layer (a stage's
  cache under a split). Then the arena hands VRAM-held experts back to the OS (`srcp->release`, unless
  `STRATA_ARENA_RELEASE=0`) and prefetches the rest (`srcp->prefetch`). The table is uploaded to every stage.
- `TokenHits thits` lets the captured token graph decide hits on the device. Verify windows get the same table via
  `VerifyHits vh` (`d_res`, `h_res`, `cache_base`, `slot_off`, `n_slots`).

The GPU computes cached experts with a different float order than the CPU; the start-up message says replies may
differ slightly from a cache-off run (bench/results/2026-09-27-cache-parity, cited at `:3776-3785`).

## Lending cache slots to the prompt path

The batched prompt path needs device buffers that scale with the chunk. By default (a profile and no
`--no-prefill-borrow`, `pf_borrow`, `:3050`) it **borrows the cache's last slots** for the duration of a prompt:

- `lend_slots(c)` = slots whose bytes cover `Prefill::bytes_needed(g, ss, c)`; `slots_from_bytes` /
  `bytes_from_slots` convert using per-slot offsets for sized slots (`:4740-4761`).
- `plan_lend(chunk)` (`:4805-4880`, one-shot mode): fixed chunk halves until it leaves 128 slots; `--prefill auto`
  picks the largest chunk on the 256-token grid (`biggest_chunk` bisection up to `auto_ceiling`) that still leaves
  the streamed **ring** at its full byte budget, falling back to 0.1.39's list (`kAutoChunks`) and a 16-slot minimum
  ring; it never takes more than `kAutoLendPct` of the slots (90% when >= 90% of expert bytes are pinned, else 85%;
  `STRATA_PREFILL_LEND_PCT` overrides). `STRATA_RING_BYTES=0` (read in prefill) selects the old list.
- Serve mode has the same logic per participant (`PfPart`, `:4976-5230`) - see [serve-mode.md](serve-mode.md) and
  [layer-split.md](layer-split.md).
- Lending marks the lent experts `kNotResident` in `host_res` (they become CPU/PCIe misses during the prompt) and
  `Prefill::relayout` places the buffers in those slots; afterwards they are refilled from `srcp` (queued copies and
  one wait, or blocking with `STRATA_REFILL_BLOCKING=1`) and the table restored.

## Resident RAM mode (`:4881-4954`)

With `--resident-cpu-experts` (and therefore `--mmap-experts`): `src.pin_cache_complement(xcache, err, pin, {},
lend_from, headroom, budget, &profile)` copies the experts no VRAM slot holds - and the slots the prompt path may lend
- into RAM, page-locked when `resident_pin`. If the whole complement does not fit and the mode is soft, it retries
with `kResidentWhatFits` (the hottest by profile) (#467); otherwise it warns and falls back to plain mmap, or exits
for an explicit hard request. With adaptive swaps on, `reserve_exchanges(min(adapt_swaps, 96))` allocates exchange
buffers so swaps trade bytes between RAM and VRAM without reading the file (`resident_stage_swaps`, `:234-263`).
On Windows the unbuffered-read choice is re-checked against the RAM actually held (#577).

## Adaptive expert swaps (`--adapt-every`, `--adapt-swaps`, `--adapt-decay`)

The VRAM tier follows the conversation. It runs in verify-window mode only (one-shot speculative loop and serve).

- **Counting.** `drive.d.usage` (one float per `(layer, expert)`) is allocated when `--adapt-every > 0` and
  `--adapt-swaps > 0` (serve: and not every expert is already resident, `:5784-5787`); the dispatch increments it as
  experts are routed.
- **When.** Every `adapt_every` verify rounds, `adapt()` runs on its own `std::thread` beside the commit and the MTP
  draft (`:7565-7568`, `:8541-8544`), and is joined before the next window.
- **What.** Per layer: candidates are non-resident experts with usage `>= 2.0` (serve: not held by the peer or a
  remote-opt helper); victims are resident ones, least used first. Pairs swap while `cand >= victim + 1.5`. All swaps
  are sorted by gain and truncated to `--adapt-swaps`. Usage is then multiplied by `--adapt-decay`.
- **How.** `resident_stage_swaps` first copies evicted experts back into exchange buffers (resident RAM mode). AMD
  with `STRATA_ARENA_MMAP=1` page-locks the source pages (`pin_blobs`). Each swap is a `cudaMemcpyAsync` into the
  victim's slot on `adapt_stream` (a stage's own stream under a split); the victim is marked non-resident at once (the
  CPU computes it meanwhile) and the pair goes into `pending`.
- **Landing.** `apply_pending(wait)` before each window: when `adapt_ev` has completed, pending pairs become resident,
  `srcp->release` frees their RAM pages, exchanges are committed, the table is re-uploaded. Since #463 the window
  **waits** for the copies (so whether an expert runs on GPU or CPU does not depend on timing);
  `STRATA_ADAPT_NOWAIT=1` restores the non-blocking query. `STRATA_TRACE_ADAPT=1` (one-shot) logs each round.
- **Peer.** After the primary's swaps, `peer.adapt` takes the next most-routed CPU misses (`--peer-adapt-swaps`).
- **Learned profile (#477).** With `--expert-profile-save`, `heat` accumulates usage at each adaptation; `save_profile`
  ranks resident experts first, then by heat, with the loaded profile as prior (`rank_learned_profile`), and writes it
  with `write_expert_profile` periodically between requests and at exit.

## VRAM give-back (`VRAM` command, `--vram-elastic`)

`vram_command` (`:6159-6253`) shrinks the segmented cache from its end until the requested reserve is free (never
below the prompt path's minimal loan + 128 slots), marking evicted slots' experts non-resident and remembering them in
`vram_evicted`; a grow maps segments again and refills each slot with its old expert or, if that one was already
swapped back, the layer's most-routed miss. `relend()` then recomputes the prompt chunk. Refused with the resident RAM
mode, a peer, an outstanding loan, or batch slots decoding. Protocol details: [serve-mode.md](serve-mode.md).

## Configuration summary

Flags: `--expert-cache`, `--expert-profile`, `--expert-profile-save[-every]`, `--expert-cache-per-layer`,
`--vram-reserve-mib`, `--vram-cap`, `--ram-cap`, `--vram-elastic`, `--vram-segment-mib`, `--mmap-experts`,
`--shared-expert-arena`, `--resident-*`, `--pcie-frac`, `--pcie-mode`, `--adapt-*`, `--expert-cache-device*`,
`--peer-*`, `--prefill`, `--no-prefill-borrow`. Env: `STRATA_ARENA_PIN_GIB`, `STRATA_ARENA_MMAP`,
`STRATA_ARENA_RELEASE`, `STRATA_FETCH_THREADS`, `STRATA_UNBUFFERED_LOAD`, `STRATA_LOOKAHEAD[_K]`,
`STRATA_PEER_HOT[_AT]`, `STRATA_PREFILL_LEND_PCT`, `STRATA_REFILL_BLOCKING`, `STRATA_ADAPT_NOWAIT`,
`STRATA_TRACE_ADAPT`, `STRATA_TEST_CACHE_FAIL` ([env-vars.md](env-vars.md)).

## Gotchas

- Under WDDM an over-subscribed allocation pages instead of failing; a verify graph spinning on a host flag then
  stalls forever. Hence the post-open re-measure and the serve-time "VRAM free with everything loaded" warning
  (`:6001-6023`).
- `--cpu-oracle-q8-0` cannot be combined with the cache: the GPU expert contract does not match it.
- The comment block at `:385-392` records that an earlier "the cache has no consumer" note was false (round 328).

## See also

- [startup.md](startup.md), [speculative-decoding.md](speculative-decoding.md), [layer-split.md](layer-split.md)
- [../core/README.md](../core/README.md), [../prefill/README.md](../prefill/README.md)
