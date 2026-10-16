[Codebase docs](../../README.md) › [Core](../README.md) › [Experts](README.md) › Multi-GPU expert tiers

# Expert tiers on other GPUs

## Purpose

The primary GPU (device 0) keeps the dense weights, the KV cache, the MTP layer, its own `ExpertCache` and the
captured verify graph. These classes add expert-only tiers on other CUDA devices. Experts that the primary does not
hold, but another card does, are computed there instead of on the CPU. The results are written into the same pinned
host rows that the CPU pool fills. Each tier is an `ExpertCache` on its device, filled with the next ranks of the expert
profile.

| Class | Flag | Residency | Kernels |
|---|---|---|---|
| `PeerExperts` | `--peer-device N` | Adaptive (swaps like the primary) | `native_expert_grouped`, native packs only |
| `RemoteExperts` | `--expert-cache-device1..3 N\|auto` | Static (profile fill); adaptive only with `RemoteExpertOpt` | `native_expert_grouped` or `moe_grouped_s2` (canonical) |
| `RemoteExpertOpt` | `--remote-expert-opt` | Swaps for the remote caches | Weighted reduction on the helper GPU |

**This fork's front end refuses `--peer-device`, `--peer-slots` and any `--remote-expert*` engine flag**
(`frontend/main.cpp:632-639`, "one GPU only"). These tiers are reachable only by running the engine binary directly.
`--expert-cache-device1..3` is not in that refusal list (it does not match any of the refused prefixes).

Files: [include/strata/core/peer_experts.hpp](../../../../include/strata/core/peer_experts.hpp),
[src/core/peer_experts.cpp](../../../../src/core/peer_experts.cpp),
[include/strata/core/remote_experts.hpp](../../../../include/strata/core/remote_experts.hpp),
[src/core/remote_experts.cpp](../../../../src/core/remote_experts.cpp),
[include/strata/core/remote_expert_opt.hpp](../../../../include/strata/core/remote_expert_opt.hpp),
[src/core/remote_expert_opt.cu](../../../../src/core/remote_expert_opt.cu).

## `PeerExperts` (a second, adaptive tier)

- `set_peer_portable(on)` (`peer_experts.hpp:32`): set once before any allocation (`generate.cpp:1595`), so mapped
  host buffers use `cudaHostAllocPortable` when a peer exists.
- `open(device, ranked, primary, src, n_layers, n_expert, reserve_mib, max_slots, err)` (`peer_experts.cpp:81-179`):
  1. Enables P2P both ways if the hardware allows it (`p2p()`).
  2. Allocates its staging and scratch buffers first, sized for `CAP = MAXT * 10` entries.
  3. Picks the profile's pairs that the primary does not hold, in rank order, until free VRAM minus `reserve_mib`
     (at least 128 MiB) is used up or `max_slots` is reached.
  4. Opens a sized (native) or uniform cache, waits for the legacy-stream zeroing, fills the slots on its refill
     stream, and verifies the first slot.
- `launch(layer, x, ids, n_tok, k, kind, err, out)` (`:181-248`): takes the entries with `kind == 2`, grouped by
  expert, and uploads the activations and one `Meta` block. It quantizes to q8_1 and runs `native_expert_grouped`.
  - **Direct mode** (default; `STRATA_PEER_DIRECT=0` disables it): `scatter_rows_f32` writes the rows straight into the
    host `out` rows. That is why the dispatcher must not zero those rows (`launched_direct()`).
  - Otherwise the rows are copied D2H into compact staging.
  - Canonical packs are refused ("only native packs are supported").
- `finish(out, err)` (`:250-267`): spins on `cudaStreamQuery` (a blocking sync would wake the pool thread late). In
  non-direct mode it then copies the compact rows into place. Wait time accumulates in `ms_wait`.
- `adapt(usage, res0, max_swaps, err)` (`:269-312`): the same rule as the primary's adaptive tier. Candidates are
  experts held by neither card with `usage >= 2`; victims are the peer's resident experts. A swap needs
  `cand >= victim + 1.5` and is ranked by gain. Copies run on the refill stream, and the victim is non-resident at once.
  `apply_pending(wait)` (`:314-323`) admits the new experts after `refill_ev_`. The serve loop calls it with
  `--peer-adapt-swaps` (default `--adapt-swaps`), after the primary's swaps (`generate.cpp:5910-5917`).
- Counters: `entries()`, `experts()`, `swaps()`, `resident()`, `gib()`.

The dispatcher marks kind 2 for experts `peer->has(l, e)` holds (`expert_source.cpp:2097-2098`, `:2154`), launches
before the CPU starts and finishes after it.

## `RemoteExperts` (static helper tiers on CUDA1..3)

- `preflight(device, free_gib, err)` (`remote_experts.cpp:67-89`): must run before the host arena registers tens of
  GiB with CUDA. Unless `STRATA_REMOTE_SPIN=0`, it creates the context with `cudaDeviceScheduleSpin | cudaDeviceMapHost`.
  The comment says a sleeping wait costs about 0.3 ms per round trip on Windows. `STRATA_EARLY_REMOTE_CONTEXTS=1`
  preflights every helper early (WSL2's pinned-host budget, `generate.cpp:2345-2359`).
- `open(device, slots, layers, experts, ranked, primary, source, claimed, err, auto_size)` (`:121-237`): selects ranked
  pairs that neither the primary nor an earlier helper has claimed (`claimed` is shared across helpers and layer-split
  stages). With `auto_size`, it stops before leaving less than 512 MiB free. Fills slots with `fill_slot_blocking`,
  verifies the first, and allocates staging plus one metadata block. **Zero-copy** (default; `STRATA_REMOTE_ZEROCOPY=0`
  disables it): the helper reads its input from, and writes its compact rows into, the mapped host buffers directly.
- How `generate.cpp` assigns ranks (`:3900-3990`): remote tiers need `--expert-profile`, a CUDA0 expert cache and the
  pool. The ranking is the profile; with several helpers or an `auto` size, every unranked pair is appended in
  expert-then-layer order. With several helpers, `--expert-cache-remote-placement stripe` (default) deals ranks round-robin, and `layer`
  gives layer `l` to helper `l % devices`.
- `begin(layer, x, ids, n_tok, k, kind, primary_res, err)` (`:239-326`): requires `k == 10`. It claims the entries the
  primary has not taken (`kind == -1`, not in `primary_res`) whose expert it holds, groups them, packs the output rows
  densely (`original_row_`) and launches. Native packs use `quantize_q8_1_rows` and `native_expert_grouped`; canonical
  packs use `quantize_q8_0_scaled` and `moe_grouped_s2`. `owns(i)` tells the dispatcher which rows to leave alone.
- `finish(out, err)` (`:328-339`): `cudaStreamSynchronize`, then scatters the compact rows into `out`, or hands over to
  `RemoteExpertOpt::accumulate`.
- Reported: `computed()`, `launched_layers()`, `returned_bytes()` vs `full_row_bytes()` (the dense packing's saving),
  `ms_begin()`, `ms_wait()`.

## `RemoteExpertOpt` (`--remote-expert-opt`)

`remote_expert_opt.cu` (194 lines). It is attached to each `RemoteExperts` before `open`, and `init` runs after all
helpers are open (`generate.cpp:3976`, `:4008`).

- `begin(weights, token_begin, tokens)` / `end()` bracket the pool callback with the router weights.
- When active, `RemoteExperts::begin` appends a `ReduceMeta` block (row and weight per entry). The helper GPU runs
  `reduce_experts`, which sums `w * row` over a token's owned entries. Only `tokens x H` floats return instead of one
  row per entry.
- `accumulate` adds each helper's sum into a mapped host buffer and ORs a per-entry mask.
- `copy_rows` (`copy_cpu_rows` kernel) copies the CPU rows into the device `parts` and zeros rows the mask or the hit
  list marks.
- `combine` adds the remote sum into `parts` (`copy_or_zero_from_mapped` first when a device-planned group must ignore
  a stale sum).
- `adapt(usage, primary, pending, max_swaps, source)` (`:136-193`): the same candidate rule, plus one more condition. A
  victim is only taken while it is no hotter than the primary's coldest resident expert of that layer. Fills run
  synchronously on the helper's stream, followed by `ExpertCache::replace`. It runs at the start of the primary's
  `adapt` (`generate.cpp:5841`).
- In the dispatcher, tokens whose entries all run on GPUs skip the CPU activation quantisation
  (`expert_source.cpp:2186-2189`).

## Configuration

| Flag / env | Default | Effect |
|---|---|---|
| `--peer-device N`, `--peer-reserve-mib N`, `--peer-slots N`, `--peer-adapt-swaps N`, `--peer-prefill-rows N` | off, 600, fit, `--adapt-swaps`, half a chunk | Peer tier |
| `--expert-cache-device1..3 N\|auto` | off | Remote tiers |
| `--expert-cache-remote-placement stripe\|layer` | stripe | Rank distribution |
| `--remote-expert-opt` | off | Reduction, mask and remote adaptivity |
| `STRATA_PEER_DIRECT=0` | direct on | Peer rows via D2H and host copy |
| `STRATA_REMOTE_SPIN=0`, `STRATA_REMOTE_ZEROCOPY=0`, `STRATA_EARLY_REMOTE_CONTEXTS=1` | | Helper context, I/O and start-up order |

## Model-specific behaviour

`PeerExperts` supports native packs only. `RemoteExperts` supports both. Both size slots per layer for native packs
(`open_sized`), and Qwen3.6 packs are always native. The code assumes `k == 10` routed experts (`CAP = MAXT * 10`;
`RemoteExperts::begin` refuses any other `k`).

## Tests

No test in `tests/core` exercises these classes. They need two or more GPUs.

## Gotchas

- The peer's `launch` in direct mode writes host rows while the CPU pool runs. The dispatcher skips zeroing those rows.
- `RemoteExperts::open` leaves 512 MiB free on the helper (WDDM), and refuses otherwise ("reduce --expert-cache-deviceN").
- The peer is not compatible with `--layer-split` or `--expert-cache-device1..3` (help text, `generate.cpp:720-722`).

## See also

[vram-cache.md](vram-cache.md) · [dispatch.md](dispatch.md) · [resident-ram-and-adaptive.md](resident-ram-and-adaptive.md)
