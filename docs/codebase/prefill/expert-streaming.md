[Codebase docs](../README.md) › [Prefill](README.md) › Expert streaming

# Getting experts onto the card during a prompt

## Purpose

A prompt chunk of thousands of tokens routes to nearly every expert of every MoE layer (at 1024 tokens x top-10 of
512 almost all are hit). Experts that are not resident in the GPU expert cache must be copied host -> device once
per chunk. This page covers where the prompt path's own memory comes from (borrowed cache slots), how the copies are
scheduled so the copy engine overlaps compute, and how the ring and chunk are sized. It is the mechanism behind the
README's remark that at 6 GB the engine "spends the prompt copying experts to the card": with few resident slots, and
those partly lent to the prompt buffers, most experts of every layer cross PCIe once per chunk.

Files: [src/prefill/prefill.cpp](../../../src/prefill/prefill.cpp) (ring, `Stager`, streamed walk, sizing),
[src/program/generate.cpp](../../../src/program/generate.cpp) (lending, chunk choice, refill).

## Borrowing cache slots

By default (`--no-prefill-borrow` not given, an expert profile loaded) the prompt path does not allocate: the caller
picks `k` slots at the end of the expert cache, marks every expert in them non-resident in the residency table
(`host_res[i] = kNotResident`, uploaded to `d_res`), and passes `xcache.device_slot(first)` and the bytes to the end
as `borrow` / `borrow_bytes` to `Prefill::init` (`src/program/generate.cpp:7913-7950`). `Alloc` then carves every
buffer from that region. After `run`, the lent slots are refilled from the host arena (queued copies and one wait,
D-4; `STRATA_REFILL_BLOCKING=1` waits on each) and marked resident again (`:7972-7992`). In `serve` the loan is per
request and per participant (each layer-split stage lends from its own cache): `lend(tokens)` rounds the request to
256 tokens, caps it at the configured chunk, and calls `Prefill::relayout` into the slots it needs
(`:7174-7216`), so a short prompt lends little.

Constraints in the lending rule: at least 128 slots must stay with the cache (`k + 128 <= slots`), and with
`--prefill auto` at most `kAutoLendPct` percent of slots may be lent: 90 when (nearly) every expert byte is pinned,
85 otherwise, `STRATA_PREFILL_LEND_PCT` overrides (`:4777-4784`). An expert whose slot is lent is simply streamed
(or, on the decode side meanwhile, computed on the CPU).

## Two ways a layer's experts are read

| Mode | When | How |
| --- | --- | --- |
| Routed-only staging | chunks below `stream_all_min()` (1024 tokens, `STRATA_PREFILL_STREAM_MIN`), or a ring of `STAGE` (8) slots | after routing, the host groups `(token, k)` pairs (one host sync per layer), then for each routed expert in id order: resident -> read from its cache slot; else DMA'd into one of 8 staging slots, up to 7 ahead (`stage_one`, `prefill.cpp:2925`) |
| Stream-all walk | chunk `>= stream_all_min()`, `ring_slots(T) > STAGE`, an `ExpertSource` present | at chunk start a sequence of EVERY non-resident expert of every layer in `[LB, LE)` in `(layer, id)` order is built (`prefill.cpp:1992-2038`); entry `k` lands in ring slot `k % ring`; the copy stream runs ahead through the attention halves instead of waiting for each layer's routing |

The sequence skips experts resident in the cache (`host_res >= 0`), and on `--peer-device` those held by the peer.
Experts the router did not pick still occupy a ring entry; their slots are given back without compute
(`release_to`, `prefill.cpp:3086`).

### Slot reuse protocol

Each ring slot has a `copied` event (recorded on the copy stream after its DMA) and a `used` event (recorded on the
compute stream once its blob has been read). Before a slot is refilled, the copy stream waits on
`used[used_of[slot]]` (`prefill.cpp:2050`, `:2097`). `used_of` lets one event release several slots: with
`STRATA_PREFILL_GROUP_GATHER` (default on, native packs on MMQ in the streamed walk) a 16-expert MMQ group is gathered
in one launch after one wait on the group's last copy, and its slots share one `used` record (`prefill.cpp:2954-2993`),
which removed about 10 us of GPU bubble per expert under WDDM.

### The issuer thread (D-5)

With the stream-all walk the copies are issued by a dedicated host thread (`prefill.cpp:2084-2112`), never by the
thread launching kernels: it issues entry `idx` once `idx < consumed + ring`, waiting for an unpinned blob's host copy
if needed. The consumer side calls `wait_issued(k)` before waiting on `copied` and `give_back(consumed)` after.
`STRATA_PREFILL_ISSUER=0` issues inline (`issue_until`). Same copies, same order, same results.

### The `Stager` (unpinned blobs)

Experts in pinned host memory are DMA'd straight from the arena (`experts_dma`). The rest (the share the arena could
not pin; a GGUF read in place, `ExpertSource::transient`) are first copied into a ring of pinned host buffers by
`Stager` threads (`prefill.cpp:296-422`): job `j` uses buffer `j % kRing`, which is free once job `j - kRing`'s DMA
(recorded via `issued_one`) is done. Claims are a CAS on `gen << 32 | n << 16 | next`, so a thread late from the
previous layer cannot take a job of the next. `kRing` is `STRATA_STAGER_RING` (default 16, or `4 x threads` for
in-place GGUF reads); threads `STRATA_STAGER_THREADS` (default `max(2, min(4, hw/4))`, 32 for in-place GGUF reads:
measured 71 s -> 29 s for a 4K chunk on an NVMe PC). A transient blob is assembled by the source itself
(`copy_blob`); a failure aborts the process.

## Sizing the ring

`ring_slots(T)` (`prefill.cpp:206`) returns `STAGE` (8) below `stream_all_min()`, else a count clamped to
`[16, ring_cap()]`:

1. `STRATA_PREFILL_RING` if set (exactly 8 keeps routed-only staging even for large chunks);
2. else `set_ring_override` (a layer split without borrowing uses 96);
3. else the auto scan's byte budget (`set_ring_budget`) for chunks above the size 0.1.39's rule would have picked;
4. else the pinned rule: 1024 slots with the fused Q2_0 ring, 384 otherwise, 96 when less than 90 % of expert bytes
   are pinned (host copies, not the copy engine, are then the limit).

`ring_cap()` is 1024 for the fused Q2_0 path, else 512. A slot is one whole `max_blob`, so the budget is kept in
bytes: `ring_bytes()` is the measured slot count times the Q2_0 blob (1,382,400 B), and `ring_budget_slots()` divides
it by this pack's `max_blob` (384 on Q2_0, 199 on a 2.54 MiB IQ3_S blob). Measured motivation in the comments: a
Q8_0 pack's 384 slots were 1.9 GiB, which held an 8 GB card at 1024-token chunks; budgeting in bytes moved it to
6144 (87 -> 402 tok/s) (`prefill.cpp:137-148`). `STRATA_RING_BYTES=0` restores 0.1.39's ring, loan and chunk list.
On HIP with `STRATA_HIP_WMMA=1` the ring is 96 (`prefill.cpp:208-216`).

`fused_ring()` (`prefill.cpp:163`) is true when the fused experts are enabled and no peer is used, and for a native
pack only if every layer is covered by the fused IQ kernels; then MoE buffers are also smaller (`moe_bufs`).

## Choosing the chunk (`generate.cpp`)

`plan_lend` (`src/program/generate.cpp:4805`): a fixed `--prefill CHUNK` is halved until its lend fits. `--prefill
auto` (default since 0.1.39b, #583) treats chunk and ring as one byte budget: it finds the largest chunk (bisection up
to `auto_ceiling`: 8192, or 16384 / 32768 when asked and the context allows) whose non-ring buffers leave room for a
full ring (`ring_cap_for(old_chunk)`: the byte-budget ring when 0.1.39's rule picked a chunk below 6144, else 0.1.39's
ring), falling back to a ring of at least 16 slots, and keeps 0.1.39's choice when the scan buys nothing. Comments
record why: on a 4-GPU IQ3_S rig a ring slot was worth about 0.53 tok/s against 0.05 for a chunk token.

## Why small cards are slow at prompts

Each chunk streams every expert that is not resident, and the bytes per chunk are fixed by the pack, not by the chunk.
Bigger chunks amortize them over more tokens, but a small card has (a) few resident slots, (b) less room to lend for
large chunk buffers, and (c) a ring that competes with both. The README's Qwen3.6 tables show the cliff: on the RTX
4090, the 32K prompt reads at 3,003 tok/s (Q4_K_XL) under an 8 GB cap and 517 at 6 GB; IQ3_XXS 4,320 and 1,121.
The `strata generate: prefill ... experts streamed N (M by DMA, host X ms), resident R` line and
`STRATA_PREFILL_TIMING=1` ("wait copy" phase) show how much of a prompt is copy-bound.

## Statistics

`PrefillStats::experts_streamed`, `experts_dma`, `ms_experts_host`, `experts_resident` (counted per expert-layer
group read from VRAM, routed or not in the fused walk).

## Configuration

| Knob | Effect |
| --- | --- |
| `--prefill CHUNK|auto|auto:N`, `STRATA_PREFILL_AUTO_MAX` | chunk choice |
| `--no-prefill-borrow` | own buffers (about `160 + chunk * 680 / 1024` MiB reserved, `generate.cpp:2258`) |
| `STRATA_PREFILL_LEND_PCT` | auto lend cap in percent |
| `STRATA_PREFILL_STREAM_MIN` | chunk size from which the stream-all walk runs (default 1024; below it Q2_0 output changed, so 1024 is the floor) |
| `STRATA_PREFILL_RING` | ring slots |
| `STRATA_RING_BYTES=0` | 0.1.39's ring and chunk list |
| `STRATA_PREFILL_ISSUER=0` | issue copies inline |
| `STRATA_PREFILL_GROUP_GATHER=0` | one gather / wait / record per expert |
| `STRATA_STAGER_RING`, `STRATA_STAGER_THREADS` | unpinned-blob staging |
| `STRATA_GROUP_COPY=1` | grouping tables via `cudaMemcpyAsync` instead of mapped memory (A/B: the copies queue behind expert blobs; measured 4.2 s of a 128K prompt) |
| `STRATA_REFILL_BLOCKING=1` | refill lent slots one blocking copy at a time |
| `STRATA_KV_STAGE_OWN` | KV streaming's staging pool in its own allocation, not borrowed slots (A/B) |
| `STRATA_SPLIT_SMALL_OWN`, `STRATA_SPLIT_SMALL_MAX` | layer split: keep a small chunk's buffers per stage so short requests lend nothing |

## Gotchas

- Lent slots hold prompt buffers, not experts, until refilled; in `serve` only the lending participant's loan is
  returned (`refill_one`), never another stage's (`generate.cpp:7192-7196`).
- A ring entry the routing skipped inside an open MMQ group first flushes the group, so no more than a group's
  entries are ever held back from the issuer.
- At the end of a run the copy stream is synchronized separately so a failed expert copy is reported by this prompt,
  not the next request (`prefill.cpp:3269-3273`).

## See also

[pipeline.md](pipeline.md), [moe.md](moe.md), [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) ("Faster prompts
(engine 0.1.13)"), [README.md](../../../README.md) (Qwen3.6 speed tables). The expert cache, residency table and
`ExpertSource` (`blob`, `pinned`, `transient`, `copy_blob` in `include/strata/core/expert_source.hpp:131-162`) are
documented in the core area ([Core › Experts](../core/experts/README.md)).
