[Codebase docs](../README.md) › [Core](README.md) › VRAM plan

# The VRAM planner (`strata/plan/vram_plan.hpp`)

## Purpose

On one consumer card the binding constraint is VRAM. The planner decides, before any weight moves, how a card's
budget is split between the **fixed set** (dense weights, drafter, KV cache for the requested context, session state,
reserve, prompt buffers) and **expert slots**, whether the context must be lowered, and how many of the remaining
experts the host RAM tier can hold; the rest stay mapped from the model file on disk. It is a pure, header-only,
GPU-free function so it can be unit-tested against the design study's table.

Ownership: the header is part of the `strata_plan` INTERFACE library in `CMakeLists.txt` (with
`include/strata/plan/plan.hpp`, the older Phase-1 planner used by `strata-device` and `strata-plan`). Its only engine
caller is `src/program/generate.cpp`; it is documented here because it sizes the core's memory. The header comment
says `strata-plan` can print it, but `src/plan/plan_main.cpp` does not include `vram_plan.hpp` in this checkout.

File covered: [include/strata/plan/vram_plan.hpp](../../../include/strata/plan/vram_plan.hpp); test
[tests/core/vram_plan_test.cpp](../../../tests/core/vram_plan_test.cpp).

## Types and functions

### `struct VramPlanInput` (`vram_plan.hpp:24`)

| Field | Meaning (as generate.cpp fills it for Qwen3.6) |
| --- | --- |
| `budget_bytes` | free VRAM from `cudaMemGetInfo` (via `vram_info`, so `--vram-cap` applies) |
| `weights_bytes` | census of the GGUF's non-expert tensors of trunk layers, `token_embd.weight` excluded (it lives in mapped host memory) |
| `drafter_bytes` | the nextn block's tensors (or 504 MiB when the file has none and the `-MTP` pack provides it); 0 without `--mtp` |
| `drafter_kv_bytes_per_token` | the drafter's one attention layer, whole context, in the `--kv` format (plain INT8 under `k8v4`) |
| `drafter_head_bytes` | draft-vocabulary head rows (only with a `draft_vocab.bin` subset) + `8 x n_vocab x 4` logits |
| `drafter_ids_bytes_per_token` | `max(--spec, 1) x 4` |
| `kv_bytes_per_token` | `n_attention_layers x 2 x n_head_kv x head_dim x bytes/elem` (fp16 2, int8 1+2/64, q4_0 18/32, k8v4 816/1024) |
| `session_bytes` | RoPE table (`max_context x n_rot x 4`) + GDN states + 36 MiB scratch |
| `cuda_context_bytes` | 0 in the engine (the free figure is read after the context exists); the test uses 300 MiB because its budgets are card sizes |
| `reserve_bytes` | `--vram-reserve-mib` if given, else 300 MiB |
| `prompt_bytes` | prompt chunk buffers `(160 + chunk*680/1024) MiB`, or 0 when prefill borrows expert-cache slots (an expert profile is given and `--no-prefill-borrow` is not) |
| `context`, `context_floor`, `context_step` | requested `--max-context`; floor `min(131072, request)`; lowered contexts are multiples of 4096 |
| `n_experts`, `mean_blob` | `n_layers x n_expert`; mean bytes per expert (native layout total / count, or the census) |
| `ram_available_bytes`, `ram_headroom_bytes` | `MemAvailable` (clamped by `--ram-cap`); headroom 4 GiB or the resident mode's own |
| `smaller_file` | named in a refusal (`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` unless the preset already is IQ3) |

### `make_vram_plan` (`vram_plan.hpp:85`)

```
fixed(ctx) = weights + drafter(ctx) + kv_per_token*ctx + session + cuda_context + reserve + prompt
if fixed(request) > budget:
    ctx = floor((budget - fixed(0)) / per_token) rounded down to context_step
    if ctx < floor:  REFUSE  (fits = false, refusal text with every term and the bytes short at the floor)
    else:            lowered = true, context = ctx
expert_bytes = budget - fixed(context)
vram_slots   = min(expert_bytes / mean_blob, n_experts)      (spare_bytes when every expert fits)
ram_copies   = min((ram_available - headroom) / mean_blob, n_experts - vram_slots)
mapped       = the rest (read from the model file on a miss)
```

Helpers: `vram_plan_per_token` (`:71`), `vram_plan_drafter` (`:75`), `vram_plan_fixed` (`:80`), and
`vram_plan_text` (`:146`) for the start-up log (budget, fixed-set breakdown, LOWERED / REFUSED lines, the three
tiers).

`VramPlan` (`:54`) returns `fits`, `context`, `lowered`, `kv_bytes`, `fixed_bytes`, `expert_bytes`, `vram_slots`,
`spare_bytes`, `ram_copies`, `mapped`, `ram_tier_bytes`, `context_needed_bytes_short`, `refusal`.

## How the engine uses it

`src/program/generate.cpp` (around lines 2184-2290), only when `g.arch == Qwen35Moe`:

1. builds the input from the GGUF census, the `--kv` format, `--mtp`, the session formula, free VRAM and host RAM;
2. prints `vram_plan_text`;
3. exits with status 2 and the refusal if `!fits`;
4. if `lowered`, sets `o.max_context = plan.context` so every later allocation uses the lowered context;
5. chooses the RAM tier mode from the plan (the following code picks resident vs mapped-plus-copies experts).

The slot count is at the **mean** blob; the engine's own per-layer sizing (`sized_slots` in generate.cpp, expert
side) prices each profile entry's own blob and "lands within a few slots of it" (header comment).

KV streaming is not a lever here: Qwen3.6's dense attention reads every cell, so the KV stays in VRAM
(`qsa_kv_streaming_applies`, [session.md](session.md)). Flash-Next starts do not use this planner.

## Configuration

`--max-context`, `--kv`, `--mtp`, `--spec`, `--vram-cap MIB`, `--vram-reserve-mib`, `--ram-cap GIB`,
`--prefill CHUNK`, `--no-prefill-borrow`, `--expert-profile`, `--resident-experts` (generate.cpp).

## Tests

`vram_plan_test` (`tests/core/vram_plan_test.cpp`, CTest, no GPU; built with `STRATA_BUILD_TESTS`): feeds the study's
table 4.1 inputs (`docs/QWEN36-35B-A3B.md`) and checks the table 4.3 ladder for budgets from 6 GB to 32 GB in three
arms (Q4 without borrow, Q4 with borrow, IQ3 with borrow), each exactly against `floor((budget - fixed) / blob)` and
within the table's rounding; then the drafter's terms, a lowered context (e.g. 196,608), the refusal naming the IQ3
file, the RAM tier at 9.5 GB, the disk tier on a 1080 Ti, and a `--vram-cap` below the fixed set.

## Gotchas

* The refusal never chooses a context below the floor for you; it names the smaller file and, when prompt buffers
  are counted, suggests an expert profile (prefill then borrows cache slots).
* `cuda_context_bytes` must be 0 when the budget is a free figure, otherwise the driver's share is counted twice.

## See also

* [session.md](session.md) (what `session_bytes` and the KV formats cost), [mtp-draft-head.md](mtp-draft-head.md).
* [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) sections 4.1-4.3, [docs/qwen36-port/step2-notes.md](../../qwen36-port/step2-notes.md), [step4-drafter.md](../../qwen36-port/step4-drafter.md) section 7.
* [experts/](experts/README.md) for how the slots are filled.
