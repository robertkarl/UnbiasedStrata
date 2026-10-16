[Codebase docs](../README.md) › [Libraries](README.md) › Memory planner (`strata-plan`)

# Memory planner and the `strata-plan` tool

## Purpose

`strata_plan` is a header-only CMake target (`CMakeLists.txt:263-268`) holding two planners:

- **`include/strata/plan/plan.hpp`**: the original phase-1 ("P1.S9") census and memory planner for
  Qwen3.8-Flash-Next. It splits a fixed VRAM pool between the KV cache (plus indexer keys), fixed costs and expert
  cache slots, and throws `DoesNotClose` rather than overcommit. This page documents it and the `strata-plan` CLI
  that prints it.
- **`include/strata/plan/vram_plan.hpp`**: the VRAM planner the engine actually uses today (Qwen3.6 port, step 2). It
  is a pure function from measured budget, weight census, drafter and KV costs to a plan that lowers the context
  toward a floor and refuses below it, then turns the remaining bytes into expert slots and a RAM tier. It is
  documented by the core writer at [../core/vram-plan.md](../core/vram-plan.md).

The design rule stated in both headers: the plan is arithmetic you can check by hand, and it must refuse at startup
rather than fail at token 4000.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/plan/plan.hpp](../../../include/strata/plan/plan.hpp) | 168 | `Geometry`, `Costs`, `state_bytes`, `kv_bytes_per_token`, `Plan`, `DoesNotClose`, `vram_pool_bytes`, `make_plan`, `to_string` |
| [src/plan/plan_main.cpp](../../../src/plan/plan_main.cpp) | 73 | `strata-plan` CLI |
| [include/strata/plan/vram_plan.hpp](../../../include/strata/plan/vram_plan.hpp) | 178 | The engine's VRAM planner (see [../core/vram-plan.md](../core/vram-plan.md)) |

## Types and functions (`plan.hpp`)

| Symbol | Line | What it is |
|---|---|---|
| `struct Geometry` | `include/strata/plan/plan.hpp:23` | Flash-Next constants: 48 layers, 12 QSA (full-attention) layers, 2 KV heads of 256, INT8 KV grouped by 64, indexer pooling one key per 4-token block, indexer with 4 query heads but **one shared 128-wide key head**, GDN state 128 x 48 v-heads, conv kernel 4, 10,240 conv channels. `n_gdn_layers()` = 48 - 12 = 36. |
| `struct Costs` | `:62` | Byte costs: `expert_blob = 1,382,400` (one expert's gate_up + down, codes and scales), `dense_bytes`, `embd_bytes`, `mtp_bytes`, `workspace_bytes`, `state_bytes` (all 0 by default). |
| `state_bytes(Geometry)` | `:83` | GDN recurrence (128 x 48 x 128 fp32 = 3,145,728 B/layer) plus conv history (3 x 10,240 fp32 = 122,880 B/layer), times 36 layers = about 117.7 MB per sequence. Fixed, not evictable. |
| `kv_bytes_per_token(Geometry)` | `:91` | Per QSA layer: INT8 K and V (2 x 2 x 256 = 1,024 B) plus fp16 scales per 64 elements (32 B) plus the indexer key (1 x 128 / 4 = 32 B), times 12 layers = 13,056 B per token. |
| `struct Plan` | `:103` | `max_context`, `kv_bytes`, `state_bytes`, `cache_bytes`, `cache_slots`, `vram_budget`, `vram_used`, `dram_experts`, `notes`. |
| `struct DoesNotClose` | `:118` | `std::runtime_error` thrown when the plan cannot be honoured. |
| `vram_pool_bytes()` | `:125` | 5,943,000,000 B: the one number taken from a planning table rather than derived (the KV + cache sum that table holds constant). |
| `make_plan(max_context, g, c, pool)` | `:129` | `kv = kv_bytes_per_token * max_context`; `fixed = dense + embd + mtp + workspace + state`; throws if `kv + fixed > pool`; else `cache_bytes = pool - kv - fixed`, `cache_slots = cache_bytes / expert_blob`; throws if 0 slots. `vram_used` counts the slot remainder as used. |
| `to_string(Plan)` | `:154` | The human-readable report `strata-plan` prints (values in decimal GB). |

Per-token KV arithmetic, from the code: `kv_elems = 2 * 2 * 256 = 1024`, `kv_scales = (1024 / 64) * 2 = 32`,
`idx_layer = 1 * 128 / 4 = 32`, so `12 * (1024 + 32 + 32) = 13,056` bytes per token.

The long comment at `plan.hpp:30-47` records that the indexer-key term was wrong twice (2x and 4x too big) because a
metadata head count was multiplied in without checking which tensor it counted: `indexer.q_proj` has 4 query heads,
`indexer.k_proj` has one key head, and only the key is cached. Likewise `state_bytes` was missing from plans until
"round 198" because nothing passed `--state`.

## The `strata-plan` CLI

`src/plan/plan_main.cpp:13`. Built as `strata-plan` (`CMakeLists.txt:270-271`); it needs no GPU.

```
strata-plan [--max-context N] [--vram BYTES] [--dense B] [--embd B] [--workspace B] [--state B] [--pool B]
```

| Flag | Default | Effect |
|---|---|---|
| `--max-context N` | 20,480 | Context the KV cache must hold. |
| `--vram BYTES` | 12,000,000,000 (RTX 5070 nominal) | Card's free VRAM. |
| `--dense B`, `--embd B`, `--workspace B` | 0 | Fill `Costs`. |
| `--state B` | `state_bytes(Geometry{})` (about 117.7 MB) | GDN state; deliberately not 0 (`plan_main.cpp:56-58`). |
| `--pool B` | | Sets the same variable as `--vram` (`:40-43`). |
| `--help` | | Usage text. |

Flow (`plan_main.cpp:59-72`): `pool = free_vram - dense - embd` (0 if negative); `make_plan(ctx, Geometry{}, costs,
pool ? pool : vram_pool_bytes())`; print `to_string(plan)` and exit 0, or print the `DoesNotClose` message to stderr
and exit 1. Unknown arguments exit 2.

## Callers

- `strata-plan` (above).
- `src/core/device_main.cpp:87` (`strata-device`): prints `make_plan(20480, Geometry{}, Costs{})` next to the card's
  free VRAM with a FITS / DOES NOT FIT line.
- The engine (`src/program/generate.cpp:31`) includes only `vram_plan.hpp`, not `plan.hpp`.

## Configuration

Only the CLI flags above. No environment variables or macros.

## Model-specific behaviour

`plan.hpp`'s `Geometry` is Qwen3.8-Flash-Next only (QSA layers, indexer, 48 layers, 1,382,400-byte Q2_0-era expert
blob). It does not describe Qwen3.6-35B-A3B; the Qwen3.6 port added `vram_plan.hpp`, which takes every size from the
model file's census instead of constants.

## Tests

There is no unit test of `plan.hpp` in this checkout. `vram_plan_test` (`tests/core/vram_plan_test.cpp`, registered at
`CMakeLists.txt:273-277`) tests `vram_plan.hpp` only. `cuda_device_selftest` runs `strata-device --selftest`, which
prints this plan but does not assert on it.

## Gotchas

- When `--dense` or `--embd` is given, they are subtracted from the VRAM figure in `plan_main.cpp` **and** counted
  again inside `make_plan`'s `fixed`, so they are charged twice. With the defaults (0) this has no effect.
- `--pool` is not a separate pool override: it sets `free_vram`, from which dense and embedding are still subtracted
  (the comment says it is for reproducing standard-tier numbers exactly).
- `Plan::dram_experts` is set to `dense_bytes + embd_bytes` (`plan.hpp:136`, commented "costs that never enter the
  pool") but `to_string` labels it "expert arena (DRAM)". The field does not hold the expert arena's size.
- `Plan::notes` is printed but `make_plan` never fills it.
- `vram_plan.hpp`'s header says "`strata-plan` can print it for any budget", but `plan_main.cpp` does not include or
  call `vram_plan.hpp`; it prints the older `plan.hpp` plan only.
- The header refers to `phases/00-README.md` and `docs/semantics.md`, which are not in this checkout.

## See also

- [../core/vram-plan.md](../core/vram-plan.md): the planner the engine uses
- [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) section 4.2 (the VRAM plan for Qwen3.6)
- [docs/qwen36-port/step2-notes.md](../../qwen36-port/step2-notes.md)
- [Libraries overview](README.md)
