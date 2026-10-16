[Codebase docs](../README.md) › [Program](README.md) › Start-up sequence

# Start-up sequence

**Purpose:** what `main` does between `argv` and the line `strata generate: session is up` (`:4215`), in order.
**Covers:** `src/program/generate.cpp:1281-4322` (plus the helper functions at `:108-1277` it calls). The order is
load-bearing: several comments explain allocations that must happen before others (WDDM limits, kernel lazy
loading, graph capture). Expert-cache sizing details are on [expert-tiers.md](expert-tiers.md) and the multi-GPU
parts on [layer-split.md](layer-split.md).

## 1. Process environment (`:1282-1304`)

- `stdout` is made unbuffered so a crash does not lose already-printed lines.
- AMD + `STRATA_ARENA_MMAP=1` (Linux): `GPU_PINNED_MIN_XFER_SIZE=1000000` unless already set, so ROCclr stages large
  copies instead of pinning mapped expert pages permanently.
- `CUDA_MODULE_LOADING=EAGER` unless set: every kernel is loaded at context creation (~30 MB), so a kernel first used
  mid-prompt cannot fail for lack of VRAM after the expert cache took the rest.

## 2. Options and validation (`:1305-2076`)

1. argv into `Options` ([cli-flags.md](cli-flags.md)); `set_coupled_draft`, `set_peer_portable` (`:1594-1595`).
2. Layer-split parsing and its device checks, `STRATA_SPLIT_OWN`, helper-GPU device numbering (`:1600-1768`); on HIP
   every used GPU's architecture is checked against the binary (`gpu_arch_problem`, `:1769-1784`).
3. `--prefill auto` without a profile or with `--no-prefill-borrow` becomes a fixed 2048 chunk. In `--serve` without
   `--tokens`, the dummy prompt is `{248045}` with `--max-new 1` and `--stop-eos` (`:1789-1794`).
4. **KV mode** (`:1813-1835`): `--kv` sets the process-wide QSA flags `qsa_set_kv_int8`, `qsa_set_kv_q4`,
   `qsa_set_kv_hybrid`; `STRATA_KV_ROT=1` routes INT8 K/V through the Hadamard rotation (`qsa_set_kv_int8_rotate`);
   `--kv-resident` sets `qsa_set_kv_resident`. These must be set before any session is initialised.
5. The suffix-draft window adjustment (`:1839-1842`, see [speculative-decoding.md](speculative-decoding.md)) and
   `layer_set_shared_early(!--shared-late)`.
6. **`--native` expansion** (`:1844-1905`): all shards, PLE shard discovery, and the model-has-PLE probe (a quick
   `read_model_geometry`); for a model with no per-layer token embedding (Qwen3.6, `qwen35moe`) `--no-ple` is set
   and `--ple-gguf` is refused.
7. Cross-flag checks (`:1906-2041`), rope-knob range checks (`:1946-1982`).
8. **CPU ISA** (`:2043-2076`): AVX2 + FMA + F16C is required unless the binary is the older-CPU build
   (`isa_floor_build()` non-empty, built with `STRATA_ISA_FLOOR=avx|none`), which instead checks AVX or SSE4.2 and
   warns. `expert_set_oracle_q8_0` is applied here.

## 3. Model detection and geometry (`:2083-2127`)

There is no model-type flag. With `--native`, `read_model_geometry(o.native_shards.front(), g, err)` reads the
`<arch>.*` metadata keys of the first shard (`include/strata/core/model_geometry.hpp:64-174`):

| `general.architecture` | `g.arch` | Model |
| --- | --- | --- |
| `qwen4exp` | `ModelArch::Qwen4Exp` | Qwen3.8-Flash-Next (and its pruned/fine-tuned variants) |
| `qwen35moe` | `ModelArch::Qwen35Moe` | Qwen3.6-35B-A3B |

Without `--native` the default-constructed `ModelGeometry` (canonical Flash-Next) is used, as for the original Q2_0
pack. For a non-Flash-Next geometry the full geometry text is printed. Then:

- `--kv-resident` is refused when `qsa_kv_streaming_applies` says the model has no selected attention (`:2101`).
- `check_model_tensor_forms` refuses fused tensor layouts by name (`:2106-2115`).
- `K = g.n_expert_used` (router top-k).
- `expert_layout_load(pack, n_layers, n_expert, n_embd, n_ff)` reads the pack's expert layout; `native_pack =
  expert_layout().native` (an IQ/GGUF-form pack) decides much of what follows (`:2121-2127`).

Model-specific branches later in start-up:

An unknown architecture string is an error that names both supported ones. The geometry itself differs in ways the
driver depends on: Qwen3.6 has `hc = 1` (plain residual), `has_ple = false`, and no sparse indexer (dense attention,
so `--kv-resident` streaming does not apply).

| Where | Flash-Next (`Qwen4Exp`) | Qwen3.6 (`Qwen35Moe`) |
| --- | --- | --- |
| VRAM plan (`:2190-2328`) | not made | `strata::plan::make_vram_plan`; refuses or lowers the context |
| No drafter, one-shot | native pack needs `--spec >= 2` | `--spec` silently raised to 2 (`:2323-2327`) |
| `check_all` (`:2606-2614`) | not run | every weight the kernels read checked by name/form |
| PLE | required (`--ple-gguf`) | none |
| MTP geometry (`:3355-3356`) | canonical (the canonical model's MTP head) | follows the target `g` |
| Native dense load range | | bounded to `[0, n_layers)` so the `-MTP` file's `blk.40` is skipped (`:2595`) |
| Control vector hint | warns unless `controlvector.model_hint == qwen4exp` | same warning |

## 4. VRAM cap, RAM cap (`:2128-2183`)

`--vram-cap MIB` allocates a blocker of `total - cap` before anything else, sets `g_vram_cap_bytes` and
`g_vram_blocker_bytes`, and checks that `vram_info` (`:128-139`) now reports what a card of that size would (within
64 MiB) or exits. From here every planner reads free VRAM through `vram_info`. `--ram-cap` calls
`set_host_ram_cap` and clamps an explicit `--resident-budget-gib`.

## 5. The Qwen3.6 VRAM plan (`:2190-2328`)

Only for `qwen35moe`. A `VramPlanInput` is assembled from a census of the GGUF tensors: dense weights (excluding
`token_embd.weight`, which lives in mapped host memory), experts (`*_exps.weight`), the draft block (`blk.N` with
`N >= n_layers`); the drafter's bytes, head subset (from `MTP_DIR/draft_vocab.bin`) and logits when `--mtp` is given;
KV bytes per token from `--kv` (fp16 2 B, int8 1 + 2/64, q4_0 18/32, k8v4 816/1024 per element; the drafter keeps
INT8 under k8v4); the session term (RoPE table, GDN state, 36 MiB scratch); a 300 MiB reserve unless
`--vram-reserve-mib` is given; prompt buffers only when not borrowing; free VRAM; available RAM and headroom.
`make_vram_plan` (`include/strata/plan/vram_plan.hpp:85`) returns `fits`, `lowered`, `context` and a refusal; the
plan text is printed. A plan that does not fit exits 2; a lowered plan overwrites `--max-context`.

Still in this block: if the resident arena (`lay.total + max_blob`) exceeds the cgroup limit or available RAM, the
engine switches itself to `--mmap-experts` with profile-ranked RAM copies (the `--resident-experts` mode), unless a
shared arena, peer device or remote-expert opt is in use (`:2291-2312`). The multi-GPU helpers are refused for a
model whose `n_embd` is not the canonical expert width (`:2313-2318`).

## 6. Pre-load checks and probes (`:2329-2487`)

- Every layer's expert formats must have GPU kernels (`native_expert_supported`, `:2333-2343`).
- Helper GPU contexts are created early (CUDA1 always, others with `STRATA_EARLY_REMOTE_CONTEXTS=1`) so their
  context creation precedes the weight arena (`:2348-2360`).
- **PCIe probe** (`:2366-2380`): with no `--pcie-frac`, a native pack measures H2D bandwidth as the best of four
  256 MiB bursts (`probe_pcie_h2d_gbps`, `:1212`) and sets `pcie_frac = 0.55 * min(1, GB/s / 20)`
  (`pcie_frac_for_gbps`, `:1275`); the canonical pack keeps 0.2.
- The canonical pack requires AVX-512 CPU kernels (`cpu_require_expert_support`); a native pack logs which kernel
  family it falls back to (`STRATA_NO_IQ256` forces ggml-cpu vec_dot).
- **Rope config** resolution (`:2394-2487`), then `rope_scaling_set`.

## 7. Weights (`:2488-2614`)

1. Native pack: `NativeEmbed::load` (from `--embd-gguf` or the shards), `set_native_embed` (`:2488-2506`). A native
   pack refuses unless `--native`, `--spec >= 2`, no `--keep-canonical`, and `--prefill` for a multi-token prompt.
2. The **skip set**: tensors served natively are not loaded canonically (`NativeDense::served_names`), the head when
   `--native-head-gguf`, `token_embd.weight` for a native pack; `keep_unquantized_ple_key` for BF16-key packs.
   Stage trimming adds other stages' dense tensors (`add_foreign`, `:2541-2560`).
3. `WeightTable::pool_bytes` then one `cudaMalloc` for the **weight arena** - the first large allocation, with a
   message naming free VRAM on failure - and `WeightTable::load` (`:2561-2588`).
4. `NativeDense::load` for the projections, bounded to the trunk's layers (`:2590-2601`).
5. Qwen3.6: `check_all(wt, g)` (`:2606-2614`).

## 8. Kernel switches, vision table, control vectors (`:2616-2662`)

The A/B switches are pushed into the layer/kernel modules (`layer_set_fast_attn`, `layer_set_fused_gr` - off when a
per-layer dump or `--stage-timing`/`--gpu-stages` needs the unfused layout - `native_*_set_enabled`, ...). With
`--vision` an identity M-RoPE table of `max_context + 64` cells is uploaded and registered (`mrope_table_set`) before
any graph captures a rope kernel. Control vectors are loaded before `session_init` so captured graphs include them.

## 9. Stream, doorbell, PLE (`:2669-2846`)

- `main_stream` is a non-blocking stream; all session calls use it (the comment records an ~8x per-launch penalty
  on the legacy default stream under WDDM).
- `--dump-halves` staging is armed here, before capture.
- `doorbell_init(g, K, db)`: the pinned-memory channel through which routed ids reach the CPU pool.
- **PLE** (`:2722-2839`): `PleTable::open` with `PleIoOptions` (mode, lock, inflight, row cache, I/O thread, SSD
  keep-alive from `STRATA_SSD_KEEPALIVE` / `STRATA_SSD_KEEPALIVE_WINDOW`), a rotational-disk warning for
  `--ple-io direct` on Linux, then the six `blk.1.ple_*` weights wired into `ss.ple.w` (a BF16 or quantized key,
  optional native key, BF16 value, norms, and a conv1d that must be stored as F16 - refused otherwise). The history
  pointer waits for `session_init`.
- `d_parts` (`K x n_embd` floats): the CPU pool's output staging.

## 10. Layer-split stages, profile, sessions (`:2848-3340`)

Covered in [layer-split.md](layer-split.md): `--split-skip-if-fits`, per-stage weights, the profile read (with
`STRATA_PEER_HOT` reordering for `--peer-device`), the `auto` placement search, and the profile's distribution to
stages. Then **CUDA0's session**: `session_bytes(g, max_context, K, 0, hi0)` and `session_init` sized to CUDA0's own
layer range (the "carve"); the KV-streaming line; `ss.ple.hist` wired; `session_zero` (`:3210-3243`). Each stage
gets its session; the last stage loads the native head. **`--batch`** carves `o.batch` more sessions per stage,
reducing the count when they do not fit and turning batching off below 2 (`:3273-3340`).

## 11. MTP drafter, head, logits (`:3342-3397`)

These are allocated **before** the expert cache and before the host expert arena is registered (WDDM refused
allocations after tens of GiB were mapped; issue #620). `MtpDrafter::load(o.mtp, draft_geometry, session, o.spec,
err, o.mtp_window)` on the last stage's device; `mtp.set_ple_session(&ss)`. The native head is loaded on CUDA0 unless
split; `n_vocab = output.weight ne1`; `d_logits`. Remaining helper contexts are created.

## 12. Expert source and CPU pool (`:3399-3543`)

The GPU name/compute capability (or HIP arch) is printed, a CUDA runtime/headers mismatch is warned about (#542), and
`device_code_error()` refuses a binary without code for the card. Then one of:

- `--mmap-experts`: `FileExpertSource` (`src.open`), mapping `experts.bin` or, for a native pack without it, the GGUF
  shards in place; `STRATA_FETCH_THREADS`; unbuffered reads with `--resident-budget-gib` or
  `STRATA_UNBUFFERED_LOAD`.
- default: `ArenaExpertSource` (`arena_src.open(..., threads=6, err, pin_limit, shared_arena)`), a resident copy of
  every expert, pinned unless capped (`STRATA_ARENA_PIN_GIB`, the 8 GiB WDDM multi-GPU cap, `--ram-cap`). Its load
  rate is printed; on Windows a rate under 0.2 GiB/s prints a Task Scheduler hint.

`ExpertPool pool(o.pool_workers, pin=true, host_works=!--no-host-worker, affinity)`.

## 13. Expert cache, tiers, hit path (`:3545-4153`)

Detailed in [expert-tiers.md](expert-tiers.md): `--expert-cache auto` sizing, sized slots for native packs,
`--vram-elastic`, the open/zero/re-measure loop, profile fill with a read-back verify, stage caches, helper and peer
tiers, the routing-aware prefetch (`RouterLookahead`), the R4 hit buffers, `--stage-timing` names, the routing dump
file, and the `pool_fn`/`hit_fn` hooks.

## 14. Graphs and run-time buffers (`:4169-4322`)

- `session_capture` captures the per-layer graphs for a **canonical** pack only (a native pack runs verify windows
  only), covering CUDA0's layer range under a split (`:4171-4180`).
- Refusals for `--no-capture` without `--no-pool`, and with `--dump-layers`/`--dump-halves`.
- `run_head`: `lm_head`, or `lm_head_mix` + `NativeHead::run` (`:4216-4221`); `d_emb`, `d_next`.
- `put_input` (`:4242-4258`): `embed_row`, then `session_zero` at position 0 or a broadcast of the embedding into the
  `hc` residual streams.
- `SamplerParams` from the CLI; the sampling line is printed in one-shot mode.
- `--dump-logits` header prepared (`row_count`), `--dump-layers` pinned staging.

After this point `main` branches: measurement modes ([diagnostics.md](diagnostics.md)), the token graph and device
residency table (`:4584-4689`, [expert-tiers.md](expert-tiers.md)), prompt-lend planning and the resident RAM mode,
then `--serve` ([serve-mode.md](serve-mode.md)) or one-shot generation ([generation-loop.md](generation-loop.md)).

## Gotchas

- `mem_mark` lines (`STRATA_TRACE=1`) print free VRAM after each major allocation; useful when a card ends LOW.
- The PLE/head/drafter-before-cache ordering exists because the auto cache takes "free minus reserve": anything
  allocated after it eats the reserve.
- Most failures here exit 1 (runtime) or 2 (bad configuration); the front end reports the stderr text.

## See also

- [../core/README.md](../core/README.md) (sessions, weights, MTP), [../libs/README.md](../libs/README.md)
  (VRAM planner, GGUF reader, platform memory)
- [docs/qwen36-port/step4-runner.md](../../qwen36-port/step4-runner.md) - how the Qwen3.6 gate was lifted
- [docs/qwen36-port/step2-notes.md](../../qwen36-port/step2-notes.md), [geometry-inventory.md](../../qwen36-port/geometry-inventory.md)
