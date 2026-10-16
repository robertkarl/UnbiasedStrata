[Codebase docs](../README.md) › Core

# Core engine runtime

The core is the GPU-side inference engine that every model run goes through. It reads the model's geometry
from the GGUF metadata, loads the dense (non-expert) weights into one VRAM arena, carves per-sequence state
(GDN recurrences, KV caches, scratch), and runs one token, or a speculative window of up to 8 tokens, through
every layer as captured CUDA graphs. It also hosts the MTP draft layer. The routed experts are **not** computed
here: each layer publishes its router output through a mapped-pinned "doorbell", and the expert side
(CPU pool, VRAM expert cache, remote/peer experts; see [experts/](experts/README.md)) fills the `parts`
buffer that the layer's second graph combines. The program layer (`src/program/generate.cpp`, which runs the
CLI and `--serve`) drives the core: it reads the geometry, makes the VRAM plan, loads weights, builds a
`SessionState`, and then calls `session_loop` / `session_run_token` (plain decode) or `Verifier::run` +
`MtpDrafter::draft` (speculative decode). The prompt path (`src/prefill/`) and the conversation cache
([conversation/](conversation/README.md)) reuse the same `SessionState`.

## Contents

| Page | What it covers |
| --- | --- |
| [model-geometry.md](model-geometry.md) | `ModelGeometry`, `read_model_geometry`, Flash-Next (`qwen4exp`) against Qwen3.6 (`qwen35moe`), the QSA shape builder |
| [weights-and-layout.md](weights-and-layout.md) | The dense-weight loader (`WeightTable`, `index.txt`, engine forms) and the per-layer shape/form checks (`check_all`) |
| [dense-and-head.md](dense-and-head.md) | Native GGUF projections (`NativeDense`), the native output head (`NativeHead`) and embedding table (`NativeEmbed`) |
| [layer-forward.md](layer-forward.md) | One layer's decode forward pass: GDN mixer, QSA/dense attention, router, shared expert, combine, the GR block and the plain block, PLE, embedding and head, the doorbell |
| [session.md](session.md) | `SessionState`, the session arena carve, KV cache formats (fp16 / int8 / q4_0 / k8v4), KV streaming, and the three decode drivers |
| [cuda-graphs.md](cuda-graphs.md) | How graphs are captured and replayed: per-layer pre/post graphs, the whole-token graph, the verify/MTP graphs, `GraphRegistry`, and the rules that keep them correct |
| [verify.md](verify.md) | The speculative verify window (`Verifier`): T tokens in one graph, commit, batch windows, layer split |
| [mtp-draft-head.md](mtp-draft-head.md) | The MTP draft layer (`MtpDrafter`) for both models and coupled draft sampling |
| [vram-plan.md](vram-plan.md) | `strata/plan/vram_plan.hpp`: the fixed set, context lowering, expert slots, RAM tier |
| [device-and-memory.md](device-and-memory.md) | `DeviceArena`, device checks, `PinnedArena` and expert-file loading, `OnDevice`, `STRATA_EMULATE_CC` |
| [standalone-tools.md](standalone-tools.md) | `strata-device`, `strata-load`, `strata-concurrent`, `strata-overlap` |
| [experts/](experts/README.md) | (other writer) expert sources, the VRAM expert cache, remote and peer experts, the hit hook, progress |
| [conversation/](conversation/README.md) | (other writer) conversation buffer, cache, snapshot and memory |

## Source map

Paths are relative to this page. Line counts are `wc -l` at the time of writing.

| File | Lines | Role |
| --- | ---: | --- |
| [include/strata/core/layout.hpp](../../../include/strata/core/layout.hpp) | 139 | `ModelArch`, `ModelGeometry`, `is_qsa_layer`, `LayerView`, `check_layer` / `check_all` declarations |
| [src/core/layout.cpp](../../../src/core/layout.cpp) | 189 | Per-layer tensor name/shape/engine-form checks |
| [include/strata/core/model_geometry.hpp](../../../include/strata/core/model_geometry.hpp) | 262 | Header-only: geometry from GGUF metadata, fused-form refusal, geometry text |
| [include/strata/core/qsa_shapes.hpp](../../../include/strata/core/qsa_shapes.hpp) | 30 | The single builder of the attention kernels' `QsaShapes` from a geometry |
| [include/strata/core/weights.hpp](../../../include/strata/core/weights.hpp) | 144 | `WeightKind`, `WeightRef`, `WeightTable` |
| [src/core/weights.cpp](../../../src/core/weights.cpp) | 479 | The pack loader (`index.txt` parser, plane segments, staged upload) |
| [include/strata/core/native_dense.hpp](../../../include/strata/core/native_dense.hpp) | 45 | `NativeDense`: GGUF-form projections attached to canonical `WeightRef`s |
| [src/core/native_dense.cpp](../../../src/core/native_dense.cpp) | 210 | Eligibility, GGUF validation, upload |
| [include/strata/core/native_head.hpp](../../../include/strata/core/native_head.hpp) | 68 | `NativeHead`, `NativeEmbed`, `set_native_embed` |
| [src/core/native_head.cpp](../../../src/core/native_head.cpp) | 198 | Head upload/run, embedding table in mapped pinned memory |
| [include/strata/core/layer.hpp](../../../include/strata/core/layer.hpp) | 658 | Buffers, states and functions of one layer (GDN, QSA, MoE, block, doorbell, PLE, head) |
| [src/core/layer.cpp](../../../src/core/layer.cpp) | 1517 | Their implementation |
| [include/strata/core/session.hpp](../../../include/strata/core/session.hpp) | 436 | `SessionState`, session graphs, `session_loop`, the token graph |
| [src/core/session.cpp](../../../src/core/session.cpp) | 967 | Session carve, capture, host loop, token graph |
| [include/strata/core/graph.hpp](../../../include/strata/core/graph.hpp) | 117 | `CapturedGraph`, `GraphRegistry` |
| [src/core/graph.cpp](../../../src/core/graph.cpp) | 146 | Their implementation |
| [include/strata/core/verify.hpp](../../../include/strata/core/verify.hpp) | 371 | `Verifier`: the speculative verify window |
| [src/core/verify.cpp](../../../src/core/verify.cpp) | 2268 | Window recording, run, commit, batch windows, layer split |
| [include/strata/core/mtp.hpp](../../../include/strata/core/mtp.hpp) | 228 | `MtpDrafter` |
| [src/core/mtp.cpp](../../../src/core/mtp.cpp) | 1372 | Draft layer load, bind, forward recording (both models), prefill, draft rounds |
| [include/strata/core/coupled_draft.hpp](../../../include/strata/core/coupled_draft.hpp) | 90 | Coupled draft sampling's counter and penalty-ring arithmetic |
| [src/core/coupled_draft_test.cpp](../../../src/core/coupled_draft_test.cpp) | 115 | CPU unit test of that arithmetic |
| [include/strata/core/device.hpp](../../../include/strata/core/device.hpp) | 95 | `DeviceInfo`, device checks, `CudaError`, `DeviceArena` |
| [src/core/device.cu](../../../src/core/device.cu) | 344 | Device queries, arch/cc checks, arena, Windows HIP free-VRAM budget |
| [include/strata/core/pinned.hpp](../../../include/strata/core/pinned.hpp) | 152 | `PinnedArena`, expert-file loaders, `stream_bandwidth` |
| [src/core/pinned.cu](../../../src/core/pinned.cu) | 725 | Large pages / THP, CUDA registration (whole or sliced), shared-file backing, readers |
| [src/core/pinned_shared_test.cpp](../../../src/core/pinned_shared_test.cpp) | 61 | Linux test of the `MAP_SHARED` backing |
| [include/strata/core/on_device.hpp](../../../include/strata/core/on_device.hpp) | 22 | `OnDevice` RAII device switch (multi-GPU) |
| [include/strata/core/emulate.hpp](../../../include/strata/core/emulate.hpp) | 36 | `STRATA_EMULATE_CC` |
| [src/core/device_main.cpp](../../../src/core/device_main.cpp) | 122 | `strata-device` |
| [src/core/load_main.cpp](../../../src/core/load_main.cpp) | 122 | `strata-load` |
| [src/core/concurrent_main.cpp](../../../src/core/concurrent_main.cpp) | 197 | `strata-concurrent` |
| [src/core/overlap_main.cpp](../../../src/core/overlap_main.cpp) | 169 | `strata-overlap` |
| [include/strata/plan/vram_plan.hpp](../../../include/strata/plan/vram_plan.hpp) | 178 | Header-only VRAM planner (part of the `strata_plan` interface library) |
| [tests/core/model_geometry_test.cpp](../../../tests/core/model_geometry_test.cpp) | 273 | Geometry from synthetic and real GGUFs |
| [tests/core/vram_plan_test.cpp](../../../tests/core/vram_plan_test.cpp) | 223 | VRAM planner against the study's table 4.3 |

Build targets (`CMakeLists.txt`): `strata_core` (static; `device.cu`, `pinned.cu`, `src/platform/memory.cpp`,
`graph.cpp`, `weights.cpp`, `layout.cpp`) and `strata_engine` (static; `layer.cpp`, `session.cpp`, `native_head.cpp`,
`native_dense.cpp`, `verify.cpp`, `mtp.cpp` plus the expert and conversation sources). The split exists because
`layer.cpp` calls kernels and `strata_kernels` already links `strata_core` (CMake comment above `strata_engine`).
Both are built only with `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP`. `strata_plan` and the header-only
geometry code build without a GPU toolkit.

## Key concepts and data flow

### Two models, one engine

| | Qwen3.8-Flash-Next (`qwen4exp`) | Qwen3.6-35B-A3B (`qwen35moe`) |
| --- | --- | --- |
| Trunk layers | 48 (36 GDN + 12 attention) | 40 (30 GDN + 10 attention) |
| Attention | QSA: sparse, with an indexer and top-k block selection | Dense, no indexer |
| Residual | Hyper-connections, `hc = 4` streams (`gr_read` / `gr_write`) | Plain pre-norm residual, `hc = 1` |
| PLE (layer-1 n-gram embedding) | Yes | No |
| Experts | 512, top-10, `n_ff` 640 | 256, top-8, `n_ff` 512 |
| GDN output gate | sigmoid(z) | SiLU(z) |

Everything that differs is decided by `ModelGeometry` (see [model-geometry.md](model-geometry.md)); the code
branches on `g.hc == 1`, `g.has_indexer()`, `g.has_ple` and `g.gdn_gate()`.

### Start-up order (as `src/program/generate.cpp` does it)

```
read_model_geometry (GGUF metadata)  ->  check_model_tensor_forms
   -> [Qwen3.6] make_vram_plan (may lower --max-context or refuse)
   -> NativeEmbed::load, NativeDense::served_names (skip set)
   -> WeightTable::pool_bytes -> cudaMalloc(arena) -> WeightTable::load  (dense weights in VRAM)
   -> NativeDense::load (attach GGUF-form projections)  -> [Qwen3.6] check_all
   -> session_bytes / session_init / session_zero        (per-sequence state)
   -> MtpDrafter::load, NativeHead::load                 (optional)
   -> expert tier sized from what is left (experts/)
   -> session_capture or session_capture_token, Verifier::init, MtpDrafter::bind
```

### One decode token (per-layer graph form, `session_loop`)

```
for each layer l:
  pre[l]  (captured):  [PLE at l==1] -> read/norm -> mixer (GDN | attention) -> residual write
                       -> read/norm -> router top-k -> DOORBELL RING (x, ids, weights to mapped memory)
                       -> shared expert (overlaps the host's pool)
  host:   poll ring -> hits(Launch) [GPU-resident experts] -> pool() [CPU experts]
          -> H2D misses into parts -> hits(Combine)
  post[l] (captured):  combine(parts, weights) + shared -> residual write
head: lm_head_mix -> output projection (canonical or NativeHead) -> logits
```

The whole-token graph (`session_capture_token` / `session_run_token`) puts all layers into one graph and makes
the GPU itself wait for the host's flag (`doorbell_wait`), which removes ~96 graph launches per token. The
speculative path replaces all of this with one window graph for T tokens (`Verifier`) plus the MTP drafter.

### Memory tiers handled here

* **VRAM:** the dense-weight arena (one `cudaMalloc` sized by `WeightTable::pool_bytes`), the session arena, native projections, the native head,
  the MTP draft layer; what remains goes to the expert cache (sized by [vram-plan.md](vram-plan.md) for Qwen3.6).
* **Pinned host:** the expert arena (`PinnedArena`), the doorbell and all mapped staging buffers, KV streaming's
  host copy, the native embedding table.

## See also

* [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works" - the upstream summary of the tiers.
* [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) and [docs/PLAN-qwen36-35b-a3b.md](../../PLAN-qwen36-35b-a3b.md) - the Qwen3.6 port design and plan.
* [docs/qwen36-port/](../../qwen36-port/) - per-step notes (geometry inventory, block, GDN, attention, drafter, runner).
