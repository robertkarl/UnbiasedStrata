[Codebase docs](../README.md) › [Core](README.md) › CUDA graphs

# CUDA graphs in the core

## Purpose

The engine is launch-bound at decode: a layer is ~45 small kernels, and the header of `session.hpp` records that
replaying a captured block (1.585 ms) beat the same work as direct launches (2.393 ms). Every hot path in the core
is therefore recorded once with stream capture and replayed. This page collects where graphs are captured, how
per-token data reaches them, and the generic helper `GraphRegistry`.

Files covered: [include/strata/core/graph.hpp](../../../include/strata/core/graph.hpp),
[src/core/graph.cpp](../../../src/core/graph.cpp); the capture sites in `session.cpp`, `verify.cpp` and `mtp.cpp`.

## The three rules (from `graph.hpp`'s header, each backed by a measurement)

1. **The host's wait must call into the driver.** A host spin that only reads memory may never see the GPU's write
   (on Windows the driver batches submission). Every wait in the core polls `cudaEventQuery` / `cudaStreamQuery`
   alongside the memory flag.
2. **A graph re-reads its input buffers at replay but not its kernel arguments.** Anything that changes per token
   (position, step counts, token ids, the doorbell sequence) must be data in a fixed device or mapped buffer.
   Examples: `QsaState::host_step` / `host_pos`, the verify window's mapped staging (`h_tok_`, `h_step_`, `h_pos_`,
   `h_commit_`), the MTP drafter's `h_step_` / `h_pos_` / `h_row_`.
3. **Fixed addresses are the caller's obligation.** Every buffer a captured body touches must exist before capture
   and never move. `session_capture` records `gr.parts_dev` because the graphs baked that pointer;
   `session_run_token` refuses a staging buffer other than the captured one.

Additional facts recorded in the code: `cudaEventRecord` inside a capture is silently dropped (so the doorbell is a
kernel, and stage timers work only uncaptured); host functions that must run per token (PLE hash and gather,
sampling with per-request parameters) are kept outside graphs; capture uses `cudaStreamCaptureModeThreadLocal`.

## Where graphs are captured

| Owner | Graphs | Captured by | Replayed by |
| --- | --- | --- | --- |
| `SessionGraphs` (`session.hpp:137`) | `execs[l]` (pre), `posts[l]` (post); with `split`: `preA`, `preB`, `preP[0..4]` | `session_capture` (`session.cpp:188`), one stream per capture | `session_loop`, `session_replay*` |
| `TokenGraph` (`session.hpp:399`) | one graph for all layers with device-side waits | `session_capture_token` (`session.cpp:826`) | `session_run_token` |
| `Verifier` (`verify.hpp:65`) | `exec_[T]` per window size T (2..8), `commit_exec_`; batch windows in `exec_bm_` / `commit_bm_` keyed by slot rows | `Verifier::capture` (`verify.cpp:1306`), `capture_commit` (`:1369`), `capture_batch` (`:1888`), `capture_commit_batch` (`:1920`) | `Verifier::run`, `commit`, `run_slots`, `batch_launch` |
| `MtpDrafter` (`mtp.hpp:37`) | `prefill_exec_[T]`, `prefill_dev_exec_[T]`, `round_exec_[T]`, `step_exec_[j]`, and coupled variants `round_exec_c_`, `step_exec_c_` | `capture_prefill`, `capture_prefill_dev`, `capture_round`, `capture_step` (`mtp.cpp:1052-1104`) | `prefill`, `draft`, `draft_first` |

All captures are lazy (first use) and idempotent. Graphs bake weight pointers, state pointers and the session
layout, so a new session or a re-carved arena needs new captures (`session_graphs_free`, `token_graph_free`,
`~Verifier`, `~MtpDrafter` destroy them).

## Host/device handshakes inside graphs

* **Doorbell ring** (`doorbell_publish` / `doorbell_ring`): a kernel writes `x`, `ids`, `weights` and increments
  `seq` in mapped memory; the host polls it ([layer-forward.md](layer-forward.md)).
* **Device wait** (`doorbell_wait`, token graph; `wait_flag_ge`-style flags in the verify window): a kernel spins on
  a mapped flag the host writes after it has served the layer. The verify window uses several flags per layer and
  group (`h_flag_`, `h_flagA_` = GPU plan published, `h_flagB_` = PCIe DMA landed).
* `Verifier::release_gpu_waits` (`verify.hpp:83`) raises every flag to `UINT32_MAX` so a window whose host side
  gave up drains instead of leaving spin kernels resident (#267).

## `GraphRegistry` and `CapturedGraph` (`graph.hpp:51`, `:92`)

A generic capture-once / replay-many registry keyed by `(LayerType, n_tokens)`:

* `CapturedGraph::begin` / `end` wrap `cudaStreamBeginCapture` / `EndCapture` + instantiate, record a completion
  event, and treat a zero-node capture as an error (`graph.cpp:63`).
* `launch` replays without synchronising; `wait_ms(timeout)` polls the event with a real-clock deadline
  (`graph.cpp:84`).
* `GraphRegistry::record` runs the body only on the first call for a key; it checks `cudaGetLastError` after the
  body and abandons the capture on error (`graph.cpp:104`). `captures()` counts real captures.
* `reset` destroys the exec before the event, with a comment explaining why order matters.

No engine code outside `graph.cpp` uses `GraphRegistry` (the session, verify and MTP code capture directly with
`cudaStreamBeginCapture`). Its test, `graph_registry_test`, is registered from `bench/micro/graph_registry_test.cu`
only if that file exists, and `bench/` is not in this checkout. `LayerType` (`graph.hpp:46`) is still the GDN/QSA
naming used in comments.

## Configuration

| Flag / variable | Effect |
| --- | --- |
| `--no-capture` | no graphs: `session_token` direct launches |
| `--no-token-graph` | per-layer graphs + host loop instead of the token graph |
| `--gpu-stages` | capture the split/prefix graphs for stage timing |
| `STRATA_TG_FLUSH_US` | token-graph flush interval (default 2000 us) |
| `STRATA_VERIFY_NODES` | print the verify window graph's node list (CUDA builds) |
| `STRATA_VERIFY_PROFILE`, `STRATA_VERIFY_TRACE` | GPU stamps / host+GPU breadcrumbs for the verify window |

CMake: `graph.cpp` is a host translation unit that calls the runtime, so `strata_core` links the GPU runtime
publicly (comment in `CMakeLists.txt` near `strata_core`).

## Gotchas

* A captured body must only launch; a body that changes host state would do so once, at capture time.
* `layer_set_*` and `qsa_set_kv_*` switches are read at capture; changing them afterwards has no effect on existing
  graphs.
* Measurement graphs (`--gpu-stages`) disable the fused GR read (generate.cpp sets `layer_set_fused_gr` off with
  `--gpu-stages`).

## See also

* [session.md](session.md), [layer-forward.md](layer-forward.md), [verify.md](verify.md), [mtp-draft-head.md](mtp-draft-head.md).
