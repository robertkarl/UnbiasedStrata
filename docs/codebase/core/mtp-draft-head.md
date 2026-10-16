[Codebase docs](../README.md) › [Core](README.md) › MTP draft head

# The MTP draft layer (`MtpDrafter`) and coupled draft sampling

## Purpose

Both models ship one multi-token-prediction (MTP) block. `MtpDrafter` runs it on the GPU to propose up to T-1 draft
tokens per round, which the verify window then checks exactly ([verify.md](verify.md)). Draft quality only affects
speed: the verify window decides every emitted token.

Cell convention (both models): MTP cell `i` pairs the trunk's final residual at position `i` with the token at
`i + 1`, at RoPE position `i`, and predicts the token at `i + 2`; its own residual feeds the next draft step.

Files covered: [include/strata/core/mtp.hpp](../../../include/strata/core/mtp.hpp), [src/core/mtp.cpp](../../../src/core/mtp.cpp),
[include/strata/core/coupled_draft.hpp](../../../include/strata/core/coupled_draft.hpp),
[src/core/coupled_draft_test.cpp](../../../src/core/coupled_draft_test.cpp).

## The two draft blocks

**Flash-Next (`qwen4exp`, `record_forward`, `mtp.cpp:600`)** - transcribed from vLLM's `qwen4_exp` MTP (header):

```
e = fc_embedding(rms(embed(tok)) * (1+w))        h = fc_hidden per stream(rms_10240(R) * (1+w))
R = h + e -> attention hyper-connection -> attention (own K/V) -> MLP hyper-connection
  -> MoE (512 experts, top-10, shared expert) -> R += y * 2 sigmoid(inject / hc) -> final mixer -> head
```

Runtime simplifications (affect drafts only): large projections as Q8_0 through multi-column MMVQ
(`tools/mtp_rt.py` quantizes them); attention is **dense** over the cells the layer has seen (identical to the
model's sparse selection below 2,051 cells), so speculative cells never touch indexer state; all routed experts live
in VRAM and run through the grouped hit kernels.

**Qwen3.6 (`qwen35moe`, `record_forward_qwen36`, `mtp.cpp:794`)** - the `-MTP` file's nextn block through the plain
graph: `concat(rms_norm(embed(tok)) * enorm, rms_norm(h) * hnorm)` (embedding first) -> `eh_proj` -> one plain
attention + MoE block over the drafter's own dense KV -> `shared_head_norm` -> the main head. `h` is the trunk's
output-normed hidden state: the drafter applies `output_norm` itself, so callers pass the same final residual rows as
for Flash-Next. `qwen36()` (`mtp.cpp:176`) selects this path from the geometry.

## Lifecycle

| Call | Line | What it does |
| --- | --- | --- |
| `load(rt_dir, g, ss, max_t, err, window)` | `mtp.cpp:279` | Reads `dense.txt` + `dense.bin` (all dense tensors, one `cudaMalloc`), for Qwen3.6 checks `mtp.json` and the required tensors (`check_qwen36_pack`, `:179`) and reads `native_experts.txt`; loads `experts.bin` (all experts resident); creates the drafter's own `QsaState` K/V borrowing the session's RoPE table; carves all buffers for `max_t` rows; creates a non-blocking stream. Records the device it ran on |
| `bind_bytes(head_row_bytes, n_vocab)` | `mtp.cpp:451` | VRAM `bind` will take (draft head subset + logits), so the expert cache can reserve it first |
| `bind(wt, head, window_R, err)` | `mtp.cpp:561` | Requires the native head; allocates `max_t x n_vocab` logits; if `rt_dir/draft_vocab.bin` exists, gathers a **draft head** over that token subset from the native head's rows; sets up coupled sampling when enabled. `window_R` is the verify window's final residuals (`Verifier::final_R_all`) |
| `prefill(R_rows, next_tokens, n, cell0, err)` | `mtp.cpp:1131` | Writes the prompt's K/V in batches of up to `max_t` rows, skipping cells the attention window can never reach again (`first_needed`) |
| `draft(T, tokens, p, a, drafts, err, probs, min_p, n_drafts)` | `mtp.cpp:1233` | One round: a catch-up over the window's T cells from `p` (rows = the window's final residuals, tokens = its argmaxes), then the draft chain from accepted row `a` |
| `draft_first(T, R_row, token, cell, drafts, ...)` | `mtp.cpp:1357` | The first round from one cell |
| `debug_row(...)` | `mtp.cpp:1004` | Test harness: one row outside any graph, handing every seam (by llama.cpp's cb() names) to a callback; `DebugOpts` holds negative controls |

`draft` launches `round_exec_[T]` and then `step_exec_[j]` graphs. With `min_p <= 0` all steps are launched at once
and the host reads each draft as it lands in mapped memory (`h_out_`), prefetching the PLE rows of each draft; with
`min_p > 0` it launches one step at a time and stops when a draft's probability falls below `min_p`. Unused draft
slots are zeroed; `n_drafts` returns how many were made.

## The drafter's K/V

Created in `load` with `qsa_state_init(..., share_rope = primary state, ring_cells)`:

* `window` (`--mtp-window`, default 32768; 0 = every cell): with KV streaming on and a window shorter than the
  context, the K/V is a ring of `window + 4*max_t + 64` cells over a pinned host copy (`kv_mode` 2), refilled by
  `kv_restore(upto)` on a conversation resume. If no pinned RAM is left, it falls back to fully resident.
* Under `--kv k8v4` the drafter's state is plain INT8 (the globals are toggled around `qsa_state_init`,
  `mtp.cpp:357-378`).
* `kv_state()` / `kv_state_rw()` expose it to the serve state hash and to `Prefill::draft_kv`
  (`src/prefill/prefill.cpp:1057`), which computes the prompt K/V in batches using `tensor_f32`, `tensor_bf16`,
  `tensor_q8`, `tensor_native`.

## Coupled draft sampling (`coupled_draft.hpp`)

Off by default; on with `STRATA_SPEC_COUPLED=1` or `--coupled-draft` (`set_coupled_draft`, `coupled_draft.hpp:65`).
For a sampled request the drafter **samples** with the target's sampler chain and the **same Philox uniform** the
verify row will use, instead of proposing its argmax, so draft and target picks agree more often; verification is
still exact-match against the target's sample, so the text depends only on the seed.

The arithmetic, all in the header and checked by the test:

* `coupled_draft_cell(p, a, j) = p + a + j` (`:70`): the MTP cell of chain step j after a window at p with a drafts
  accepted.
* `coupled_draft_counter(cell) = cell + 1` (`:73`): the Philox counter of the verify row that checks that draft.
* Penalties: a device ring of `kCoupledHistCap` (4096) + `max_t` slots; the host stages the base history
  (`coupled_hist_base`, `set_draft_history`) and draft j writes at `cap + j`, so draft j's window equals the verify
  row's `penalty_rows`.

`set_draft_sampling(sp)` installs the request's parameters; greedy requests keep the argmax graphs. Coupled rounds
use their own graphs (`round_exec_c_`, `step_exec_c_`).

## Callers

`src/program/generate.cpp`: `mtp.load` (~3359, before the expert cache is sized), `mtp.bind` (~5483, ~8272),
`mtp.prefill` / `draft_first` / `draft` in the speculative loops (~7598, ~8568); `src/prefill/prefill.cpp`
(`draft_kv`).

## Configuration

| Flag / variable | Effect |
| --- | --- |
| `--mtp DIR` | the draft pack (`tools/mtp_rt.py` for Flash-Next, `tools/mtp_qwen36_pack.py` for Qwen3.6); ignored without `--spec T` (T >= 2) |
| `--spec T`, `--mtp-max-t M` | window size; cap on MTP windows (longer ones come from the suffix drafter) |
| `--mtp-window N` | the drafter's attention window (default 32768) |
| `--coupled-draft` / `--no-coupled-draft`, `STRATA_SPEC_COUPLED` | coupled draft sampling |
| `STRATA_MTP_PREFILL_SYNC` | 1 = per-group synced prompt loop (default on HIP), 0 = batched upload (default on CUDA) |
| `STRATA_FUSE_HEAD_GR` | fuse the head's GR read (A/B) |

## Model-specific behaviour

* Flash-Next: 512 canonical Q2_0 expert blobs, hyper-connections, PLE prefetch for drafts.
* Qwen3.6: 256 native GGUF expert blobs (`expert_gu_type`, `expert_d_type`, `expert_blob_bytes`), plain block, the
  pack's `mtp.json` must match the trunk geometry field by field.
* VRAM plan: the drafter's weights, K/V per token, head and ids are part of the fixed set ([vram-plan.md](vram-plan.md)).

## Tests

* `qwen36_mtp_test` (`tests/cuda/qwen36_mtp_test.cpp`, fixture `tests/python/qwen36_mtp_fixture.py`, needs CUDA):
  every seam of `debug_row` against the oracle's dumps for 75 positions; `--real` mode compares the drafter's top-1
  against llama.cpp's MTP graph on the real file.
* `coupled_draft_test` (`src/core/coupled_draft_test.cpp`, CPU only; built with `STRATA_BUILD_TESTS` or
  `STRATA_BUILD_CONVERSATION_TESTS`): simulated decodes check the counters, cells and penalty windows.

## Gotchas

* `load` must run before the VRAM expert tier is sized (~0.9 GB for Flash-Next's drafter); `bind` needs the native
  head (`--native`).
* The drafter uses its own stream; `idle()` synchronises it. Async verify commits rely on the drafter reading only
  the last window's final rows and its own K/V.

## See also

* [verify.md](verify.md), [session.md](session.md), [cuda-graphs.md](cuda-graphs.md).
* [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md).
