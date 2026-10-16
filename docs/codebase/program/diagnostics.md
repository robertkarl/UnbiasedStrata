[Codebase docs](../README.md) › [Program](README.md) › Diagnostics

# Diagnostics: dumps, measurement modes, teacher forcing, traces

**Purpose:** the driver doubles as the project's measurement and bisection tool. This page lists every diagnostic
output it can produce, what file format it writes, and its preconditions.
**Covers:** `include/strata/program/logits_selection.hpp`, and in `src/program/generate.cpp` the dump options
(`:275-351`), arming (`:2685-2707`, `:4095-4144`, `:4277-4322`), the measurement modes (`:4342-4582`), per-token
dumps (`:8087-8156`), the speculative-loop dumps (`:8503-8538`), the report (`:8662-8830`) and serve-side traces.

## `logits_selection.hpp` (`--logits-stride`)

| Function | Meaning |
| --- | --- |
| `parse_stride(text, out)` | Strict positive decimal int64: no sign, whitespace, suffix or overflow |
| `selected(pos, total, stride)` | Row `pos` is stored when `pos % stride == 0` or `pos == total - 1` |
| `row_count(total, stride)` | Number of selected rows: `(total - 1) / stride + 1`, plus one if the last row is not a multiple |

"Storage selection only": every input token is still conditioned on. `--logits-stride` may be given once and, above
1, requires `--max-new 1` and `--dump-logits` (`:1906-1909`). The header's row count is `row_count(n_prompt - 1 +
max_new, stride)` (`:4300-4301`).

## Dump files

| Flag | Written by | Format | Preconditions |
| --- | --- | --- | --- |
| `--dump-logits P` | token loop (`:8129-8156`); verify windows (`:8503-8524`) | `int32 {n_vocab, n_rows}` header written **with the first row**, then `n_vocab` f32 per row. Token path: one row per selected position from 0. Windows: one row per emitted position (every row under `--teacher-force`); a native pack's header then promises `--max-new` rows | rows above int32 refused |
| `--dump-residual P` | report (`:8720-8743`) | `int32 {hc, n_embd}` + final `R` | the head can then be re-checked in Python |
| `--dump-mixed P` | report (`:8699-8717`) | `n_embd` f32 of `ss.block.mixed` (the head input) | |
| `--dump-layers P` | `session_loop` staging, written per position (`:8106`) | `(n_layers + 1) x hc x n_embd` f32 per position: the C1 bisection ladder vs `llama-debug` | captured path; not with `--no-capture`; disables the token graph and fused GR read |
| `--dump-halves P` | captured into the layer graphs; written per position (`:8107-8109`) | `n_layers x half_stride` f32, `half_stride = 2 n_embd + 2 hc + n_head x head_dim + 5 n_head_kv x head_dim + 8` | armed before `session_capture`; not with `--no-capture` |
| `--dump-routing P` | `drive_pool` / `drive_pool_multi` (`:792-844`) | records `int32 layer, int32 k, k x int32 ids, k x f32 weights`; verify windows write unit weights (zeros) | needs the CPU pool; serve flushes it after each request |
| `--dump-final-r P` | token loop (`:8087-8092`) | `int64 {pos, tok}` + `hc x n_embd` f32 per position (the MTP head's input) | token path only |

The routing trace's layer index is taken as `d.layers - 1` after the dispatch, a fix for an off-by-one that
`tools/make_profile.py` caught (`:803-809`).

## Measurement modes (print and exit)

| Flag | What it times | Notes |
| --- | --- | --- |
| `--gpu-stages` | Three graphs per layer (mixer / ffn+router / post) timed from outside the capture, 20 reps; then the mixer split into GDN vs attention layers; five stages by differencing prefixes; the same by sweeping prefixes, with the bias between the two (`:4354-4500`) | Canonical pack only: a native pack has no per-layer graphs and the error points at `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1` with `--serve` instead (#610). Refused across a split |
| `--graph-only` | The `pre` graphs back to back (`session_replay`) (`:4502-4530`) | Omits `post` graphs and the head: **not** the GPU floor |
| `--gpu-only-full` | `session_replay_full` + head, 20 reps after a warm pass (`:4542-4582`) | "the true per-token GPU floor"; refused across a split |
| `--stage-timing` | CUDA events inside the layer path, reported at the end (`stage_timing_report`) | Needs `--no-capture` (events in a capture are dropped); shares are of kernel count, not time |
| `--stats` | The one-shot report's breakdown ([generation-loop.md](generation-loop.md)) | |

## Teacher forcing (`--teacher-force`)

Qwen3.6 step 4 (`docs/qwen36-port/step4-runner.md`): with `--spec-oracle FILE` the engine is held to a reference
continuation (llama.cpp's greedy text). Every window row is committed (`a = T - 1`), the next window starts from the
oracle's token, windows never run past the oracle, `--max-new` is lowered to the oracle's length, and each row's pick
is compared with the oracle (`top-1 agreement X of Y positions` is printed). `--dump-logits` then writes one row per
position. Refused with `--serve` and with `--mtp` (drafted rows would put tokens into the state that the oracle never
had) (`:1910-1920`, `:8247-8251`, `:8493-8502`, `:8577-8578`). Note the separate, always-on teacher forcing over the
prompt in the token loop (`:8219-8225`).

## Other one-shot diagnostics

- `--spec-oracle` without `--teacher-force`: oracle tokens are drafts; `--spec-corrupt N` corrupts every Nth.
- `--prefill-until P`: batch `[0, P)`, run the rest teacher-forced through the token path to score many positions.
- `--no-pool`: the GPU-only floor (routed experts contribute nothing; wrong tokens by design).
- `--no-ple`: PLE ablation, with a loud warning that the tokens are not this model's.
- `STRATA_DUMP_FIRST_LOGITS=PATH`: the first verify window's logits (f32, no header), for comparing prompt paths.
- `STRATA_TRACE_ADAPT=1`: adaptive-tier rounds and landing.

## Serve-side diagnostics (environment)

| Variable | Output |
| --- | --- |
| `STRATA_TRACE=1` | `strata trace:` lines per request step (positions built, window, refill, lend, read timings) and `mem_mark` VRAM lines at start |
| `STRATA_DECODE_TIMING=1` | per request: windows, avg T, ms per window split into verify (GPU-reach wait, host plan/actq/jobs/CPU, staging), commit/emit, draft; per-layer CPU experts, hits, PCIe; plus the verifier's GPU stage profile |
| `STRATA_SERVE_LOGIT_TRACE=FILE` | appends `L <req> <tok#> <id> row i of T [sfx] miss pcie hits top <id:logit> x5` per emitted token |
| `STRATA_SERVE_LOGIT_DUMP=PREFIX` | with the trace: every emitted token's full logit row to `PREFIX.<request>.f32` (for `tools/qwen36_compare_logits.py --no-engine-header`) |
| `STRATA_LOGPOS=PATH` | for prompt tokens read through windows: teacher-forced log-probabilities appended via `Verifier::window_logprobs`; `STRATA_LOGPOS_EXTRA` (default 248046) adds one extra id's log-prob column |
| `STRATA_STATE_HASH=1` | FNV-1a fingerprints of the session after each request (GDN, PLE, tails, pooled keys, KV, MTP KV, stale cells, dead keys); `STRATA_STATE_HASH_GDN=1` per GDN layer |
| `STRATA_CKPT_REREAD=1` | re-read checkpoints instead of restoring them (an exactness check) |
| `STRATA_SNAPSHOT_VERIFY=1` | drafter-KV path lines and a restored-draft-KV hash |
| `STRATA_SPLIT_TIMING=1` | per-stage verify timings |
| `STRATA_RSS_TRACE=1` | `RssFile` at start-up steps (who reads the arena back) |
| `STRATA_WATCHDOG_S=N` | stall watchdog period (default 60; 0 = off); the stall report is described on [serve-mode.md](serve-mode.md) |

`STRATA_VERIFY_PROFILE` is mentioned in an error message here but read elsewhere (the verifier), not in this file.

## Tests

No automated test covers the dumps; `tools/logits_identical.py` and `tools/qwen36_compare_logits.py` consume them
(cited in comments). `logits_selection.hpp` has no dedicated unit test in the tree (no reference outside
`generate.cpp` was found).

## See also

- [generation-loop.md](generation-loop.md), [speculative-decoding.md](speculative-decoding.md), [env-vars.md](env-vars.md)
- [docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md),
  [step5-serve.md](../../qwen36-port/step5-serve.md) ("The same greedy request, different tokens")
