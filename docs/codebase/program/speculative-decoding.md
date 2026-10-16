[Codebase docs](../README.md) › [Program](README.md) › Speculative decoding

# Speculative decoding: verify windows, MTP and suffix drafts

**Purpose:** decode several tokens per pass of the model. Each round runs one **verify window** of `T` rows - the
last emitted token followed by `T - 1` drafts - through every layer; the rows' picks equal what plain decode would
emit, so drafts are accepted up to the first mismatch and the round emits `accepted + 1` tokens.
**Covers:** the one-shot loop `src/program/generate.cpp:8230-8660` and the serve decode loop `:7428-7622`; the option
adjustment at `:1836-1842`; the drafter load/bind (`:3342-3361`, `:5482-5486`, `:8271-8275`). The verifier is
`strata::core::Verifier` (`src/core/verify.cpp`), the drafter `strata::core::MtpDrafter` (`src/core/mtp.cpp`), and
the lookup drafter/policy are `strata::spec::SuffixDrafter` / `DraftPolicy` ([libs](../libs/README.md)).

## Window sizes

| Quantity | Source |
| --- | --- |
| `kVerifyMaxT` | 8 (`include/strata/kernels/verify_kernels.hpp:23`) |
| `S` (`o.spec`) | `--spec T`. If `--suffix-draft > 0`, `T >= 2` and `--mtp-max-t` is 0, the engine sets `mtp_max_t = T` and `spec = min(T + 2, 8)` (`:1839-1842`): lookup windows may be two tokens longer than MTP windows. The front end's `--spec 4` therefore runs MTP windows of 4 and lookup windows up to 6 |
| `S_mtp` | `min(mtp_max_t, S)` or `S`; `mtp.set_max_drafts(S_mtp - 1)` when smaller |
| Verifier capacity | `Verifier::init(..., max(o.spec, o.batch))` in serve, `o.spec` one-shot |

## One round

```
T = S_mtp
if spec_min_p > 0: T = 1 + number of leading drafts with draft prob >= spec_min_p   (dprob from the last draft)
if first window:   T = 1                    (the last prompt token alone)
lookup: k = sfx.propose(S - 1, sbuf); if k > 0 and sbuf[0] == drafts[0]:
            pick = policy.choose(T, k, sfx.last_match()); if pick.lookup: T = pick.t, drafts from sbuf
window = [x, drafts...]                      (x = last emitted token)
apply_pending(wait)                           (adaptive swaps land first)
ver.run(T, window, p, pool_fn, user, outv)    (outv[i] = pick after row i)
a = number of i with window[i+1] == outv[i], counting from 0 until the first mismatch
[adapt() on a thread every --adapt-every rounds]
ver.commit(a + 1)                             (keep the state of the emitted tokens)
emit outv[0..a]; stop on EOS / max_new
mtp.draft(T, outv, p, a, drafts, err, dprob, spec_min_p)   (drafts for the next window)
policy.observe(from_sfx, T, a, match, round_ms)
x = outv[a]; p += a + 1
```

(serve: `:7467-7622`; one-shot: `:8437-8586`.)

## Draft sources

- **MTP drafter** (`--mtp DIR`). Loaded before the expert cache (`MtpDrafter::load(dir, draft_geometry, session,
  spec, err, mtp_window)`); for Flash-Next the draft geometry is the canonical model's, for Qwen3.6 the target's
  (`:3355-3356`). Bound to the head and the verifier's final residual rows (`mtp.bind(wt, head, ver.final_R_all())`),
  so drafting reads the window's residuals directly. During prefill the drafter's own KV is filled chunk by chunk
  (`on_chunk` -> `Prefill::draft_kv` or `MtpDrafter::prefill`); prompt tokens read through windows call
  `mtp.prefill(ver.final_R_all(), next, T, q)`. One-shot without a native pack uses `mtp.draft_first(spec, ss.R, x,
  p - 1, ...)` to seed the first drafts from the token loop's residual (`:8416-8420`). `--mtp-window` limits the
  draft layer's attention span.
- **Suffix drafter (prompt lookup).** `SuffixDrafter sfx(max(1, --suffix-draft), 64, max_context + 4096)`; the whole
  prompt is appended at the start of a request (serve) or run (one-shot), and every emitted token after. When the
  recent tokens repeat an earlier stretch of at least `--suffix-draft` tokens, `propose` returns what followed it.
  It is used only when the MTP's own first draft agrees with the lookup's first token (one-shot without MTP: always
  eligible) and the `DraftPolicy` predicts more tokens per ms (it learns acceptance and window costs from
  `observe`). Counts are reported (`suffix drafts: N windows, X of Y accepted`).
- **Oracle** (`--spec-oracle FILE`, one-shot): drafts are the file's tokens at the matching positions;
  `--spec-corrupt N` breaks every Nth draft (an exactness test of rejection).
- **No source** (no `--mtp`, no oracle): `T = 1`, one token per window (`:8447`) - the Qwen3.6 drafter-off path.

## Sampling in windows

`ver.set_sampling(SamplerParams)` makes each row's pick a sample instead of argmax: CLI parameters in one-shot mode
(`:8277`, no penalties), per-request keys in serve mode (`:7320-7341`, with repetition/frequency/presence penalties
over a per-row history buffer `d_hist`, `penalty_rows`, up to `penalty_last_n <= 4096`). `mtp.set_draft_sampling`
makes drafts sampled too when coupled drafting is on (`--coupled-draft` / `STRATA_SPEC_COUPLED=1`; with penalties
`mtp.set_draft_history` keeps the drafter's history aligned, `:7594-7595`). Acceptance is still "draft equals the
row's pick".

## Configuration

`--spec`, `--mtp`, `--mtp-window`, `--mtp-max-t`, `--suffix-draft`, `--spec-min-p` (serve: per-request `spec_min_p=`
key), `--coupled-draft`, `--spec-split`, `--pcie-frac` / `--pcie-mode` (how the window's missed experts are read:
`ver.set_pcie_mode`, `drive.d.pcie_num = round(frac x 256)`), `--adapt-*`, `--spec-oracle`, `--spec-corrupt`,
`--teacher-force`. Env: `STRATA_ADAPT_NOWAIT`, `STRATA_TRACE_ADAPT`, `STRATA_DECODE_TIMING`,
`STRATA_DUMP_FIRST_LOGITS`, `STRATA_SPEC_COUPLED`.

## Model-specific behaviour

- Native packs (all Qwen3.6 packs, and IQ Flash-Next packs) decode **only** through windows; the first window is the
  last prompt token alone (`first_window = native_pack`, `:8415`).
- Qwen3.6 without `--mtp` in one-shot mode raises `--spec` to 2 (`:2323-2327`) and runs `T = 1` windows.

## Reports

One-shot prints (`:8592-8659`): rounds, drafts accepted / offered, tokens per round; teacher-forced agreement; window
size histogram (with `--spec-min-p`); suffix-draft counts; accepted-per-round histogram; verify-window
wait/pool/host/commit ms per round and CPU experts per layer; pool multi phases and GB/s; dispatch phases; adaptive
swaps; resident RAM and file-tier counters; routing-prefetch hits; PCIe experts per layer; MTP ms per round and VRAM.
Serve prints the request summary (drafts accepted / offered, hit rate) and, with `STRATA_DECODE_TIMING=1`, a
per-window breakdown and the verifier's GPU stage profile.

## Gotchas

- The window must not run past the context: one-shot exits 2 ("ran out of context"), serve stops the request
  (`if (p + T > o.max_context) break;`).
- `ver.wait_commit` must be called before anything else reads the session: commits can be asynchronous
  (`Verifier::set_commit_async(!multi_gpu)`, `:2894`).
- Adaptive swaps landing between windows change whether an expert runs on the GPU or the CPU, which round
  differently; the default waits for them so output does not depend on copy timing (#463).
- `--spec` needs the device residency table in one-shot mode (`--expert-profile`, a cache and the token graph,
  `:8253-8257`).

## See also

- [serve-mode.md](serve-mode.md), [expert-tiers.md](expert-tiers.md), [diagnostics.md](diagnostics.md)
- [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md), [step4-runner.md](../../qwen36-port/step4-runner.md)
- [../core/README.md](../core/README.md), [../libs/README.md](../libs/README.md)
