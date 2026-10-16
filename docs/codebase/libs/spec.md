[Codebase docs](../README.md) › [Libraries](README.md) › Speculative-decoding policy

# Speculative-decoding policy (`strata_spec`)

## Purpose

`strata_spec` is a small static library with no GPU or model dependency. It answers two questions for the verify
loop in `src/program/generate.cpp`:

1. **What could we draft without the MTP head?** The `SuffixDrafter` ("prompt lookup") proposes the tokens that
   followed an earlier occurrence of the current context suffix.
2. **Which drafts are worth verifying this round?** The `DraftPolicy` compares the MTP's window with a lookup window
   by expected committed tokens per millisecond, learned online.

A third component, `Controller`, is an earlier and more general cost-model-driven chooser (none / lookup / MTP, with
k up to 8). It is built into the library and unit-tested, but **nothing outside its test calls it** (verified by
grep: only `src/spec/controller_test.cpp` includes `strata/spec/controller.hpp`). The engine uses `SuffixDrafter`
plus `DraftPolicy`.

None of these change the output: every draft goes through the normal verify pass, which accepts only the prefix that
matches the model's own picks. They only change how many tokens one verify pass can commit.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/spec/suffix_drafter.hpp](../../../include/strata/spec/suffix_drafter.hpp) | 53 | `SuffixDrafter` API: trigram-indexed history, `append`, `propose`, `last_match` |
| [src/spec/suffix_drafter.cpp](../../../src/spec/suffix_drafter.cpp) | 88 | Hash table, append, longest-match proposal |
| [src/spec/suffix_drafter_test.cpp](../../../src/spec/suffix_drafter_test.cpp) | 141 | Unit tests plus an offline `--simulate` replay over token-id files |
| [include/strata/spec/draft_policy.hpp](../../../include/strata/spec/draft_policy.hpp) | 52 | `DraftPolicy` API: MTP window vs lookup window per verify round |
| [src/spec/draft_policy.cpp](../../../src/spec/draft_policy.cpp) | 101 | Online cost / acceptance estimates and the choice rule |
| [src/spec/draft_policy_test.cpp](../../../src/spec/draft_policy_test.cpp) | 63 | Simulated rounds with an RTX 5070-shaped cost curve |
| [include/strata/spec/controller.hpp](../../../include/strata/spec/controller.hpp) | 73 | `CostModel`, `Source`, `Choice`, `Controller` (unused by the engine) |
| [src/spec/controller.cpp](../../../src/spec/controller.cpp) | 73 | Step-time model, expected-tokens math, EMA updates |
| [src/spec/controller_test.cpp](../../../src/spec/controller_test.cpp) | 74 | Decision tests and an acceptance-sweep printout |

Build: `add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)`
(`CMakeLists.txt:1052`). It is linked into the `strata` and `unbiased-strata` executables (`CMakeLists.txt:578`,
`CMakeLists.txt:1442`).

## SuffixDrafter

`class SuffixDrafter` (`include/strata/spec/suffix_drafter.hpp:19`), namespace `strata::spec`.

| Member | Where | What it does |
|---|---|---|
| `SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19)` | `src/spec/suffix_drafter.cpp:18` | Clamps `min_match` to at least 3 and `max_match` to at least `min_match`. Allocates an open-addressing table of the smallest power of two that is at least `2 * capacity_tokens` slots (load factor at most 0.5 at nominal capacity), and reserves the history. |
| `reset()` | `src/spec/suffix_drafter.cpp:27` | Clears history and table. |
| `append(const int32_t*, size_t)` / `append(int32_t)` | `src/spec/suffix_drafter.cpp:51` | Pushes tokens to `hist_`. For each new position `end >= 2`, hashes the trigram `hist_[end-2..end]` and records `end` in that slot's `pos[WAYS]` list (most recent first, `WAYS = 4`). |
| `propose(int max_k, int32_t* out)` | `src/spec/suffix_drafter.cpp:64` | Looks up the trigram ending at the current last token. For each recorded earlier end position `p < cur`, extends the match backwards while tokens agree (up to `max_match`). Takes the longest; ties keep the most recent (candidates are scanned newest first and replaced only on a strictly longer match). If the best is shorter than `min_match`, proposes nothing. Otherwise copies up to `max_k` tokens that followed `p`. |
| `last_match()` | header `:34` | Length of the match behind the last proposal (0 if none); the draft policy buckets on it. |

Details and invariants:

- **Hash and table.** `Slot { uint64_t key; uint32_t pos[WAYS]; uint8_t n; }` (header `:38`). The key is a 64-bit
  mix of the three token ids with the low bit forced to 1 so that 0 means "empty" (`key_at`,
  `src/spec/suffix_drafter.cpp:33`). Collisions are handled by linear probing (`find_slot`, `:38`). Different
  trigrams that collide on the full 64-bit key would share a slot; the backwards extension compares real tokens, so
  a false hit only produces a short or zero match.
- **Bounded memory.** The table is fixed at construction. When it is full, `find_slot` returns `nullptr` and that
  trigram is simply not indexed (`:48`, `:57`). The bounded-memory unit test appends 5,000 tokens into a drafter
  built for 1,024.
- **Periodic text.** A continuation may run into the current suffix itself; the copy loop reads history up to `cur`
  (`:83-84`), so text like `1 2 3 1 2 3 1 2 3` proposes `1 2 3 ...`.
- `propose` needs at least 4 tokens of history (`:67`).
- Positions are stored as `uint32_t`, so the history is limited to 2^32 tokens.

## DraftPolicy

`class DraftPolicy` (`include/strata/spec/draft_policy.hpp:22`). The engine creates one per process (serve) or per
run (one-shot) and calls it once per verify round.

```
Pick choose(int t_mtp, int lookup_k, int match) const;          // which window to verify
void observe(bool lookup, int t, int accepted, int match, double round_ms);  // what happened
```

`t` is the window size (1 committed token + drafts), capped at `max_t` (at most `kMaxT = 8`). `match` is
`SuffixDrafter::last_match()`.

### What it learns (`observe`, `src/spec/draft_policy.cpp:84`)

| State | Update |
|---|---|
| `cost_[t]`, `cost_n_[t]`: round time by window size | EMA with weight `kCostAlpha = 0.1` (first sample taken as is). The round time covers verify, commit and draft. |
| `mtp_tok_[t]`, `mtp_n_[t]`: tokens committed by MTP windows of size t | EMA of `accepted + 1` with weight `kTokAlpha = 0.05` |
| `ok_[b]`, `bad_[b]`: lookup outcomes per match-length bucket | Decayed counts (`kDecay = 0.97`): `ok += accepted`; `bad += 1` when the window was cut short (`accepted < t - 1`) |

Buckets (`bucket`, `:27`): match < 6, < 12, < 24, and 24 or more.

### Estimates

- **Lookup acceptance** `q(match)` (`lookup_rate`, `:31`) = `(ok + 4 * prior) / (ok + bad + 4)`, with priors
  `kPriorQ = {0.75, 0.88, 0.93, 0.96}` worth 4 observations each (`:18-19`). Longer matches start out more trusted.
- **Round cost** `cost_ms(t)` (`:36`): the measured EMA if size t has been seen; otherwise scaled from the sizes
  that have been seen, by a prior shape `kShape = {_, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6}` (`:12`),
  weighting each seen size by its sample count (capped at 20). With no data at all it returns the shape itself.
- **MTP tokens** `mtp_tokens(t)` (`:50`): the EMA if seen, else `1 + 0.7 * (t - 1)`.

### The choice (`choose`, `src/spec/draft_policy.cpp:55`)

```
base     = mtp_tokens(t_mtp) / cost_ms(t_mtp)
for k = 1 .. min(lookup_k, max_t - 1):
    E(k) = 1 + q + q^2 + ... + q^k              # expected committed tokens
    rate = E(k) / cost_ms(k + 1)
best_t   = argmax rate
take lookup window best_t  if  best_rate > base * (1 + margin)      # margin default 0.03
```

If the lookup does not win, there is a **probe** rule (`:74-80`): a guessed cost could stop the policy from ever
measuring a large window, so when the full lookup window `t_full` is larger than the MTP window, has been measured
fewer than `kProbes = 3` times, and `q >= 0.85`, it is taken anyway. Only that round's speed is at stake, because
verification keeps the output unchanged.

With no lookup proposal (`lookup_k <= 0`) the MTP window is returned unchanged (clamped to `[1, max_t]`).

## Controller (not wired into the engine)

`include/strata/spec/controller.hpp` describes a "plan v0.3 P6" controller that maximizes
`E[tokens committed] / T(step)` over none, lookup with `k <=` its proposal, and MTP with `k <= K_MAX = 8`.

- `CostModel::step_ms(k, mtp)` (`src/spec/controller.cpp:8`): `dense_ms * dense_ratio[n-1]` plus a CPU-miss term
  `(1 - hit_rate) * cpu_all_miss_ms * U(n) * (1 + extra_use_cost * (n/U(n) - 1))`, plus `sync_ms`, plus `k` times
  the per-draft cost of the source. Defaults (`controller.hpp:25-34`) are measurements dated 23 Sep cited in the
  header comment (dense 11 ms, 480-expert all-miss CPU pass 15.8 ms, hit rate 0.55, sync 2.4 ms, MTP draft 1.2 ms per
  token). With these defaults a plain one-token step is about 20.5 ms.
- `Controller::choose(lookup_available, lookup_match, mtp_ready)` (`:36`): evaluates every k for lookup (one
  acceptance `q` per match bucket, raised to the k-th power) and for MTP (per-position conditional acceptances
  `p_i`), keeps the best rate, and falls back to no speculation unless it beats plain decoding by `min_gain`
  (default 5%).
- `Controller::observe(choice, accepted, lookup_match)` (`:57`): EMA updates (default weight 0.05) of the positions
  actually tested. When every MTP draft is accepted, the untested deeper positions are pulled toward the deepest
  observed rate so that the window can grow (`:67-70`).
- Priors (`:20-25`): MTP 0.86 per position; lookup by bucket `{0.35, 0.6, 0.8, 0.92}` for matches `<5, <8, <16, >=16`
  (`controller.hpp:64`). Note that these buckets and priors differ from `DraftPolicy`'s.

## How the engine uses it

All call sites are in [src/program/generate.cpp](../../../src/program/generate.cpp):

| Where | What |
|---|---|
| `generate.cpp:553-557` | Option `suffix_draft` (default 3), `--suffix-draft N` (`:1498`); 0 turns prompt lookup off. |
| `generate.cpp:558-559`, `:1499` | `--mtp-max-t M`: cap on the MTP's own window. |
| `generate.cpp:1836-1842` | When `suffix_draft > 0`, `--spec >= 2` and `--mtp-max-t` is unset, the MTP keeps `--spec` as its cap and the verify window grows by 2 (up to 8, `kVerifyMaxT`), so lookup windows can be up to 2 tokens longer than the MTP's. |
| `generate.cpp:6098-6099` | Serve loop: one `SuffixDrafter(max(1, suffix_draft), 64, max_context + 4096)` and one `DraftPolicy(S)` for the whole process ("learned over the whole process"). |
| `generate.cpp:7433-7436` | Per request: `sfx.reset()` and append the prompt ids. |
| `generate.cpp:7476-7485` | Per round: `propose(S - 1)`; the lookup is only considered when its first token equals the MTP's first draft (`sbuf[0] == drafts[0]`), then `policy.choose(T, k, sfx_match)`. |
| `generate.cpp:7586` | Every emitted token is appended to the drafter. |
| `generate.cpp:7615-7617` | `policy.observe(from_sfx, T, a, sfx_match, round_ms)` with the measured round time (skipped on the first window and at EOS). |
| `generate.cpp:8427-8458` | One-shot (non-serve) path: same pattern, but without an MTP head (`!use_mtp`) a lookup can be used without the first-token agreement check; lookup is skipped under `--teacher-force`. |
| `generate.cpp:8582`, `:8605-8607` | `observe`, and a final "suffix drafts" stats line. |

Because the `SuffixDrafter` constructor clamps `min_match` to at least 3, `--suffix-draft 1` and `--suffix-draft 2`
behave like 3. The engine always passes `max_match = 64`.

## Configuration

| Knob | Effect |
|---|---|
| `--suffix-draft N` | Minimum match length for prompt lookup (effective minimum 3); 0 = MTP only. Default 3. |
| `--spec S` | Verify window size; with lookup on and `--mtp-max-t` unset, widened by 2 (max 8). |
| `--mtp-max-t M` | The MTP's own cap; longer windows can only come from lookup. |
| `--teacher-force` | Disables lookup in the one-shot runner. |

No environment variables or compile-time macros affect this library.

## Model-specific behaviour

The library is model-agnostic: it sees token ids and timings only. It is used identically for Qwen3.8-Flash-Next and
Qwen3.6-35B-A3B. For Qwen3.6 the one-shot runner's drafter-off path (no MTP) can still use lookup windows, see
[docs/qwen36-port/step4-runner.md](../../qwen36-port/step4-runner.md).

## Tests

All three are registered with CTest when `STRATA_BUILD_TESTS` is on (`CMakeLists.txt:1054-1063`). They need no GPU
or model. Note that `STRATA_BUILD_TESTS` defaults to OFF in this checkout because `tests/CMakeLists.txt` and
`bench/micro` are absent (`CMakeLists.txt:88-92`); configure with `-DSTRATA_BUILD_TESTS=ON`.

| Test | Covers |
|---|---|
| `suffix_drafter_test` | Verbatim repeat, no match, longest-over-most-recent, `min_match`, periodic continuation, bounded memory. `suffix_drafter_test --simulate [--k K] FILE.ids ...` replays token-id files in two regimes ("continue": first half as history; "copy": regenerate the middle third) and prints tokens/step, proposal rate and accepted per proposal. |
| `draft_policy_test` | No proposal keeps the MTP window; rejected short matches stop being taken; accepted long matches take the full window; buckets learn separately; window cap; a mediocre lookup does not replace a strong MTP; an unmeasured size is probed, then the measured cost decides. |
| `controller_test` | MTP chosen at the prior with k in 2..5; persistent rejection turns speculation off; long lookup match beats MTP; a cheap wide verify takes the whole lookup window; a trigram-only lookup alone is not used; full acceptance widens the window. Also prints a sweep of chosen k for MTP acceptance 0.5 to 0.95. |

Run: `ctest -R "suffix_drafter_test|draft_policy_test|controller_test"` from the build directory.

## Gotchas and limitations

- `Controller` is dead code from the engine's point of view; its buckets, priors and cost model are not what the
  engine uses. Change `DraftPolicy` to change behaviour.
- The `suffix_drafter.hpp` comment says the index costs "~20 bytes per history token". By the code, the table has at
  least `2 * capacity_tokens` slots of `sizeof(Slot)` (8 + 16 + 1 bytes before padding) each, so the real figure is
  higher. The engine sizes it at `max_context + 4096` tokens. Not measured here.
- When the table is full, every append of a new trigram probes the entire table before giving up
  (`find_slot`, `src/spec/suffix_drafter.cpp:39-48`). The engine avoids this by resetting per request and sizing for
  the full context.
- `DraftPolicy` cost estimates are wall-clock round times, so they adapt to the machine, the context length and
  background load. Learned state persists across requests in serve mode.
- The header comments cite benchmark directories (`bench/results/2026-09-23-*`, `2026-09-27-spec`) that are not in
  this checkout.

## See also

- [Libraries overview](README.md)
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works" (speculation and prompt lookup, engine 0.1.7)
- [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md) and
  [step4-runner.md](../../qwen36-port/step4-runner.md) for the MTP drafter these windows compete with
