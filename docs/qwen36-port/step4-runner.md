# Qwen3.6 port, Phase 2 step 4, area `runner`: the gate lifted, the first token held to the oracle and llama.cpp

Working notes, kept as the step was done (2026-10-11, branch `qwen36-step4-runner`, from the step-4 integration
branch `qwen36-step4` = step 3 complete + the pack).  The design is `docs/QWEN36-35B-A3B.md` sections 3.5 and 6
(step 4) and 3.5; what the pack carries is `docs/qwen36-port/step4-pack.md` (section 2 the layout, section 6 the load
order); the references are `docs/qwen36-port/oracle-crosscheck.md` (the cases, the tolerance policy, the quantized-mode
gates).  This file is the runner half; the drafter is `qwen36-step4-drafter` (mtp.cpp, blk.40), not touched here.

## 1. What this makes the engine do

With `--native <qwen35moe gguf> --pack <dir>` the start no longer stops at the step-2 gate
(`qwen35moe_not_yet`).  It makes and prints the VRAM plan as before, then proceeds through the native pack's load
order, the session at hc = 1 with no PLE and no indexer, the expert tiers the plan sized, the prompt path, and the
decode windows, and produces tokens.  Measured on the real IQ3_XXS file (VM 221, the 4090), the first token of
"Hello, world! How are you?" (ids 9419,11,1814,0,2500,513,488,30) is 271, and the 16 greedy tokens that follow are
coherent English; the engine's own greedy continuation of the test20 prompt is byte-identical to llama.cpp's.

`--spec 0` / no drafter: the decode windows hold one token each (a window padded with token 0 would verify a draft
nobody made).  A native-pack start with `--spec` below 2 raises it to 2 (the verifier's smallest window) and says so;
the drafter plugs in at `generate.cpp`'s `--mtp DIR` block (the `mtp.draft` / `mtp.draft_first` calls in the spec
loop), which is the drafter agent's.

## 2. What changed, where (file:line against `qwen36-step4`)

`generate.cpp` is shared with the drafter agent; every hunk below is in the gate region, the load order, or the
session/decode wiring, in its own commit.

| Site | What | file:line |
| --- | --- | --- |
| the gate | the step-2 gate removed: the qwen35moe block now prints the plan, refuses a plan that does not fit (exit 2), and for a drafter-off start raises `--spec` to 2; it no longer returns at `qwen35moe_not_yet` | `generate.cpp` the `if (g.arch == Qwen35Moe)` block at ~2175 and the drafter-off note at ~2246-2258 |
| the load order | `check_all(wt, g)` after `NativeDense::load`, which `generate.cpp` had never called (step4-pack.md section 6): a pack missing a row or carrying it in the wrong form is refused by name before the first window | `generate.cpp:2532` (after the native projections) |
| the runner | `--teacher-force` (with `--spec-oracle FILE`): every window row commits, the next window starts from the oracle's token, `--dump-logits` writes one row of head logits per position (the window path wrote none), and the run reports the top-1 agreement | `generate.cpp` option 495, flag 1462, usage 671, validation 1910, oracle-length clamp 8128, the window loop 8305-8480 |
| the drafter-off path | one token per window when there is no drafter and no oracle; `--suffix-draft` skipped under teacher forcing | `generate.cpp:8326-8333` |
| `[shared]` verify | the two commit captures skip the indexer key-tail restore when `!g.has_indexer()`: a dense model has no tail, and the first real-file window stopped at "blk.3.indexer.k_norm.weight is missing" after the window had already run | `verify.cpp` `capture_commit` ~1407, `capture_commit_batch` ~1957 |

The Flash-Next path is untouched: the generate.cpp hunks are guarded by `g.arch == Qwen35Moe` or by the native-pack
/ teacher-force conditions, and the verify.cpp change is guarded by `!g.has_indexer()`.  The step-3 parity suite is
re-run below as the regression proof.

## 3. The tests (CMake targets)

`cmake/` nothing new; the fixtures are python, in `tests/python/`.

- **`qwen36_runner_fixture.py`** writes the synthetic model at the real per-layer geometry (the prefill harness's
  `full4`: 4 layers, three Gated DeltaNet and one full attention, n_embd 2048, GDN 16/32 heads of 128, attention
  16/2 of 256, 16 experts top-8 of 512) as a file a native pack can be built from (`token_embd.weight` Q8_0, a type
  `iq_embed_rows` serves), builds the pack with `tools/iq_pack.py` and a 4 x 16 expert profile ranking every pair,
  and dumps the oracle for the 8-, 11- and 40-token cases both faithfully (`oN`) and with `--act-quant q8`
  (`oN_q8`), each with `--all-logits`.
- **`qwen36_verify_test.cpp`** (CTest `qwen36_verify_test`, fixture `qwen36_runner_fixtures`) drives `Verifier` on
  that pack as `generate` drives it (the load order, `check_all`, the decode config `layer_verify_compatible`
  requires, the cache holding every expert), the step-3 review's HIGH finding (no test drove the window on
  qwen35moe).  Four parts: one 8-token window's residual after every layer and its head logits per row; the same
  rows from eight one-token windows, **bit for bit** equal to the 8-row window (verify.hpp's discipline); 8 + 3
  with a commit between; 40 tokens one per window (the oracle's `base_ar` shape).
- **`qwen36_runner_check.py`** (CTest `qwen36_runner_check`) drives the `strata` binary end to end: the 40-token
  sequence teacher-forced to the oracle in three chunkings (the oracle's `base_c7`, one chunk, and `base_ar`),
  `--dump-logits` per position, and the **mutation-sensitivity gate** (4 KB of `blk.1.ffn_gate_inp.weight` flipped
  in a copy of dense.bin must move the logits by more than 1e-2 relative L1).
- The real files: **`tools/qwen36_teacher_llama.cpp`** is llama.cpp's own greedy continuation with the head's
  logits per position and the routed experts at probed layers; **`tools/qwen36_compare_logits.py`** is top-1
  agreement, KL(ref || engine) per position, the drift gate (one-sided: only a KL that *rises* with position is a
  leaking state) and the routed-selection overlap.  `ref/qwen36.py` grew `--all-logits` (dump
  `result_output_all`, the head at every position); `qwen36_prefill_fixture.py`'s `write` grew `q8_embd`.

### Tolerances

The oracle is F32 weights x F32 activations; the engine runs Q8_0 projections (FP16 GEMV with q8_1 activations) and
Q8_0 experts on int8 dot products, so the reference for the window and runner tests is the **Q8 oracle**
(`--act-quant q8`, oracle-crosscheck.md "Tolerance policy" items 3-4).  Per-layer forced (the input forced from the
Q8 oracle's `l_out` of the layer before): 1.5e-2 relative L2 (4x the real file's Q8 `l_out` class, the fixture's
16-expert MoE spreads more).  Accumulated over the 4 layers and the recurrence: 5e-2.  Argmax must agree with the
Q8 oracle at every position; positions where the F32 and Q8 oracles themselves disagree (near ties) are reported,
not gated.

## 4. Results

### VM 221, the RTX 4090 (sm_89), 2026-10-11

Build: the common configure with `-DSTRATA_MMQ_KQUANTS=ON`, 25 min for the whole tree, 0 errors.

**`qwen36_verify_test`** (the synthetic native pack, 4 layers, 16 experts, every expert resident): PASS, 0 failures.

| check | measured | gate |
| --- | --- | --- |
| 8-token window, `l_out-N` per row accumulated vs the Q8 oracle | 1.9e-3 .. 2.4e-2 | 5e-2 |
| each layer alone, input forced from the Q8 oracle, `l_out-N` | 1.1e-3 .. 3.1e-3 | 1.5e-2 |
| eight one-token windows vs the 8-row window (R and logits) | 0.0 (bitwise) | bitwise |
| 8 + 3 with a commit, `l_out-3` / logits per position (11) vs Q8 | 2.4e-2 / 2.7e-2 | 5e-2 |
| 40 one-token windows, `l_out-3` / logits per position vs Q8 | 3.5e-2 / 3.5e-2 | 5e-2 |
| argmax vs the Q8 oracle (8 / 11 / 40 positions) | 8/8, 11/11, 40/40 | equal |

(Against the F32 oracle the 40-token argmax differs at 3 near-tie positions where the F32 and Q8 oracles disagree.)

**`qwen36_runner_check`** (the `strata` binary, teacher-forced): PASS, 0 failures.  Logits per position vs the Q8
oracle: `c7` 2.7e-2, `c40` 2.3e-2, `ar` 4.4e-2 (all <= 5e-2); argmax equal at every unambiguous position (the `ar`
run has 3 near-tie positions excluded, the same 1/4/23); the two prompt chunkings agree (1.4e-2), prompt vs decode
agree (2.1e-2); **the 4 KB mutation moves the logits by 2.1e-1 relative L1** (> 1e-2).

**The real files, greedy, teacher-forced to llama.cpp's own greedy continuation** (3 prompts from the regina
corpus x 64 tokens, context 8192, `--kv fp16`, the full profile-filled cache):

IQ3_XXS on the 4090:

| prompt | top-1 agreement | mean KL | max KL | drift (Spearman) |
| --- | --- | --- | --- | --- |
| test20 | 64 / 64 | 1.1e-3 | 2.1e-2 | -0.63 (falls) |
| task-prime-add-test | 64 / 64 | 6.6e-3 | 1.5e-1 (one near tie) | +0.12 (flat) |
| task-psim-collide | 64 / 64 | 1.9e-3 | 1.8e-2 | -0.06 (flat) |

No monotone positional drift on any prompt (Maya's quantized-mode gate: the drift is one-sided; a falling or flat
KL is not a leaking state).  The engine's own greedy text for test20 is byte-identical to llama.cpp's (the
qualitative check).  Q4_K_XL on the 4090 (the plain 40-block file, pack `qwen36-q4`):

| prompt | top-1 agreement | mean KL | max KL | drift (Spearman) |
| --- | --- | --- | --- | --- |
| test20 | 61 / 64 (3 near-tie quant flips) | 7.4e-3 | 1.5e-1 | -0.57 (falls) |
| task-prime-add-test | 64 / 64 | 1.7e-3 | 3.5e-2 | -0.08 (flat) |
| task-psim-collide | 64 / 64 | 4.2e-4 | 7.2e-3 | -0.20 (falls) |

The 1080 Ti was not used for the Q4 fit (the card was under a sibling's lock); the 4090 Q4 run is `--expert-cache
auto`, the whole file resident.

The step-3 parity suite, re-run to prove the Flash-Next path and the kernels are unchanged by the gate lift and the
`[shared]` verify edit (`ctest -R` over qsa_parity, qsa_prompt_attn_parity, kv_hybrid_parity, kv_stream_parity,
qsa_topk_parity, qsa_dense_parity, gdn_parity, router_top10_parity, shared_expert_parity, qwen36_seam_parity,
qwen36_block_test, prefill_qwen36, native_expert_parity): **100% of 33 tests passed, 0 failed** (pack_load_check
skipped, it needs the real-file STRATA_QWEN36_GGUF / _PACK env).  Every step-3 parity target and both new runner
tests pass; the Flash-Next geometry's parity is unchanged.

### VM 211, the GTX 1080 Ti (sm_61), 2026-10-11 21:38-21:42 UTC

Run after the notes above were first written, once the card's lock freed (held by sibling agents with nothing
running; released 21:34).  Tree `/home/rk/qwen36-step4-runner`, logs `/home/rk/logs/step4-runner-*.log`.

- Configure (the common line, `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`) and an incremental
  build from the pack tree's objects: 3 min 4 s, 0 errors.
- `qwen36_verify_test`: PASS, 0 failures.  `result_output_all` per position (40): worst rel L2 F32 5.65e-02, Q8
  3.46e-02 (gate Q8 <= 5e-02); argmax equals the Q8 oracle's at 40 of 40, the F32 oracle's at 37 of 40 (the same
  three near ties, positions 1, 4, 23, as on the 4090).
- `qwen36_runner_check.py`: PASS, 0 failures.  c7 worst rel L2 F32 2.05e-02 / Q8 2.55e-02; c40 2.32e-02 / 2.26e-02;
  ar (positions 1..39) 4.35e-02 / 4.37e-02, argmax 36 of 36 unambiguous; c7 vs c40 rows 35..39 rel L2 1.32e-02; c7
  vs ar 2.12e-02; the 4 KB mutation of `blk.1.ffn_gate_inp.weight` moves the logits by rel L1 2.05e-01.
- The real-file comparison against llama.cpp (three prompts, IQ3_XXS) did **not** run: the teacher fixtures
  (`/home/rk/qwen36-step4-teacher/<prompt>/prompt_ids.txt`, `eng/*.f32`) were never synced to VM 211, so all three
  engine runs exit 2 on "invalid token id" and the comparisons on a missing file.  Owed to step 5 along with the Q4
  fit at 11 GB; the card has since been taken by another service (section 5).

### VM 221, round 8 (2026-10-11 21:41 UTC): the drafter's VRAM term and the speculation rate

`vram_plan_test` with the drafter's whole share (its K/V over the context, the draft head subset, the ids): all
passed.  The real drafter (`--mtp`, 4-token windows, 64 greedy tokens on the `iq3-test20` prompt, `--prefill 2048
--max-context 8192 --kv fp16 --expert-cache auto --suffix-draft 0`):

| file, draft pack | drafts accepted | tokens per round | drafting ms/round | drafter VRAM | decode tok/s |
| --- | --- | --- | --- | --- | --- |
| IQ3_XXS, `qwen36-mtp-draft-full` | 44 of 57 (0.772) | 3.20 | 2.30 | 524 MiB | 317 |
| IQ3_XXS, `qwen36-mtp-draft` (subset head) | 43 of 60 (0.717) | 3.05 | 1.38 | 695 MiB | 326 |
| Q4_K_XL (plain file), `qwen36-mtp-draft` | 40 of 75 (0.533) | 2.54 | 1.54 | 745 MiB | 246 |
| IQ3_XXS, no `--mtp` (one-token windows) | - | 1.00 | - | 0 | 112 (1 token) |

The Q4 file's lower acceptance is expected: the draft pack comes from the Q4_K_XL-MTP file's `blk.40`, drafted
against the plain Q4_K_XL trunk; the drafter notes (step4-drafter.md) measured 77% teacher-forced on the MTP file
itself.  Logs `/models/work/logs/spec-*.log`, `r8-vram-plan-test.log`.

## 5. Not done, surprises

- **The 1080 Ti pass** of steps 2-4 ran at 21:38 (section 4): build, verify test and runner check pass; the
  llama.cpp real-file comparison is owed (fixtures never synced).  Since 2026-10-12 22:04 UTC the card runs another
  session's Flash-Next coder service (`unbiased-strata.service`, 10.9 GB, 100% busy) and is off limits.
- **The Q4 fit at 11 GB** was not measured (the same lock, then the service); the Q4 run above is the 4090's.  The pack notes
  (step4-pack.md section 5) record the plan on the 11 GB card (Q4_K_XL fixed set 4,461 MiB, 34% of slots resident),
  so a reduced-context Q4 run there is the remaining 1080 Ti item.
- **The chat template** (`make test-chat-parser MODEL=<gguf>`): on Qwen3.6's template a tool call whose required
  parameter is not written first was silently dropped (0 tool calls).  Filed REGINA-104.  Resolved at integration
  (e73adc6): the template does route to the Qwen3-Coder parser the any-argument-order patch covers; the patch was
  simply never applied, because only `make build` applied it and every VM tree was configured with cmake directly on
  a copied submodule.  CMake now applies it at configure time; with the rebuilt `libllama-common` the test passes on
  VM 221 for all four orders.
- **The argmax near ties.** On the synthetic 40-token `base_ar` case the F32 and Q8 oracles disagree at 3 positions
  (1, 4, 23); the engine matches the Q8 oracle at all 40.  A position where the two references disagree cannot
  discriminate the engine, so the tests gate against the Q8 oracle and report those positions.
- **The real file's single-position KL spikes** (task-prime-add-test, 1.5e-1 at position 40) are near ties where
  top-1 still agrees; the mean KL is 1e-3 to 7e-3.

## 6. The interface other agents consume

- `strata --native FILE --pack DIR [--spec T] [--mtp DIR] --prefill CHUNK` now runs a Qwen3.6 pack to tokens.
  Drafter off: omit `--mtp`; the windows are one token each.  `--spec-oracle FILE --teacher-force` holds the engine
  to a reference continuation and writes one logits row per position with `--dump-logits`.
- `ref/qwen36.py --all-logits` dumps `result_output_all` (n_tok x n_vocab, the head at every position of the last
  chunk); `tools/qwen36_teacher_llama.cpp` is the llama.cpp reference (logits.f32, continuation.txt, routes.txt);
  `tools/qwen36_compare_logits.py` scores top-1, KL and the drift gate.
- The runner tests: `qwen36_verify_test <fixture dir>`, `qwen36_runner_check.py --strata S --fixture DIR`, both from
  `qwen36_runner_fixture.py --out DIR`.
