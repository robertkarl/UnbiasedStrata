[Codebase docs](../README.md) › [Tools](README.md) › Python tests and fixtures

# Python tests and fixture generators

## Purpose

Two groups of Python test code:

- **`tools/test_*.py`**: `unittest` suites for the offline tools and the oracle. They need no GPU and (except
  where noted) no model.
- **`tests/python/qwen36_*.py`**: fixture generators and one checker that CTest runs around the engine's CUDA
  tests for the Qwen3.6 port. They write synthetic GGUFs, packs and oracle dumps into the build directory.

`tests/python/serve_max_tokens_check.py` (the HTTP front end check run by `make test-serve`) is covered in
[../frontend/README.md](../frontend/README.md).

| File | Lines | Kind | Needs |
| --- | --- | --- | --- |
| [`tools/test_iq_pack.py`](../../../tools/test_iq_pack.py) | 840 | unittest, 4 classes | numpy, gguf-py |
| [`tools/test_make_profile.py`](../../../tools/test_make_profile.py) | 145 | unittest | stdlib |
| [`tools/test_draft_vocab.py`](../../../tools/test_draft_vocab.py) | 47 | unittest | stdlib (imports `strata_tokenizer`) |
| [`tools/test_mtp_fetch.py`](../../../tools/test_mtp_fetch.py) | 201 | unittest, mocked HTTP | stdlib |
| [`tools/test_qwen36_oracle.py`](../../../tools/test_qwen36_oracle.py) | 419 | unittest, 24 methods | numpy, gguf-py |
| [`tests/python/qwen36_prefill_fixture.py`](../../../tests/python/qwen36_prefill_fixture.py) | 186 | fixture | numpy, gguf-py |
| [`tests/python/qwen36_mtp_fixture.py`](../../../tests/python/qwen36_mtp_fixture.py) | 158 | fixture | numpy, gguf-py |
| [`tests/python/qwen36_runner_fixture.py`](../../../tests/python/qwen36_runner_fixture.py) | 109 | fixture | numpy, gguf-py |
| [`tests/python/qwen36_runner_check.py`](../../../tests/python/qwen36_runner_check.py) | 186 | checker (drives the `strata` binary) | numpy, CUDA device |

## Running the tool tests

From the repository root (gguf-py found through `STRATA_GGUF_PY` or `third_party/llama.cpp/gguf-py`):

```
python3 -m unittest tools.test_iq_pack tools.test_make_profile tools.test_draft_vocab tools.test_mtp_fetch
python3 -I tools/test_qwen36_oracle.py
```

Observed in this checkout (submodule not checked out, no `regex`): `test_make_profile`, `test_draft_vocab` and
`test_mtp_fetch` run (21 tests) with one error, `test_mtp_fetch.Setup.test_setup_rebuilds_a_corrupt_install`,
because it imports upstream's removed `setup.py`. The gguf-py-dependent suites exit at import through
`_paths.gguf_py()` when gguf-py is absent.

### What each suite pins

- **`test_iq_pack.py`**. `write_gguf` builds tiny (optionally sharded) GGUFs with gguf-py.
  `CompatibilityTests`: BF16 halfway rounding to even, non-finite refusal, `--compat-bf16` conversions and
  `compat-bf16.json`, inexact F32 router refused by default, byte-identical output for an already-BF16 pack, native
  PLE key, `experts.bin` reuse only with the same blobs, `--layout-only`, the nextn block left out, fused gate/up
  refused, HF snapshot symlinks. `ConversionTests`: exact F32 to BF16, F32 to F16 by the loader (kind 3), Q8_0
  projections. `SplitArtifactTests`: four shards with a layer split per role (v4 file), unsplit layers keep v3,
  missing / duplicate / foreign / truncated shards, a refused model leaves the pack alone, interrupted packs, a
  stale same-size `experts.bin`. `Qwen36PackTests`: forms and index contents, the BF16-exact rule with the count,
  native projection types refused by name, the nextn block left to the drafter, the expert profile check and copy,
  and that the Flash-Next `FORM` table is unaffected.
- **`test_make_profile.py`**: `rank_profile` modes (complete base hides traces, `--reorder`, frequency and tie
  order, append-only default, `--no-base`), end-to-end CLI runs, text traces with `--n-layer`, the base header
  deciding the layer count.
- **`test_draft_vocab.py`**: `covering` (most frequent first until the share is covered, ties to the lower id),
  `outside`, `corpus_counts` over several files.
- **`test_mtp_fetch.py`**: a fake safetensors checkpoint behind a mocked `urlopen`: range checks, a mirror that
  ignores `Range`, honest mirror, wrong bytes re-fetched once then refused, inventory from another revision,
  `verify` cache and revision skip, every tensor hashed.
- **`test_qwen36_oracle.py`**: see [reference-model.md](reference-model.md) and
  [qwen36-verification.md](qwen36-verification.md); it writes synthetic files to a temporary directory and checks the
  oracle's invariants and that every rival reading moves its seam by more than 5 % relative L1.

## The `tests/python` fixtures

All three generators share the skip protocol: if numpy or gguf-py cannot be imported they write `<out>/SKIP` with
the reason and exit 0, and the consuming C++ test exits 77 (CTest `SKIP_RETURN_CODE`). CTest sets
`STRATA_GGUF_PY=<repo>/third_party/llama.cpp/gguf-py` for them, and they pass it on to the oracle subprocesses.

### `qwen36_prefill_fixture.py --out DIR [--seed 0] [--cases gdn3,full4] [--no-oracle]`

Synthetic `qwen35moe` models at the **real per-layer geometry** (`n_embd 2048`, GDN 16/32 heads of 128, attention
16/2 heads of 256, `n_ff 512`, shared 512) with 16 experts top-8 (`cfg`, `:74`): `gdn3` (3 DeltaNet layers, no
attention layer) and `full4` (3 DeltaNet + 1 attention). Its own `write` (`:86`) stores projections, experts and the
head as Q8_0, the routers / `ssm_alpha` / `ssm_beta` / shared gate as BF16-exact F32 (router gain halved so the
renormalization rival is visible), the rest F32. For 40 tokens it runs the oracle into `<case>/o`, `<case>/o_q8` and
negative controls `<case>/neg_gate|pair|norm|attngate`. CTest: `prefill_qwen36_fixtures` ->
`prefill_qwen36_test` (`gdn3`) and `prefill_qwen36_attn_test` (`full4`).

### `qwen36_mtp_fixture.py --out DIR [--seed 0] [--no-oracle]`

`full4` plus a nextn block (`blk.4`), embedding and `nextn.eh_proj` also Q8_0 -> `mtp5.gguf`; the draft pack from
`tools/mtp_qwen36_pack.py --draft-vocab none` -> `pack/`; oracle `--mtp` dumps for 76 tokens (75 nextn positions,
enough for the prompt-side draft pass) in `o`, `o_mtp`, `o_q8`, `o_q8_mtp`. CTest: `qwen36_mtp_fixtures` ->
`qwen36_mtp_test`.

### `qwen36_runner_fixture.py --out DIR [--seed 0] [--no-oracle]`

`full4` with a Q8_0 `token_embd` -> `full4.gguf`; a 4 x 16 `profile.bin` ranking every pair
(`make_profile.write_profile`); a native pack from `tools/iq_pack.py --expert-profile profile.bin` into `pack/`
(stub `tokenizer/vocab.json` and `chat_template.jinja` are written first so the tokenizer export is skipped);
oracle dumps with `--all-logits` for `o8` ("hello wo"), `o11` ("hello world"), `o40` (40 tokens), each also
`_q8`. CTest: `qwen36_runner_fixtures` -> `qwen36_verify_test` and `qwen36_runner_check`.

### `qwen36_runner_check.py --strata BIN --fixture DIR [--keep]`

Runs the engine binary (`strata --native full4.gguf --pack pack --tokens ... --greedy --spec 4 --kv fp16
--expert-profile ... --expert-cache 64 --spec-oracle FILE --teacher-force --dump-logits FILE --suffix-draft 0`,
`run_engine`, `:44`) three ways: `c7` (prompt chunks of 7 up to position 35), `c40` (one prompt chunk of 35), `ar`
(prompt of one token, then decode windows). Each dumped logits row must be within 5e-2 relative L2 of the Q8
oracle's `result_output_all` and agree on the argmax wherever the F32 and Q8 oracles agree; `c7`, `c40` and `ar`
must agree with each other. A mutation gate then flips 4 KB of `blk.1.ffn_gate_inp.weight` in a copy of the pack's
`dense.bin` (located through `index.txt`) and requires the logits to move by more than 1e-2 relative L1. Exit 0, 1,
or 77 (no fixture, or no CUDA device).

## CTest wiring

These are registered only when `STRATA_BUILD_TESTS` is on (default on only if `tests/CMakeLists.txt` and
`bench/micro` exist, `CMakeLists.txt:87-92`) and an interpreter with numpy is found; the CUDA consumers also need
`STRATA_ENABLE_CUDA`. The Python `unittest` suites in `tools/` are not registered with CTest.

## Gotchas

- `tests/python` fixtures can take long (CTest `TIMEOUT 1800`): the oracle runs a real-width model in numpy.
- The SKIP path covers a missing numpy, not a missing gguf-py: importing `qwen36_synth_gguf` calls
  `_paths.add_gguf_py()`, which calls `sys.exit` (a `SystemExit`, not an `ImportError`). Verified in this checkout
  (no submodule): `python3 -I tests/python/qwen36_prefill_fixture.py --out X --no-oracle` prints "llama.cpp's gguf-py
  was not found" and exits 1 without writing `X/SKIP`, so CTest would report the fixture as failed rather than
  skipped. (The `cmake/qwen36_*.cmake` fixture scripts do match the "gguf-py was not found" message in some cases.)

## See also

- [reference-model.md](reference-model.md), [qwen36-verification.md](qwen36-verification.md),
  [pack-builders.md](pack-builders.md), [mtp-tools.md](mtp-tools.md)
- [docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md),
  [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md),
  [docs/qwen36-port/step4-runner.md](../../qwen36-port/step4-runner.md)
