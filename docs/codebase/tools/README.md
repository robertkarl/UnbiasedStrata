[Codebase docs](../README.md) › Tools

# Offline tooling and the numpy reference model

`tools/` and `ref/` hold everything that runs **before** or **beside** the engine rather than inside it. The pack
builders turn a model GGUF into the directory the engine loads with `--pack` (`index.txt`, `dense.bin`,
`native_experts.txt`, `conversions.json`, `tokenizer/`, `expert-profile.bin`) and the draft-head directory it loads
with `--mtp` (`dense.txt`, `dense.bin`, `experts.bin`, `native_experts.txt`, `mtp.json`, `draft_vocab.bin`); the
engine's loaders (`src/core/weights.cpp`, `src/kernels/cpu/expert_layout.cpp`, `src/core/expert_cache.cpp`,
`src/core/mtp.cpp`) are their only consumers. The Qwen3.6 verification tools and `ref/qwen36.py`, a numpy oracle
written from the pinned llama.cpp graph, produce the reference tensors that the engine's CUDA parity tests are held
to, and compare the oracle with llama.cpp itself through small C++ programs linked against
`third_party/llama.cpp`. Nothing here is compiled into the `strata` binary; the Python tools need Python 3 with
numpy and, for most of them, llama.cpp's gguf-py (located by `STRATA_GGUF_PY` or the submodule).

## Contents

| Page | What it covers |
| --- | --- |
| [pack-format.md](pack-format.md) | **Authoritative** on-disk formats: trunk native pack, Qwen3.6 draft pack, Flash-Next draft runtime, `expert-profile.bin`, `draft_vocab.bin`, legacy canonical pack |
| [pack-builders.md](pack-builders.md) | `iq_pack.py` (tensor classification, conversions, refusals, CLI), `gguf_reader.py`, `gguf_writer.py`, `_paths.py`, `strata_tokenizer.py`, legacy `strata_pack.py` / `pack_layer.py` / `pack_index.py` |
| [profile-and-draft-vocab.md](profile-and-draft-vocab.md) | `make_profile.py` (expert-cache ranking) and `draft_vocab.py` (draft-head token subset) |
| [mtp-tools.md](mtp-tools.md) | Flash-Next `mtp_fetch.py` -> `mtp_pack.py` -> `mtp_rt.py`; Qwen3.6 `mtp_qwen36_pack.py` |
| [qwen36-verification.md](qwen36-verification.md) | Synthetic GGUFs, llama.cpp dumper and teacher, dump / eval-callback / logits comparators, dequant check, spread and router studies, `iq_fixture.py`, `qwen36_crosscheck.sh` |
| [reference-model.md](reference-model.md) | `ref/load.py` and `ref/qwen36.py`: structure, rival readings, CLI, how the oracle feeds the cross-check and the CUDA tests |
| [python-tests.md](python-tests.md) | `tools/test_*.py` suites and the `tests/python/qwen36_*` fixture generators and runner check |

`tests/python/serve_max_tokens_check.py` belongs to the front end: see [../frontend/README.md](../frontend/README.md).

## Source map

| Path | Lines | Role |
| --- | --- | --- |
| [tools/iq_pack.py](../../../tools/iq_pack.py) | 763 | Native pack builder for both models (index, dense floats, expert map, profile) |
| [tools/gguf_reader.py](../../../tools/gguf_reader.py) | 142 | GGUF v3 header reader without gguf-py (knows `Q2_0` = 42) |
| [tools/gguf_writer.py](../../../tools/gguf_writer.py) | 234 | Minimal GGUF writer and the `Q2_0` codec |
| [tools/_paths.py](../../../tools/_paths.py) | 28 | Finds gguf-py: `STRATA_GGUF_PY`, then the submodule |
| [tools/strata_tokenizer.py](../../../tools/strata_tokenizer.py) | 339 | Reference byte-level BPE; exports `<pack>/tokenizer/` |
| [tools/strata_pack.py](../../../tools/strata_pack.py) | 423 | Legacy canonical Q2_0 pack build/verify/info (missing `canonical_xcheck` import) |
| [tools/pack_layer.py](../../../tools/pack_layer.py) | 123 | Legacy canonical expert arena (1,382,400-byte blobs) |
| [tools/pack_index.py](../../../tools/pack_index.py) | 260 | Legacy: `manifest.json` -> 19-column `index.txt` |
| [tools/make_profile.py](../../../tools/make_profile.py) | 176 | `expert-profile.bin` from a base and routing traces |
| [tools/draft_vocab.py](../../../tools/draft_vocab.py) | 146 | Extends `draft_vocab.bin` by script or corpus coverage |
| [tools/mtp_fetch.py](../../../tools/mtp_fetch.py) | 308 | Range-reads Flash-Next's `mtp.*` tensors from Hugging Face, SHA-256 checked |
| [tools/mtp_pack.py](../../../tools/mtp_pack.py) | 193 | Flash-Next MTP tensors -> MTP GGUF (Q2_0/Q4_0/Q8_0 experts) |
| [tools/mtp_rt.py](../../../tools/mtp_rt.py) | 111 | Flash-Next MTP GGUF -> `experts.bin`, `dense.bin`, `dense.txt` |
| [tools/mtp_qwen36_pack.py](../../../tools/mtp_qwen36_pack.py) | 246 | Qwen3.6 `-MTP` GGUF draft block -> draft pack with `mtp.json` |
| [tools/qwen36_synth_gguf.py](../../../tools/qwen36_synth_gguf.py) | 373 | Tiny deterministic `qwen35moe` GGUFs |
| [tools/qwen36_dump_llama.cpp](../../../tools/qwen36_dump_llama.cpp) | 171 | llama.cpp graph, every named tensor dumped (`qwen36-dump`) |
| [tools/qwen36_teacher_llama.cpp](../../../tools/qwen36_teacher_llama.cpp) | 199 | llama.cpp greedy continuation with logits and routes (`qwen36-teacher`) |
| [tools/qwen36_dequant_ggml.cpp](../../../tools/qwen36_dequant_ggml.cpp) | 30 | ggml `to_float` on one tensor slab (`qwen36-dequant`) |
| [tools/qwen36_crosscheck.sh](../../../tools/qwen36_crosscheck.sh) | 82 | Oracle vs llama.cpp on all synthetic cases, one command |
| [tools/qwen36_compare_dumps.py](../../../tools/qwen36_compare_dumps.py) | 85 | Seam-by-seam dump comparison |
| [tools/qwen36_check_eval_callback.py](../../../tools/qwen36_check_eval_callback.py) | 130 | `llama-eval-callback` printout vs oracle dumps |
| [tools/qwen36_check_dequant.py](../../../tools/qwen36_check_dequant.py) | 55 | `ref/load.py` vs ggml dequantization on a real file |
| [tools/qwen36_spread.py](../../../tools/qwen36_spread.py) | 37 | Per-seam-class error table on a real file |
| [tools/qwen36_router_bf16_flips.py](../../../tools/qwen36_router_bf16_flips.py) | 111 | Top-k flips from BF16 router rounding |
| [tools/qwen36_compare_logits.py](../../../tools/qwen36_compare_logits.py) | 155 | Engine vs reference logits: top-1, KL, drift, routes |
| [tools/iq_fixture.py](../../../tools/iq_fixture.py) | 137 | i-quant block fixtures for `iq_parity` |
| [tools/test_iq_pack.py](../../../tools/test_iq_pack.py) | 840 | Tests of `iq_pack.py` (incl. Qwen3.6) |
| [tools/test_make_profile.py](../../../tools/test_make_profile.py) | 145 | Tests of `make_profile.py` |
| [tools/test_draft_vocab.py](../../../tools/test_draft_vocab.py) | 47 | Tests of `draft_vocab.py` corpus mode |
| [tools/test_mtp_fetch.py](../../../tools/test_mtp_fetch.py) | 201 | Tests of `mtp_fetch.py` (mocked HTTP) |
| [tools/test_qwen36_oracle.py](../../../tools/test_qwen36_oracle.py) | 419 | Oracle and synthetic-fixture invariants, rival readings |
| [ref/load.py](../../../ref/load.py) | 217 | Lazy GGUF -> FP32 tensors, per expert and per row |
| [ref/qwen36.py](../../../ref/qwen36.py) | 683 | Qwen3.6 numpy oracle and dump writer |
| [tests/python/qwen36_prefill_fixture.py](../../../tests/python/qwen36_prefill_fixture.py) | 186 | Real-width synthetic models + oracle dumps for the prompt-path test |
| [tests/python/qwen36_mtp_fixture.py](../../../tests/python/qwen36_mtp_fixture.py) | 158 | Same with a nextn block, plus its draft pack, for the drafter test |
| [tests/python/qwen36_runner_fixture.py](../../../tests/python/qwen36_runner_fixture.py) | 109 | Synthetic model, native pack, profile and oracle dumps for the runner tests |
| [tests/python/qwen36_runner_check.py](../../../tests/python/qwen36_runner_check.py) | 186 | Runs `strata` on that pack and holds its logits to the oracle |

## Workflow: preparing model files (UPSTREAM.md)

```
                    Qwen3.6 (this repo's tools)                         Qwen3.8 Flash-Next (upstream's tools)
  Qwen3.6-35B-A3B-UD-{Q4_K_XL,IQ3_XXS}.gguf                     Flash-Next GSQ-RCO IQ3_XXS shards
            │ tools/iq_pack.py --gguf F --out qwen36-{q4,iq3}             │ iq_pack.py --gguf shard1 --out pack-iq3-xxs
            │   ├─ strata_tokenizer.py -> tokenizer/                      │ cp data/expert-profile.bin
            │   └─ copies data/qwen36-expert-profile.bin                  │
            ▼     (built by make_profile.py from a route trace)           ▼
  index.txt dense.bin native_experts.txt conversions.json         same file set
  tokenizer/ expert-profile.bin

  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf (-MTP repo, blk.40)            HF BF16 checkpoint, 31 mtp.* tensors
            │ tools/mtp_qwen36_pack.py --gguf F --out qwen36-mtp-draft    │ mtp_fetch.py fetch/verify -> mtp_pack.py
            ▼                                                             ▼   -> mtp_rt.py; cp data/draft_vocab.bin
  dense.txt dense.bin experts.bin native_experts.txt mtp.json     mtp/{experts.bin, dense.bin, dense.txt,
  draft_vocab.bin                                                      draft_vocab.bin}

  run: ./unbiased-strata --model <model.gguf> --pack <pack> --mtp <draft dir> ...   (top-level README.md)
       the front end starts the engine, whose own flags are --native, --pack, --mtp, --expert-profile
       (src/program/generate.cpp)
```

UPSTREAM.md records the exact commands and the SHA-256 of every produced file; building twice gives the same bytes.
`dense.bin` is identical for the Qwen3.6 Q4_K_XL and IQ3_XXS packs, and one draft pack serves both trunks.

## Workflow: verifying the Qwen3.6 port

```
 qwen36_synth_gguf.py ─► toy GGUF ─┬─► ref/qwen36.py --dump ──────┐
                                   ├─► qwen36-dump (llama.cpp) ───┴─► qwen36_compare_dumps.py   (qwen36_crosscheck.sh)
                                   └─► llama-eval-callback ──────────► qwen36_check_eval_callback.py
 CMake / tests/python fixtures ─► synthetic GGUF + ref/qwen36.py dumps ─► CUDA parity tests (seam, attention, block,
                                                                           prefill, drafter, verify window, runner)
 real GGUF ─► qwen36_check_dequant.py, qwen36_spread.py, qwen36_router_bf16_flips.py (tolerances, pack decisions)
 real GGUF ─► qwen36-teacher ─► logits ─┐
 strata --dump-logits ──────────────────┴─► qwen36_compare_logits.py
```

## Key concepts

- **Native pack.** Quantized weights are never copied; the engine reads them from the GGUF. The pack holds the
  float tensors in the form each kernel reads (BF16 routers and GDN gates, F32 norms), an index whose native rows
  carry only a shape, and the per-layer expert map with absolute shard offsets.
- **Exact by default.** A conversion that would round (F32 to BF16 with nonzero low bits, quantized to float) is
  refused unless `--compat-bf16`; every conversion is recorded in `conversions.json`. Both Qwen3.6 files convert
  exactly.
- **Refuse before writing; publish last.** The expert table, the profile and every tensor's type are validated
  before any file changes, and `native_experts.txt` is the completion marker written last.
- **Oracle with rivals.** `ref/qwen36.py` names each plausible misreading of the graph (`Model.opts`); tests and
  negative-control dumps require each rival to change its seam by more than 5 %.
- **Shared dump format.** `<stem>.f32` + `manifest.txt`, stems are llama.cpp `cb()` names with `#N` for repeats,
  written identically by the oracle and the llama.cpp dumper.

## Configuration

| Variable / option | Used by |
| --- | --- |
| `STRATA_GGUF_PY` | `_paths.py` callers: `iq_pack.py`, `mtp_*.py`, `qwen36_synth_gguf.py`, `ref/qwen36.py`, `qwen36_check_dequant.py`, `test_iq_pack.py`, the `tests/python` fixtures (CTest sets it to the submodule's gguf-py) |
| `STRATA_MTP_REVISION`, `HF_ENDPOINT` | `mtp_fetch.py` |
| `QWEN36_THREADS` | `qwen36-dump` |
| `LLAMA_CPP_DIR`, `PYTHON` | `qwen36_crosscheck.sh` |
| `IQ_FIXTURE_PYTHON`, `STRATA_BUILD_TESTS` | CMake: interpreter for `iq_fixture.py`; registration of the fixture tests |

## Known limitations in this checkout

- `third_party/llama.cpp` is a submodule; without it gguf-py and every C++ verification tool are unavailable, and
  the gguf-py-dependent tools exit with "llama.cpp's gguf-py was not found".
- `tools/strata_pack.py` imports `canonical_xcheck`, which is not in the repository.
- `tools/test_mtp_fetch.py` has one test that imports the removed `setup.py`.
- `tools/gguf_writer.py`'s self-check passes an integer where `add_q2_0` expects a shape list and raises.

## See also

- [UPSTREAM.md](../../../UPSTREAM.md) (commands and checksums), [README.md](../../../README.md)
- [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) (what was restored from upstream),
  [docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md),
  [docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md),
  [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md)
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md)
