[Codebase docs](../../README.md) › [Kernels](../README.md) › [Parity tests](README.md) › Fixtures and running

# Test fixtures, oracle data and how to run the parity tests

## Purpose

Where each parity test's reference data comes from, how CTest orders the generators before the tests, which inputs
are checked into the repository and which must be supplied, and the build switches and exit-code conventions. Files
covered: the four `cmake/qwen36_*` fixture scripts, the two generated `src/kernels/*.inc` tables, the
`tests/data/native_experts/` layer tables, and the python generators the CMake tests call.

## Files

| File | Lines | Role |
|---|---|---|
| [cmake/qwen36_synth_fixtures.cmake](../../../../cmake/qwen36_synth_fixtures.cmake) | 27 | `tools/qwen36_synth_gguf.py` -> `toy.gguf`, `toy-nonextn.gguf`, `toy-fused-gate-up.gguf`, `toy-fused-qkv.gguf` |
| [cmake/qwen36_block_fixtures.cmake](../../../../cmake/qwen36_block_fixtures.cmake) | 28 | `ref/qwen36.py` dumps of `toy.gguf` on 8 tokens (`o8/`) |
| [cmake/qwen36_attn_fixtures.cmake](../../../../cmake/qwen36_attn_fixtures.cmake) | 39 | Three attention GGUFs at head_dim 256 (g8 16/2, g12 24/2, g2 4/2) and their dumps on 40 tokens |
| [cmake/qwen36_oracle_dumps.cmake](../../../../cmake/qwen36_oracle_dumps.cmake) | 35 | `k.gguf` at the engine's GDN state size, 256 experts top-8, Q8_0 shared expert 512; dumps `o_f32/` and `o_q8/` on 11 tokens |
| [src/kernels/ple_oracle_vectors.inc](../../../../src/kernels/ple_oracle_vectors.inc) | 105 | Hash cases, probe rows of the PLE table, dense-pack regions (for `ple_parity`) |
| [src/kernels/qsa_oracle_vectors.inc](../../../../src/kernels/qsa_oracle_vectors.inc) | 134 | `{f32 bits, numpy float16 bits}` pairs (for `qsa_parity`) |
| [tests/data/native_experts/](../../../../tests/data/native_experts/) | 8 files, 376 lines | Copies of real packs' `native_experts.txt` (for `tests/core/expert_layout_test.cpp`) |

The python generators (`tools/iq_fixture.py`,
`tools/qwen36_synth_gguf.py`, `ref/qwen36.py`, `tests/python/qwen36_{runner,prefill,mtp}_fixture.py`,
`tests/python/qwen36_runner_check.py`) belong to the tools area ([../../tools/qwen36-verification.md](../../tools/qwen36-verification.md)).

## Build switches

| Switch | Default here | Effect on tests |
|---|---|---|
| `STRATA_BUILD_TESTS` | **OFF** (it defaults ON only if `tests/CMakeLists.txt` and `bench/micro/` exist; neither is in this checkout) | Gates the CPU tests, `qsa_topk_active_parity`, `native_expert_parity_*` ctests, the `tests/cuda/qwen36_block/verify` harnesses, the prompt-expert tests, `direct_file_async_test`, `dequant_bf16_test`, and the `_q36_python` probe |
| `STRATA_ENABLE_CUDA` / `STRATA_ENABLE_HIP` / `STRATA_HIP_GFX906` | OFF | Any GPU backend builds all the `src/kernels/*_parity` executables and registers their ctests |
| `STRATA_NATIVE_EXPERTS` | ON | Native expert tests and `router_dot_parity`; requires `third_party/llama.cpp` |
| `STRATA_MMQ_KQUANTS` | OFF | `prefill_mmq_kquant_test` |
| `STRATA_PARITY_PROMPT_ATTN` | unset | Builds `qsa_prompt_attn_parity` (CUDA) / `qsa_select_bench` (HIP) without the rest of the tests |
| `IQ_FIXTURE_PYTHON` | unset | Overrides the numpy-python search for `iq_parity_fixtures` and the prefill / MTP fixtures |
| `STRATA_PLE_FIXTURE_DIR` | `bench/micro` | Where `ple_parity` looks for `ple_in.bin` / `ple_out.bin` |
| `STRATA_ORACLE_SOURCE_DIR`, `STRATA_ORACLE_BUILD_DIR` | empty | With CUDA, builds `bench/micro/native_*_parity.cpp` against a separate llama.cpp build; those sources are not in this checkout |

## Python probes

Two independent probes look for an interpreter that can `import numpy`: the configured `Python3_EXECUTABLE`, then
`.venv/bin/python`, `/usr/bin/python3`, `python3`.

- `_iq_python` (`CMakeLists.txt:676-703`, inside the GPU block; `.venv/Scripts/python.exe` also tried): gates
  `iq_parity_fixtures` / `iq_parity`, `prefill_qwen36_fixtures` and `qwen36_mtp_fixtures`.
- `_q36_python` (`CMakeLists.txt:239-249`, inside `if(STRATA_BUILD_TESTS)`): gates `qwen36_synth_fixtures`,
  `model_geometry_test`, the block / runner harnesses, `qwen36_oracle_dumps` and `qwen36_attn_fixtures`.

The generators also need the vendored **gguf-py** from `third_party/llama.cpp/gguf-py` (passed as `STRATA_GGUF_PY`
to the tests/python scripts). Without numpy or gguf-py each generator writes a `SKIP` marker (python scripts exit 0
or 3; the cmake scripts exit 0) and the dependent test exits 77.

## CTest fixture graph

```
qwen36_synth_fixtures ──FIXTURES_SETUP qwen36_synth_data──┬─> model_geometry_test
                                                          └─> qwen36_block_fixtures ─ qwen36_block_data ─> qwen36_block_test
qwen36_runner_fixtures ─ qwen36_runner_data ─> qwen36_verify_test, qwen36_runner_check
qwen36_oracle_dumps    ─ qwen36_oracle_data ─> qwen36_seam_parity
qwen36_attn_fixtures   ─ qwen36_attn_data   ─> qsa_dense_parity --fixtures
iq_parity_fixtures     ─ iq_fixture_data    ─> iq_parity
prefill_qwen36_fixtures ─ qwen36_prefill_data ─> prefill_qwen36_test, prefill_qwen36_attn_test
qwen36_mtp_fixtures    ─ qwen36_mtp_data    ─> qwen36_mtp_test
```

All generated data goes into the build directory (`<build>/qwen36_synth`, `qwen36_block`, `qwen36_runner`,
`qwen36_oracle`, `qwen36_attn`, `iq_fixture`, `qwen36_prefill`, `qwen36_mtp`); the source tree is not touched.

## Inputs that are not generated

| Input | Needed by | Notes |
|---|---|---|
| `pack/full/experts.bin` (canonical Flash-Next Q2_0 pack) | `expert_parity`, `pool_test` (working dir = source root); `expert_arena_load` is registered only if it exists | Missing = exit 2 (failure) for the first two |
| `pack/full/dense.bin` | `ple_parity` | Regions listed at the end of `ple_oracle_vectors.inc` |
| Flash-Next Q2_0 GGUF shard 2 (the IQ4_NL PLE table) | `ple_parity` | `--gguf`, `$STRATA_PLE_GGUF`, else `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`; the file must be exactly 192 + 320,001,536 x 90 = 28,800,138,432 bytes |
| `ple_in.bin`, `ple_out.bin` (a ggml capture of the PLE layer) | `ple_parity` | From upstream's `bench/micro/ple_layer_xcheck.cpp`, which is not in this checkout; `--check-fixtures` validates them on the CPU only |
| Real model shards | `dequant_bf16_test`, `native_expert_bench`, `native_expert_parity <shard>`, `ple_fp8_parity`, `ple_q5_parity`, `q5_projection_parity`, `qwen36_mtp_test --real` | Manual runs only |

## The `.inc` oracle vectors

- `ple_oracle_vectors.inc` says it is generated by `tests/python/gen_ple_oracle.py` from `ref/ngram.py::ngram_rows`,
  the original GGUF's bytes read with numpy, and `pack/full/manifest.json`. It holds six hash cases (window
  direction, cut propagation, NULL reads as EOS, token id 0 is not missing, own EOS does not cut, real vocabulary
  ids), probe values and L1 sums of rows 0, 12345 and 320001535, the expected table geometry and the dense-pack
  offsets of the PLE key/value weights.
- `qsa_oracle_vectors.inc` says it is generated by `scripts/_gen_f16_oracle.py`: f32 -> f16 conversions as numpy
  performs them (zeros, overflow to inf, subnormals, ties).
- Neither generator script (nor `ref/ngram.py`, `ref/gdn.py`, `ref/qsa.py`, `ref/moe.py`, `ref/quant.py` that several
  test comments cite) is in this checkout; `ref/` holds only `load.py` and `qwen36.py`. The vectors are frozen data.

## `tests/data/native_experts/`

Eight `native_experts.txt` tables copied from real packs: Flash-Next `q2_0` (v2), `iq2_xs`, `iq3_xxs`, `iq3_s`,
`coder-iq1_m`, `swift-iq2_xs`, and Qwen3.6-35B-A3B `qwen36-iq3_xxs`, `qwen36-q4_k_xl` (v3, 40 layers, 256 experts).
`expert_layout_test` (`CMakeLists.txt:474-476`, passed the directory as its argument) parses each against the
pre-v4 reader's rules. The format is described in [../cpu/expert-layout.md](../cpu/expert-layout.md).

## Exit codes

| Code | Meaning | Where |
|---|---|---|
| 0 | Pass (also used by some tests for an explicit skip on missing CPU features, e.g. `expert_parity`, `pool_stress`, `decode_cluster_parity` below sm_90) | all |
| 1 | Failure | all |
| 2 | Bad usage or a missing required input (a failure) | `expert_parity`, `pool_test`, `ple_parity` |
| 3 | Fixture generator could not run (no numpy / gguf-py): `SKIP_RETURN_CODE 3`; `iq_parity` uses 3 too; the python `prefill` / `mtp` generators | `iq_parity_fixtures`, `iq_parity`, `prefill_qwen36_fixtures`, `qwen36_mtp_fixtures`, `model_geometry_test` |
| 77 | Skip: no device, unsupported architecture, or fixture `SKIP` marker (`SKIP_RETURN_CODE 77`) | `tests/cuda/*`, `qsa_dense_parity`, `qwen36_seam_parity`, `qsa_topk_active_parity`, `qsa_attn_bench`, `prefill_fused_*` |
| 3 (`pool_stress`) | Watchdog: a batch did not return within 10 s | `pool_stress` |

## How to run

```
git submodule update --init                       # third_party/llama.cpp: ggml, gguf-py
cmake -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON [-DSTRATA_MMQ_KQUANTS=ON]
cmake --build build -j
ctest --test-dir build --output-on-failure        # everything registered
ctest --test-dir build -R 'parity'                # the kernel parity tests
ctest --test-dir build -R 'qwen36'                # the Qwen3.6 harnesses with their fixtures
build/native_expert_parity --geom 2048x512 --synthetic iq2_s/iq3_xxs   # one case by hand
```

Most GPU parity tests take `--selftest` (their ctest form); several take `--bench` for timings
(`decode_cluster_parity`, `iq_multi_parity`, `native_grouped_parity`, `s2_expert_grouped_parity`). Tests that need a
model or a pack (table above) fail with exit 2 rather than pass when the input is absent.

## Gotchas

- With a GPU backend on but `STRATA_BUILD_TESTS` off, `_q36_python` is never set, and `if(NOT _q36_python STREQUAL "")`
  is then **true** (CMake compares an undefined name as its literal text; checked with `cmake -P`). So
  `qwen36_oracle_dumps` and `qwen36_attn_fixtures` are registered with an empty `PYTHON`, and `qwen36_seam_parity` /
  `qsa_dense_parity --fixtures` depend on them. Configure with `-DSTRATA_BUILD_TESTS=ON` to get the probe.
- `iq_parity_fixtures` and the python fixture scripts exit 3 for a missing numpy / gguf-py and that counts as a skip;
  a fixture that silently produced nothing is what the `SKIP` marker / exit 77 discipline prevents.
- Some test comments cite upstream paths that are absent here (`bench/micro/...`, `ref/*.py` other than `qwen36.py`,
  `scripts/`, `docs/semantics.md`, `docs/activation-contract.md`).

## See also

- [README.md](README.md), [model-tests.md](model-tests.md), [../cpu/tests.md](../cpu/tests.md)
- [../../build/README.md](../../build/README.md), [../../tools/qwen36-verification.md](../../tools/qwen36-verification.md)
- [docs/qwen36-port/oracle-crosscheck.md](../../../qwen36-port/oracle-crosscheck.md) (tolerance policy)
