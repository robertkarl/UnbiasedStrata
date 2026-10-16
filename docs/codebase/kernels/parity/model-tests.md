[Codebase docs](../../README.md) › [Kernels](../README.md) › [Parity tests](README.md) › Model-level CUDA tests

# Model-level CUDA harnesses (`tests/cuda/`) and the async file test

## Purpose

Where the `src/kernels/*_parity.cpp` tests check one kernel family on synthetic data, the harnesses in `tests/cuda/`
drive whole engine paths - a block, the speculative verify window, the MTP drafter, the prompt path - on small
**synthetic GGUFs at the real per-layer geometry** and hold every intermediate ("seam", named as llama.cpp's `cb()`
names it) to the numpy oracle `ref/qwen36.py`. Most were written for the Qwen3.6-35B-A3B port (steps 3 and 4 in
`docs/qwen36-port/`). The prompt-expert tests (`prefill_fused_*`, `prefill_mmq_kquant_test`) compare the prompt
path's int8 expert kernels against a double-precision reference and against llama.cpp's MMQ path. Also covered:
`tests/platform/direct_file_async.cpp`.

## Files

| File | Lines | Links |
|---|---|---|
| [tests/cuda/qwen36_block_test.cpp](../../../../tests/cuda/qwen36_block_test.cpp) | 531 | `strata_engine`, `strata_artifact` |
| [tests/cuda/qwen36_verify_test.cpp](../../../../tests/cuda/qwen36_verify_test.cpp) | 511 | `strata_engine`, `strata_artifact` |
| [tests/cuda/qwen36_mtp_test.cpp](../../../../tests/cuda/qwen36_mtp_test.cpp) | 622 | `strata_prefill`, `strata_engine`, `strata_artifact`, cudart |
| [tests/cuda/prefill_qwen36_test.cpp](../../../../tests/cuda/prefill_qwen36_test.cpp) | 483 | `strata_prefill`, cudart |
| [tests/cuda/prefill_fused_moe_test.cpp](../../../../tests/cuda/prefill_fused_moe_test.cpp) | 420 | `strata_mmq`, `ggml-base`, cudart |
| [tests/cuda/prefill_fused_iq_test.cpp](../../../../tests/cuda/prefill_fused_iq_test.cpp) | 642 | `strata_mmq`, `ggml-base`, cudart |
| [tests/cuda/prefill_mmq_kquant_test.cpp](../../../../tests/cuda/prefill_mmq_kquant_test.cpp) | 209 | `strata_mmq`, `ggml-base`, cudart |
| [tests/platform/direct_file_async.cpp](../../../../tests/platform/direct_file_async.cpp) | 60 | compiled with `src/platform/direct_file.cpp`, Threads |

## The harnesses

### `qwen36_block_test` - the plain block (hc = 1)

- **Command / ctest**: `qwen36_block_test <build>/qwen36_synth <build>/qwen36_block/o8` (`TIMEOUT 120`,
  `SKIP_RETURN_CODE 77`).
- **Condition**: `STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND _q36_python` (`CMakeLists.txt:507-520`).
- **Fixtures**: `qwen36_synth_fixtures` (`cmake/qwen36_synth_fixtures.cmake`: `tools/qwen36_synth_gguf.py` writes
  `toy.gguf`, `toy-nonextn.gguf`, `toy-fused-gate-up.gguf`, `toy-fused-qkv.gguf`) then `qwen36_block_fixtures`
  (`cmake/qwen36_block_fixtures.cmake`: `ref/qwen36.py toy.gguf --tokens 104,101,108,108,111,32,119,111 --dump o8`).
- **Checks**: through the engine's own functions on a `WeightTable` loaded from a pack the test writes -
  `block_layer_pre` stage 0 (`attn_norm-N`), the residual add, `block_layer_pre` half 2 (`attn_post_norm-N`, router
  `ffn_moe_topk-N` and `ffn_moe_weights_norm-N`), `block_layer_post` (`ffn_out-N`, `l_out-N`), `lm_head_mix`
  (`result_norm`) and logits; the verify window's T = 8 row kernels (`plain_norm_rows`, `plain_residual_rows`,
  `native_moe_combine_multi`) bitwise the one-row calls; negative cases (the GR read at hc = 1 is not the plain norm,
  post-attention norm skipped, top-10 on the 8-expert file refused, missing norm refused).
- **Teacher forcing**: mixers and experts are taken from the dumps (the fixture's projections are F32 and the engine
  has no F32 projection kernel); their own errors are reported, not asserted.
- **Tolerances**: relative L2 1e-6 (norms, adds, MoE composition), 1e-5 (logits); router weights 1e-2 reported, picks
  identical.

### `qwen36_verify_test` and `qwen36_runner_check` - the verify window on a native pack

- **Command / ctest**: `qwen36_verify_test <build>/qwen36_runner` (`TIMEOUT 600`, skip 77); `qwen36_runner_check`
  runs `tests/python/qwen36_runner_check.py --strata $<TARGET_FILE:strata> --fixture ...` (`TIMEOUT 900`, skip 77).
- **Condition**: `STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND _q36_python` (`CMakeLists.txt:526-540`).
- **Fixture**: `qwen36_runner_fixtures` = `tests/python/qwen36_runner_fixture.py --out <build>/qwen36_runner`
  (`TIMEOUT 1800`, env `STRATA_GGUF_PY=third_party/llama.cpp/gguf-py`): the prompt-path fixture's `full4` model (4
  layers, three DeltaNet + one attention, n_embd 2048, 16 experts top-8 of 512) with a Q8_0 token embedding, the
  native pack built by `tools/iq_pack.py`, and oracle dumps `o8` ("hello wo"), `o11` ("hello world"), `o40`, each
  F32 and `_q8`.
- **Checks**: one 8-token window: every row's residual after every layer vs `l_out-N` and logits vs
  `result_output_all`; eight one-token windows **bit for bit** equal to the 8-row window; 8 + 3 windows with a commit
  between; the 40-token sequence one token per window with argmax vs the oracle. Gates `l_out` 1e-2 and logits 2e-2
  relative L2 per row. `qwen36_runner_check` then runs the real `strata` binary teacher-forced (chunkings c7, c40,
  ar) at 5e-2, plus a mutation gate: 4 KB of `blk.1.ffn_gate_inp.weight` flipped in a copy of `dense.bin` must move
  the logits by more than 1e-2.

### `prefill_qwen36_test` - the prompt path at Qwen3.6 geometry

- **Command / ctests**: `prefill_qwen36_test <build>/qwen36_prefill gdn3` and `prefill_qwen36_attn_test` with
  `full4` (`TIMEOUT 600`, skip 77). Options `--chunk N`, `--no-force`.
- **Condition**: `STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906` (target, no `STRATA_BUILD_TESTS` guard); ctests need
  `_iq_python` (`CMakeLists.txt:738-763`).
- **Fixture**: `prefill_qwen36_fixtures` = `tests/python/qwen36_prefill_fixture.py` (skip code 3, `TIMEOUT 1800`):
  `gdn3` (3 DeltaNet layers) and `full4`, Q8_0 projections and experts, BF16-exact routers, oracle dumps `o`, `o_q8`
  and negative-control dumps `neg_*` (sigmoid GDN gate, interleaved head pairing, unnormalized router, attention gate
  off).
- **Checks**: `Prefill::run` as one 40-token chunk and in chunks of 7, every `Prefill::on_seam` seam vs the oracle
  with the layer input teacher-forced to the oracle's previous `l_out`; router top-k exact; each negative-control dump
  more than 5% relative L1 from the engine; `Prefill::init` refuses top-10 on the 16-expert file and an unsupported
  GDN geometry.

### `qwen36_mtp_test` - the MTP drafter

- **Command / ctest**: `qwen36_mtp_test <build>/qwen36_mtp` (`TIMEOUT 900`, skip 77); by hand
  `qwen36_mtp_test --real --gguf FILE --pack DIR --dump D --mtp-dump D_mtp --tokens ...` against llama.cpp's own MTP
  graph dumped by `tools/qwen36_dump_llama.cpp`.
- **Condition**: same as `prefill_qwen36_test` (`CMakeLists.txt:764-778`).
- **Fixture**: `qwen36_mtp_fixtures` = `tests/python/qwen36_mtp_fixture.py`: `mtp5.gguf` (the `full4` trunk + nextn
  block `blk.4`), oracle dumps for 76 tokens (`o`, `o_mtp`, `o_q8`, `o_q8_mtp`) and a draft pack cut by
  `tools/mtp_qwen36_pack.py`.
- **Checks**: for k = 1..75, `MtpDrafter::debug_row` seams vs the Q8 oracle row k-1, top-8 identical as a set, draft
  equal to the oracle's argmax; negative cases (hidden state first in the concat, hnorm skipped, wrong position
  pairing); the captured `draft_first` equals the uncaptured draft; `Prefill::draft_kv` for cells 0..63. Gates from
  oracle-crosscheck.md's Q8 column with 4x margin (`attn_out` 3e-3, `moe_out`/`ffn_out` 2e-2, `l_out` 6e-3).

### `prefill_fused_moe_test`, `prefill_fused_iq_test`, `prefill_mmq_kquant_test` - prompt-path experts

- **Condition**: `STRATA_NATIVE_EXPERTS` and `STRATA_ENABLE_CUDA` and `STRATA_BUILD_TESTS` (the `strata_mmq`
  section); `prefill_mmq_kquant_test` additionally `STRATA_MMQ_KQUANTS`. Timeouts 300 / 600 / 120.
- `prefill_fused_moe_test`: the fused int8 kernels (`moe_fused.hpp`, opt-in `STRATA_PF_FUSED=1` in the engine) on
  random Strata Q2_0 blobs, 256 tokens over 64 experts with skewed routing, vs a double reference and vs the MMQ
  sequence of `prefill.cpp`; pass if the fused error is at most 1.5x MMQ's RMS and 2x its worst row; then a timing at
  2048 tokens. Exit 77 without an sm_80+ device.
- `prefill_fused_iq_test`: the same for native packs (`moe_fused_iq.hpp`), gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S
  IQ4_XS with down Q2_0 or IQ4_NL, at Flash-Next 2560/640 top-10 and Qwen3.6 2048/512 top-8; `--no-ref`,
  `--no-timing`, `--only=NAME`, `--chunks=A,B`. Exit 77 without sm_80+.
- `prefill_mmq_kquant_test`: MMQ for UD-Q4_K_XL's formats (Q4_K/Q5_K gate/up, Q5_1/Q8_0 down at 2560 x 640; Q5_K/Q6_K
  down at 2048 x 512) on ggml-quantized synthetic weights vs ggml's dequantized weights; a layout / stride / row-id
  screen, not bit-exact.

### `direct_file_async_test`

- **Condition**: `STRATA_BUILD_TESTS AND UNIX AND NOT APPLE`; built from the test plus `src/platform/direct_file.cpp`
  with Threads (no GPU).
- **Checks** on a temporary `/tmp/strata-direct-test-XXXXXX` of 16 pages + 17 bytes with `STRATA_IO_THREADS=4`: an
  unaligned offset is refused; 17 aligned 4 KiB reads complete once each with the right tags, the last as a 17-byte
  short read at EOF; `wake()` produces a `WAKE_TAG` completion; `close()` drains outstanding reads; a reopened file
  has no stale completion and reads again. Needs C++20 (`std::jthread`). The API is documented in
  [../../libs/platform.md](../../libs/platform.md).

## Model-specific behaviour

All `qwen36_*` and `prefill_qwen36_*` harnesses are Qwen3.6-35B-A3B (`qwen35moe`, hc = 1, no PLE, no indexer) only.
The prompt-expert tests cover both models' expert geometries.

## Gotchas

- The fixture generators need python3 with numpy **and** the vendored gguf-py from `third_party/llama.cpp`; without
  them they write `SKIP` and the tests exit 77, which CTest reports as skipped, not passed.
- `prefill_qwen36_test` and `qwen36_mtp_test` targets are built in every non-gfx906 CUDA build (no `STRATA_BUILD_TESTS`
  guard); only their ctests depend on the python probe.
- The `CMakeLists.txt` comment that `full4` exits 77 "until the attention branch's G = 8 kernels are in the tree" is
  older than the code: the test's own skip paths are only "no device" and the `SKIP` marker
  (`prefill_qwen36_test.cpp:338-347`).

## See also

- [README.md](README.md), [fixtures.md](fixtures.md)
- [docs/qwen36-port/step3-block.md](../../../qwen36-port/step3-block.md),
  [step3-prefill.md](../../../qwen36-port/step3-prefill.md), [step4-runner.md](../../../qwen36-port/step4-runner.md),
  [step4-drafter.md](../../../qwen36-port/step4-drafter.md), [oracle-crosscheck.md](../../../qwen36-port/oracle-crosscheck.md)
- [../../prefill/README.md](../../prefill/README.md), [../../core/README.md](../../core/README.md)
