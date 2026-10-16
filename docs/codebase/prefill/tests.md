[Codebase docs](../README.md) › [Prefill](README.md) › Tests

# Tests and diagnostics for the prompt path

## Purpose

What checks the prefill subsystem, what each test holds it to, and how to run it. Two of the test programs live in
`src/prefill/` itself (`gdn_rec_parity.cu`, `gemm_bf16_parity.cu`); the rest are in `tests/cuda/` and
`tests/python/`. All are synthetic (no model file) unless stated. Build with `STRATA_BUILD_TESTS=ON` (the fork's
`make build` turns tests off, `Makefile:31`); run with `ctest -R <name>` from the build directory or the binary
directly. Exit code 77 is CTest's skip.

## Inventory

| Test (CTest name) | Source | Registered | What it checks |
| --- | --- | --- | --- |
| `gdn_rec_parity` | [src/prefill/gdn_rec_parity.cu](../../../src/prefill/gdn_rec_parity.cu) (1094 lines) | `CMakeLists.txt:739` (CUDA, not gfx906) | the geometry table (48/16/10240 and 32/16/8192 built; 32/16/10240, 32/8/8192, 48/16/8192 refused); `gdn_rec_kh_kernel` vs `gdn_rec_cols_pipe_kernel` bit for bit (output, state, FP16 output) for T = 1 .. 4099 from a non-zero state at both geometries; the engine's `gdn_recurrence` / `gdn_conv` launches against a double-precision host transcription (decay before the rank-1 update, v-head h with k-head h % HK, gated RMS-normed output); rival readings (the other model's gate, interleaved head pairing) required to be more than 5 % away. `--bench`: timings at T = 8192 and 32768, block-per-SM scaling, and fewer-SM runs |
| `gemm_bf16_parity` | [src/prefill/gemm_bf16_parity.cu](../../../src/prefill/gemm_bf16_parity.cu) (155 lines) | `CMakeLists.txt:744` | `Gemm::bf16` vs `cublasGemmEx` BF16 within fp32-accumulation rounding over the prompt path's shapes (hc down with activation slicing, hc up, PLE value, router / indexer, hc inject, a single row, `ldy > N`, `beta = 1`). Exercises the FP16 path below sm_80 (`STRATA_BF16_TC=2` forces it on any card) and the FP32 path on Pascal. `--bench` adds timings |
| `prefill_qwen36_fixtures` | [tests/python/qwen36_prefill_fixture.py](../../../tests/python/qwen36_prefill_fixture.py) | `CMakeLists.txt:754` (needs a numpy interpreter) | writes `gdn3` (3 GDN layers) and `full4` (3 GDN + 1 attention layer) synthetic qwen35moe GGUFs at the real per-layer widths (2048, GDN 32/16 x 128, attention 16/2 x 256, 16 experts top-8 of 512, Q8_0 projections and experts), and the oracle's dumps (`ref/qwen36.py`) for 40 tokens: faithful, `--act-quant q8`, and rival readings. Writes `SKIP` without numpy / gguf-py |
| `prefill_qwen36_test` / `prefill_qwen36_attn_test` | [tests/cuda/prefill_qwen36_test.cpp](../../../tests/cuda/prefill_qwen36_test.cpp) (483 lines) | `CMakeLists.txt:759`, `:761` | `Prefill::run` at Qwen3.6's geometry (one chunk of 40 and chunks of 7) with every layer's input teacher-forced through `on_seam("l_in")`; per seam relative L1 gates: `attn_norm` 1e-4, `attn_post_norm` 2e-3, `linear_attn_out` / `attn_output` 1e-2, `ffn_out` 2e-2, `l_out` 1e-2; router top-k equal except near-tie swaps; negative controls more than 5 % away; `Prefill::init` refusals (top-10 on a top-8 file, a GDN geometry with no kernels). Usage: `prefill_qwen36_test <fixture dir> gdn3|full4 [--chunk N] [--no-force]` |
| `prefill_fused_moe_test` | [tests/cuda/prefill_fused_moe_test.cpp](../../../tests/cuda/prefill_fused_moe_test.cpp) (420 lines) | `CMakeLists.txt:1247` | the fused Q2_0 int8 experts vs a double-precision reference and vs the MMQ sequence (at most 1.5x MMQ's RMS error, 2x its worst row); skewed routing, an all-zero token, unrelated blob addresses; timing at 2048 tokens x 512 experts. Exit 77 below sm_80 |
| `prefill_fused_iq_test` | [tests/cuda/prefill_fused_iq_test.cpp](../../../tests/cuda/prefill_fused_iq_test.cpp) (642 lines) | `CMakeLists.txt:1252` | the same for the native fused kernels, every gate/up x down pair (IQ2_XXS .. IQ4_XS over Q2_0 / IQ4_NL), at both geometries (2560/640 top-10 and 2048/512 top-8); timing on 2048 / 3584 / 8192-token layers. `--no-ref`, `--no-timing`. Exit 77 below sm_80 |
| `prefill_mmq_kquant_test` | [tests/cuda/prefill_mmq_kquant_test.cpp](../../../tests/cuda/prefill_mmq_kquant_test.cpp) (209 lines) | `CMakeLists.txt:1259` (with `STRATA_MMQ_KQUANTS`) | MMQ products for Q4_K / Q5_K gate/up and Q5_1 / Q8_0 / Q5_K / Q6_K down, synthetic ggml-quantized weights vs a double-precision product: a screen for layout / stride / bound / row-id errors, and an uncovered format (Q3_K) must be reported |
| `qwen36_mtp_test` | [tests/cuda/qwen36_mtp_test.cpp](../../../tests/cuda/qwen36_mtp_test.cpp) | `CMakeLists.txt:768` | the Qwen3.6 drafter; part 4 runs `Prefill::init` + `Prefill::draft_kv` on 64 cells and checks it takes them batched |

`prefill_qwen36_test` and `prefill_qwen36_attn_test` share the fixture through CTest `FIXTURES_REQUIRED
qwen36_prefill_data`; run the fixture test first (`ctest -R prefill_qwen36`). The fixture generation takes minutes
(4m55s for both cases on the 4090 VM per the step-3 notes).

HIP tests referenced by CMake (`hip_prefill_gemm`, `hip_prefill_mmq_parity`, `hip_prefill_native_batch`,
`hip_prefill_hipblaslt_gemm`) have no sources in this tree (`tests/hip/` is absent).

## Recorded results

[docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md), section 3, has the numbers on the RTX 4090
(sm_89) and GTX 1080 Ti (sm_61): `gdn_rec_parity` 0 failures with identical digits on both cards; the fused tests
pass on the 4090 and skip on Pascal; `prefill_qwen36_test` passes both chunkings with `ffn_out` about 1e-2 (the MMQ
int8 rounding) and `l_out` 2e-3 to 4e-3 relative L1 against the F32 oracle. On Pascal the prompt MoE takes MMQ
(dp4a); `STRATA_PREFILL_MMQ=0` switches to the FP16 GEMM path, which is closer to the F32 oracle.

## Runtime diagnostics (env vars)

| Variable | Output |
| --- | --- |
| `STRATA_PREFILL_TIMING=1` | per-prompt GPU timeline by phase (embed, hc read, gdn, qsa proj / indexer / select / attn, router, host grouping, gather, wait copy, dequant, gemm gate/up, gemm down, combine, ple, kv stage, gdn conv / recurrence / out proj), host shares, and the peer GPU's timeline |
| `STRATA_PF_STEP_SYNC=1` | sync after each named step, log steps over 250 ms |
| `STRATA_PREFILL_ALLOC_TRACE=1` | which buffer did not fit (a count / carve mismatch, a failed `cudaMalloc`) |
| `STRATA_DBG_NAN=1` | first layer of a chunk with non-finite MoE values; state at prompt end |
| `STRATA_PREFILL_DUMP_R=<file>` | final residual rows every 64th position |
| `STRATA_STATE_HASH_GDN=1` | a 16-bit hash per GDN state after the prompt |
| `STRATA_TRACE=1` | a line per chunk |
| `STRATA_SEL_OVERLAP`, `STRATA_IDX_FP16_CHECK`, `STRATA_QSA_DUMP` | Flash-Next selection experiments |
| `STRATA_DRAFT_TIMING=1` | where `draft_kv`'s time goes |
| `STRATA_HIPBLASLT_VERBOSE=1` | HIP: hipBLASLt choices and fallback shapes |

## See also

[pipeline.md](pipeline.md), [gdn-and-attention.md](gdn-and-attention.md), [moe.md](moe.md), [gemm.md](gemm.md),
[docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md) (tolerance policy).
