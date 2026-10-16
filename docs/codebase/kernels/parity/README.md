[Codebase docs](../../README.md) › [Kernels](../README.md) › Parity tests

# Kernel parity, bench and unit tests

The engine's correctness rests on parity tests: every GPU kernel is held to a reference that was itself validated
against ggml / llama.cpp or a numpy oracle, usually with a **negative control** (the plausible wrong reading must be
computed and must differ by a stated margin before the kernel is judged). The tests live as standalone executables at
the top level of `src/kernels/` (one `*_parity.cpp`, `*_test.cpp` or `*_bench.cpp` per kernel family, plus two
generated oracle tables `*.inc`), and model-level CUDA harnesses live in `tests/cuda/`. They are built and registered
with CTest by the top-level `CMakeLists.txt`; they call the kernels in `strata_kernels`
([../cuda/README.md](../cuda/README.md)), `strata_kernels_cpu` ([../cpu/README.md](../cpu/README.md)),
`strata_prefill` and `strata_engine`, and use the numpy oracle `ref/qwen36.py`, `tools/*.py` fixture generators and
ggml (from `third_party/llama.cpp`) as references. The CPU-only tests next to the CPU kernels are in
[../cpu/tests.md](../cpu/tests.md).

## Contents

| Page | Covers |
|---|---|
| [model-tests.md](model-tests.md) | The `tests/cuda/` harnesses (Qwen3.6 block, verify window, MTP drafter, prompt path, fused/MMQ prompt experts) and `tests/platform/direct_file_async.cpp`, with their fixture chains |
| [fixtures.md](fixtures.md) | Build switches, how CTest fixtures are wired, every fixture generator and data file, the `.inc` oracle vectors, exit-code conventions, and how to run |

## Source map

| File | Lines | Role |
|---|---|---|
| [src/kernels/bf16_bits_test.cpp](../../../../src/kernels/bf16_bits_test.cpp) | 80 | f32 -> bf16 rounding vs ggml over all 2^32 inputs |
| [src/kernels/bf16_gemv_parity.cpp](../../../../src/kernels/bf16_gemv_parity.cpp) | 148 | BF16 GEMV (naive and split) |
| [src/kernels/cvec_parity.cpp](../../../../src/kernels/cvec_parity.cpp) | 223 | Control-vector kernel |
| [src/kernels/decode_cluster_parity.cpp](../../../../src/kernels/decode_cluster_parity.cpp) | 424 | sm_90 cluster top-k / argmax vs one-CTA kernels |
| [src/kernels/dequant_bf16_test.cpp](../../../../src/kernels/dequant_bf16_test.cpp) | 108 | Device dequantizers vs CPU on real shards (manual) |
| [src/kernels/dequant_s2_parity.cpp](../../../../src/kernels/dequant_s2_parity.cpp) | 143 | S2 (Q2_0 canonical) decode |
| [src/kernels/elementwise_parity.cpp](../../../../src/kernels/elementwise_parity.cpp) | 475 | Layer glue (softplus, gates, SiLU, ...) |
| [src/kernels/fused_gr_bench.cpp](../../../../src/kernels/fused_gr_bench.cpp) | 116 | AMD fast GR kernels vs old, bitwise + timing (manual) |
| [src/kernels/gdn_parity.cpp](../../../../src/kernels/gdn_parity.cpp) | 532 | Gated DeltaNet recurrence, conv, norms, output gate |
| [src/kernels/gr_parity.cpp](../../../../src/kernels/gr_parity.cpp) | 770 | Gated residual / hyper-connection read and write |
| [src/kernels/iq_multi_parity.cpp](../../../../src/kernels/iq_multi_parity.cpp) | 415 | Decode-once IQ kernels vs per-column ones |
| [src/kernels/iq_parity.cpp](../../../../src/kernels/iq_parity.cpp) | 125 | i-quant dequant and MMVQ vs gguf-py fixtures |
| [src/kernels/kv_hybrid_parity.cpp](../../../../src/kernels/kv_hybrid_parity.cpp) | 371 | K8V4 hybrid KV |
| [src/kernels/kv_q4_parity.cpp](../../../../src/kernels/kv_q4_parity.cpp) | 346 | Hadamard-rotated Q4_0 KV |
| [src/kernels/kv_q8_parity.cpp](../../../../src/kernels/kv_q8_parity.cpp) | 141 | INT8 KV |
| [src/kernels/kv_stream_parity.cpp](../../../../src/kernels/kv_stream_parity.cpp) | 234 | Streamed vs resident KV |
| [src/kernels/mmvq_multi_parity.cpp](../../../../src/kernels/mmvq_multi_parity.cpp) | 263 | `native_mmvq` multi-column exactness |
| [src/kernels/native_expert_bench.cpp](../../../../src/kernels/native_expert_bench.cpp) | 170 | Grouped native experts, layout A vs B on real rows (manual) |
| [src/kernels/native_expert_parity.cpp](../../../../src/kernels/native_expert_parity.cpp) | 670 | Native expert: float reference vs CPU vs GPU |
| [src/kernels/native_grouped_parity.cpp](../../../../src/kernels/native_grouped_parity.cpp) | 401 | `native_expert_grouped` group stride and fused SwiGLU+q8_1 vs v1 |
| [src/kernels/ple_fp8_parity.cpp](../../../../src/kernels/ple_fp8_parity.cpp) | 84 | FP8 PLE table through `PleTable` (manual) |
| [src/kernels/ple_oracle_vectors.inc](../../../../src/kernels/ple_oracle_vectors.inc) | 105 | Generated hash / table oracle vectors for `ple_parity` |
| [src/kernels/ple_parity.cpp](../../../../src/kernels/ple_parity.cpp) | 1184 | PLE hash, table read, PLE block |
| [src/kernels/ple_q5_parity.cpp](../../../../src/kernels/ple_q5_parity.cpp) | 59 | Q5_0 PLE rows vs ggml (manual) |
| [src/kernels/q5_projection_parity.cpp](../../../../src/kernels/q5_projection_parity.cpp) | 62 | Q5_1 shared-expert projection vs ggml (manual) |
| [src/kernels/qsa_attn_bench.cpp](../../../../src/kernels/qsa_attn_bench.cpp) | 151 | Split-K decode attention timing (manual) |
| [src/kernels/qsa_dense_parity.cpp](../../../../src/kernels/qsa_dense_parity.cpp) | 616 | Dense attention (Qwen3.6) vs host and oracle |
| [src/kernels/qsa_oracle_vectors.inc](../../../../src/kernels/qsa_oracle_vectors.inc) | 134 | Generated f32 -> f16 bit pairs for `qsa_parity` |
| [src/kernels/qsa_parity.cpp](../../../../src/kernels/qsa_parity.cpp) | 1555 | QSA cache, indexer, selection, attention |
| [src/kernels/qsa_prompt_attn_parity.cpp](../../../../src/kernels/qsa_prompt_attn_parity.cpp) | 274 | Tensor-core prompt attention vs FP32 kernel and FP64 |
| [src/kernels/qsa_select_bench.cpp](../../../../src/kernels/qsa_select_bench.cpp) | 174 | Prompt-path QSA selection timing + accuracy gate (HIP) |
| [src/kernels/qsa_topk_active_parity.cpp](../../../../src/kernels/qsa_topk_active_parity.cpp) | 95 | Active-block-bounded top-k vs capacity rule |
| [src/kernels/qsa_topk_parity.cpp](../../../../src/kernels/qsa_topk_parity.cpp) | 139 | Dispatched QSA top-k vs reference kernel |
| [src/kernels/quantize_act_parity.cpp](../../../../src/kernels/quantize_act_parity.cpp) | 407 | Q8_0 activation quantizer bytes |
| [src/kernels/qwen36_seam_parity.cpp](../../../../src/kernels/qwen36_seam_parity.cpp) | 526 | Qwen3.6 decode GDN / router / shared expert vs oracle seams |
| [src/kernels/rope_parity.cpp](../../../../src/kernels/rope_parity.cpp) | 442 | NEOX partial RoPE, scaled tables |
| [src/kernels/route_window_parity.cpp](../../../../src/kernels/route_window_parity.cpp) | 75 | Batched verify-window routing vs per-token |
| [src/kernels/router_top10_parity.cpp](../../../../src/kernels/router_top10_parity.cpp) | 186 | MoE router (512/10 and 256/8) |
| [src/kernels/s2_expert_grouped_parity.cpp](../../../../src/kernels/s2_expert_grouped_parity.cpp) | 679 | Rewritten S2 expert kernels vs previous |
| [src/kernels/s2_gemv_parity.cpp](../../../../src/kernels/s2_gemv_parity.cpp) | 173 | S2 GEMV |
| [src/kernels/s2_gemv_q8_parity.cpp](../../../../src/kernels/s2_gemv_q8_parity.cpp) | 181 | S2 GEMV with Q8_0 activations |
| [src/kernels/s_gemv_parity.cpp](../../../../src/kernels/s_gemv_parity.cpp) | 634 | S2/S4/S8 GEMV (Q2_0, Q4_0, IQ4_NL, Q8_0) |
| [src/kernels/s_gemv_q8k_parity.cpp](../../../../src/kernels/s_gemv_q8k_parity.cpp) | 343 | S-GEMV with Q8_K activations |
| [src/kernels/sampler_parity.cpp](../../../../src/kernels/sampler_parity.cpp) | 1284 | Sampler chain order and paths |
| [src/kernels/shared_expert_parity.cpp](../../../../src/kernels/shared_expert_parity.cpp) | 673 | Shared expert structure (SiLU on gate, scalar gate) |
| [tests/cuda/prefill_fused_iq_test.cpp](../../../../tests/cuda/prefill_fused_iq_test.cpp) | 642 | Fused int8 prompt experts, native formats |
| [tests/cuda/prefill_fused_moe_test.cpp](../../../../tests/cuda/prefill_fused_moe_test.cpp) | 420 | Fused int8 prompt experts, Q2_0 blobs |
| [tests/cuda/prefill_mmq_kquant_test.cpp](../../../../tests/cuda/prefill_mmq_kquant_test.cpp) | 209 | MMQ for UD-Q4_K_XL formats |
| [tests/cuda/prefill_qwen36_test.cpp](../../../../tests/cuda/prefill_qwen36_test.cpp) | 483 | Prompt path at Qwen3.6 geometry vs oracle |
| [tests/cuda/qwen36_block_test.cpp](../../../../tests/cuda/qwen36_block_test.cpp) | 531 | Plain block (hc = 1) vs oracle |
| [tests/cuda/qwen36_mtp_test.cpp](../../../../tests/cuda/qwen36_mtp_test.cpp) | 622 | MTP drafter vs oracle |
| [tests/cuda/qwen36_verify_test.cpp](../../../../tests/cuda/qwen36_verify_test.cpp) | 511 | Verify window on a native pack vs oracle and single-token path |
| [tests/platform/direct_file_async.cpp](../../../../tests/platform/direct_file_async.cpp) | 60 | `DirectFile` async reads |

## Conditions used in the tables

- **GPU**: built inside `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` (`CMakeLists.txt:281-1038`) with no
  `STRATA_BUILD_TESTS` guard; the ctest is registered because that block calls `enable_testing()`
  (`CMakeLists.txt:457`). `STRATA_HIP_GFX906` builds count as CUDA (it forces `STRATA_ENABLE_CUDA=ON`).
- **CUDA**: additionally `if(STRATA_ENABLE_CUDA)`.
- **TESTS**: `STRATA_BUILD_TESTS=ON` (default OFF in this checkout: `tests/CMakeLists.txt` and `bench/micro/` are
  missing, `CMakeLists.txt:86-92`).
- **NATIVE**: inside `if(STRATA_NATIVE_EXPERTS)` (default ON; needs `third_party/llama.cpp`).
- **py-iq** / **py-q36**: a python3 with numpy was found at configure time (`_iq_python`, probed in the GPU block;
  `_q36_python`, probed only when **TESTS** is on). See [fixtures.md](fixtures.md).
- "manual": an executable with no `add_test`; run by hand.

## Tests at the top level of `src/kernels/`

| Test (CTest name, command) | Build condition | What it checks | Reference | Fixtures / data |
|---|---|---|---|---|
| `bf16_bits_test` | TESTS or `STRATA_BUILD_CONVERSATION_TESTS` (CPU, header-only) | `bf16_bits.hpp` f32 -> bf16 equals ggml's rule on every f32 pattern; unchanged on non-NaN vs the old rule; NaN stays quiet NaN; bf16 -> f32 -> bf16 round trip | `ggml_compute_fp32_to_bf16` transcribed | None |
| `bf16_gemv_parity --selftest` | GPU | Naive and split BF16 GEMV vs a host dot on the same bf16 bits; naive vs split; an fp16 activation must differ (activation contract observable) | Host double reference | Synthetic |
| `cvec_parity` | GPU | Control vector: project (`h - s(h.v)v`), add, switched-off is bitwise the fused GR write, layers without a direction untouched | Host double | Synthetic |
| `decode_cluster_parity --selftest` | CUDA | `qsa_block_topk_cluster` and `sample_greedy_cluster` bitwise vs the one-CTA kernels, adversarial scores (ties, NaN, +-inf, -0, denormals), up to 262,144 cells, graph replay; exit 0 (skip) below sm_90; `--bench` | `qsa_block_topk`, `qsa_block_topk_ref`, greedy sampler | Synthetic |
| `dequant_bf16_test SHARD1 [SHARD2]` (manual) | TESTS and GPU | Device dequantizers (FP32 path rel 1e-6, BF16 within half an ulp) on the first rows of one real tensor per type | `strata/artifact/dequant.hpp` CPU dequantizers | **Real model shards** |
| `dequant_s2_parity --selftest` | GPU | GPU canonical S2 decode bit-exact | `strata::dequantize_q2_0` (cross-checked against ggml upstream) | Synthetic finite fp16 scales |
| `elementwise_parity --selftest` | GPU | softplus large-x branch, negative `ssm_a` decay, SiLU, other glue conventions; rivals asserted observable | Host transcriptions of `ref/gdn.py` rules | Synthetic |
| `fused_gr_bench [iters] [Tmin] [Tmax]` (manual) | GPU | `fused_gr_read_multi` AMD fast kernels (`STRATA_GR_FAST`) bitwise vs old on every output, plus timings | The previous kernels | Synthetic |
| `gdn_parity --selftest` | GPU | GDN: modulo head pairing, decay before update, conv slide direction, `l2_norm` eps placement, state layout; the output gate per model (sigmoid Flash-Next, SiLU Qwen3.6) with the other gate required > 5% off | Host reference of `ref/gdn.py` (not in this checkout) | Synthetic |
| `gr_parity --selftest` | GPU | Hyper-connection read/write: per-stream RMSNorm, `/hc` inside SiLU, SiLU vs sigmoid placement, mean not sum, BF16 vs FP32 activation contracts, `2*sigmoid` gate; graph replay keeps the precision (`STRATA_GR_V3` env read) | Host reference of `ref/gr.py` rules | Synthetic |
| `iq_multi_parity` | GPU | `iq_mmvq` (1..8 and 11 columns) and `native_expert_grouped` (0..11 entries, scattered destinations) bitwise new vs old; gross-error guard vs double | Old kernels (`iq_set_old_kernels`), double ref from the GPU dequantizer | Synthetic random blocks |
| `iq_parity <dir>` (+ fixture test `iq_parity_fixtures`) | GPU, ctest only with py-iq | i-quant dequant vs gguf-py (measured rel error, 0 on the fixtures); MMVQ vs float matvec for 1..8 columns, multi-column bitwise single-column | gguf-py dequantization (Q2_0 via `tools/gguf_writer.py`) | `tools/iq_fixture.py --out build/iq_fixture` (needs vendored gguf-py; exit 3 = skip) |
| `kv_hybrid_parity` | GPU | K8V4: INT8 K codes/scales and rotated Q4_0 V bitwise; composed gathers; `qsa_decode_attn<3>` + inverse FWHT vs host attention and bounded vs fp32-true | Host references (`kv_q8.hpp`, `kv_q4.hpp` rules) | Synthetic |
| `kv_q4_parity` | GPU | FWHT-256 vs exact orthonormal Hadamard; Q4_0 packing bitwise; paged append/gather; dot-product invariance under rotation | Host FWHT (Sylvester, as `ggml_gen_hadamard`) | Synthetic |
| `kv_q8_parity` | GPU | INT8 codes + fp16 scales bitwise; gather bitwise; vs FP16 path within 0.624 quantization steps | Host reference | Synthetic |
| `kv_stream_parity` | GPU | Streamed attention bitwise vs resident; residency map consistent; counters; ring restore (MTP layout); INT8, FP16, Q4_0 pools | Fully resident pool | Synthetic |
| `mmvq_multi_parity` (+ `mmvq_multi_parity_cuda_layout` with `STRATA_MMVQ_WAVE=0` on gfx906) | CUDA | `native_mmvq` multi_exact: a T-column call bitwise T single-column calls on the same Q8_1 bytes; negative control (multi_exact off) must differ for T > 4 | Single-column calls | Synthetic, finite scales |
| `native_expert_bench <shard1.gguf> <layers> <groups> <tok> <ref mode> <mode>` (manual) | NATIVE and GPU | Grouped native experts under two `STRATA_EXP_MODE` layouts, bitwise, with timings | Reference mode | **Real GGUF rows** |
| `native_expert_parity_<gu>_<down>` (7 pairs at 2560x640), `native_expert_parity_qwen36_<gu>_<down>` (5 pairs at `--geom 2048x512`), `native_expert_parity_q5_1_min`, `native_expert_parity_bf16_embd`, `native_expert_parity_refuses_q6_K_down_at_640` (`WILL_FAIL`) | NATIVE and GPU builds the target; ctests need TESTS and CUDA | One native expert three ways: (a) float reference from ggml `to_float`, (b) CPU path, (c) GPU `native_expert_grouped` (both kernel variants bitwise); (d) GPU dequantizers vs `to_float`; (e) negative control: a flipped scale sign moves exactly one row; also the multi-token CPU kernels | ggml `to_float` + float SwiGLU | `--synthetic GU/DOWN` (ggml-quantized random weights) for the ctests; real shards by hand |
| `native_grouped_parity` | GPU | `native_expert_grouped` group stride (`grid_groups`) and fused SwiGLU + q8_1 bitwise vs v1, rows it must not write included; `--bench`, `--bench-down` | v1 launches (`native_grouped_set_v1`) | Synthetic |
| `ple_fp8_parity <ple-fp8.gguf> <reference.bin> [<iq4nl shard>]` (manual) | NATIVE, TESTS and CUDA | FP8 PLE rows bit for bit through Direct, Mmap and `gather_batch`; correlation with the IQ4_NL table | Values decoded from the checkpoint (torch) | **Real files** |
| `ple_parity --selftest --in .../ple_in.bin --out .../ple_out.bin` (working dir: source root) | GPU | A: hash vs `ref/ngram.py` vectors with rivals observable; B: table rows vs numpy over the original GGUF; C: the PLE block vs ggml's graph, per stage; NaN-safe comparisons | `ple_oracle_vectors.inc`; numpy; ggml capture | **Required**: PLE shard (`STRATA_PLE_GGUF` / `--gguf`, default `../../Q2_0/...-00002-of-00002.gguf`), `pack/full/dense.bin`, `${STRATA_PLE_FIXTURE_DIR}` (default `bench/micro`) `ple_in.bin` / `ple_out.bin`; missing input = exit 2 |
| `ple_q5_parity` (manual) | NATIVE, TESTS and CUDA | Q5_0 PLE rows of a real artifact vs ggml's dequantizer | ggml | **Real file** (#296) |
| `q5_projection_parity` (manual) | NATIVE, TESTS and CUDA | Q5_1 shared-expert projection: Strata CUDA dequantization vs ggml | ggml | **Real file** (#296) |
| `qsa_attn_bench` (manual) | GPU | Split-K decode attention time per call at several depths (asserts nothing); exit 77 without a device | - | Synthetic |
| `qsa_dense_parity [--fixtures DIR]` (+ fixture test `qwen36_attn_fixtures`) | GPU; `--fixtures` form with py-q36 | Dense attention for G = 2, 8, 12, null ids as identity, bounded scratch, sparse width refused; oracle seams `Qcur-3`, `Kcur-3#2`, `kq_soft_max-3`, `kqv-3`, `attn_gated-3`; rivals (gate skipped, adjacent-pair RoPE, `h % n_kv` GQA) > 5% | Host reference; `ref/qwen36.py` dumps | `cmake/qwen36_attn_fixtures.cmake` (g8/g12/g2 GGUFs + dumps); exit 77 when missing |
| `qsa_parity --selftest` | GPU | QSA: pooled key rotated at the first cell, pool-then-norm, spare slot from cell 0, per-head ReLU, `q / 12` head map, only selected cells attended, gate from the second half, paged KV with reversed page tables | Host reference of `ref/qsa.py` / `ref/model.py` rules (not in this checkout); `qsa_oracle_vectors.inc` for f16 rounding | Synthetic |
| `qsa_prompt_attn_parity [ctx] [queries] [reps]` (manual on CUDA); HIP: ctest `hip_prompt_attn_wmma 32768 2048 3` | CUDA: (TESTS or `STRATA_PARITY_PROMPT_ATTN`); HIP: always in a HIP build | Tensor-core prompt attention vs `qsa_decode_attn_batch` (rel 1e-4) and FP64 (error no worse than a small multiple of the old kernel's), INT8/FP16/Q4_0 pools; timing; HIP WMMA variant skips (77) off gfx12 | FP32 kernel, FP64 host | Synthetic |
| `qsa_select_bench 32768 256 3 262144` | (TESTS or `STRATA_PARITY_PROMPT_ATTN`) and HIP | Prompt-path QSA selection per stage, scorer vs scorer, FP64 accuracy gate; non-zero exit on a failed gate | FP64 host | Synthetic |
| `qsa_topk_active_parity` | TESTS and CUDA | Top-k with the active-block bound (forced via `STRATA_TOPK_ACTIVE_ANY=1`) selects the same ids as the capacity rule, capacity 262,144; exit 77 without a device | Capacity-rule dispatch | Synthetic |
| `qsa_topk_parity --selftest` | GPU | Dispatched `qsa_block_topk` ids identical to `qsa_block_topk_ref` across the register kernel's reach; continuous, 64-level and all-equal scores; tail blocks of 0-3 cells | Reference kernel | Synthetic |
| `quantize_act_parity --selftest` | GPU | Q8_0 activation quantizer compared on the 34-byte blocks, not values | Host transcription of `ref/quant.py::q8_0` with `f16_bits.hpp` conversions | Synthetic |
| `qwen36_seam_parity <dir>` (+ fixture test `qwen36_oracle_dumps`) | GPU; ctests with py-q36 | Decode GDN (legacy, native, fused, verify multi-token across 8 + 3 windows), router 256/8 top-k and normalized weights, shared expert 512 on Q8_0 vs oracle seams; rivals (sigmoid gate, unnormalized weights, gate skipped) > 5% | `ref/qwen36.py` dumps (F32 and Q8) | `cmake/qwen36_oracle_dumps.cmake` -> `k.gguf`, `o_f32/`, `o_q8/`; exit 77 when missing |
| `rope_parity --selftest` | GPU | RoPE table vs float64; rotation bit-exact vs host NEOX rotation with the same table; scaled tables (linear, YaRN) vs a float64 `rope_yarn` transcription; table path vs native fast-math path | `ref/qsa.py` formulas in double | Synthetic |
| `route_window_parity` | GPU | `moe_route_window` (one multi-column BF16 projection + one n-token top-k) bitwise vs per-token calls, router (2560 -> 256, top-10) and shared gate shapes | Per-token kernels | Synthetic |
| `router_top10_parity --selftest` | GPU | Expert ids exactly and weights to tolerance at 512/10 and 256/8, ties and near-ties, the 2^-14 clamp proven unreachable | Host `ref/moe.py::router` transcription | Synthetic |
| `s2_expert_grouped_parity --selftest` | GPU | Every rewritten S2 expert entry point bitwise vs `moe_grouped_select_old(1)`, output and scratch; fallbacks on misaligned inputs verified via `moe_grouped_last_path`; double-precision sanity check | Previous kernels | Synthetic Q2_0 blobs |
| `s2_gemv_parity --selftest` | GPU | S2 GEMV relative error (measured, FP16 tolerance 1e-3) | Dot on `strata::dequantize_q2_0` | Synthetic, full-mantissa fp16 scales |
| `s2_gemv_q8_parity --selftest` | GPU | Q8_0-activation GEMV vs host; FP16 vs Q8_0 activations must differ (~1%); the gap is reported | Host `ref/quant.py::q8_0` rule | Synthetic |
| `s_gemv_parity --selftest` | GPU | S2 Q2_0, S4 Q4_0 and IQ4_NL, S8 Q8_0 GEMV; the offset branch (Q4_K/Q5_K) is reported as not exercised | Raw GGUF block rebuilt from canonical planes, decoded by `strata/artifact/dequant.hpp` | Synthetic |
| `s_gemv_q8k_parity --selftest` | GPU | Q8_K activation path of `s_gemv` vs double on the same bytes; Q8_K vs FP16 gap measured | Host double; `quantize_q8_K` | Synthetic |
| `sampler_parity --selftest`, `sampler_parity_one_block`, `sampler_parity_old` | GPU | The order top_k -> top_p -> temperature is observable (the wrong order must give a different pick) and the kernel follows it; same fixtures on the one-block (`STRATA_SAMPLER_ONE_BLOCK=1`) and 0.1.20 (`STRATA_OLD_SAMPLER=1`) paths | Host reference chain | Synthetic |
| `shared_expert_parity --selftest` | GPU | SiLU on gate (not up), scalar `ffn_gate_inp_shexp` gate (not elementwise or per-expert), each rival required to differ; tolerance set by the fp16 intermediate | Host reference | Synthetic |

## Tests in `tests/cuda/` and `tests/platform/`

| Test (CTest name, command) | Build condition | What it checks | Reference | Fixtures / data |
|---|---|---|---|---|
| `qwen36_block_test <synth> <dump>/o8` | TESTS, CUDA, py-q36 | Plain block (hc = 1) through the engine's own functions on a pack the test writes: norms, residual, router, MoE combine, head and logits per layer; T = 8 row kernels bitwise one-row; negative cases; mixers and experts teacher-forced | `ref/qwen36.py` dumps on 8 tokens | `qwen36_synth_fixtures` (`cmake/qwen36_synth_fixtures.cmake`) then `qwen36_block_fixtures` (`cmake/qwen36_block_fixtures.cmake`); exit 77 |
| `qwen36_verify_test <dir>`; also `qwen36_runner_check` (python + the `strata` binary) | TESTS, CUDA, py-q36 | `Verifier` on a qwen35moe native pack: 8-row window vs `l_out-N` and logits; eight one-token windows bitwise the 8-row window; 8 + 3 windows; 40 tokens one per window with argmax | Oracle dumps o8/o11/o40; the single-token path | `qwen36_runner_fixtures` (`tests/python/qwen36_runner_fixture.py`, `STRATA_GGUF_PY`); exit 77 |
| `prefill_qwen36_test <dir> gdn3`, `prefill_qwen36_attn_test <dir> full4` | CUDA and not gfx906; ctests with py-iq | `Prefill::run` as one chunk and in chunks of 7, every seam teacher-forced per layer; router top-k exact; rivals > 5%; geometry refusals | Oracle dumps of 40 tokens | `prefill_qwen36_fixtures` (`tests/python/qwen36_prefill_fixture.py`); exit 77 |
| `qwen36_mtp_test <dir>` (`--real ...` by hand) | CUDA and not gfx906; ctest with py-iq | MTP drafter `debug_row` per seam and position, top-8 set, draft = oracle argmax; negative pairings; captured `draft_first`; prompt-side `draft_kv` | Oracle `--mtp` dumps (Q8 mode), llama.cpp MTP graph in `--real` | `qwen36_mtp_fixtures` (`tests/python/qwen36_mtp_fixture.py`); exit 77 |
| `prefill_fused_moe_test` | NATIVE, CUDA, TESTS | Fused int8 prompt experts on Strata Q2_0 blobs vs double reference and vs MMQ (<= 1.5x MMQ's RMS, 2x its worst row); timing | Double reference; MMQ path | Synthetic; exit 77 below sm_80 |
| `prefill_fused_iq_test` | NATIVE, CUDA, TESTS | Same for native formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS gate/up; Q2_0 IQ4_NL down) at 2560/640 and 2048/512 | ggml `to_float` double reference; MMQ | Synthetic; exit 77 below sm_80 |
| `prefill_mmq_kquant_test` | NATIVE, CUDA, TESTS, `STRATA_MMQ_KQUANTS` | MMQ products for Q4_K/Q5_K gate/up and Q5_1/Q8_0 (and Qwen3.6's Q5_K/Q6_K) down vs double; uncovered formats (Q3_K) must say so | ggml dequantized weights | Synthetic |
| `direct_file_async_test` | TESTS, Unix not Apple | `DirectFile`: unaligned request refused, 17 queued 4 KiB reads with a short EOF read, `wake`, close drains, reopen has no stale completion (`STRATA_IO_THREADS=4`) | The file's known bytes | A temp file in `/tmp` |

Details of each harness: [model-tests.md](model-tests.md). Fixture generation and run instructions:
[fixtures.md](fixtures.md).

## Key concepts

- **The reference is a chain, not a second opinion.** Kernels are compared against code already validated against
  ggml (`strata/artifact/dequant.hpp`, `quantize_q8_K`, ggml's own `to_float`), or against the numpy oracle
  `ref/qwen36.py`, which agrees with the pinned llama.cpp graph seam by seam
  ([docs/qwen36-port/oracle-crosscheck.md](../../../qwen36-port/oracle-crosscheck.md)).
- **Negative controls.** Every structural reading with a plausible rival is computed the wrong way and required to
  differ (often by > 5% relative L1) before the kernel is judged; a test whose control has no power reports it.
- **Bitwise where the contract is bitwise.** Multi-token and rewritten kernels must equal their single-token or
  previous forms bit for bit (the verify window reproducing greedy decode depends on it); tolerances are used only
  where arithmetic order legitimately differs, and the measured error is printed.
- **NaN-safe comparisons** (`ple_parity.cpp`): `x <= t` silently passes NaN, so comparisons require finiteness.
- **Exit codes**: 0 pass, 1 fail, 2 missing required input (counts as a failure), 77 skip where CTest's
  `SKIP_RETURN_CODE 77` is set, 3 for a fixture generator that lacks numpy / gguf-py.

## See also

- [../cuda/README.md](../cuda/README.md), [../cpu/tests.md](../cpu/tests.md)
- [../../build/README.md](../../build/README.md) (build options and targets)
- [docs/qwen36-port/upstream-tooling.md](../../../qwen36-port/upstream-tooling.md) (the negative-test discipline),
  [docs/qwen36-port/oracle-crosscheck.md](../../../qwen36-port/oracle-crosscheck.md)
