[Codebase docs](../../README.md) › [Kernels](../README.md) › CUDA

# CUDA kernels (`src/kernels/cuda/`, `include/strata/kernels/`)

The CUDA kernels are the GPU half of the engine's per-token work. They build into one static library,
`strata_kernels` (`CMakeLists.txt:323-365`, together with the host-only `src/kernels/ngram.cpp`,
`src/ngram/ple_reader.cpp` and `src/platform/direct_file.cpp`). Every entry point is a plain host function in
`namespace strata::kernels`, declared in a header under `include/strata/kernels/`. It takes device pointers and a
`void* stream` (a `cudaStream_t` behind a `void*`, so the headers need no CUDA types). The callers are the
layer and session code in `src/core/` (`layer.cpp`, `verify.cpp`, `mtp.cpp`, `session.cpp`, `native_dense.cpp`,
`native_head.cpp`, `peer_experts.cpp`, `remote_experts.cpp`), the prompt path in `src/prefill/` (`prefill.cpp`,
`gemm.cu`, `kernels.cu`) and the run-time switches in `src/program/generate.cpp`. The kernels call only the CUDA
runtime and ggml's block definitions (`third_party/ggml/ggml-common.h`, included by `iq_kernels.cu`). Many of them are
line-for-line transcriptions of llama.cpp at commit `3cf03257` (the commit `UPSTREAM.md` pins), because the
engine's correctness contract is "the same numbers as llama.cpp's CUDA backend". All 44 files came from the upstream
Strata snapshot. The Qwen3.6 port changed eight of them and added none (see "Qwen3.6 port" below).

## Contents

| Page | Covers |
| --- | --- |
| [attention-qsa.md](attention-qsa.md) | Full-attention layers: the QSA KV cache, indexer, block scores and top-k selection, split-K decode attention, the tensor-core prompt attention, dense attention for Qwen3.6, and the pinned `native_qsa*` / `native_flash_attn` adapters |
| [gdn.md](gdn.md) | Gated DeltaNet (linear-attention) layers: conv + SiLU + L2 norm, the delta-rule recurrence, the output norm with its sigmoid (Flash-Next) or SiLU (Qwen3.6) gate, and the verify-window step/commit kernels |
| [moe-and-experts.md](moe-and-experts.md) | Routing (`router_top10`, `native_router`), the MoE combine, the shared expert, the GPU expert tiers (`s2_expert_grouped` for S2 packs, `native_expert_grouped` for GGUF i-quant/K-quant packs) and the native MMVQ adapters |
| [gemv-and-quant.md](gemv-and-quant.md) | Dense GEMVs over the canonical S2/S4/S8 forms and BF16, activation quantizers (Q8_0, Q8_K, q8_1), GGUF dequantizers, `dp4a.hpp` and `q8_1_finite.hpp` |
| [norms-rope-elementwise.md](norms-rope-elementwise.md) | Hyper-connection read/write (`gr`, `fused_gr`, `native_gr_*`), RoPE (table, analytic, M-RoPE, scaling), elementwise glue and doorbells, control vectors, and the PLE block |
| [kv-cache.md](kv-cache.md) | KV storage formats beyond FP16 (INT8, Hadamard-rotated Q4_0, hybrid K8V4) and KV streaming from pinned host memory |
| [sampling-and-verify.md](sampling-and-verify.md) | The sampler chain (greedy, split top-k, cluster argmax, coupled draft sampling) and the speculative-verify helper kernels |
| [bits-helpers.md](bits-helpers.md) | `bf16_bits.hpp` and `f16_bits.hpp`: the bit-exact float conversions every kernel shares |

## Source map

Paths are relative to this page. Line counts are from the tree at the time of writing.

### Kernels (`src/kernels/cuda/`)

| File | Lines | Role | Page |
| --- | ---: | --- | --- |
| [bf16_gemv.cu](../../../../src/kernels/cuda/bf16_gemv.cu) | 141 | BF16-weight GEMV, BF16 activations (naive / warp-per-row / split) | [gemv](gemv-and-quant.md) |
| [cvec.cu](../../../../src/kernels/cuda/cvec.cu) | 166 | Control vector on the residual stream (project / add) | [norms](norms-rope-elementwise.md) |
| [dequant_bf16.cu](../../../../src/kernels/cuda/dequant_bf16.cu) | 261 | GGUF blocks to BF16/FP16/FP32 for the prompt-path GEMMs | [gemv](gemv-and-quant.md) |
| [dequant_s2.cu](../../../../src/kernels/cuda/dequant_s2.cu) | 51 | Naive S2 decode (reference) | [gemv](gemv-and-quant.md) |
| [elementwise.cu](../../../../src/kernels/cuda/elementwise.cu) | 397 | Glue ops, embedding gather, weighted RMSNorm, doorbells, mapped-memory copies | [norms](norms-rope-elementwise.md) |
| [fused_gdn.cu](../../../../src/kernels/cuda/fused_gdn.cu) | 169 | GDN conv+L2, alpha/beta, step+output norm in one kernel each | [gdn](gdn.md) |
| [fused_gr.cu](../../../../src/kernels/cuda/fused_gr.cu) | 1371 | Two-kernel hyper-connection read, multi-token variants, per-card self-check | [norms](norms-rope-elementwise.md) |
| [gdn.cu](../../../../src/kernels/cuda/gdn.cu) | 253 | Reference GDN step, conv, L2 norm, beta gate, output norm | [gdn](gdn.md) |
| [gr.cu](../../../../src/kernels/cuda/gr.cu) | 432 | Hyper-connection read/write (unfused) | [norms](norms-rope-elementwise.md) |
| [iq_kernels.cu](../../../../src/kernels/cuda/iq_kernels.cu) | 2637 | i-quant / K-quant MMVQ, q8_1 quantizer, dequantizers, grouped native experts | [moe](moe-and-experts.md) |
| [kv_q4.cu](../../../../src/kernels/cuda/kv_q4.cu) | 231 | FWHT-256 and Q4_0 KV append/gather | [kv](kv-cache.md) |
| [kv_q8.cu](../../../../src/kernels/cuda/kv_q8.cu) | 127 | INT8 KV append/gather | [kv](kv-cache.md) |
| [kv_stream.cu](../../../../src/kernels/cuda/kv_stream.cu) | 276 | KV streaming: residency resolve (CLOCK), ring tables, staging | [kv](kv-cache.md) |
| [native_bf16.cu](../../../../src/kernels/cuda/native_bf16.cu) | 199 | Pinned llama.cpp BF16-weight / F32-activation MMVF (1..8 columns) | [gemv](gemv-and-quant.md) |
| [native_flash_attn.cu](../../../../src/kernels/cuda/native_flash_attn.cu) | 212 | Diagnostic pinned FlashAttention-vec, context <= 256 | [attention](attention-qsa.md) |
| [native_gdn.cu](../../../../src/kernels/cuda/native_gdn.cu) | 122 | Pinned `gated_delta_net` recurrence | [gdn](gdn.md) |
| [native_gdn_preprocess.cu](../../../../src/kernels/cuda/native_gdn_preprocess.cu) | 202 | Pinned conv+SiLU, L2 norm, beta, gate, output norm | [gdn](gdn.md) |
| [native_gr_norm.cu](../../../../src/kernels/cuda/native_gr_norm.cu) | 102 | Pinned weighted F32 RMSNorm (HC read) | [norms](norms-rope-elementwise.md) |
| [native_gr_postops.cu](../../../../src/kernels/cuda/native_gr_postops.cu) | 115 | Pinned HC down-SiLU, pre-gated mean, post (write) | [norms](norms-rope-elementwise.md) |
| [native_mmvq.cu](../../../../src/kernels/cuda/native_mmvq.cu) | 1619 | Pinned llama.cpp MMVQ for Q2_0/Q4_0/Q5_0/Q8_0/Q3_K..Q6_K/IQ4_NL/IQ4_XS, q8_1 quantize, dispatcher | [moe](moe-and-experts.md) |
| [native_moe.cu](../../../../src/kernels/cuda/native_moe.cu) | 86 | Pinned MoE weighted reduction (combine) | [moe](moe-and-experts.md) |
| [native_ple_postops.cu](../../../../src/kernels/cuda/native_ple_postops.cu) | 221 | Pinned PLE arithmetic after its projections (single and batch) | [norms](norms-rope-elementwise.md) |
| [native_qsa.cu](../../../../src/kernels/cuda/native_qsa.cu) | 127 | Pinned weighted RMSNorm and sigmoid output gate | [attention](attention-qsa.md) |
| [native_qsa_indexer.cu](../../../../src/kernels/cuda/native_qsa_indexer.cu) | 290 | Pinned indexer key cache, pooling, norm and IMRoPE | [attention](attention-qsa.md) |
| [native_qsa_score.cu](../../../../src/kernels/cuda/native_qsa_score.cu) | 213 | Pinned TF32-MMA indexer scores (no engine caller) | [attention](attention-qsa.md) |
| [native_rope.cu](../../../../src/kernels/cuda/native_rope.cu) | 147 | Pinned analytic IMRoPE; M-RoPE table and RoPE-table registry | [norms](norms-rope-elementwise.md) |
| [native_router.cu](../../../../src/kernels/cuda/native_router.cu) | 127 | Pinned `topk-moe` for 512 experts / top-10 | [moe](moe-and-experts.md) |
| [ple.cu](../../../../src/kernels/cuda/ple.cu) | 386 | PLE block (Flash-Next n-gram embedding layer) | [norms](norms-rope-elementwise.md) |
| [qsa.cu](../../../../src/kernels/cuda/qsa.cu) | 841 | QSA cache append/gather, indexer pooling, per-cell scores, top-k, reference attention, gate | [attention](attention-qsa.md) |
| [qsa_decode_attn.cu](../../../../src/kernels/cuda/qsa_decode_attn.cu) | 793 | Split-K decode attention straight from the KV pools | [attention](attention-qsa.md) |
| [qsa_prompt_attn.cu](../../../../src/kernels/cuda/qsa_prompt_attn.cu) | 1499 | Tensor-core prompt attention (mma.sync, cp.async; Volta and RDNA4 variants) | [attention](attention-qsa.md) |
| [qsa_select.cu](../../../../src/kernels/cuda/qsa_select.cu) | 1237 | Block scores (warp, TF32 tensor-core) and radix top-k (register, wide, cluster) | [attention](attention-qsa.md) |
| [quantize_act.cu](../../../../src/kernels/cuda/quantize_act.cu) | 313 | Q8_0 / Q8_K activation quantizers and inverses | [gemv](gemv-and-quant.md) |
| [rope.cu](../../../../src/kernels/cuda/rope.cu) | 136 | Host float64 RoPE tables (with scaling), table-driven NEOX rotation | [norms](norms-rope-elementwise.md) |
| [router_top10.cu](../../../../src/kernels/cuda/router_top10.cu) | 497 | Softmax / top-k / renormalise router, any expert count up to 32,768 | [moe](moe-and-experts.md) |
| [s2_expert_grouped.cu](../../../../src/kernels/cuda/s2_expert_grouped.cu) | 1190 | GPU expert tier for S2 (Q2_0) packs: per-hit and grouped kernels | [moe](moe-and-experts.md) |
| [s2_gemv.cu](../../../../src/kernels/cuda/s2_gemv.cu) | 63 | Naive S2 GEMV (reference) | [gemv](gemv-and-quant.md) |
| [s2_gemv_fast.cu](../../../../src/kernels/cuda/s2_gemv_fast.cu) | 153 | Experiment kept as a negative result (not faster) | [gemv](gemv-and-quant.md) |
| [s2_gemv_q8.cu](../../../../src/kernels/cuda/s2_gemv_q8.cu) | 96 | S2 GEMV over Q8_0 activations (the engine's S2 dense path) | [gemv](gemv-and-quant.md) |
| [s2_gemv_quads.cu](../../../../src/kernels/cuda/s2_gemv_quads.cu) | 93 | S2 GEMV, four elements per code byte | [gemv](gemv-and-quant.md) |
| [s_gemv.cu](../../../../src/kernels/cuda/s_gemv.cu) | 694 | S-family GEMV (S2/S4/S8) over FP16, Q8_K and Q8_0 activations | [gemv](gemv-and-quant.md) |
| [sampler.cu](../../../../src/kernels/cuda/sampler.cu) | 1081 | Sampler chain, greedy argmax (cluster on sm_90+), coupled draft sampling | [sampling](sampling-and-verify.md) |
| [shared_expert.cu](../../../../src/kernels/cuda/shared_expert.cu) | 384 | Shared expert (single and 1..8 tokens) and `moe_combine` | [moe](moe-and-experts.md) |
| [verify_kernels.cu](../../../../src/kernels/cuda/verify_kernels.cu) | 685 | Verify-window GDN multi/commit, MTP helpers, resident plans, small copies | [gdn](gdn.md), [sampling](sampling-and-verify.md) |

### Headers (`include/strata/kernels/`, excluding `cpu/`)

| File | Lines | Role | Page |
| --- | ---: | --- | --- |
| [bf16_bits.hpp](../../../../include/strata/kernels/bf16_bits.hpp) | 61 | Host/device bf16 <-> f32 (RNE, NaN-quieting) | [bits](bits-helpers.md) |
| [bf16_gemv.hpp](../../../../include/strata/kernels/bf16_gemv.hpp) | 56 | BF16 GEMV and pinned MMVF API | [gemv](gemv-and-quant.md) |
| [cvec.hpp](../../../../include/strata/kernels/cvec.hpp) | 61 | `Cvec` and control-vector API | [norms](norms-rope-elementwise.md) |
| [dequant_bf16.hpp](../../../../include/strata/kernels/dequant_bf16.hpp) | 27 | GGUF dequantizer API | [gemv](gemv-and-quant.md) |
| [dequant_s2.hpp](../../../../include/strata/kernels/dequant_s2.hpp) | 18 | S2 decode API | [gemv](gemv-and-quant.md) |
| [dp4a.hpp](../../../../include/strata/kernels/dp4a.hpp) | 45 | `STRATA_DP4A`, `strata_spin_pause` (pre-sm_61 / pre-sm_70 fallbacks) | [gemv](gemv-and-quant.md) |
| [elementwise.hpp](../../../../include/strata/kernels/elementwise.hpp) | 128 | Glue ops and doorbell API | [norms](norms-rope-elementwise.md) |
| [f16_bits.hpp](../../../../include/strata/kernels/f16_bits.hpp) | 104 | Host/device fp16 <-> f32 (RNE, subnormals, saturation) | [bits](bits-helpers.md) |
| [fused_gdn.hpp](../../../../include/strata/kernels/fused_gdn.hpp) | 40 | Fused GDN API | [gdn](gdn.md) |
| [fused_gr.hpp](../../../../include/strata/kernels/fused_gr.hpp) | 67 | `FusedGrArgs`, fused HC read API | [norms](norms-rope-elementwise.md) |
| [gdn.hpp](../../../../include/strata/kernels/gdn.hpp) | 110 | `GdnShapes`, `GdnGate`, reference GDN API, state layout | [gdn](gdn.md) |
| [gr.hpp](../../../../include/strata/kernels/gr.hpp) | 122 | `GrShapes`, `GrWorkspace`, HC read/write API | [norms](norms-rope-elementwise.md) |
| [iq_kernels.hpp](../../../../include/strata/kernels/iq_kernels.hpp) | 81 | i-quant API, `NativeExpertLayout`, grouped experts | [moe](moe-and-experts.md) |
| [kv_q4.hpp](../../../../include/strata/kernels/kv_q4.hpp) | 61 | `block_q4_0`, FWHT and Q4_0 KV API | [kv](kv-cache.md) |
| [kv_q8.hpp](../../../../include/strata/kernels/kv_q8.hpp) | 41 | INT8 KV API | [kv](kv-cache.md) |
| [kv_stream.hpp](../../../../include/strata/kernels/kv_stream.hpp) | 104 | `KvHostPools`, `KvStreamMap`, `KvFormat`, streaming API | [kv](kv-cache.md) |
| [mrope.hpp](../../../../include/strata/kernels/mrope.hpp) | 64 | M-RoPE position table, opt-in RoPE angle table | [norms](norms-rope-elementwise.md) |
| [native_flash_attn.hpp](../../../../include/strata/kernels/native_flash_attn.hpp) | 36 | Short-context FlashAttention adapter | [attention](attention-qsa.md) |
| [native_gdn.hpp](../../../../include/strata/kernels/native_gdn.hpp) | 22 | Pinned GDN step and its switch | [gdn](gdn.md) |
| [native_gdn_preprocess.hpp](../../../../include/strata/kernels/native_gdn_preprocess.hpp) | 40 | Pinned GDN pre/post-processing | [gdn](gdn.md) |
| [native_gr_norm.hpp](../../../../include/strata/kernels/native_gr_norm.hpp) | 18 | Pinned weighted RMSNorm | [norms](norms-rope-elementwise.md) |
| [native_gr_postops.hpp](../../../../include/strata/kernels/native_gr_postops.hpp) | 28 | Pinned HC post-ops | [norms](norms-rope-elementwise.md) |
| [native_mmvq.hpp](../../../../include/strata/kernels/native_mmvq.hpp) | 115 | Pinned MMVQ API and type IDs | [moe](moe-and-experts.md) |
| [native_moe.hpp](../../../../include/strata/kernels/native_moe.hpp) | 21 | Pinned MoE combine | [moe](moe-and-experts.md) |
| [native_ple_postops.hpp](../../../../include/strata/kernels/native_ple_postops.hpp) | 38 | Pinned PLE post-ops | [norms](norms-rope-elementwise.md) |
| [native_qsa.hpp](../../../../include/strata/kernels/native_qsa.hpp) | 30 | Pinned QSA norm and gate | [attention](attention-qsa.md) |
| [native_qsa_indexer.hpp](../../../../include/strata/kernels/native_qsa_indexer.hpp) | 46 | Pinned indexer append | [attention](attention-qsa.md) |
| [native_qsa_score.hpp](../../../../include/strata/kernels/native_qsa_score.hpp) | 31 | Pinned indexer score | [attention](attention-qsa.md) |
| [native_rope.hpp](../../../../include/strata/kernels/native_rope.hpp) | 20 | Pinned RoPE | [norms](norms-rope-elementwise.md) |
| [native_router.hpp](../../../../include/strata/kernels/native_router.hpp) | 19 | Pinned 512/top-10 router | [moe](moe-and-experts.md) |
| [ngram.hpp](../../../../include/strata/kernels/ngram.hpp) | 216 | PLE n-gram hash and table reader (host code, `src/kernels/ngram.cpp`) | [CPU: n-gram](../cpu/ngram.md), [norms](norms-rope-elementwise.md) |
| [ple.hpp](../../../../include/strata/kernels/ple.hpp) | 123 | `PleWeights`, `PleOut`, PLE block API | [norms](norms-rope-elementwise.md) |
| [q8_1_finite.hpp](../../../../include/strata/kernels/q8_1_finite.hpp) | 33 | Finite clamp of q8_1 scale/sum (#606) | [gemv](gemv-and-quant.md) |
| [qsa.hpp](../../../../include/strata/kernels/qsa.hpp) | 393 | `QsaShapes`, `QsaStep`, cache layouts, QSA API | [attention](attention-qsa.md) |
| [qsa_decode_attn.hpp](../../../../include/strata/kernels/qsa_decode_attn.hpp) | 79 | `QsaAttnPools`, split-K decode attention API | [attention](attention-qsa.md) |
| [qsa_prompt_attn.hpp](../../../../include/strata/kernels/qsa_prompt_attn.hpp) | 32 | Tensor-core prompt attention API | [attention](attention-qsa.md) |
| [qsa_select.hpp](../../../../include/strata/kernels/qsa_select.hpp) | 57 | Block scores / top-k API | [attention](attention-qsa.md) |
| [quantize_act.hpp](../../../../include/strata/kernels/quantize_act.hpp) | 58 | Q8_0 / Q8_K quantizer API | [gemv](gemv-and-quant.md) |
| [rope.hpp](../../../../include/strata/kernels/rope.hpp) | 64 | `rope_neox_pair`, table builders, `rope_neox_apply` | [norms](norms-rope-elementwise.md) |
| [rope_scaling.hpp](../../../../include/strata/kernels/rope_scaling.hpp) | 193 | `RopeScaling` (none / linear / YaRN), `rope_scaled_angle` | [norms](norms-rope-elementwise.md) |
| [router_top10.hpp](../../../../include/strata/kernels/router_top10.hpp) | 26 | Router API | [moe](moe-and-experts.md) |
| [s2_expert_grouped.hpp](../../../../include/strata/kernels/s2_expert_grouped.hpp) | 131 | S2 GPU expert tier API | [moe](moe-and-experts.md) |
| [s2_gemv.hpp](../../../../include/strata/kernels/s2_gemv.hpp) | 25 | S2 GEMV API | [gemv](gemv-and-quant.md) |
| [s2_gemv_q8.hpp](../../../../include/strata/kernels/s2_gemv_q8.hpp) | 22 | S2 x Q8_0 GEMV API | [gemv](gemv-and-quant.md) |
| [s_gemv.hpp](../../../../include/strata/kernels/s_gemv.hpp) | 133 | `SForm`, `Codebook`, S-family GEMV API | [gemv](gemv-and-quant.md) |
| [sampler.hpp](../../../../include/strata/kernels/sampler.hpp) | 86 | `SamplerParams`, sampler API, `penalty_rows` | [sampling](sampling-and-verify.md) |
| [shared_expert.hpp](../../../../include/strata/kernels/shared_expert.hpp) | 103 | `NativeSharedWeights`, shared expert and `moe_combine` API | [moe](moe-and-experts.md) |
| [verify_kernels.hpp](../../../../include/strata/kernels/verify_kernels.hpp) | 101 | Verify-window and MTP helper API, `kVerifyMaxT` | [gdn](gdn.md), [sampling](sampling-and-verify.md) |

## Key concepts

### Two models, one kernel set

| | Flash-Next (`qwen4exp`) | Qwen3.6-35B-A3B (`qwen35moe`) |
| --- | --- | --- |
| Residual | hyper-connections, `hc` = 4 streams of 2,560 (`gr*`, `fused_gr`) | plain pre-norm residual: `native_qsa_rms_norm_weighted` + `add_inplace` (`src/core/layer.cpp:1231-1240`) |
| Linear-attention layers | GDN, h_k 16 / h_v 48, gate `sigmoid(z)` | GDN, h_k 16 / h_v 32, gate `SiLU(z)` (`GdnGate::Silu`) |
| Full-attention layers | QSA: 24 q heads / 2 KV / 256, sparse selection through a 4-head indexer | dense attention: 16 q heads / 2 KV / 256, no indexer (`qsa_dense`) |
| Router | 512 experts, top-10 (`native_router` only applies here) | 256 experts, top-8 (`router_top10`) |
| Experts | n_ff 640; S2 packs (`s2_expert_grouped`) or GGUF i-quants (`native_expert_grouped`) | n_ff 512; GGUF IQ2_S/IQ3_XXS/IQ4_XS or Q4_K/Q5_K/Q6_K (`native_expert_grouped`) |
| PLE | yes (`ple`, `native_ple_postops`) | none |

The figures are from the kernel headers and `docs/QWEN36-35B-A3B.md` section 1.

### "native_*" kernels

The files named `native_*.cu` are not Qwen3.6-specific. They are upstream Strata's adapters that reproduce
llama.cpp's CUDA backend at commit `3cf03257` operation for operation (same reduction trees, same `--use_fast_math`
arithmetic). CMake compiles them with `--use_fast_math` (`CMakeLists.txt:399-412`). The engine enables each one
through a CLI flag (`--native-gdn`, `--native-qsa`, `--native-rope`, `--native-router`, `--native-moe-combine`,
`--native-qsa-indexer`, `--native-ple-postops`, `--gr-native-mmvf`, `--native-bf16`, `--native-bf16-extra`,
`--native-flash-attn-short`). `--native SHARD1` turns all of them on at once (`src/program/generate.cpp:1844-1890`,
switches applied at `:2616-2653`). The front end always starts the engine with `--native <model>`
(`frontend/main.cpp:677`), so in this fork the native paths are the default runtime paths. The other files are the
older "legacy" kernels (canonical S-forms, BF16-rounded activations). They remain the fallback, the reference
the parity tests compare against, and the path for S2 packs.

### Qwen3.6 port changes in this area

`git diff 12a7825 HEAD -- src/kernels/cuda include/strata/kernels` (12a7825 is the upstream import) touches only:

| File | Change |
| --- | --- |
| `gdn.hpp`, `gdn.cu`, `fused_gdn.*`, `native_gdn_preprocess.*`, `verify_kernels.*` | `GdnGate` parameter: SiLU output gate for `qwen35moe`; `h_v` generic (32 or 48) |
| `qsa.hpp`, `qsa.cu` | `qsa_dense`, `qsa_shapes_valid`, `qsa_selection_cap`; the indexer entry points refuse a dense shape |
| `qsa_decode_attn.*` | `G` (query heads per KV head) a template parameter (2, 8, 12); null `ids` = identity; split-K chunk cap of 128 for dense shapes; `qsa_identity_ids` |
| `qsa_prompt_attn.cu` | `G` templated (2, 8, 12), null `ids` |
| `iq_kernels.*` | Q6_K MMVQ / dequant; Q5_K, Q6_K and IQ3_XXS added to the grouped-expert down list (Qwen3.6 n_ff 512) |

### Per-token data flow (decode, one layer)

```
             Flash-Next                                   Qwen3.6
R (hc x 2560) --fused_gr_read--> mixed        R (2048) --native_qsa_rms_norm_weighted--> mixed
                    |                                             |
       +------------+--------------+                  +-----------+------------+
   GDN layer                   QSA layer          GDN layer                dense attention
 native_mmvq (qkv,z)       native_mmvq (q,k,v)   (same kernels,           native_mmvq (q,k,v)
 fused_gdn_conv_l2         norm + rope           GdnGate::Silu)           norm + rope
 fused_gdn_ab              kv_append_{q8,q4}                              kv_append_*
 fused_gdn_step_norm       qsa_block_scores                               qsa_decode_attn_step(ids=null)
 native_mmvq (ssm_out)     qsa_block_topk                                 native_qsa_gate_apply
                           [kv_stream_resolve]                            native_mmvq (o)
                           qsa_decode_attn_step
                           gate, native_mmvq (o)
       +------------+--------------+                  +-----------+------------+
                    |                                             |
           fused_gr_read (ffn) --> bf16 MMVF router logits --> native_router_top10 | router_top10
                    |
           doorbell_publish --> CPU pool computes misses      (host side, src/core)
           VRAM-resident experts: native_expert_grouped | moe_grouped_s2 | moe_hit_grouped_s2_dev
           shared_expert(_multi) --> native_moe_combine | moe_combine --> residual write (+ cvec_apply)
```

The verify window (up to `kVerifyMaxT` = 8 tokens, `verify_kernels.hpp:23`) runs the same layer with the
multi-column forms: `fused_gr_read_multi`, `gdn_*_multi`, `native_mmvq` with `ncols` > 1,
`qsa_decode_attn_batch`, `shared_expert_multi`, `native_router_top10_multi`, `native_moe_combine_multi`.
According to `docs/QWEN36-35B-A3B.md` section 2, decode for a native pack runs only through the verify window. The
prompt path (`src/prefill/`) uses `qsa_prompt_attn_batch`, `qsa_block_scores_tc`, `dequant_f16` /
`iq_dequant_gu_f16` (for cuBLAS GEMMs) and `native_ple_postops_batch`.

### Conventions shared by all kernels

- Graph capture. Every per-token value (position, cell count, selection width, hit counts) is read from device
  memory, never passed as a host scalar. A CUDA graph bakes kernel arguments in at capture time, and a frozen
  host scalar would give plausible but wrong output. The `QsaStep` record (`qsa.hpp:176-182`) is the canonical
  example. Grids are sized by a constant capacity, and the kernels skip the surplus.
- No allocation on the token path. Scratch is caller-owned: `*_scratch_bytes()` helpers (`gr_workspace_init`,
  `shared_expert_scratch_bytes`, `ple_block_scratch_bytes`, `native_expert_scratch_bytes`,
  `moe_hit_grouped_scratch_bytes`, `qsa_decode_attn_scratch_floats`). The one exception is the sampler's split
  scratch (`sampler.cu:881`), which is never used under capture.
- Kernel selection is fixed at capture. Every `*_set_enabled` / `*_set_*` switch must be set before session
  capture, and existing graphs keep the kernels they captured.
- Bitwise multi-token forms. The `_multi` / `_batch` forms are written to equal the single-token call bit for bit,
  and the parity tests check that. Speculative decoding is exact only if a token's result does not depend on its
  window.
- Error handling. Legacy entry points print and `std::exit(1)` on bad geometry. The `native_*` adapters throw
  `std::invalid_argument` / `std::runtime_error`.

### Architecture-specific paths (CUDA)

| Feature | Requires | Files | Fallback |
| --- | --- | --- | --- |
| `__dp4a` | sm_61 | `dp4a.hpp` (`STRATA_DP4A`), all q8 dot products | bit-exact byte loop below sm_61 |
| `__nanosleep` | sm_70 | `dp4a.hpp` (`strata_spin_pause`) | plain spin |
| `mma.sync` FP16 m16n8k16 + `cp.async` | sm_80 | `qsa_prompt_attn.cu`, `fused_gr.cu` (staged read) | Turing: two m16n8k8 MMAs, v1 kernel; plain copies for `cp.async` |
| `mma.sync` m8n8k4 | sm_70 only, `STRATA_EXPERIMENTAL_SM60` builds | `qsa_prompt_attn.cu` (`prompt_attn_v70_kernel`) | prompt attention falls back to `qsa_decode_attn_batch` below sm_70 |
| `mma.sync` TF32 m16n8k8 | sm_80 | `qsa_select.cu` (3xTF32 scores), `native_qsa_score.cu` | warp FP32 kernel; FP32 FMA path |
| Thread-block clusters (`cudaLaunchKernelEx`, DSMEM) | sm_90 and code built for it | `qsa_select.cu` (top-k), `sampler.cu` (argmax) | one-CTA kernels |
| Pre-sm_75 decode attention (PR #540) | `STRATA_EXPERIMENTAL_SM60`, card below sm_75 | `qsa_decode_attn.cu` | `STRATA_ATTN_PRE75=0` keeps the standard kernel |

CMake refuses architectures below 75 unless `-DSTRATA_EXPERIMENTAL_SM60=ON` (`CMakeLists.txt:138-176`). That
option defines `STRATA_EXPERIMENTAL_SM60` for every target and allows sm_60, sm_61 and sm_70 (CUDA 12.x
toolkits only). The default architecture is 120. `STRATA_EMULATE_CC=75|80|86|89` (`include/strata/core/emulate.hpp`,
tests only) makes the run-time capability checks answer as another generation would. The source also carries HIP/AMD
paths (`__HIPCC__`, `STRATA_HIP_GFX906`). This fork does not build or support them (`UPSTREAM.md`). These pages
mention them only where they share a dispatcher with the CUDA path.

### Environment variables (read by the kernels)

| Variable | Default | Effect | Page |
| --- | --- | --- | --- |
| `STRATA_OLD_IQ_MMVQ` | off | per-column i-quant kernels instead of decode-once | [moe](moe-and-experts.md) |
| `STRATA_IQ_STAGE_GRID` / `_MMVQ` | on / off | stage i-quant codebook grids in shared memory | [moe](moe-and-experts.md) |
| `STRATA_GROUPED_V1` | off | older grouped-expert launch (separate SwiGLU + quantize) | [moe](moe-and-experts.md) |
| `STRATA_OLD_GROUPED`, `STRATA_GROUPED_PAIR_MIN_HITS` | off, 0 | previous S2 grouped kernels | [moe](moe-and-experts.md) |
| `STRATA_FUSED_SWIGLU_Q81`, `STRATA_DEC_BATCH` | on, on | shared-expert fusions | [moe](moe-and-experts.md) |
| `STRATA_ROUTER_BLOCK`, `STRATA_HIP_ROUTER_*` | - | HIP-only router variants | [moe](moe-and-experts.md) |
| `STRATA_MMVQ_WAVE` | - | HIP-only (gfx906) MMVQ layout | [moe](moe-and-experts.md) |
| `STRATA_ATTN_PRE75` | on | PR #540 kernel below sm_75 (SM60 builds) | [attention](attention-qsa.md) |
| `STRATA_PROMPT_ATTN_Q4`, `STRATA_PROMPT_ATTN_V1` | on, off | prompt attention for Q4_0 KV; first int8 version | [attention](attention-qsa.md) |
| `STRATA_QSA_WARP` | off | `1`, `attn` or `select`: pre-sm_80 kernels on any card | [attention](attention-qsa.md) |
| `STRATA_SCORES_MULTI` | on | block-scores kernel that reads each key once for <= 8 queries | [attention](attention-qsa.md) |
| `STRATA_TOPK_OLD`, `STRATA_QSA_CLUSTER`, `STRATA_TOPK_CAPACITY_GUARD`, `STRATA_TOPK_ACTIVE_ANY` | off, on, off, off | top-k kernel choice | [attention](attention-qsa.md) |
| `STRATA_HC_SPLIT`, `STRATA_GR_V3`, `STRATA_GR_DOWN_MAX4` | staged, off, off | fused HC read variants | [norms](norms-rope-elementwise.md) |
| `STRATA_ROPE_TABLE` | off | rotate by the float64 angle table instead of fast-math `cosf/sinf` | [norms](norms-rope-elementwise.md) |
| `STRATA_GDN_COMMIT_SPLIT` | on | lighter commit kernel when the commit writes no outputs | [gdn](gdn.md) |
| `STRATA_OLD_SAMPLER`, `STRATA_SAMPLER_ONE_BLOCK`, `STRATA_ARGMAX_MULTI` | off, off, on | sampler kernel choice | [sampling](sampling-and-verify.md) |
| `STRATA_EMULATE_CC` | off | answer capability queries as another card (tests) | this page |

## Tests

The parity programs live next to the kernels in `src/kernels/*_parity.cpp` and are registered as CTest entries in
`CMakeLists.txt` (most run with `--selftest`; run one with `ctest --test-dir build -R <name>`). Each leaf page lists
the ones that cover it. The oracle-linked `native_*_parity` tests (`CMakeLists.txt:420-445`) are built only when
`STRATA_ORACLE_SOURCE_DIR` and `STRATA_ORACLE_BUILD_DIR` point at a pinned llama.cpp build. Their sources are
expected in `bench/micro/`, which this repository does not contain, so those tests cannot be built from this tree.
The parity tests themselves are documented in [../parity/README.md](../parity/README.md).

## See also

- [Kernels overview](../README.md), [CPU kernels](../cpu/README.md), [parity tests](../parity/README.md)
- [Core](../../core/README.md) (the layer and session code that calls these kernels), [Prefill](../../prefill/README.md)
- `docs/QWEN36-35B-A3B.md` sections 3.2-3.4 (the kernel gap analysis), `docs/qwen36-port/step3-attn.md`,
  `step3-gdn.md`, `step3-experts.md`, `step3-prefill.md`, `pascal-build.md` (the sm_61 guard inventory)
- `docs/upstream/DETAILS.md`, "How it works"
