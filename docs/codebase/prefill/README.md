[Codebase docs](../README.md) › Prefill

# Prefill (batched prompt processing)

The prefill subsystem (`strata::prefill`, library target `strata_prefill`, plus `strata_mmq` for the int8 expert
kernels) reads a prompt in chunks of `T` tokens (256 to 32768, usually 1024 to 8192) through every layer at once,
instead of one token at a time through the decode path. It leaves the session state exactly where the token path
would have left it: the Gated DeltaNet (GDN) recurrent and conv state of every linear-attention layer, the K/V cells
of every full-attention layer (and on Flash-Next the QSA indexer state and the PLE n-gram history), and, through a
callback, the MTP draft layer's K/V. Decode (the token path in `src/core/layer.cpp`, or for native packs the verify
window in `src/core/verify.cpp`) then continues from the next position. The only caller in the engine is
`src/program/generate.cpp`: the one-shot `generate` path and the `serve` loop both build a `strata::prefill::Prefill`,
lend it the tail of the GPU expert cache as device memory, call `Prefill::run`, and refill the lent slots afterwards.
Prefill in turn calls `core` (weights, layer views, session state, expert source and cache, embedding), the shared
GPU kernels in `src/kernels/cuda/` (QSA attention, indexer, KV formats, PLE, rope scaling, i-quant dequantizers),
cuBLAS / hipBLAS(Lt), and llama.cpp's MMQ kernels compiled from the ggml checkout.

The per-layer arithmetic in one sentence: the projections are tensor-core GEMMs (quantized dense weights dequantized
to FP16 on the fly, BF16 weights as they are), the GDN conv and recurrence walk the chunk inside one kernel, the
prompt attention runs on tensor cores over the paged KV pools, and the routed experts are grouped by expert and
multiplied by int8 kernels (llama.cpp MMQ, or Strata's own fused kernels) with each expert read either from its
VRAM cache slot or streamed host -> device through a pinned ring on a copy stream.

## Contents

| Page | What it covers |
| --- | --- |
| [pipeline.md](pipeline.md) | `Prefill` class: `init` / `carve` / `relayout` / `bytes_needed`, the chunk and layer loop of `run_impl`, embeddings, hooks (`on_chunk`, `on_seam`, `should_stop`), the state it hands to decode, `draft_kv`, layer split and the peer GPU, timing and stats, and how `generate.cpp` drives it |
| [expert-streaming.md](expert-streaming.md) | How experts reach the card during a prompt: borrowed cache slots, the staging ring, the stream-all walk, the `Stager` host-copy threads, the issuer thread, ring sizing in bytes, the auto chunk scan, refill after the prompt; why small-VRAM cards spend the prompt copying experts |
| [gemm.md](gemm.md) | `Gemm` (`gemm.cu`): cuBLAS BF16 / FP16 / native-dequantized products, the sub-sm_80 BF16 paths, hipBLASLt with a tuning table (`hipblaslt_tuning.hpp`) |
| [moe.md](moe.md) | The MoE half: router, shared expert, the four routed-expert paths (FP16 GEMM fallback, llama.cpp MMQ, fused Q2_0 int8, fused native IQ int8), how one is chosen per layer, combine, peer-GPU share |
| [ggml-bridge.md](ggml-bridge.md) | `moe_mmq.cu` and `ggml_cuda_host.cu`: compiling ggml-cuda's MMQ into `strata_mmq`, `supported` / `fits`, `STRATA_MMQ_KQUANTS`, the HIP `STRATA_PREFILL_MMQ` build |
| [gdn-and-attention.md](gdn-and-attention.md) | `kernels.cu`: hyper-connection and plain-block kernels, the chunked GDN gates / conv / recurrence (all variants), the attention half (Q/K norm, rope, `kv_append`, Flash-Next indexer and selection, prompt attention), how the state lands for decode |
| [tests.md](tests.md) | `gdn_rec_parity`, `gemm_bf16_parity`, `prefill_qwen36_test` and its fixture, the fused / MMQ expert tests, debug env vars |

## Source map

| Path | Lines | Role |
| --- | ---: | --- |
| [src/prefill/prefill.cpp](../../../src/prefill/prefill.cpp) | 3359 | `Prefill` implementation: geometry checks, buffer layout, expert streaming, the layer loop, peer / layer-split help, draft-layer K/V |
| [src/prefill/gemm.cu](../../../src/prefill/gemm.cu) | 556 | `Gemm`: cuBLAS `cublasGemmEx` BF16 / FP16 products, dequantize-then-GEMM for GGUF weights, Volta/Turing/Pascal BF16 fallbacks, hipBLASLt dispatch |
| [src/prefill/kernels.cu](../../../src/prefill/kernels.cu) | 1130 | Element-wise and stateful prompt kernels: hyper-connection, plain-block norm, GDN gates/conv/recurrence, router, SwiGLU, combine, rope, KV append, conversions |
| [src/prefill/moe_fused.cu](../../../src/prefill/moe_fused.cu) | 465 | Fused int8 tensor-core experts for the Strata Q2_0 pack (sm_80+), GPU-side grouping |
| [src/prefill/moe_fused_iq.cu](../../../src/prefill/moe_fused_iq.cu) | 646 | Fused int8 experts for native GGUF i-quant packs (gate/up IQ2_XXS..IQ4_XS, down Q2_0 / IQ4_NL), opt-in |
| [src/prefill/moe_mmq.cu](../../../src/prefill/moe_mmq.cu) | 268 | Wrapper over llama.cpp's MMQ (`mul_mat_q_case`, `quantize_mmq_q8_1_cuda`), expert gathers, SwiGLU, iota |
| [src/prefill/ggml_cuda_host.cu](../../../src/prefill/ggml_cuda_host.cu) | 147 | The few ggml-cuda host symbols MMQ needs (`ggml_cuda_info`, error hook, buffer pool) without linking ggml-cuda |
| [src/prefill/gdn_rec_parity.cu](../../../src/prefill/gdn_rec_parity.cu) | 1094 | Test/bench executable: GDN recurrence variants bit for bit, engine launches vs a double-precision reference, both geometries |
| [src/prefill/gemm_bf16_parity.cu](../../../src/prefill/gemm_bf16_parity.cu) | 155 | Test/bench executable: `Gemm::bf16` vs cuBLAS's own BF16 GEMM |
| [src/prefill/hipblaslt_tuning.hpp](../../../src/prefill/hipblaslt_tuning.hpp) | 123 | Parser and lookup for the `STRATA_HIPBLASLT_TUNING_V1` solution table (HIP only) |
| [include/strata/prefill/prefill.hpp](../../../include/strata/prefill/prefill.hpp) | 195 | Public `Prefill` API and `PrefillStats` |
| [include/strata/prefill/gemm.hpp](../../../include/strata/prefill/gemm.hpp) | 66 | `Gemm` class |
| [include/strata/prefill/kernels.hpp](../../../include/strata/prefill/kernels.hpp) | 123 | Kernel launch functions, `GdnGeom` |
| [include/strata/prefill/moe_fused.hpp](../../../include/strata/prefill/moe_fused.hpp) | 64 | `strata::prefill::fused` API (Q2_0 path), `Batch`, `kTileRows`, `kMaxBatch` |
| [include/strata/prefill/moe_fused_iq.hpp](../../../include/strata/prefill/moe_fused_iq.hpp) | 43 | `NativeGeom`, `native_supported`, `experts_native` |
| [include/strata/prefill/moe_mmq.hpp](../../../include/strata/prefill/moe_mmq.hpp) | 88 | `strata::prefill::mmq` API: `Product`, `Context`, `GatherGroup` |

Build: `strata_prefill` = `gemm.cu` + `kernels.cu` + `prefill.cpp` (`CMakeLists.txt:565`); `strata_mmq` =
`moe_mmq.cu`, `moe_fused.cu`, `moe_fused_iq.cu`, `ggml_cuda_host.cu` and ggml's `quantize.cu` plus one
`mmq-instance-<type>.cu` per covered type (`CMakeLists.txt:1218`). Linking `strata_mmq` defines
`STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1` on `strata_prefill` (`CMakeLists.txt:1244`); without them
`prefill.cpp:48-84` compiles stub namespaces so the FP16 expert path is used everywhere.

## Data flow of one chunk

```
tokens[c0 .. c0+T)                                              (generate.cpp: Prefill::run)
   |  embed (batched gather or per token; image rows from embd_rows)
   v
 R [T x hc*n_embd]  (Flash-Next: broadcast to 4 hyper-connection streams; Qwen3.6: R = emb)
   |
   |  for layer l in [LB, LE):                         experts of every layer streamed in id order
   |    half 0: norm/read -> mixer -> write            on the copy stream (chunks >= 1024 tokens)
   |       GDN layer : qkv,z (FP16 GEMM) alpha,beta (BF16 GEMM) -> gates -> conv -> recurrence -> ssm_out
   |                   writes ss.gdn_state (recurrent state + conv history)
   |       attn layer: K,V,Q (FP16 GEMM) -> k/q norm, rope -> kv_append into paged pools
   |                   [Flash-Next: indexer append, block scores, top-k selection]
   |                   -> prompt attention (tensor cores) -> gate -> attn_output
   |    half 1: norm/read -> MoE -> write
   |       router (BF16 GEMM) -> top-k -> shared expert (FP16 GEMMs)
   |       routed experts: group by expert -> MMQ / fused int8 / FP16 GEMM, each expert from
   |                       a resident cache slot or a ring slot just copied from host RAM
   |       moe_combine -> bo;  R += bo (plain) / gr_write (hyper-connection)
   v
 R rows of the chunk -> on_chunk (MTP draft layer K/V) ; next chunk
```

## Key concepts

- **Chunk `T`.** Chosen by `generate.cpp` (`--prefill CHUNK` or `--prefill auto[:N]`). Every non-resident expert a
  chunk routes to (from 1024 tokens on: every non-resident expert, routed or not) is copied over PCIe once per chunk,
  so a bigger chunk streams fewer bytes per token. See [expert-streaming.md](expert-streaming.md).
- **Borrowed memory.** By default every prompt buffer is carved out of the top slots of the GPU expert cache
  (`init(..., borrow, borrow_bytes)`), the experts in those slots are marked non-resident for the prompt, and the
  slots are refilled from host RAM after it. `--no-prefill-borrow` allocates instead.
- **One scratch region for three phases.** The GDN set, the attention set and the MoE set of buffers are never live
  at once, so they share one region sized as the max of the three (`prefill.cpp:677-700`, `carve` at
  `prefill.cpp:964-1005`).
- **Two model geometries.** `Geom::of` (`prefill.cpp:95-117`) turns `ModelGeometry` into the widths used
  everywhere; `init` refuses anything other than Flash-Next (`qwen4exp`: hc 4, 2560 wide, top-10 of 512, GDN
  48/16/10240) or Qwen3.6 (`qwen35moe`: hc 1, 2048 wide, top-8 of 256, GDN 32/16/8192), with head_dim 256, 2 KV heads,
  n_rot 64 (`prefill.cpp:805-834`).
- **Numerics differ from decode by design.** The prompt path uses BF16 / FP16 activations for dense GEMMs and q8_1 /
  int8 activations for experts; decode uses FP32 activations and its own kernels. The difference is gated by the
  tests in [tests.md](tests.md), not removed.

## Flash-Next vs Qwen3.6 at a glance

| Aspect | Flash-Next (`qwen4exp`) | Qwen3.6 (`qwen35moe`) |
| --- | --- | --- |
| Residual block | 4 hyper-connection streams, BF16 `hc_*` down/up/inject GEMMs (`gr_*` kernels) | plain block: `rms_norm_images`, `residual_add` |
| GDN geometry, gate | 48 v / 16 k heads, 10240 conv channels, `sigmoid(z)` | 32 / 16, 8192, `z * sigmoid(z)` (SiLU) |
| Attention | sparse QSA with indexer (append, block scores, top-k), 24 heads (G = 12) | dense, no indexer, null `ids`, 16 heads (G = 8) |
| MoE | top-10 of 512, n_ff 640; Strata Q2_0 pack or native packs | top-8 of 256, n_ff 512; native GGUF packs only (IQ3_XXS / UD-Q4_K_XL) |
| Expert kernel taken | fused Q2_0 int8 (default on sm_80+ for chunks of 1024+ tokens), else MMQ; fused IQ for native packs (opt-in) | MMQ (its down types are outside the fused IQ kernels' list), FP16 GEMM if MMQ cannot |
| PLE n-gram block at layer 1 | yes (when the PLE shard is loaded) | no |
| `draft_kv` (MTP K/V) | `fc_embedding` / `fc_hidden` + hyper-connection read | `nextn` block (`eh_proj`, enorm / hnorm) |

## See also

- [docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md): the port of this subsystem to Qwen3.6's geometry, the harness and its numbers.
- [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md), section 3.6: the study's table of literal sites.
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md): upstream's measurements ("Faster prompts (engine 0.1.13)", the 0.1.36 fused-kernel note).
- [README.md](../../../README.md), the Qwen3.6 speed tables: prompt throughput by VRAM budget.
- Other areas of these docs: [Core](../core/README.md) (session state, expert cache and source, layer layout),
  [Kernels](../kernels/README.md) (QSA attention, KV formats, PLE), [Program](../program/README.md) (`generate.cpp`).
