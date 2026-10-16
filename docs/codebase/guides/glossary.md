[Codebase docs](../README.md) › [Guides](README.md) › Glossary

# Glossary

## Purpose

The code, the comments and the port notes use a dense private vocabulary: model names, upstream plan identifiers,
tier names, quantization formats, ticket ids. Each entry below defines a term the way **this repository** uses it,
with the file that is the best place to read more. Links are relative to this page. Where the source does not spell
out an acronym, the entry says so rather than guessing.

"Flash-Next" and "Qwen3.6" are used throughout as short names for the two supported models.

## 1. Models and architecture

| Term | Meaning here | Read more |
| --- | --- | --- |
| **Flash-Next**, Qwen3.8-Flash-Next | The model Strata was written for: 125B-parameter MoE, llama.cpp architecture `qwen4exp` (HF `qwen4_exp_text`), 48 layers of width 2560 (36 GDN + 12 QSA), 512 experts top-10 of width 640 plus a shared expert, hyper-connections (`hc = 4`), a PLE block, one MTP layer, vocabulary 248,320. Served here only as ISTA-DASLab's GSQ-RCO IQ3_XXS GGUF. | [README.md](../../../README.md), [layout.hpp](../../../include/strata/core/layout.hpp) |
| **Qwen3.6-35B-A3B**, `qwen35moe` | The second supported model (llama.cpp arch `qwen35moe`): 35B parameters, 40 layers of width 2048 (30 GDN + 10 dense attention), 256 experts top-8 of width 512 plus a shared 512 with a scalar gate, plain residual (`hc = 1`), no PLE, no indexer; the `-MTP` files add a draft block `blk.40`. Served from Unsloth's UD-Q4_K_XL and UD-IQ3_XXS files. | [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) section 1 |
| **A3B** | "3B active": about 3B of the 35B parameters are used per token (README: "35B parameters, 3B active"). | [README.md](../../../README.md) |
| **Coder**, **Swift** | Upstream's Flash-Next variants: the GSQ-RCO Coder (pruned to 256 experts per layer) and Swift 1.5 (a fine-tune). Not supported by this fork; their `native_experts.txt` remain as test fixtures. | [data-files.md](data-files.md) |
| **GSQ-RCO** | Part of the ISTA-DASLab Flash-Next GGUF names; its routed-expert encoding is Q2_0. The acronym is not expanded in this tree. | [tools/gguf_reader.py](../../../tools/gguf_reader.py) line 19 |
| **`ModelGeometry`** | The per-model shape struct (widths, layer pattern, heads, experts, `hc`, `has_ple`, indexer) read from the GGUF's `<arch>.*` metadata since port step 2; before, it was a struct of Flash-Next defaults. | [model_geometry.hpp](../../../include/strata/core/model_geometry.hpp) |
| **MoE** | Mixture of experts: each layer's feed-forward is a set of small "expert" MLPs of which the router picks `k` per token, plus a shared expert. | [layer.hpp](../../../include/strata/core/layer.hpp) |
| **expert** | One routed MLP of one layer, identified by `(layer, expert)`; stored as a **blob** `[gate rows, up rows, down rows]`. Flash-Next has 48 x 512 = 24,576, Qwen3.6 40 x 256 = 10,240. | [expert_layout.hpp](../../../include/strata/kernels/cpu/expert_layout.hpp) |
| **shared expert** | The always-on MLP beside the routed ones (`ffn_*_shexp`), computed on the GPU; Qwen3.6 scales it by a 1-D sigmoid gate (`ffn_gate_inp_shexp`). | [shared_expert.hpp](../../../include/strata/kernels/shared_expert.hpp) |
| **router**, **top-k** | The per-layer projection (`ffn_gate_inp`, read as BF16) and selection: softmax over all experts, the `k` largest (10 for Flash-Next, 8 for Qwen3.6), weights renormalised. `router_top10` is generic in `n_expert <= 512`, `k <= 32`. | [router_top10.hpp](../../../include/strata/kernels/router_top10.hpp) |
| **GDN**, Gated DeltaNet | The recurrent ("linear attention") mixer of 3 of every 4 layers: a short causal conv (kernel 4) then a gated delta-rule state update with state 128 per head; the output gate is `sigmoid(z)` for Flash-Next and `SiLU(z)` for Qwen3.6 (study 3.2). | [gdn.hpp](../../../include/strata/kernels/gdn.hpp) |
| **QSA** | Flash-Next's full-attention layer (every 4th layer, `is_qsa_layer`): an **indexer** scores the cached cells and a top-k keeps at most 2,051 of them for attention. Qwen3.6's attention is dense (no indexer) but still runs through the `qsa_*` code with identity ids. The acronym is not expanded in the source. | [qsa.hpp](../../../include/strata/kernels/qsa.hpp), [step3-attn.md](../../qwen36-port/step3-attn.md) |
| **hc**, hyper-connections, **GR** | Flash-Next's residual is a stack of `hc = 4` streams; `gr_read` / `gr_write` (gated residual) mix them around each mixer and MoE. Qwen3.6 has `hc = 1` (plain residual), a separate block path since step 3. | [gr.hpp](../../../include/strata/kernels/gr.hpp), [step3-block.md](../../qwen36-port/step3-block.md) |
| **PLE** | Flash-Next's per-layer token embedding block: 16 rows per token gathered from `per_layer_token_embd.weight` and mixed into the residual. `model_geometry.hpp` describes Qwen3.6 as having "no per-layer token embedding, no n-gram table". | [ple.hpp](../../../include/strata/kernels/ple.hpp) |
| **n-gram table** | The 28.8 GB IQ4_NL table behind PLE, in Flash-Next's second GGUF shard; its 16 row indices per token come from a host hash of the last three token ids. Read per token from disk (`--ple-io direct`), mapped, or held in RAM (`--ple-io ram`, which the front end passes). | [ngram.hpp](../../../include/strata/kernels/ngram.hpp) |
| **MTP**, nextn | The model's multi-token-prediction layer, used as the drafter. Flash-Next's comes from the checkpoint's draft-head tensors (packed in `mtp/`); Qwen3.6's is the GGUF's `nextn` block `blk.40` (packed in `qwen36-mtp-draft/`). | [mtp.hpp](../../../include/strata/core/mtp.hpp) |
| **draft head** | The output projection the MTP layer drafts with: rows of the main model's `output.weight`, the whole vocabulary or a subset. | [src/core/mtp.cpp](../../../src/core/mtp.cpp) |
| **draft vocab** | The subset of token ids the draft head scores (`draft_vocab.bin`, int32 ids); smaller is less VRAM, but a token outside it is never drafted. | [data-files.md](data-files.md) |

## 2. Experts, memory and placement

| Term | Meaning here | Read more |
| --- | --- | --- |
| **pack** | A directory the engine loads beside the GGUF (`--pack`): `index.txt`, `dense.bin`, `native_experts.txt`, `conversions.json`, `expert-profile.bin`, `tokenizer/` (and optionally `experts.bin`). Made by `tools/iq_pack.py`. A **native pack** keeps experts in their GGUF encodings and serves them from the GGUF; the **canonical** (Q2_0/S2) pack has `experts.bin` and no `native_experts.txt` and is Flash-Next-only. | [../tools/README.md](../tools/README.md), [step4-pack.md](../../qwen36-port/step4-pack.md) |
| **draft pack** | The `--mtp` directory: the MTP layer's dense weights, experts and optional `draft_vocab.bin`. | [step4-drafter.md](../../qwen36-port/step4-drafter.md) |
| **profile** | `expert-profile.bin` (`STRP`): every `(layer, expert)` pair ranked hottest first; it decides which experts start in VRAM. `--expert-profile-save` writes a learned one. | [data-files.md](data-files.md) |
| **expert cache** | The VRAM tier of experts (`ExpertCache`): an arena of slots plus a residency table `(layer, expert) -> slot or -1`; the GPU computes resident experts, the rest are **misses**. `--expert-cache N|auto`. | [expert_cache.hpp](../../../include/strata/core/expert_cache.hpp) |
| **slot** | One blob-sized place in the expert cache. Native packs use **sized** slots (blob size differs per layer, `open_sized`). | [expert_cache.hpp](../../../include/strata/core/expert_cache.hpp) |
| **miss** | A routed expert not in VRAM: computed by the CPU expert pool from RAM, or copied to the GPU over PCIe (see PCIe fraction), or read from the file when RAM does not hold it. | [expert_source.hpp](../../../include/strata/core/expert_source.hpp) |
| **pinned RAM**, pinned arena | The page-locked host arena the experts live in (large pages when available) so the copy engine reaches full bandwidth. Low-RAM variants: `--mmap-experts`, `--resident-cpu-experts` / `--resident-experts`, `--ram-cap`. | [pinned.hpp](../../../include/strata/core/pinned.hpp) |
| **expert pool** | The CPU worker pool that computes missed experts in place while the GPU continues (`--pool-workers`); the front end's `--threads`. | [../core/experts/README.md](../core/experts/README.md) |
| **doorbell** | The flag the host loop polls so the CPU pool starts on a layer's misses as soon as the GPU has routed. | [session.hpp](../../../include/strata/core/session.hpp) |
| **PCIe fraction** | `--pcie-frac F`: the share (0..1) of each layer's distinct missed experts the GPU fetches over PCIe from pinned RAM instead of the CPU computing them; default 0.2 (direct) for the Q2_0 pack, 0.55 (DMA) for native packs; `--pcie-mode auto|dma|kernel|direct`. | [generate.cpp](../../../src/program/generate.cpp) lines 499-502 |
| **adaptive swaps**, adaptive tier | Every `--adapt-every` rounds (default 4) up to `--adapt-swaps` (96) of the most-routed missing experts replace the least-routed resident ones, by decayed counts (`--adapt-decay` 0.7); 0 = static residency. The counts are the **heat** that `--expert-profile-save` ranks by. | [generate.cpp](../../../src/program/generate.cpp) lines 503-528 |
| **fixed set**, VRAM planner | What must fit before any expert slot: dense weights, drafter, KV for the requested context, session, CUDA context, reserve, prompt buffers. The planner lowers the context toward 131,072, then refuses; every remaining byte becomes slots. `--vram-cap MiB` / `--ram-cap GiB` make a big card plan like a small one. | [vram_plan.hpp](../../../include/strata/plan/vram_plan.hpp) |

## 3. Attention and KV cache

| Term | Meaning here | Read more |
| --- | --- | --- |
| **KV streaming** | Flash-Next only: `--kv-resident N` keeps N cells per QSA layer in VRAM (minimum 20,480) and the authoritative K/V in pinned host memory; selected blocks are made resident on demand with a clock sweep. Refused for Qwen3.6, whose dense attention reads every cell. | [kv_stream.hpp](../../../include/strata/kernels/kv_stream.hpp) |
| **resident window** | The cells of a streamed layer held in VRAM (`--kv-resident N`). | [kv_stream.hpp](../../../include/strata/kernels/kv_stream.hpp) |
| **int8 KV** | `--kv int8`: int8 codes with one FP16 scale per 64 values, half of FP16's memory; the front end passes it when `--ctx` > 8192. | [kv_q8.hpp](../../../include/strata/kernels/kv_q8.hpp) |
| **q4_0 KV** | `--kv q4_0`: K and V rotated by a 256-point Hadamard matrix, then ggml q4_0 (576 B per cell vs int8's 1,056); the README's Qwen3.6 command line uses it. | [kv_q4.hpp](../../../include/strata/kernels/kv_q4.hpp) |
| **K8V4** | `--kv k8v4`: INT8 K with rotated Q4_0 V, 816 B per cell; not with `--kv-resident`. | [layer.hpp](../../../include/strata/core/layer.hpp) |

## 4. Speculation (drafting and verification)

| Term | Meaning here | Read more |
| --- | --- | --- |
| **verify window** | T consecutive tokens (the last accepted one plus T-1 drafts, `T <= kVerifyMaxT = 8`) through all layers in one captured graph; each row's argmax equals what greedy decode would produce, so a draft is accepted exactly when greedy would have produced it. The only decode path for native packs. `--spec N` sets the window (the front end passes `--spec 4`). | [verify.hpp](../../../include/strata/core/verify.hpp) |
| **acceptance** | The share of drafted tokens the verify window accepts, and tokens committed per round; e.g. Qwen3.6 IQ3_XXS 0.77 accepted / 3.20 tokens per round with the full head (step4-integration.md). The controller and draft policy keep running estimates of it. | [controller.hpp](../../../include/strata/spec/controller.hpp) |
| **suffix draft**, prompt lookup | A weight-free drafter that proposes what followed the longest earlier occurrence of the current suffix (trigram index); `--suffix-draft N` (default 3, 0 = MTP only). The draft policy takes its window only when it pays. | [suffix_drafter.hpp](../../../include/strata/spec/suffix_drafter.hpp), [draft_policy.hpp](../../../include/strata/spec/draft_policy.hpp) |
| **`--spec-min-p`** | A draft enters the window only while it and every draft before it have at least this probability under the draft layer (front end: 0.5). | [generate.cpp](../../../src/program/generate.cpp) lines 506-509 |
| **coupled draft** | Opt-in sampling mode (`STRATA_SPEC_COUPLED=1`, `--coupled-draft`) that draws drafts with the same Philox counters as the verify row that checks them. | [coupled_draft.hpp](../../../include/strata/core/coupled_draft.hpp) |

## 5. Quantization formats

| Term | Meaning here | Read more |
| --- | --- | --- |
| **Q2_0** | ggml type 42, the GSQ-RCO routed-expert encoding (block of 64 values in 18 bytes per `tools/gguf_reader.py`); not in gguf-py, so `ref/load.py` decodes it itself. | [tools/gguf_reader.py](../../../tools/gguf_reader.py) |
| **S2** | Q2_0's canonical form inside Strata's canonical pack: 2-bit codes packed 4 per byte and one FP32 scale per 64 values, decoded as `(code - 1) * scale`. The `s2_*` kernels work on it. | [dequant_s2.hpp](../../../include/strata/kernels/dequant_s2.hpp) |
| **IQ2_XS, IQ3_XXS, IQ3_S** (and IQ1_M, IQ2_XXS, IQ2_S, IQ4_XS, IQ4_NL) | llama.cpp's "i-quant" codebook formats. Native packs keep experts in them; the CPU computes them with ggml-cpu's own `vec_dot`, the GPU with `iq_kernels.cu`. As file names (`...-IQ3_XXS.gguf`) they name a whole quantization whose experts mix several of these per layer. | [native_expert.hpp](../../../include/strata/kernels/cpu/native_expert.hpp), [data-files.md](data-files.md) |
| **K-quants** | ggml's Q4_K, Q5_K, Q6_K (256-value super-blocks; a down projection in them needs `n_ff % 256 == 0`, which Qwen3.6's 512 meets and Flash-Next's 640 does not). The Q4_K_XL file's experts are K-quants, which the README calls "the fast AVX2 path for the experts the CPU computes". `STRATA_MMQ_KQUANTS` builds their prompt-path MMQ kernels. | [step3-experts.md](../../qwen36-port/step3-experts.md) |
| **UD quant** | Unsloth's "dynamic" GGUF quantizations (`UD-Q4_K_XL`, `UD-IQ3_XXS`): per-layer, per-role type choices rather than one type throughout (CMake contrasts them with "a plain llama-quantize _M tier"). | [README.md](../../../README.md) |
| **Q8_0 / Q8_1 / Q8_K** | ggml 8-bit formats: Q8_0 for the Q4_K_XL file's dense projections (the IQ3_XXS file's are Q6_K) and the synthetic fixtures; Q8_1 / Q8_K are the activation formats the native dot products quantize to. | [native_expert.hpp](../../../include/strata/kernels/cpu/native_expert.hpp) |
| **BF16** | The 16-bit float the engine reads routers and other small tensors as; `iq_pack.py` writes an F32 tensor as BF16 only when every value is exactly representable (or with `--compat-bf16`, lossy). | [step4-pack.md](../../qwen36-port/step4-pack.md) |

## 6. Running a request

| Term | Meaning here | Read more |
| --- | --- | --- |
| **prefill** | Processing the prompt in chunks (`--prefill CHUNK|auto`, up to 8,192 by default) with batched GEMMs and experts streamed to the GPU through a ring that borrows expert-cache slots. | [../prefill/README.md](../prefill/README.md) |
| **decode** | Producing output tokens one verify window at a time after prefill. | [verify.hpp](../../../include/strata/core/verify.hpp) |
| **prompt cache**, checkpoint | `--prompt-cache N` keeps N conversation checkpoints (GDN state, indexer tails, PLE history) between requests so a continued chat reads only its new tokens; `--conversation-cache-mib` parks whole conversations in RAM. | [../core/conversation/README.md](../core/conversation/README.md) |
| **mmproj** | The vision projector GGUF (`--mmproj PATH`), the model's own image encoder. | [../frontend/README.md](../frontend/README.md) |
| **mtmd** | llama.cpp's multimodal library (`tools/mtmd` in the submodule) that the front end uses to run the mmproj encoder on the CPU. | [frontend/vision.hpp](../../../frontend/vision.hpp) |
| **control vector**, cvec | A per-layer direction added to or projected out of the residual (`--control-vector-scaled`, `--cvec-mode`), llama.cpp's format. | [cvec.hpp](../../../include/strata/kernels/cvec.hpp) |

## 7. Testing, references and process

| Term | Meaning here | Read more |
| --- | --- | --- |
| **parity test** | A program that holds a kernel to a reference (host code, the previous kernel bit for bit, ggml, or the oracle's dumps), with negative controls. | [../kernels/parity/README.md](../kernels/parity/README.md) |
| **oracle** | `ref/qwen36.py`, the numpy forward pass of `qwen35moe` written from the pinned llama.cpp graph and cross-checked against it; the "Q8 oracle" is its `--act-quant q8` mode. Upstream also used the name for llama.cpp itself (`STRATA_ORACLE_*`). | [oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md) |
| **seam** | A named intermediate tensor (llama.cpp's `cb()` name, e.g. `attn_norm-0`, `ffn_moe_topk-3`) at which the engine and the oracle are compared. | [tools/qwen36_compare_dumps.py](../../../tools/qwen36_compare_dumps.py) |
| **rival reading** | A plausible wrong interpretation of the graph (sigmoid for SiLU, adjacent-pair RoPE, ...); a fixture must make every rival differ by more than 5% relative L1 before a pass means anything. | [tools/test_qwen36_oracle.py](../../../tools/test_qwen36_oracle.py) |
| **teacher forcing** | Running the engine along a fixed reference continuation: `--spec-oracle FILE --teacher-force` commits every window row and starts the next window from the reference's token, so per-position logits (`--dump-logits`) line up with llama.cpp's. | [generate.cpp](../../../src/program/generate.cpp) lines 491-495 |
| **KL** | Kullback-Leibler divergence KL(reference \|\| engine) in nats between the softmaxed logits at each position, reported with top-1 agreement by `qwen36_compare_logits.py`; e.g. "mean KL 1e-3 to 7e-3" in the README. | [tools/qwen36_compare_logits.py](../../../tools/qwen36_compare_logits.py) |
| **E0 trace** | Phase 0's route trace of the IQ3_XXS file through llama.cpp on a 110-prompt corpus; the source of `data/qwen36-expert-profile.bin`. | [step4-pack.md](../../qwen36-port/step4-pack.md) section 3 |
| **the gate** | The step-2 stop (`qwen35moe_not_yet`) after the VRAM plan for a Qwen3.6 file, lifted in step 4 (`generate.cpp:2187`). | [step4-runner.md](../../qwen36-port/step4-runner.md) |
| **step 1..7** | The Qwen3.6 port's Phase 2 steps (oracle, geometry, kernels, pack/drafter/runner, serve, Pascal build, README). | [project-notes.md](project-notes.md) |
| **REGINA-NNN** | Ticket ids of the "regina" project, the author's measurement and tracking repository (not in this tree). Seen here: 104 (the chat-parser patch applied at configure time), 106 (Pascal: wrong activations in the multi-token CPU dispatch), 108 (the teacher-forced logits check against llama.cpp: 64 positions, pass = top-1 at least 62 and mean KL below 1e-2), 110 (streaming `content` deltas through the front end), 111 (the remaining token differences on the 1080 Ti), 112 (`max_tokens` and the engine's 8-slot reserve). | [step5-serve.md](../../qwen36-port/step5-serve.md) |
| **#NNN**, PR #NNN | Upstream Strata issue and PR numbers carried in comments (e.g. #477 profile save, #533 elastic VRAM, #136 fused prompt MoE). | [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) |
| **P0.S1, P2.S5, plan v0.3 P6, R4, ...** | Upstream Strata's own plan/phase identifiers in file headers; the plans themselves are not in this tree. | [UPSTREAM.md](../../../UPSTREAM.md) |
| **Pascal**, SM60 build | GPUs below compute capability 7.5 (sm_61 GTX 10, sm_70 Volta), built with `STRATA_EXPERIMENTAL_SM60` (FP32 and dp4a paths, no tensor cores); the Makefile turns it on for any arch below 75 in `CUDA_ARCHS`. | [pascal-build.md](../../qwen36-port/pascal-build.md) |

## See also

- [project-notes.md](project-notes.md) - where each term's history is written up.
- [data-files.md](data-files.md) - draft vocab, profile and `native_experts.txt` formats.
- [testing.md](testing.md) - parity tests, fixtures, the oracle tools.
