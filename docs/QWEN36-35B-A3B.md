# Running Qwen3.6-35B-A3B on UnbiasedStrata - the study

What it takes for Qwen3.6-35B-A3B (llama.cpp architecture `qwen35moe`) to run on this tree's Strata
engine, component by component, with the numbers the decision needs. Written 2026-10-11 against engine
0.1.39 (`CMakeLists.txt:11`, the `6f32ec0` snapshot; tree `8634357`). It is the Phase 2 design document
of `docs/PLAN-qwen36-35b-a3b.md` and the counterpart of Project Maya's `docs/GLM5-FLASH.md`, whose shape
it copies. Every `file:line` is this tree; "row N" is `docs/qwen36-port/geometry-inventory.md`; the model
facts come from `docs/qwen36-port/gguf-metadata.md` and `ref/qwen36.py` on branch `qwen36-oracle`; the
measurements from regina's `phase0-notes.md` and `replay/phase0-replay.md`.

Strata is a native implementation of one architecture, Qwen3.8-Flash-Next (`qwen4exp`), with its memory
tiers and its speculation stack built around that shape. Qwen3.6-35B-A3B is the same family one generation
earlier with three things removed (the hyper-connection residual, the sparse-attention indexer, the PLE
table) and one numerical change in the GDN gate. The kernel gap is small. The plumbing gap is not: the
removed pieces are assumed structurally in every block, in the verify window that is the only decode path
for native packs, in the drafter and in prefill, and `ModelGeometry` is never populated from the model file.

---

## 1. The model, exactly

From the shipped GGUFs (`gguf-metadata.md`, dumped with `tools/gguf_reader.py`; sizes and SHA-256s in
`phase0-notes.md`):

| | |
| --- | --- |
| Architecture | `general.architecture = qwen35moe`; `block_count` 40 (41 in the MTP files); `embedding_length` 2048; `context_length` 262,144 |
| Layer pattern | `full_attention_interval` 4: layer `il` is recurrent unless `(il+1) % 4 == 0`, so attention at 3, 7, ..., 39 (10 layers), GDN at the other 30 |
| GDN | `ssm.conv_kernel` 4, `ssm.state_size` 128, `ssm.group_count` 16 (k-heads), `ssm.time_step_rank` 32 (v-heads), `ssm.inner_size` 4096; conv channels 8192 = 2x2048 + 4096. Output gate `rms_norm(o) * ssm_norm * SiLU(z)` (`qwen35moe.cpp:267-276`) |
| Full attention | 16 heads, 2 KV heads, `key_length`/`value_length` 256, `rope.dimension_count` 64 (0.25 of 256), `rope.dimension_sections` [11, 11, 10, 0], `freq_base` 1e7, eps 1e-6; `attn_q` is 2x wide (query then gate per head, interleaved); output `* sigmoid(gate)`. **Dense**: no indexer |
| MoE | 256 experts, top-8, `expert_feed_forward_length` 512, shared expert 512 with a 1-D `ffn_gate_inp_shexp` scalar gate. Softmax over all 256, top-8 weights renormalised to sum 1 (`norm_w = true` hard-coded, `qwen35moe.cpp:506`); no `expert_weights_scale` (never loaded for this architecture) |
| Residual | plain: `attn_norm`, `post_attention_norm`, `output_norm`, all multiplicative RMS weights (no "+1") |
| MTP | `nextn_predict_layers` **absent in the plain file, 1 in the MTP files**; `blk.40` holds one attention + MoE block plus `nextn.eh_proj` [4096, 2048] over `concat(enorm(emb), hnorm(h))`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`; no `embed_tokens`, no `shared_head_head` (the draft head is the main `output.weight`) |
| Vocabulary | 248,320 tokens, `pre = qwen35`, BOS/EOS/PAD 248044/248046/248055; the same tokenizer the engine's `--eos-ids` default already names (row 152) |
| Float tensors | the router `ffn_gate_inp`, `ssm_alpha`, `ssm_beta`, the shexp gate and every norm are **F32** in the plain file; the nextn block's router and shexp gate are BF16 |
| Mixed quantization | per layer and per role: IQ3_XXS file has IQ2_S gate/up everywhere, IQ3_XXS down on 37 layers and IQ4_XS on 3 (34, 38, 39); the nextn block in the MTP files is Q3_K/Q4_K experts with Q8_0 projections |
| Fused forms | none shipped: gate and up are separate `ffn_gate_exps`/`ffn_up_exps`, attention layers carry separate `attn_q`/`attn_k`/`attn_v`; `attn_qkv` exists only in the 30 GDN layers (the conv input) |

The byte census of the three files on hand (gguf-py over the headers, 2026-10-11; `blk.40` separated):

| File | Non-expert on the GPU (`output` + projections + norms) | `token_embd` | Trunk experts (40 x 256) | Per expert mean / max | `blk.40` dense / experts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `UD-Q4_K_XL` (MTP repo, rev 5bc3e238, 21,784 MiB) | **2,038 MiB** (Q8_0 1,939 + F32 100) | 515 MiB Q8_0 | **18,726 MiB** (Q4_K 11,232, Q5_K 6,864, Q6_K 630) | 1.829 / 2.195 MiB (layer 39) | 40 / 464 MiB |
| `UD-IQ3_XXS` (12,589 MiB) | **1,597 MiB** (Q6_K 1,497 + F32 100) | 398 MiB Q6_K | **10,594 MiB** (IQ2_S 6,560, IQ3_XXS 3,626, IQ4_XS 408) | 1.035 / 1.172 MiB | none |
| `UD-IQ4_XS-MTP` (17,355 MiB) | 2,038 MiB | 515 MiB | 14,514 MiB | 1.417 / 1.883 MiB | 40 / 364 MiB |

The two non-expert figures are what llama.cpp's `-lv 4` reported on the 1080 Ti once the GPU-resident
expert layers are subtracted: Q4 `CUDA0 model` 5,917 MiB minus layers 32 to 39 (3,878 MiB) = 2,039; IQ3
7,475 minus layers 18 to 39 (5,878) = 1,597 (`phase0-notes.md`, smoke table). The token embedding stays in
host memory in both llama.cpp and this engine (`generate.cpp:2190`, "token embedding in mapped host
memory"), so it is not in the fixed set. The plan's Section 2 estimated ~3.1 GB for this item; the real
figure is 2.0 GiB.

## 2. What the engine already is

The implemented geometry is a struct of Flash-Next defaults (`include/strata/core/layout.hpp:27-59`):
n_embd 2560, 48 layers, full attention every 4th, GDN 16 k-heads / 48 v-heads / state 128 / conv 4 with
conv channels 10240 and value dim 6144 stored as members (row 2), QSA 24 heads / 2 KV heads / 256 with a
4-head 128-wide indexer, hyper-connections `hc` 4 with a 320-wide bottleneck, 512 experts x top-10 x 640.
`is_qsa_layer` is `layer % 4 == 3` (`layout.hpp:63`). Per layer (`src/core/layer.cpp:1170-1286`,
`1319-1342`):

    gr_read(attn) -> GDN | QSA -> gr_write(attn) -> gr_read(ffn) -> moe_route [doorbell] ... moe_finish -> gr_write(ffn)

The unit is a captured per-layer CUDA graph replayed by a host loop (`include/strata/core/session.hpp:1-15`)
that rings a doorbell at the router so a CPU expert pool starts on the misses while the GPU continues.
Experts live in three tiers: a VRAM cache of slots filled by a profile and swapped by decayed usage
(`include/strata/core/expert_cache.hpp`), a RAM tier (pinned arena, or `--resident-experts` /
`--resident-budget-gib` keeping the complement of the VRAM cache in a compact copy,
`generate.cpp:1506-1520`, `expert_source.cpp:1331`), and the model file mapped from disk
(`--mmap-experts`, `expert_source.cpp:618-680`) with a `MADV_WILLNEED` warm hint per role on predicted
ids (`expert_source.cpp:1129-1160`, called at `1264`). Decode for a *native* pack (experts served from the
GGUF in their own ggml types, `include/strata/kernels/cpu/native_expert.hpp`) runs only through the verify
window (`expert_source.cpp:1869-1875`, row 114): windows of up to `kVerifyMaxT = 8` rows
(`verify_kernels.hpp:21`) drafted by the model's own MTP layer, a separate Q8_0 pack in VRAM
(`src/core/mtp.cpp:150-216`, `mtp.hpp:1-20`), plus a prompt-lookup drafter (`generate.cpp:522-524`).
Prompts go through a chunked prefill with the experts streamed over PCIe through a ring that borrows
cache slots (`generate.cpp:2723`, `4404-4425`; `docs/upstream/DETAILS.md`, "How it works"). The front
end (`frontend/`) is model-agnostic: OpenAI API, llama.cpp's chat parser (`frontend/chat_parser_test.cpp`),
vision through `mtmd` (`frontend/vision.hpp:4-87`), and it refuses `--gpus` (`frontend/main.cpp:558`).

## 3. Component-by-component gap analysis

### 3.1 Applies unchanged

| Piece | Why |
| --- | --- |
| Layer pattern | `is_qsa_layer` (`layout.hpp:63`) gives 3, 7, ..., 39 for 40 layers; `n_qsa_layers() = 40/4 = 10` (row 55). Every per-layer loop in `session.cpp` uses `g.n_layers` (row 27). |
| GDN decode kernels | State 128, head pairing `head % h_k` (`native_gdn.cu:51`, `fused_gdn.cu:29`), grids from `h_v`, `h_v % h_k == 0` checked (`native_gdn.cu:99-101`, `fused_gdn.cu:148-153`); 32 % 16 = 0. The conv (`d_conv == 4 && state 128`, `layer.cpp:247-248`) matches. |
| Router | `router_top10` is a generic softmax top-k over runtime `n_expert <= 512`, `k <= 32` (`router_top10.hpp:21`, row 124); `moe_route` takes it whenever the pack is not 512/10 (`layer.cpp:370`). Semantics equal llama.cpp's: softmax over all, stable descending argsort, renormalise with the 2^-14 clamp (`router_top10.hpp:3-5`; `oracle-crosscheck.md`, "Router and experts"). |
| Shared expert | Shape-driven (`layer.cpp:418-451`); its Q8_K activation rule needs `n_ff % 256 == 0` (`shared_expert.cu:237-239`), which 640 failed and 512 satisfies (row 127). |
| Native expert kernels | CPU: `native_fmt` is ggml-cpu trait-driven, whole-block checks only (`native_expert.cpp:40-60`); 2048 and 512 are multiples of 256. `kNativeActBytes` 4096 / `kNativeHBytes` 1024 hold a 2048-wide Q8_K (2,336 B) and a 512-wide one (584 B) (row 111). GPU: `iq_kernels.cu:1076-1140` read `L.n_embd`/`L.n_ff` per layer; Q4_K/Q5_K gate-up and Q5_1/Q8_0 down are already present (row 120); the IQ2_S/IQ3_XXS/IQ4_XS/Q6_K of the shipped files are ggml types ggml-cpu has `vec_dot` for. |
| Expert cache and tiers | Residency `[n_layers x n_expert]`, slots, the profile header checked against `(n_layers, n_expert)` (`expert_cache.cpp:43`); the file source, the complement copy, the warm hint; all parameterised at runtime (rows 118, 109). |
| KV formats and streaming machinery | fp16 / int8 / q4_0 / K8V4 pools at `head_dim 256` (`kv_q4.cu:176`, row 103); paged pools, `--kv-resident`, the resident floor 20,480 cells (`layer.cpp:556`). The *use* of streaming changes (section 4.2). |
| RoPE | `n_rot 64`, `head_dim 256` (`native_rope.cu:121`), interleaved M-RoPE sections 11/11/10/0 with `t = h = w` for text (`mrope.hpp:3-12`); the oracle pins the same pairs `(p, p+32)` (`oracle-crosscheck.md`, "Full attention"). |
| Vocabulary, sampler, head | `n_vocab` is a kernel argument, merge handles <= 262,144 (`sampler.cu:607`); the native head takes `[n_in, n_out]` with `n_in % 256 == 0` (`native_head.cpp:21`); the drafter reads `n_vocab_ = output.weight.ne1` (`mtp.cpp:430`). |
| Speculation protocol | Verify windows, draft policy, prompt lookup; the drafter's *weights* change (3.10). |
| Front end | Tokenizer, template, tool-call parser, vision (row 154). |
| MTP drafter's attention | Already dense over every cell (`mtp.hpp:16-18`, row 76): the one dense-attention path in the tree. |

### 3.2 GDN: head pairing, SiLU gate, four kernels (small)

The geometry changes are runtime except two stored members, `ssm_conv_channels` and `ssm_value_dim`,
which must become 8192 and 4096 (row 2; `session.cpp:41-44` derives the state size from them). The one
numerical change: the output gate is `sigmoid(z)` in the engine and `SiLU(z)` in Qwen3.6 (row 39;
`layer.hpp:16` records the difference). Four kernels carry it:

| Kernel | Line | Path |
| --- | --- | --- |
| `gdn_out_norm` | `src/kernels/cuda/gdn.cu:148` | legacy decode |
| `gdn_step_norm_kernel` | `src/kernels/cuda/fused_gdn.cu:69` (`1/(1+__expf(-zz))`) | the default native decode, inside the fused step |
| out-norm kernel | `src/kernels/cuda/native_gdn_preprocess.cu:110-112` | native unfused |
| `gdn_rec_kernel`, `gdn_out_norm_kernel` | `src/prefill/kernels.cu:324, 612` (`sigm(z)`) | prompt |

The oracle's negative control shows the sigmoid reading moves the layer output by 1.07 relative L1 and the
logits by 0.92 (`oracle-crosscheck.md`, negative controls), so the trap is far above the 5% gate.
`src/kernels/gdn_parity.cpp:104` pins `h_v = 48` and its reference `sigmoid_f` (`:42`); both are re-pinned
at 32 v-heads with SiLU, the sigmoid reading becoming the negative case. The prompt GDN kernels are
specialised on `VPK = HV / HK = 3` value heads per key head (`prefill/kernels.cu:478`) and launched on the
`HV`/`HK`/`C` constants (`kernels.cu:18-19`, rows 89 to 91); they are the prefill item in 3.6.

### 3.3 Dense attention replacing QSA (the largest kernel item, and a memory item)

The decode kernel gathers K/V through `ids` and a page table and does not care how `ids` was built
(`qsa_decode_attn.cu:100-120`, row 102). So dense attention is a "select every cell" producer feeding the
existing kernel. What stands in the way:

1. **The producer.** `qsa_layer` runs the indexer unconditionally: the raw key projection (`layer.cpp:945`),
   the append (`:962-964`), the block scores and top-k (`:966-975`), with `indexer.{k,q}_proj.weight` and
   `indexer.{k,q}_norm.weight` dereferenced as required tensors (`:891-897`, row 45). The verify window
   does the same per row (`verify.cpp:757-761`, `772-870`, row 145). A dense path skips those steps and
   writes `ids = 0..n_kv-1` from `st.step`; the selection width `qsa_selection_width(n_kv, s) =
   min(n_kv, 2051)` (`qsa.hpp:144-147`) and the capacity `cap = qsa_selection_width(kTopkMaxCells, s)`
   (`layer.cpp:876`, `verify.cpp:352`, `prefill.cpp:865`) become `n_kv` and `max_cells`.
2. **The 12-versus-8 literal.** `G = 12` query heads per KV head, refused with `exit(1)` unless
   `n_head == 12 * n_head_kv` (`qsa_decode_attn.cu:17, 453, 497`; `qsa_prompt_attn.cu:22, 1447`; row 93, 94).
   Qwen3.6 is 8. `G` becomes a template parameter instantiated at 8 and 12.
3. **The 24-head short-context kernel.** `native_flash_attn.cu:196-206` refuses anything but 24/2/256 and
   contexts above 256; it is a diagnostic adapter (`layer.hpp:53`, "context <= 256") and is not on the
   prompt path. It stays as it is.
4. **State without the indexer.** `qsa_state_bytes` carves `idx_tail`, `idx_dead`, `idx_pooled` with
   `pooled_rows = max_cells/4 + 2` (`layer.cpp:559-575`, row 43): at 262,144 cells that is 65,538 x 128 x
   4 B = 32 MiB per attention layer, 320 MiB over ten, all dead. The drafter's ring already passes a state
   "which has no indexer" with `pooled_rows = 2` (`layer.cpp:527-531`); the dense state takes the same.
5. **The split-K scratch scales with the context.** `qsa_decode_attn_scratch_floats(cap, s) = (cap/64) x
   n_head x 258 + 64` (`qsa_decode_attn.cu:489-491`). Today `cap` is 2,051; with `cap = 262,144` and 16
   heads it is 64.5 MiB *per query row*: 516 MiB for the 8-row verify window (`verify.cpp:422`) and
   `attn_batch` times that in prefill (`prefill.cpp:888`). The dense kernel needs either a larger chunk
   per block or an online-softmax merge that does not keep a partial per 64-cell chunk.
6. **KV streaming no longer saves VRAM.** Streaming keeps the authoritative K/V in host memory and resolves
   the *selected* blocks into VRAM slots before the read (`layer.hpp:230-236`, `qsa_kv_resolve` at
   `layer.cpp:976`). With every cell selected every token, every block must be resident every token: the
   mechanism degenerates to copying the whole KV (1,440 MiB at q4_0) over PCIe per token. Section 4.2
   and section 8 draw the consequence for the plan's sizing rule.

KV formats, the Hadamard-rotated q4_0 pool, the page table and the RoPE tables carry over unchanged
(`layer.cpp:938-1010`, row 46). The prompt attention has three device paths: cp.async + mma on sm_80+,
mma on Turing, an m8n8k4 kernel on Volta, and for `cc < 70` the host returns false
(`qsa_prompt_attn.cu:1433`) so prefill falls back to the batched decode kernel
(`prefill.cpp:2222-2228`). On Pascal that fallback is the prompt attention; it needs `G = 8` and the
scratch fix above, nothing else.

### 3.4 Router and shared expert (small; one measured-not-asserted parity)

Both are generic already (3.1). Two form questions remain. The router kernel reads a BF16 weight
(`layer.cpp:361-366`, `project_bf16`), `fused_gdn_ab` reads BF16 `ssm_alpha`/`ssm_beta`
(`layer.cpp:285-287`), and `moe_shared` refuses a `ffn_gate_inp_shexp` that is not engine form BF16
(`layer.cpp:427-437`). All four are F32 in the shipped file (`gguf-metadata.md`, "What was unexpected").
The pack rounds them: exactly when the low 16 bits are zero, else only under `--compat-bf16`
(`upstream-tooling.md` section 4, the ORCA precedent). That is a numerical change against llama.cpp, which
multiplies in F32, and it lands on the one place where rounding is not a rounding error: a flipped top-8
selection is a different expert (`layer.hpp:131-139`). The oracle's real-file smoke already shows the
F32-versus-quantized router disagreeing on 6 of 32 (token, slot) choices in layer 0 and 12 to 23 in layers
30 to 39 (`oracle-crosscheck.md`, "Not covered"), so the harness measures a flip rate at the
`logits -> top-8` boundary with the quantized-mode gates Maya calibrated (selection equal at probed layers
on an F32 fixture; a bounded rate on the real file), not bitwise equality. `router_top10_parity.cpp`'s tie
fixtures (all-equal logits, a 12-way tie, the unreachable clamp) carry over at 256/8.

### 3.5 The block without hyper-connections: `layer.cpp` *and* `verify.cpp` (the largest plumbing item)

`hc` sizes the residual `R` (hc, n_embd), `inject`, `gr_rs` and the GR workspace (`layer.cpp:1127-1145`),
and every block reads four `hc_attn_*` and four `hc_ffn_*` tensors with required forms
(`layer.cpp:1231-1255`, row 22). No `hc = 1` meaning exists (row 4): the GR math on one stream is not a
plain residual, and the names Qwen3.6 ships (`attn_norm`, `post_attention_norm`, `output_norm`) are
referenced nowhere in `src/` (rows 22, 23). The plain block is

    x = rms_norm(R, attn_norm) -> mixer -> R += out
    x = rms_norm(R, post_attention_norm) -> MoE -> R += out            head: rms_norm(R, output_norm) -> output.weight

(`qwen35moe.cpp` layer loop; the oracle's `attn_norm-N`, `attn_residual-N`, `attn_post_norm-N`, `l_out-N`,
`result_norm` seams). The weighted RMS norm kernel exists (`native_qsa_rms_norm_weighted`, used at
`layer.cpp:879` and `mtp.cpp:488`). The places that must grow the plain path:

| Site | What assumes hc |
| --- | --- |
| `layer.cpp:1170-1286` `block_layer_pre`, `1319-1342` `block_layer_post` | the two GR halves, the fused-GR fast path (`fused = g_fused_gr && fused_gr_supported(2560, 4, 320)`, `:1195`; `fused_gr.cu:1164-1166`), `pending_ffn` folding |
| `layer.cpp:1088-1107` `lm_head_mix` | `output_hc_{norm,down,up}` |
| `session.cpp:141-148` | first token = `hc` copies of the embedding (one copy at hc = 1, row 26) |
| `verify.cpp:331-340` | refuses unless `fused_gr_supported && ss.k == 10 && state 128 && d_conv 4` and `layer_verify_compatible` demands the fused GR and the native indexer (`layer.cpp:476-486`, row 141) |
| `verify.cpp:356-434` | `R_`, `xn_` as `T*HC*N`, `inj_`, `lo_ (T*hc_lr)`, `idx_raw_L_`, `qidx_`, `tail_snap_` |
| `verify.cpp:1123-1125` | the head mix |
| `verify_kernels.cu:256-291` | `broadcast_streams` / `add_streams_broadcast` (runtime `hc`, row 140) |
| `mtp.cpp:503-507, 564-568, 613-625`; `prefill.cpp:972-1086` | the drafter's three hc groups (3.10) |

Because native packs decode only in verify windows (row 114) and the window refuses a layer without fused
GR and the indexer (row 141, 142), the plain block has to land in `verify.cpp` for a single token to come
out. The verify window's `ss.k == 10` refusal becomes `k == 8` from the file. `cvec` (the control-vector
apply over `hc*n_embd`, row 140) is runtime-shaped and needs no change.

### 3.6 Prefill (the most literal-heavy subsystem)

`Prefill::init` refuses unless `n_embd == 2560 && hc == 4 && hc_lr == 320 && k == 10`
(`prefill.cpp:737`, row 132); the file-wide constants are `N 2560, HC 4, D, LR 320, K 10, NE 512, C 10240,
ZV 6144, HV 48` (`prefill.cpp:90-91`) and `N, HC, D, LR, S, HK 16, HV 48, C 10240` in the kernels
(`prefill/kernels.cu:18-19`). The literals, each with its Qwen3.6 value:

| Site | Literal | Qwen3.6 |
| --- | --- | --- |
| `prefill.cpp:649-650, 883-884` | `Qf` as `T * 12288` (24 x 512), `q`/`attn` as `T * ZV`; `idx_raw`, `q_idx` | 8,192 (16 x 512), 4,096; the two indexer buffers dead |
| `prefill.cpp:681-685, 704-721, 896-907` | MoE buffers `T*K*1280`, `T*K*640`, `1280 * 2560`, `2560 * 640`; `mmq::fits(gt, 1280)`, `matrix_bytes(dt, N, 640)` | `2*n_ff` 1024, `n_ff` 512, `n_embd` 2048 |
| `prefill.cpp:854-856, 1000-1026, 1375-1380` | `grs`, `gated`, `inj`, `xq`, `ident` from `HC`, `D`, `LR` | removed with hc |
| `prefill/kernels.cu:211-215, 285-304, 345-364, 387-432, 478-570, 893-924` | GDN kernels indexed on `HV`, `HK`, `C`, launched `<<<HV, ...>>>`; `VPK = 3` (`:478`) | 32, 16, 8192; `VPK = 2` |
| `prefill/kernels.cu:616-657` | `route_kernel<REG>`: `REG 8` exists for 256 experts but the rank loop and `ids[t*10 + rank]`, `lane < 10` bake top-10 | `k` as a parameter |
| `prefill/kernels.cu:752-761` | `split_q_kernel`, `gate_attn_kernel` over `T * 24 * 256` and `h * 512` | 16 heads |
| `prefill/moe_fused_iq.cu:59` | `GU_ROWS_K = 2560, D_ROWS_K = 640` (the native-pack fused prompt MoE) | 2048, 512 |
| `prefill.cpp:1724` loop, `1303` `RP = 48` | layers from `g.n_layers` (fine, row 134); `RP` | check |

The GDN kernels are forked per geometry (template on `HV, HK, C`) rather than made runtime, because their
shared-memory staging (`sq[2][TB][S]`, `kernels.cu:482-484`) is sized from them. `src/prefill/gdn_rec_parity.cu`
mirrors the constants (row 91) and is re-pinned with them. The chunked-versus-single-token equivalence (a
prefix through prefill then decoded must agree with token-by-token decode) is the harness that already
exists for this path (`tests/cuda/prefill_*_test.cpp`, restored in `c00942d`).

### 3.7 The native pack path (one leak, otherwise runtime)

The canonical Q2_0 CPU expert kernel bakes `H 2560, FF 640, NE 512, BLOB 1,382,400`
(`include/strata/kernels/cpu/expert.hpp:34-47`, row 107) and `expert_pool_dispatch` refuses any other
width (`expert_source.cpp:1863-1868`); Qwen3.6 does not go through it. It rides the native path, as the
Coder (256 of 512 experts) does. The path is runtime per layer (`NativeFmt` with `n_embd`, `n_ff`, gate/up
and down ggml types, `native_expert.hpp:22-31`) with one leak: `expert_layout_load` builds every layer's
format with the compile-time `H, FF` (`src/kernels/cpu/expert_layout.cpp:284`, `native_fmt(gt, dt, H, FF,
f, err)`) and then checks `f.bytes == blob`, so a 2048/512 pack fails the "blob is X B but its formats make
Y" check (row 108). The fix is the loader taking `n_embd`/`n_ff` from the geometry the caller resolved,
which means the call at `generate.cpp:2008-2010` (today `g0.n_layers, g0.n_expert` from a default struct,
row 10) must run after the metadata read of 3.8, not before it.

`native_experts.txt` already carries a type per layer and per role (gate/up share one, down may differ,
`upstream-tooling.md` section 4; v3 header `(n_expert N)`, `expert_layout.cpp:271`), so the IQ3_XXS file's
37/3 down split and the Q4_K_XL's Q6_K layers need no format change. The engine reads experts as three
separate `ffn_{gate,up,down}_exps` tensors (`expert_source.cpp:2475-2500`, row 116); the shipped files are
separate, and a fused `ffn_gate_up_exps` is refused by name rather than split (section 8, item 5).

### 3.8 Metadata and `ModelGeometry`

`check_architecture` requires `general.architecture == "qwen4exp"` and reads six `qwen4exp.*` keys with
defaults 48/2560/24/2 (`include/strata/artifact/gguf_reader.hpp:576-597`); its callers are the native dense
loader, the native head and embedding, and the GGUF reader (`native_dense.cpp:103`, `native_head.cpp:29`,
`:113`, row 18). Every Qwen3.6 load dies there first. Beyond the guard, only `expert_count`,
`expert_used_count` and the rope keys are read, under `qwen4exp.` and only with `--native`
(`generate.cpp:2089-2096`, rows 8, 9). What `ModelGeometry` must read, all present in the file
(`gguf-metadata.md`, metadata table; row 153):

| Field | Key |
| --- | --- |
| `n_embd`, `n_layers` | `qwen35moe.embedding_length`, `block_count` minus `nextn_predict_layers` (0 when absent) |
| `qsa_interval` | `full_attention_interval` |
| `ssm_state_size`, `ssm_k_heads`, `ssm_v_heads`, `ssm_d_conv`; conv channels and value dim derived | `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank`, `ssm.conv_kernel`, `ssm.inner_size` |
| `n_head`, `n_head_kv`, `head_dim`, `n_rot`, sections, base, eps | `attention.head_count`, `head_count_kv`, `key_length`, `rope.dimension_count`, `rope.dimension_sections`, `rope.freq_base`, `attention.layer_norm_rms_epsilon` |
| `n_expert`, `k`, `n_ff` (routed and shared) | `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length` |
| `hc` | 1 (no key; the architecture string decides) |

The oracle's `read_hparams` (`ref/qwen36.py:125-150`) is the reference for defaults and derived values.
Other default-geometry sites: the routing trace's `layer >= 48` bounds (`generate.cpp:771, 795`, row 13),
the embedding load with `g0.n_embd` and the literal 248320 (`generate.cpp:2184-2186`, row 11), the draft
geometry (`generate.cpp:3026`, 3.10), and the standalone planner's `Geometry` and `expert_blob`
(`include/strata/plan/plan.hpp:22-52`, row 19), which is off the serving path.

### 3.9 PLE, optional at the CLI

The kernels already run without it: `block_layer_pre` applies the PLE only at layer 1 when `ple != nullptr
&& ple->ready()` (`layer.hpp:524-545`, row 30), prefill and verify guard on `ple_on` (rows 68, 69). The CLI
does not: `--native` refuses to start without `--ple-gguf` or a shard holding `per_layer_token_embd.weight`
(`generate.cpp:1800-1815`), and without `--native` the flag is "required" with `--no-ple` called a
"diagnostic ablation" (`:1843-1855`, rows 62, 63). The change is to make the absence of the tensor the
normal case when the architecture has no PLE. The fixed `NG_HIST x NG_HC_DIM` history buffer (about 360 KB,
`session.cpp:50-52`, row 29) and the `blk.1.ple_key` special cases (row 65) are dead but harmless.

### 3.10 The MTP drafter

The drafter is its own pack: `<rt_dir>/dense.txt`, `dense.bin` (every dense weight Q8_0), `experts.bin`
written by `tools/mtp_rt.py` (`mtp.cpp:150-216`, row 70). It expects vLLM-style names
(`fc_embedding`, `fc_hidden`, `pre_fc_norm_*`, `self_attn.*`, `mlp.shared_expert.*`, `mlp.gate`,
`mlp.shared_expert_gate`, three `*hyper_connection*` groups; `mtp.cpp:210-212, 488-495, 503-507, 575`),
canonical Q2_0 experts of `n_expert x BLOB` bytes in VRAM through `moe_group_resident` /
`moe_grouped_s2` (`mtp.cpp:191, 579-583`; `s2_expert_grouped.cu:37-46` bakes 2560/640, row 121), and the
default geometry (`generate.cpp:3026`, "the canonical model's MTP head (512 experts)", row 12).

Qwen3.6's draft block (`gguf-metadata.md`, "The nextn block"): `concat(rms_norm(emb) * enorm, rms_norm(h)
* hnorm)` (embedding first) through one `eh_proj [4096, 2048]`, then a plain attention + MoE block
(`attn_norm`, joint q+gate, q/k norm, IMRoPE, causal over its own KV, sigmoid gate, `attn_output`,
residual, `post_attention_norm`, the same top-8 MoE plus gated shared expert, residual), `shared_head_norm`,
the main `output.weight`. Pairing: token k with the trunk's output-normed hidden state of k-1
(`oracle-crosscheck.md`, "Nextn / MTP"; verified at every nextn seam, 50 of 50). What the drafter keeps:
dense attention over its own KV ring (rows 76), the main head and the `draft_vocab.bin` subset (row 77,
the same 248,320 vocabulary, so `data/draft_vocab*.bin` carry over), `token_embd` embeds (row 78). What
changes: the two input projections become one `eh_proj` over the concatenation (`mtp.cpp:488-495`); the
three hc groups become three norms (`:503-507, 564-568, 613-625`; the prefill-side pass at
`prefill.cpp:972-1086` the same); the draft experts are 256 K-quant blobs (Q5_K/Q4_K, 464 MiB in the
Q4_K_XL file; Q3_K/Q4_K, 364 MiB in the IQ4_XS file) served by the native GPU expert kernels
(`iq_kernels.cu`, row 120) instead of the Q2_0 grouped kernel; the geometry follows the target.

A Qwen3.6 draft pack: `dense.txt`/`dense.bin` with llama.cpp's names (`nextn.eh_proj`, `nextn.enorm`,
`nextn.hnorm`, `nextn.shared_head_norm`, `attn_norm`, `post_attention_norm`, `attn_{q,k,v,output}`,
`attn_{q,k}_norm`, `ffn_gate_inp` BF16, `ffn_gate_inp_shexp` BF16, `ffn_*_shexp`), the projections kept
Q8_0 as stored (39 MiB), and a `native_experts.txt`-style line for the one layer's three expert tensors.
Because the drafter is separate from the trunk file, the pack cut from the Q4_K_XL file's `blk.40` serves
the IQ3_XXS trunk too (the plain IQ3 file has no nextn block). Draft acceptance is an empirical question per
model; llama.cpp's own MTP arm (B4) is the external reference.

### 3.11 Tokenizer, chat template, vision

Same vocabulary and the same `qwen35` pre-tokenizer family (`gguf-metadata.md`; `tools/strata_tokenizer.py`
refuses schemes it has not transcribed, `upstream-tooling.md` section 2). The front end uses llama.cpp's
chat parser (`frontend/chat_parser_test.cpp`) and carries the Qwen3-Coder any-argument-order patch
(`chat_parser_test.cpp:48`); the plan's step 5 checks it against Qwen3.6's template with
`make test-chat-parser MODEL=<gguf>` (`Makefile:35-41`). Vision loads an mmproj through `mtmd`
(`frontend/vision.hpp:79-87`); the `Qwen3.6-35B-A3B-MTP-GGUF-mmproj-F16.gguf` is on the VM with its SHA-256
(`phase0-notes.md`). Nothing in the engine changes for either.

## 4. The memory plan

### 4.1 The fixed set, from measurements

| Item | Q4_K_XL, 262,144 | IQ3_XXS, 262,144 | Source |
| --- | ---: | ---: | --- |
| Non-expert weights on the GPU | 2,038 MiB | 1,597 MiB | `-lv 4` minus resident expert layers (section 1); census |
| Draft block: 40 MiB dense + 256 experts in VRAM + ring KV + 8 x 248,320 logits | ~520 MiB | ~520 MiB (the Q4 draft pack) | census; `mtp.hpp:19`; `mtp.cpp:430-436` |
| KV, q4_0, 10 layers | 1,440 MiB | 1,440 MiB | `phase0-notes.md` smoke rows (both arms) |
| Session: RoPE table 64 MiB, scratch; **plus 320 MiB of indexer state unless 3.3 item 4 removes it** | ~164 MiB | ~164 MiB | `layer.cpp:559-576` |
| CUDA context | ~230 to 300 MiB | same | nvidia-smi 9,487 (peak 9,565) minus the three buffers, Q4 row |
| Reserve | 300 MiB (`kSmallReserveMib`) to 700 (default) | same | `generate.cpp:424, 3249` |
| Prompt buffers: borrowed from the cache with a profile, else chunk-sized | 0 (borrow) / 1,904 MiB at chunk 2048 | 0 / 1,878 MiB | `generate.cpp:2723`; llama.cpp `-ub 2048` compute buffer |
| **Decode-time fixed set** | **~4,760 MiB** borrowing; ~6,660 without | **~4,320** borrowing; ~6,200 without | sum |
| Trunk experts (file) | 18,726 MiB; 1.829 MiB mean, 2.195 max | 10,594 MiB; 1.035 mean, 1.172 max | census |
| Dense split-K attention scratch at `cap = 262,144` | 64.5 MiB per query row (516 for the verify window) | same | `qsa_decode_attn.cu:489-491` |

The replay and the plan used a 5.8 GB (Q4) / 4.6 GB (IQ3) fixed set that includes the 1.9 GB prompt buffer
and excludes the drafter, and 17.4 GB / 10.5 GB of experts (`phase0-replay.md`, "Budget to slots"). The
census moves the expert total up by 7% at Q4 and the fixed set down by the prompt buffer when prefill
borrows; the two roughly cancel at the 11 GB rung (table 4.4).

### 4.2 The VRAM planner (Phase 2 step 2)

What exists: `--expert-cache auto` reads free VRAM, subtracts the reserve and the prompt estimate, grants
slots at `max_blob` (`generate.cpp:3232-3300`), then for a native pack with a profile re-prices the slots
per layer so the same VRAM holds the profile's hottest experts at their own blob sizes
(`generate.cpp:3307-3327`, "~30% more IQ3_XXS experts than slots of the largest blob"); the prompt path
borrows the last slots (`:4404-4425`); below `min_slots` the reserve shrinks to 300 MiB and then the start
refuses with the numbers (`:3267-3290`, "no VRAM is left for the expert cache"). What the step adds, in the
plan's order with one change:

1. Fixed set first: weights, drafter, KV for the requested context, session. **KV streaming is not a lever
   for this model** (3.3 item 6): the resident-window floor of 20,480 cells (`layer.cpp:556`) exists for a
   selection of 2,051 cells, and dense attention reads every cell. So the rule's first fallback, "stream
   the KV to RAM above a resident window", is replaced by: hold the KV in VRAM, and if the fixed set does
   not fit, lower the context toward 131,072, then refuse with the numbers and the IQ3 file's name. Never
   silently shorten.
2. Every remaining byte becomes expert slots, sized per layer from the profile as today.
3. The RAM tier: free RAM minus a headroom (`--resident-experts` uses 4 GiB, `generate.cpp:1508`;
   `--resident-budget-gib N` sets it, `:1516-1520`), filled with the next experts by profile rank
   (`pin_cache_complement`, `expert_source.cpp:1331`); whatever remains stays mapped from the GGUF
   (`--mmap-experts`, `:618-680`) with the `MADV_WILLNEED` warm on predicted ids (`:1129-1160`).
4. Emulation: `--vram-cap MiB` clamps every `cudaMemGetInfo` free figure the planner reads (`:3297, 3313`)
   to `cap - used` and, so the driver sees the same pressure the real card would, allocates a blocker of
   `total - cap` at start; `--ram-cap GiB` bounds `resident_budget` and the pinned arena. The page cache
   cannot be capped from inside the process; the 12 GiB VM does that physically, and on a larger host a
   cgroup `memory.max` does.
5. The start-up log and `/health` state the three counts (VRAM slots, RAM copies, mapped) and per-token
   counters: VRAM hits (`cache_hits`), RAM-copy hits, warm-hint count (`warm_count_`), and disk reads as
   the major-fault delta from `/proc/self/stat`, since a page-cache miss is not otherwise observable.

### 4.3 The 6-to-32 GB ladder

Budgets in the replay's decimal convention (6 GB = 5,722 MiB) except the 1080 Ti (11,264 MiB) and a 24 GB
card (24,576 MiB). Slots at the census mean blob; "borrow" is the decode-time fixed set of 4.1, "no borrow"
adds the chunk-2048 prompt buffer.

| Budget | Q4 replay basis (5.8 GB fixed, 1.70 MB/expert) | Q4 derived, no borrow | Q4 derived, borrow | IQ3 replay basis | IQ3 derived, borrow |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 GB | 80 (1%) | 0 (fixed set does not fit) | ~525 (5%) | 1,360 (13%) | ~1,350 (13%) |
| 8 GB | 1,280 (12%) | ~525 (5%) | ~1,570 (15%) | 3,280 (32%) | ~3,200 (31%) |
| 11 GB (decimal) | 3,040 (30%) | ~2,090 (20%) | ~3,130 (31%) | 6,240 (61%) | ~5,960 (58%) |
| 1080 Ti, 11,264 MiB | | ~2,510 (25%) | ~3,550 (35%) | | ~6,700 (66%) |
| 12 GB | 3,640 (36%) | ~2,610 (26%) | ~3,650 (36%) | 7,200 (70%) | ~6,880 (67%) |
| 16 GB | 6,000 (59%) | ~4,700 (46%) | ~5,740 (56%) | all | all |
| 24 GB (24,576 MiB) | all | ~9,790 (96%) | all, ~1,090 MiB spare | all | all |
| 32 GB | all | all | all | all | all |

The RAM tier at the replay's 9.5 GB usable: Q4 ~4,950 experts at the census blob (the replay's 5,590 was at
1.70 MB), IQ3 ~8,760. The disk tier at Q4 on the 1080 Ti is then 10,240 - 3,550 - 4,950 = ~1,740 experts
(17%) with borrowing and ~2,780 (27%) without; the replay modelled 1,610 (16%). At IQ3 everything fits VRAM
plus RAM at every rung from 6 GB (`phase0-replay.md`, finding 5). The planner's own log at `--vram-cap`
6/8/11/12/16/24/32 is the check of this table, within one slot of the derivation.

## 5. Expected envelope per budget

From `phase0-replay.md` (LFU, profile seed, second half of a 214,940-token trace of the IQ3_XXS model,
decode only; the Q4 arm reuses the IQ3 routes). The parameters are parameters: `gpu_ms` 17 (an upper
estimate from the smoke rows' busy share; 12 is a flag away), `cpu_lane_ms` 0.15 per expert (llama.cpp
IQ3 N=18 implies 0.17), `disk_ms` 1.3 (fio 1 MiB random QD4; parallel faults may halve it), `pcie_ms` 0
(promotions counted, not charged: 47.7 copies per token at Q4 11 GB, about 12 ms at 0.26 ms each, is the
largest omission; charging it takes Q4 11 GB LFU from 42.5 to 27.8 tok/s). "Overlap" is
`1000 / max(gpu, cpu, disk)`, "serial" the sum.

| Budget | Q4 VRAM hit | Q4 disk reads/tok | Q4 tok/s overlap / serial | IQ3 VRAM hit | IQ3 disk/tok | IQ3 tok/s overlap / serial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 GB | 0.0% | 70.7 | 7.7 / 6.8 | 59.7% | 0 | 51.7 / 27.5 |
| 8 GB | 57.9% | 51.5 | 12.6 / 10.4 | 84.4% | 0 | 58.8 / 40.9 |
| 11 GB | 82.6% | 13.2 | 42.5 / 24.7 | 96.0% | 0 | 58.8 / 52.9 |
| 12 GB | 86.8% | 7.1 | 58.8 / 31.8 | 97.7% | 0 | 58.8 / 55.2 |
| 16 GB | 95.5% | 0 | 58.8 / 52.2 | 100% | 0 | 58.8 / 58.8 |
| 24, 32 GB | 100% | 0 | 58.8 / 58.8 | 100% | 0 | 58.8 / 58.8 |

58.8 is `1000 / gpu_ms` and the overlap column saturates there by construction. The Belady ceiling at Q4
11 GB is 91.0% hit, 6.6 disk reads, 58.8 / 34.6. Against it, the llama.cpp smoke points on the same card:
Q4 N=32 **8.78 tok/s** decode, 128.6 prefill, GPU busy 15%; IQ3 N=18 **24.58** decode, 433.9 prefill, GPU
busy 41% (`phase0-notes.md`). The plan's floors (A2: 15/10 tok/s at 8K/32K on 11 GB, 10/7 on 6 GB, 60/45 on
24 GB; A3: 300 tok/s prefill on Pascal) sit inside these predictions except the Q4 6 GB row, which the
plan already routes to the IQ3 file. With the census tiers (4.3) the Q4 11 GB disk term is 17 to 27% of
experts rather than 16%, so its disk reads per token sit above 13.2; the replay is re-run with the census
blob sizes as part of step 2, not re-derived here. Prefill is not modelled; the engine's chunked MMQ path
amortises expert bytes across the chunk, and on Pascal the prompt MoE's fast path is a risk (section 7).

## 6. Execution plan (Phase 2, steps 2 to 7)

Step 1, the oracle and fixture, is done on `qwen36-oracle`: `ref/qwen36.py` agrees with the pinned
llama.cpp graph on 2,155 comparisons over 116 seam families at 1.1e-4 worst absolute error, the router's
integers identical, and all seven rival readings fail at the seam that makes them (`oracle-crosscheck.md`).
Every harness below holds the engine to those dumps by llama.cpp's `cb()` names. Each step has a parity
harness in the repo's negative-test discipline (`gdn_parity.cpp`: every wrong reading must differ by more
than 5% relative L1 before the right one is judged, `upstream-tooling.md` section 3), runs on the 1080 Ti
under `STRATA_EXPERIMENTAL_SM60=ON` (sm_61) from the first build and on a rented sm_86 card for compile
speed; a kernel passes when it passes on both. Steps 2 to 4 need seconds of GPU per run and proceed while
the Phase 0 sweeps hold the card. Effort weights sum to 100 for steps 2 to 7.

**Step 2, geometry, metadata, sizing (15%).** Files: `gguf_reader.hpp:576-597` (accept `qwen35moe`, read
the 3.8 key set), `layout.hpp:27-59` (`hc = 1`, derived conv channels and value dim, no indexer fields),
`generate.cpp:2008-2010, 2073-2096, 2184-2186, 771, 795` (geometry from the file before the layout load;
the embedding dims from it), `expert_layout.cpp:284` (the leak), `expert_source.cpp:2475-2500` (fused
forms refused by name), `generate.cpp:1800-1815, 1843-1855` (PLE optional), the planner
(`generate.cpp:3232-3327`) with the 4.2 rule, `--vram-cap`, `--ram-cap`, the counters, `tools/make_profile.py`
(`N_LAYER = 48`). Harness: `ModelGeometry` from the synthetic GGUF (`tools/qwen36_synth_gguf.py`) and the
real file equals the oracle's `HParams` field by field; the planner's log at the seven caps reproduces
table 4.3. Negative tests: a `qwen4exp` file refused with its name; a 48 x 512 profile refused
(`expert_cache.cpp:43`); a `--fused-gate-up` fixture refused by tensor name; `--vram-cap` below the fixed
set refuses with the context it would need. No GPU beyond the planner's run.

**Step 3, kernels and the block (45%).** In order of risk:

- *GDN (3.2):* SiLU in the four kernels; `gdn_parity.cpp` at `h_v 32` with sigmoid as the negative, plus the
  existing four traps (pairing, decay order, conv direction, l2 epsilon); `gdn_rec_parity.cu` at `VPK 2`.
  Seams: `conv_output_silu-0`, `new_state-0`, `final_output-0`.
- *Dense attention (3.3):* the identity-`ids` producer; `G` templated at 8; the indexer skipped in
  `qsa_layer` and the verify attention step; the state without indexer buffers; `cap = max_cells`; the
  split-K scratch bounded. Harness: `qsa_parity.cpp`'s attention cases with `ids` identity against the
  oracle's `kq_soft_max-3`, `kqv-3`, `attn_gated-3`, and a 300-cell case where the old width (2,051) and
  the dense width differ. Negative tests: `G = 12` on 16 heads (refused, not mis-indexed), the gate skipped
  (1.03 relative L1 at `attn_gated-3`), adjacent-pair RoPE (0.20 at `kqv-3`), both from the oracle's
  controls.
- *Router and shared expert (3.4):* `router_top10_parity.cpp` at 256/8 with its tie fixtures; the BF16
  rounding of the F32 router measured as a flip rate on the real file's first 8 tokens with the quantized-
  mode gates; `shared_expert_parity.cpp` at 512 with the 1-D F32-to-BF16 gate. Seams: `ffn_moe_topk-N`,
  `ffn_moe_weights_norm-N`, `ffn_shexp_gated-N`.
- *The block without hyper-connections (3.5):* `layer.cpp` and `verify.cpp` both, `session.cpp:141-148`,
  the head. Harness: the layer composition test, Maya's shape: layer 0 (GDN + MoE) and layer 3 (attention +
  MoE) of the synthetic model through the engine's kernels, every seam held to the dumps; then the verify
  window on an 8-token sequence against the oracle's `l_out-N` per position, with the window's row
  arithmetic required to equal the single-token path (the discipline `verify.cpp:773-774` already states).
  Negative tests: the GR read with identity weights (not a plain norm), the post-attention norm skipped,
  `k = 10` on an 8-expert file.
- *Prefill (3.6):* the constants and the eight literal sites; the GDN kernels forked at 32/16/8192; the
  route kernel at `k`; the split-q and gate kernels at 16 heads; `moe_fused_iq.cu` at 2048/512. Harness:
  the restored `tests/cuda/prefill_*_test.cpp` plus chunked-prefill-then-decode against token-by-token
  decode on a 40-token sequence (the oracle's `base_c7` case: chunks of 7). Negative tests: `VPK 3` on 32
  heads, top-10 routing on 8.

**Step 4, pack, drafter, runner (20%).** `tools/iq_pack.py`: a `qwen35moe` entry in `FORM`/`ROUTERS`
(`ssm_alpha`, `ssm_beta`, `ffn_gate_inp`, `ffn_gate_inp_shexp` F32 to BF16 under `--compat-bf16` with
`compat-bf16.json`; norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d` F32 copied; projections served natively),
`native_experts.txt` v3 per layer; `tools/test_iq_pack.py`'s `write_gguf()` fixtures at the new
architecture. The draft pack of 3.10 from the MTP file's `blk.40` (a new `tools/mtp_qwen36_pack.py`
replacing `mtp_pack.py`/`mtp_rt.py`), `mtp.cpp` and `prefill.cpp:972-1086` on the plain graph, the draft
experts through the native GPU kernels, `generate.cpp:3026` following the target. `make_profile.py` from
the E0 trace (40 x 256). Harness: the drafter's one-row forward against the oracle's `--mtp` dumps
(`mtp_eh_proj`, `mtp_attn_gated`, `mtp_ffn_out`, `result_output`; 50 nextn seams); the streaming runner
held to the oracle at every position of the 11- and 40-token cases, autoregressive and chunked
(`base_ar`, `base_c7`); then the real Q4 file, greedy, against llama.cpp's output distribution (top-1
agreement and KL per teacher-forced position, Maya's quantized-mode gates: no monotone positional drift,
selection equal at probed layers). Negative test: a 4 KB weight corruption must move the logits by more
than 1e-2 (the mutation-sensitivity gate).

**Step 5, serve and measure (10%).** The 1080 Ti at 11 GB, then `--vram-cap` 8 and 6; the other rungs when
a card frees up. `tools/calibrate.py` for `--pcie-frac`, `--spec-min-p`, `--pool-workers`; regina's harness
and the three instruments (`nvidia-smi` 100 ms, `nvprof` on Pascal, `perf stat`); the engine's points on
`ladder-baseline.png`; H5. `make test-chat-parser MODEL=<gguf>` against the template; the any-argument-order
patch checked for misfires; images through `--mmproj`. Acceptance A1 to A5 per rung.

**Step 6, the Pascal cards and the fat binary (5%).** `CMakeLists.txt:138-176`: `STRATA_EXPERIMENTAL_SM60`
default ON, `CMAKE_CUDA_ARCHITECTURES=61;70;75;86;89;120`, CUDA 12.8 (the sm_120-with-12.8 warning at
`:166-169` stays; RTX 50 is built, not measured). `cuobjdump --list-elf | grep sm_` after every build. The
1070 Ti and the 1060 against B1b; the sm_75+ paths checked on the 4090 or A6000 when one is free.

**Step 7, README (5%).** The supported model and checksums, the ladder against the baseline, the sizing
rule in words, the Pascal caveats.

The critical path runs through step 3's block-and-verify rewrite and its composition harness; the dense
attention kernel is the one new kernel, the rest is parameterisation and removal.

## 7. Risks

1. **Nobody has run this engine on Pascal.** The build refuses `cc < 75` without the experimental flag;
   the flag is "community-tested, not in the ready-made engine" for sm_60/sm_70 (`CMakeLists.txt:150-160`),
   and Maya requires Volta. The 1080 Ti is sm_61, so `__dp4a` is native and only the `__nanosleep` shim
   applies (`dp4a.hpp:27-40`).
2. **FP16 on GP102/GP104/GP106 runs at 1/64 rate.** 21 kernel files touch half types. Conversions
   (`__half2float`, the `hf()` stores in `prefill/kernels.cu:614, 761`, the fp16 K/V reads in
   `qsa_decode_attn.cu`) are full rate; half *arithmetic* is not. The audit must separate the two in
   `native_mmvq.cu`, `iq_kernels.cu`, `s_gemv.cu`, `shared_expert.cu` and the prefill kernels. The prompt
   MoE is the exposed case: the MMQ (dp4a) path is gated by `mmq::built()`/`mmq::fits`
   (`prefill.cpp:676-684`) and its tests exit 77 below sm_80 (`CMakeLists.txt:1076`); the fallback is the
   FP16 GEMM path over dequantized fp16 weights (`prefill.cpp:907`), which is exactly the slow case. The
   A3 floor of 300 tok/s prefill on Pascal depends on the dp4a MMQ running on sm_61, where llama.cpp's own
   MMQ does run.
3. **Prompt attention on Pascal is the decode kernel.** `qsa_prompt_attn_batch` returns false for
   `cc < 70` (`qsa_prompt_attn.cu:1433`) and prefill falls back to `qsa_decode_attn_batch` over the chunk
   (`prefill.cpp:2222-2228`): correct, but a 2,048-query chunk through a decode-shaped kernel is a
   throughput risk, and its split-K scratch is the 64.5 MiB-per-row item of 3.3.
4. **Memory items the plan did not count:** the draft block (~520 MiB), the dead indexer state (320 MiB
   at 256K unless removed), the dense scratch, and the uniform-slot grant before the per-layer re-pricing
   (`generate.cpp:3307-3327`), which is 17% of capacity at Q4 (2.195 versus 1.829 MiB) if the profile is
   absent and the re-pricing does not run.
5. **KV streaming does not apply** (3.3 item 6): a 6 GB card at 256K holds the KV in VRAM or lowers the
   context; there is no third option in this engine without a CPU attention path.
6. **Numerical:** BF16 rounding of the F32 router and GDN gates (3.4); the router's flip rate is measured
   on the real file, and the acceptance gates are top-1 and KL, not per-logit tolerances, as Maya's
   cross-check found (two flips in 512 explained by 1e-5 selection gaps).
7. **Verify-only decode.** Every token of a native pack goes through `verify.cpp` with `--spec T >= 2`
   (`generate.cpp:2178`); the plain IQ3 file has no draft block, so its drafter is the Q4 file's `blk.40`
   (3.10). Draft acceptance on Qwen3.6 is unmeasured; the prompt-lookup drafter alone leaves short windows.
8. **The replay's model:** `pcie_ms` 0, one IQ3 trace reused for Q4, a cache that never resets, no
   prefill, `gpu_ms` an upper estimate (`phase0-replay.md`, "Limits"); the census blob sizes shift the Q4
   disk tier up (4.3). The engine's counters, not the replay, decide H4 and H6.
9. **The host.** VM 211 has 8 vCPU of a Zen 1 shared with VM 220 and AVX2 only; the native expert path then
   runs ggml-cpu `vec_dot` for the i-quants and the AVX-2 kernels for the rest (`generate.cpp:2068-2072`).
   The 0.15 ms CPU lane is the replay's assumption; `perf stat` measures it.
10. **Upstream tooling gaps:** no published Python oracle for the Flash-Next kernels, so the parity files'
    C++ references are the only pins of the existing semantics (`upstream-tooling.md` section 1); the
    `bench/micro` llama.cpp-oracle tests cannot be restored.

## 8. Disagreements with the plan

1. **Section 2, the sizing rule's first fallback ("stream the KV to RAM above a resident window").** Not
   available for dense attention (3.3 item 6): streaming resolves the selected cells, and every cell is
   selected. The rule becomes KV in VRAM, then lower the context, then refuse. H3's "6 GB needs the KV
   streamed" is replaced by: 6 GB at Q4 holds the fixed set at 256K only if prefill borrows from the cache
   (~525 slots, table 4.3), else it takes 131,072 or the IQ3 file.
2. **Section 2, the fixed set.** Non-expert weights are 2,038 MiB at Q4 and 1,597 at IQ3, not ~3.1 / ~1.9
   GB; the prompt buffer is 1.9 GB at chunk 2048, not 0.8; the drafter (~520 MiB) is missing from the
   table. The plan says Phase 0 numbers replace the estimates; this study does so and adds the drafter.
3. **Section 2, the expert totals.** Q4 trunk experts are 18,726 MiB (19.6 GB), 1.829 MiB mean and 2.195
   max per expert, not 17.4 GB / 1.7 MiB; IQ3 agrees (10,594 MiB). The 11 GB rung's tiers at Q4 shift to
   ~3,550 VRAM / ~4,950 RAM / ~1,740 disk with borrowing, and the replay's 13.2 disk reads per token is a
   lower bound until it is re-run at the census sizes.
4. **Section 5 step 3, "`native_flash_attn.cu` is the fallback for prompts."** It is a diagnostic adapter
   for contexts up to 256 with 24 heads hard-coded (`native_flash_attn.cu:196-206`, `layer.hpp:53`). The
   prompt path is `qsa_prompt_attn.cu`, and on Pascal it is the batched decode kernel (risk 3). The dense
   prompt attention comes from parameterising those two, not from `native_flash_attn.cu`.
5. **Section 5 step 2, "accept llama.cpp's fused `ffn_gate_up_exps` and `attn_qkv` forms or split them at
   pack time."** No shipped Qwen3.6 file uses either (`gguf-metadata.md`). The study scopes this to a
   named refusal in `check_experts_gguf` and `native_dense.cpp:31-33`; the oracle's fused cases stay as
   evidence of the layouts should a file appear.
6. **Section 5 step 2, "per-layer prefetch hints as Maya's section 7 added."** This tree already has them:
   `FileExpertSource::warm` issues `MADV_WILLNEED` per role (`PrefetchVirtualMemory` on Windows) for the
   predicted ids (`expert_source.cpp:1129-1160`, `1264`). Nothing is ported from Maya; step 2 verifies the
   hint fires on the Qwen3.6 pack's offsets and counts it.
7. **Section 2, "which file" (Q4_K_M-MTP, or Q4_K_XL plus a separately packed head).** The Q4_K_XL from the
   MTP repository (rev 5bc3e238, SHA in `phase0-notes.md`) already carries `blk.40` (753 tensors). It is
   the default file, and its `blk.40` is the draft pack for both arms (3.10).
8. **Section 1, "every kernel path on these cards must be FP32 or dp4a int8."** Agreed for arithmetic;
   fp16 *storage* with conversion (the KV pools, the fp16 stores in prefill) is full rate on Pascal and
   stays. The prompt MoE's dp4a MMQ must be enabled on sm_61 (risk 2), which the plan does not list.

Nothing else in Sections 0, 2 and 5 is contradicted: the target, the three tiers on the primary rung, the
Q4 default with the IQ3 arm, the Pascal-first build, the step order and the acceptance thresholds stand.
