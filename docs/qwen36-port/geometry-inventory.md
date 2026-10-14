# Flash-Next geometry and extras baked into the Strata engine (6f32ec0, engine 0.1.39)

Scope: `src/` and `include/`. Target: Qwen3.6-35B-A3B (hidden 2048, 40 layers, GDN 16 k-heads / 32 v-heads,
16 attention heads / 2 KV heads of 256, dense attention, 256 experts top-8 of 512 plus a shared 512, no
hyper-connections, no PLE, MTP 1 layer, vocabulary 248,320). No fixes are proposed here.

Derived Qwen3.6 numbers used below: GDN conv channels 8192 (2*2048 + 4096), GDN value dim 4096, attention
`q_full` width 8192 (16 * 2 * 256), attention output width 4096, GQA group 8 (16 / 2), 40 layers = 30 GDN + 10 attention.

Classes:

- **RT**: already a runtime parameter (`ModelGeometry`, pack metadata, tensor shape). Fine as is.
- **CT**: a compile-time constant that disagrees with Qwen3.6. Must change.
- **CT-ok**: a compile-time constant that is an upper bound or a value Qwen3.6 shares. Needs no change.
- **ST**: a structural assumption. Needs a code path, not a number.

Line numbers are from the current tree. Where a path only runs for the canonical Q2_0 pack (not the `--native` GGUF
pack that Qwen3.6 would use), the note says "canonical only".

## A. The central finding: `ModelGeometry` is a default, and only two fields are ever read from the model file

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 1 | include/strata/core/layout.hpp:28-30 | `n_embd = 2560`, `n_layers = 48`, `qsa_interval = 4` | CT | Struct defaults. Nothing reads `n_embd`/`n_layers` from the file. Interval 4 already matches Qwen3.6. |
| 2 | include/strata/core/layout.hpp:33-38 | `ssm_k_heads 16`, `ssm_v_heads 48`, `ssm_conv_channels 10240`, `ssm_value_dim 6144` | CT | Conv channels and value dim are stored members, not derived from heads. Changing `ssm_v_heads` alone leaves them wrong. Heads 16 and state 128 and d_conv 4 already match. |
| 3 | include/strata/core/layout.hpp:41-45 | `n_head 24`, `idx_q_heads 4`, `idx_key_dim 128` | CT | `n_head` must be 16. `n_head_kv 2` and `head_dim 256` match. Indexer fields are meaningless for Qwen3.6. |
| 4 | include/strata/core/layout.hpp:48-49 | `hc = 4`, `hc_lr = 320` | CT / ST | No "hc = 1" meaning exists. `hc` sizes `R`, `inject`, every GR buffer. See section B. |
| 5 | include/strata/core/layout.hpp:52-53 | `n_expert = 512`, `n_ff = 640` | CT | One `n_ff` serves routed and shared experts. Qwen3.6 has 512 for both, so one field still works. |
| 6 | include/strata/core/layout.hpp:55-58 | `hc_dim() = hc*n_embd`, `n_qsa_layers = n_layers/interval`, `n_gdn_layers` | RT | Derived correctly from the fields. |
| 7 | src/program/generate.cpp:2073-2074 | `ModelGeometry g;` with the comment "canonical defaults", and `K = 10` | CT | `g` and `K` start from Flash-Next defaults. |
| 8 | src/program/generate.cpp:2089-2092 | Only `qwen4exp.expert_count` -> `g.n_expert` and `qwen4exp.expert_used_count` -> `K` are read, and only when `--native` is given | RT (partial) | The only model-file overrides of geometry. `block_count`, `embedding_length`, head counts, `ssm.*`, `expert_feed_forward_length` are never read. |
| 9 | src/program/generate.cpp:2093-2096 | Rope keys read under the `qwen4exp.` prefix | CT | Key prefix must become `qwen35moe.`. Values (base 1e7) happen to match. |
| 10 | src/program/generate.cpp:2008-2010 | `const ModelGeometry g0;` passed to `expert_layout_load(o.pack, g0.n_layers, g0.n_expert)` | CT | Runs before the line-2091 override, so it sees 48 layers and 512 experts. The `native_experts.txt` header can still correct `n_expert` (expert_layout.cpp:271). |
| 11 | src/program/generate.cpp:2184-2186 | `NativeEmbed::load(..., g0.n_embd, 248320, ...)` | CT | Default `n_embd` 2560 would reject a 2048-wide `token_embd.weight`. The 248320 literal equals Qwen3.6's vocabulary. |
| 12 | src/program/generate.cpp:3026 | `static const ModelGeometry draft_geometry{}` handed to `mtp.load` | CT | The comment says the draft layer is "the canonical model's MTP head (512 experts)". Draft geometry does not follow the target. |
| 13 | src/program/generate.cpp:771, 795 | Routing trace rejects `layer >= 48` | CT | Literal 48 in two bounds checks. |
| 14 | src/program/generate.cpp:5769 | `kImagePad = 248056` (comment: `qwen4exp.ple.image_token_id`) | CT-ok | Only feeds the PLE hash for image cells. Dead without PLE. |
| 15 | include/strata/core/session.hpp:48 | `SessionState::k = 10` | CT | Experts per token default. Overwritten from `K` at session init. |
| 16 | include/strata/core/expert_source.hpp:194 | `int k_ = 10` | CT-ok | Default only; set per layer at runtime. |
| 17 | include/strata/artifact/gguf_reader.hpp:576-578 | `Qwen4ExpGuard{block_count 48, hidden 2560, head_count 24, head_count_kv 2}` | CT | Struct defaults for the architecture guard. |
| 18 | include/strata/artifact/gguf_reader.hpp:581-597 | `check_architecture` requires `general.architecture == "qwen4exp"` and reads `qwen4exp.block_count`, `.embedding_length`, `.expert_count`, `.expert_used_count`, `.attention.head_count`, `.attention.head_count_kv` | CT / ST | Callers: native_dense.cpp:103, native_head.cpp:29 and :113, gguf_reader.cpp:57. Every Qwen3.6 load is refused here first. |
| 19 | include/strata/plan/plan.hpp:22-42, 49-52, 83, 133 | Standalone planner `Geometry` (48 layers, 12 QSA, 48 v-heads, 10240 channels, indexer cache term), `Costs::expert_blob = 1382400`, `vram_pool_bytes() = 5,943,000,000`, "slots of 1,382,400 B" | CT | Used only by src/plan/plan_main.cpp and src/core/device_main.cpp. Not on the serving path. Has no streaming-KV or RAM-tier model. |
| 20 | src/core/graph.cpp (whole file) | None | RT | No geometry. Graphs are keyed by `(LayerType, n_tokens)`. |

## B. Layer composition: `layer.hpp`, `layer.cpp`, `session.cpp`

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 21 | src/core/layer.cpp:1170-1286 (`block_layer_pre`), 1319-1342 (`block_layer_post`) | Every layer is `gr_read(attn) -> mixer -> gr_write(attn) -> gr_read(ffn) -> router ... moe_finish -> gr_write(ffn)` | ST | The gated-residual read/write wraps both mixer and FFN on every layer. No plain "rms_norm -> mixer -> residual add" branch exists. |
| 22 | src/core/layer.cpp:1231-1255 | Four GR tensors per half, prefixes `hc_attn_` and `hc_ffn_` (`norm`, `down`, `up`, `inject`), required engine forms (norm F32, other three BF16) | ST | Qwen3.6 has `attn_norm` and `post_attention_norm` (plain RMS norms) instead. Neither name is referenced anywhere in src/. |
| 23 | src/core/layer.cpp:1088-1107 (`lm_head_mix`) | Final mix is `gr_read` with `output_hc_norm/down/up` | ST | Qwen3.6 needs `output_norm.weight` (plain RMS norm). That name is not referenced anywhere in src/. Same in verify.cpp:1123-1125. |
| 24 | src/core/layer.cpp:1127-1145 (`block_buffers_bytes/init`) | `R` is `(hc, n_embd)`; `inject`, `inject2`, `gr_rs`, `gr_workspace_bytes(GrShapes{n_embd, hc, hc_lr})` | RT | Sized from `g.hc`. Works for hc = 1 in size, but the GR math is meaningless there. |
| 25 | src/core/layer.cpp:1146-1158 (`dump_stride_floats`) | C1-oracle dump layout is `2*n_embd + 2*hc + n_head*head_dim + ...` | RT | Debug only. |
| 26 | src/core/session.cpp:141-148 | The first token's residual is `hc` copies of the embedding | RT / ST | Loops `g.hc` times. With hc = 1 it is one copy. |
| 27 | src/core/session.cpp:1, 228, 331, 424 etc. | Comments say "48 layers", but loops use `g.n_layers` | RT | Verified: loops at 55, 77, 194-205, 283, 301, 331, 391, 424, 662, 788, 856, 921 all use `g.n_layers`. |
| 28 | src/core/session.cpp:41-44 | `gdn_state_floats = S*v_heads*S + conv_channels*(d_conv-1)` | RT | From `ModelGeometry`. Correct once row 2 is fixed. |
| 29 | src/core/session.cpp:50-52, 71, 124-126, 159-162 | `ple_hist_bytes() = NG_HIST * NG_HC_DIM * 4` (9 x 10240 floats) is always carved and zeroed | CT-ok | Fixed ~360 KB per session, allocated even with no PLE. Harmless but dead. |
| 30 | include/strata/core/layer.hpp:524-587 and src/core/layer.cpp:1192-1229 | PLE runs only at `layer == 1`, and only when `ple != nullptr && ple->ready()` | RT | PLE is optional in the block. See rows 62 to 66 for what forces it on. |
| 31 | src/core/layer.cpp:355 | `moe_route` requires `1 <= k <= 64`; `moe_finish`/`moe_combine_parts` the same (456, 472) | RT | K = 8 is fine. |
| 32 | src/core/layer.cpp:370 | Native fused router only when `n_expert == 512 && k == 10`, else generic `router_top10` | RT | Guarded at runtime. 256/8 takes the generic kernel (row 87). |
| 33 | src/core/layer.cpp:396 | `moe_route_window` batched path excluded when native router active at 512/10 | RT | Same guard. |
| 34 | src/core/layer.cpp:335-354 | `moe_buffers_*` sized from `n_expert`, `k`, `n_ff` | RT | |
| 35 | src/core/layer.cpp:418-451 (`moe_shared`) | Shared expert has a sigmoid scalar gate `ffn_gate_inp_shexp` that must be engine form BF16 (`Bf16InF32`); the three projections need S-forms | ST | Qwen3.6 has the same scalar gate. Its GGUF dtype (F32 or BF16) has to match what the loader produces; see the tensor tables below. |
| 36 | src/core/layer.cpp:223-333 (`gdn_layer`) | Sizes from `g.ssm_*`; recurrence head pairing is `head % h_k` | RT | Matches ggml `repeat` tiling. Needs `h_v % h_k == 0` (32 % 16 = 0). |
| 37 | src/core/layer.cpp:247-248 | Fused conv path requires `d_conv == 4 && state_size == 128`; fused step requires `state_size == 128` (306) | CT-ok | Qwen3.6 matches. |
| 38 | src/core/layer.cpp:276 | `scale_inplace(b.h, qk, 1/sqrt(S))` on the legacy (non-native) path | RT | |
| 39 | src/core/layer.cpp:288-318 and kernels (rows 76-79) | GDN output gate is `rms_norm(o) * ssm_norm * sigmoid(z)` | **ST / CT** | **Qwen3.6 uses SiLU(z), not sigmoid(z).** llama.cpp qwen35moe.cpp:273 `ggml_silu(gate)`, qwen4exp.cpp:481 says "the one numerical difference from Qwen3.5's GDN". The layer.hpp:16 comment records it. This is the only numerical (not geometric) GDN difference. |
| 40 | src/core/layer.cpp:505 (`qsa_shapes`) | Starts from `qsa_real_shapes()` then overrides `n_head`, `n_head_kv`, `head_dim`, `idx_*` from `ModelGeometry` | RT / CT | `n_rot = 64`, `idx_block = 4`, `idx_top_k = 2048`, `page_size = 4` stay compile-time (rows 41 to 44). `n_rot 64` matches Qwen3.6 (0.25 * 256). |
| 41 | include/strata/kernels/qsa.hpp:98-113 (`qsa_real_shapes`) | `n_head 24`, `idx_n_head 4`, `idx_dim 128`, `idx_block 4`, `idx_top_k 2048`, `page_size 4` | CT | Used by layer.cpp:505, mtp.cpp:71, verify.cpp:~100, prefill.cpp:862 etc. |
| 42 | include/strata/kernels/qsa.hpp:144-147, 282 | `qsa_selection_width = min(n_kv, idx_top_k + idx_block - 1)`; `kTopkMaxCells = 32768` | ST | Attention reads only the selected cells, via `ids`. With the indexer absent there is no selection, so every cell must attend. |
| 43 | src/core/layer.cpp:559-575, 761-790 | `qsa_state_bytes` / `qsa_buffers_bytes` include `idx_pooled`, `idx_tail`, `idx_dead`, `pooled_rows = max_cells/idx_block + 2`, indexer scratch, `cell_scores` | ST | The KV state always carries indexer state (about `max_cells/4 * 128 * 4` B per layer). The drafter already passes `ring_cells > 0` for a state "which has no indexer" (layer.cpp:527-531, `pooled_rows = 2`). |
| 44 | src/core/layer.cpp:876-1019 (`qsa_layer`) | Always runs: `indexer.k_proj` BF16 projection (962), indexer key append (945-948), `indexer.q_proj` + norm + rope (962-964), block scores + top-k (966-975), KV resolve, gather/attend over the selected `ids` | ST | The indexer is not optional in this function. It requires `indexer.{k,q}_proj.weight` BF16 and `indexer.{k,q}_norm.weight` (line 891 and the `Req` list). A dense path would skip steps 3, 5-indexer, 7, 8-score/select and attend over `0..n_kv`. |
| 45 | src/core/layer.cpp:891-897 | `w_idxk = v.get("indexer.k_proj.weight")` and `w_idxq` are dereferenced, with a BF16 form check | ST | Null here is an error, not a skip. |
| 46 | src/core/layer.cpp:938-1010 | KV in 4 formats (fp16 / int8 / q4_0 / k8v4), hybrid rotations (`fwht256`), paged pools | RT | `head_dim 256` only (kv_q4.cu:176). Same for Qwen3.6. |
| 47 | src/core/layer.cpp:556 | `qsa_kv_resident_min() = 20480` cells | CT-ok | Streaming floor. Independent of geometry. |

## C. Loader, layout, metadata

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 48 | src/core/layout.cpp:50-80 | `check_one` requires per layer: `hc_attn_{down,up,inject}`, `hc_ffn_{down,up,inject}` with `hc_dim x hc_lr` shapes | ST | `blk.N.hc_*` do not exist in a Qwen3.6 GGUF. |
| 49 | src/core/layout.cpp:60-63 | `ffn_gate_inp [n_embd, n_expert]`, `ffn_{gate,up}_shexp [n_embd, n_ff]`, `ffn_down_shexp [n_ff, n_embd]` | RT | Use `ModelGeometry`. |
| 50 | src/core/layout.cpp:65-70 | GDN: `attn_qkv [n_embd, conv_channels]`, `attn_gate [n_embd, value_dim]`, `ssm_out [value_dim, n_embd]`, `ssm_conv1d [d_conv, channels]`, `ssm_alpha/beta [n_embd, v_heads]` | RT | Same orientation as llama.cpp qwen35moe. |
| 51 | src/core/layout.cpp:72-79 | Attention: `attn_q [n_embd, 2*n_head*head_dim]`, `attn_k/v [n_embd, n_kv*head_dim]`, `attn_output`, and `indexer.q_proj`, `indexer.k_proj` | RT / ST | Attention shapes RT. The two indexer tensors are ST (required). |
| 52 | src/core/layout.cpp:97-106 | 1-D set: `hc_attn_norm`, `hc_ffn_norm` (`hc_dim`, F32), `ffn_gate_inp_shexp` (BF16), `ssm_a`, `ssm_dt.bias`, `ssm_norm`, `attn_q_norm`, `attn_k_norm`, `indexer.q_norm`, `indexer.k_norm` | ST | Same split as rows 48 and 51. |
| 53 | src/core/layout.cpp:162-169 | `check_all` counts `g.n_qsa_layers()` / `g.n_gdn_layers()` | RT | |
| 54 | src/core/layout.cpp:141, 149; layout.hpp:91 | `check_layer` / `check_all` have no call site in src/ | n/a | Grep finds none. The shape assertions are not enforced at load in this tree. Parity tools outside src/ may call them. |
| 55 | include/strata/core/layout.hpp:63 | `is_qsa_layer = layer % interval == interval - 1` | RT | Layers 3, 7, ..., 39 for Qwen3.6. |
| 56 | src/core/weights.cpp:91, 123, 146-191 | Loads whatever `<pack>/index.txt` lists (19 fields per row), into an arena | RT | Name-driven and shape-driven. The model-specific part is the pack builder (`tools/pack_index.py`, `tools/iq_pack.py`, not in this tree). |
| 57 | src/core/native_dense.cpp:31-33 | Native dense GGUF suffixes: `.attn_qkv`, `.attn_gate`, `.ssm_out`, `.attn_q`, `.attn_k`, `.attn_v`, `.attn_output`, `.ffn_{gate,up,down}_shexp` (all `.weight`) | RT | The names match llama.cpp. A fused attention-layer `attn_qkv` (a llama.cpp option) would not match the engine's separate q/k/v. |
| 58 | src/core/native_dense.cpp:102-104, 169-172 | Needs `general.architecture` passing `check_architecture`; each native tensor must already be in the canonical table with a matching 2-D shape | ST | See row 18. Also requires a pack index entry per tensor. |
| 59 | src/core/native_head.cpp:29, 32-39 | `output.weight` must be `[n_in, n_out]`; `n_in % 256 == 0` | RT | 2048 % 256 = 0. `n_out` comes from the pack. |
| 60 | src/core/native_head.cpp:108-128 | `token_embd.weight` must be `[n_embd, n_vocab]` | RT | Caller passes the literals from row 11. |
| 61 | src/core/layer.cpp:1033-1087 (`embed_row`) | `w->ne0 != g.n_embd` is an error | RT | Follows `g.n_embd`. |

## D. PLE and the n-gram table

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 62 | src/program/generate.cpp:1800-1815 | `--native` requires `--ple-gguf` (or a shard holding `per_layer_token_embd.weight`); `--native` plus `--no-ple` is refused | ST | Qwen3.6 has no such tensor, so every native launch is refused. |
| 63 | src/program/generate.cpp:1843-1855 | Without `--native`, `--ple-gguf` is "required"; `--no-ple` is called a "diagnostic ablation" | ST | PLE is mandatory at the CLI, optional in the kernels. |
| 64 | src/program/generate.cpp:2395-2500 | When a PLE table is opened, the pack must contain `blk.1.ple_{key,value,norm_key,norm_query,norm_conv,conv1d}.weight` or the launch fails ("a DIFFERENT MODEL") | ST | Skipped when `--no-ple`. |
| 65 | src/core/native_dense.cpp:29-30, 65-73, 156; generate.cpp:2207-2209, 1956 | Special-casing of `blk.1.ple_key.weight` | ST | Dead without PLE. |
| 66 | include/strata/kernels/ngram.hpp:31-49 | `NG_N_EMBD 2560`, `PLE_HEAD_DIM 160`, `NGRAM_SIZE 3`, `HEADS_PER_NGRAM 8`, `PLE_N_HEADS 16`, `NG_HC 4`, `NG_HC_DIM 10240`, `NG_HIST 9`, `PLE_TABLE_ROWS 320001536`, `PLE_EOS_TOKEN_ID 248044` | CT-ok | All PLE-only. Dead if PLE is skipped. `NG_HIST` x `NG_HC_DIM` still sizes a session buffer (row 29). |
| 67 | src/kernels/cuda/ple.cu:238; native_ple_postops.cu:35, 98; include/strata/kernels/native_ple_postops.hpp:6 | `static_assert(NG_N_EMBD == 2560 ...)`; `N = 2560, H = 4, HISTORY = 9` | CT-ok | Compiles regardless; never launched without PLE. |
| 68 | src/prefill/prefill.cpp:543-549, 806-814, 924-925, 1401, 1590-1599, 1683-1714, 1920 | Prefill allocates `ple_emb`, `ple_norm`, pinned PLE staging; runs the PLE at layer 1 when `ss.ple.ready()` | RT | Guarded by `ple_on`. Buffers are still allocated (about `T * 2560 * 4`, `NG_HC_DIM * 4`). |
| 69 | src/core/verify.cpp:370, 408, 559, 636-675, 1311, 1405 | Verify window stages and runs the PLE at layer 1 when `ss.ple.ready() && ple_stage()` | RT | Guarded. |

## E. MTP drafter (`mtp.cpp`, `mtp.hpp`)

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 70 | src/core/mtp.cpp:150-216 | Loads `<rt_dir>/dense.txt`, `dense.bin`, `experts.bin` produced by `tools/mtp_rt.py` (not in this tree), all dense weights as Q8_0 | ST | The drafter does not read the GGUF. |
| 71 | src/core/mtp.cpp:210-212 | Required tensors: `fc_embedding.weight`, `fc_hidden.weight`, `self_attn.{q,k,v,o}_proj.weight`, `mlp.shared_expert.{gate,up,down}_proj.weight` | ST | vLLM-style names. Qwen3.6's MTP has one `nextn.eh_proj [2*n_embd, n_embd]` over `[enorm(emb), hnorm(h)]`, not two separate projections. |
| 72 | src/core/mtp.cpp:488-495 | `pre_fc_norm_embedding` -> `fc_embedding`; `pre_fc_norm_hidden` over `hc*n_embd` -> `fc_hidden` per stream -> `R = h + e` broadcast over `hc` streams | ST | Qwen3.6: `eh_proj(concat(enorm(emb), hnorm(h)))`, one stream. |
| 73 | src/core/mtp.cpp:503-507, 564-568, 613-625 | Three hyper-connection groups: `attn_hyper_connection.*`, `mlp_hyper_connection.*`, `hyper_connection_mixer.*` | ST | Same gated-residual structure as rows 21-23. |
| 74 | src/core/mtp.cpp:191, 200, 306-308, 579 | Drafter experts are `n_expert * cpu::BLOB` bytes in VRAM, canonical Q2_0 at 2560/640, run through `moe_group_resident` | CT | `BLOB = 1,382,400`. Qwen3.6 Q2_0 blob would be `3 * 2048 * 512 * 18/64` = 884,736 B. Message text says "the 512 experts do not fit". |
| 75 | src/core/mtp.cpp:575, 598 | Draft router `mlp.gate.weight` BF16 `[n_embd, n_expert]`; `mlp.shared_expert_gate.weight` BF16 | RT | Shape-driven. |
| 76 | src/core/mtp.cpp:71-78, 218-249 | Draft KV is dense attention (no indexer state read), shapes from `ModelGeometry` | RT | The header says it is "identical to the model's sparse selection below 2,051 cells". The only existing dense-attention path. |
| 77 | src/core/mtp.cpp:428-454 | Head is the main `output.weight`; `n_vocab_ = wo->ne1`; optional `draft_vocab.bin` subset | RT | Vocabulary size is a runtime value. Subset files (`data/draft_vocab_*.bin`) are token ids into the same 248,320 vocabulary. |
| 78 | src/core/mtp.cpp:477, 488 | Draft embeds from `token_embd.weight` | RT | |
| 79 | src/core/mtp.cpp:251-268 | Buffers from `g.hc`, `g.hc_lr`, `g.n_head` | RT | |
| 80 | src/prefill/prefill.cpp:972-1086 | MTP prefill reads `attn_hyper_connection.hc_norm.weight`, uses `LR`, `HC`, `fc_hidden` per stream | ST / CT | Same vLLM-style structure as row 72. |

## F. GDN kernels

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 81 | src/kernels/cuda/native_gdn.cu:37, 100-102, 117 | `S = 128`; grid `h_v` x `S/4`; requires `h_v % h_k == 0`, `h_v <= 65535` | RT | Heads from `GdnShapes`. State layout `(S, h_v, S)`. |
| 82 | src/kernels/cuda/native_gdn.cu:51 | `q_head = head % h_k` | RT | Modulo pairing. |
| 83 | src/kernels/cuda/fused_gdn.cu:12-14, 22-24, 127-129, 148-153 | `S = 128`, `RG = 4`; grid `h_v`; `qk_heads` blocks for conv+L2; `h_v % h_k == 0` | RT | |
| 84 | src/kernels/cuda/gdn.cu:46-59, 156 | `MAX_H = 8` heads per block, `S <= 128` | RT | Legacy path; only a performance shape. |
| 85 | src/kernels/cuda/gdn.cu:148 | `y = o * rms * ssm_norm * sigmoid_f(z)` | **CT** | **SiLU for Qwen3.6.** See row 39. |
| 86 | src/kernels/cuda/fused_gdn.cu:69 | `... * (1/(1+exp(-zz)))` in `gdn_step_norm_kernel` | **CT** | **SiLU for Qwen3.6.** The default decode path for native packs. |
| 87 | src/kernels/cuda/native_gdn_preprocess.cu:110-112 | `weighted * sigmoid(z)` in the out-norm kernel | **CT** | **SiLU for Qwen3.6.** The same file's `conv_silu` (53-67) and `beta_sigmoid` (86-88) are correct for both models. |
| 88 | src/prefill/kernels.cu:324, 612 | `sigm(z)` in `gdn_rec_kernel` and `gdn_out_norm_kernel` | **CT** | **SiLU for Qwen3.6.** Prompt path. |
| 89 | src/prefill/kernels.cu:18-19 | `N = 2560, HC = 4, D = N*HC, LR = 320`; `S = 128, HK = 16, HV = 48, C = 10240` | CT | Constants for every prompt-path GDN, GR and gate kernel. |
| 90 | src/prefill/kernels.cu:211-215, 285-304, 345-364, 387-432, 478-570, 893-924 | GDN prompt kernels index with `HV`, `HK`, `C`; launched `<<<HV, ...>>>`, `<<<HV*NCB>>>`, `<<<HK*NCB>>>` | CT | Includes `VPK = HV/HK = 3` value heads per key head (478): 2 for Qwen3.6. `gdn_rec_kh_kernel` is specialised on it. |
| 91 | src/prefill/gdn_rec_parity.cu | Parity harness for the above | CT | Mirrors the same constants. |
| 92 | include/strata/kernels/gdn.hpp:11, 64; layer.hpp:10, 21, 23, 546 | Comments cite 10240 / 48 / 2560 | n/a | Comments only. |

## G. Attention kernels (the QSA path)

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 93 | src/kernels/cuda/qsa_decode_attn.cu:16-17, 110, 272, 453, 497 | `HD = 256`, `G = 12` (query heads per KV head); refuses with `exit(1)` unless `n_head == 12 * n_head_kv` | CT | The main decode kernel. Qwen3.6 has G = 8. |
| 94 | src/kernels/cuda/qsa_prompt_attn.cu:21-22, 115-116, 1442-1447 | `HD = 256`, `G = 12`; refuses unless `n_head == 12 * n_head_kv` | CT | The prompt attention kernels (tensor-core). |
| 95 | src/kernels/cuda/native_flash_attn.cu:55, 196-197, 206 | `kv = head / 12`; refuses unless `n_head == 24 && n_head_kv == 2 && head_dim == 256`; grid 24; context 1..256 | CT | Short-context adapter only. |
| 96 | src/kernels/cuda/qsa.cu:96 | Generic `qsa_attend`/`kv_append`/`kv_gather`; requires `n_head % n_head_kv == 0` | RT | Reference-quality path. |
| 97 | src/kernels/cuda/native_rope.cu:121 | `head_dim in {128, 256}`, `n_rot == 64` | CT-ok | Matches Qwen3.6 (partial rotary 0.25 of 256). |
| 98 | include/strata/kernels/mrope.hpp:3-12 | Interleaved M-RoPE sections 11/11/10/0; text has `t = h = w` | CT-ok | Qwen3.6 uses the same sections. |
| 99 | src/kernels/cuda/native_qsa_score.cu:31, 194 | Indexer: `D = 128, HEADS = 4, R = 4`, `idx_top_k == 2048` | ST | Indexer-only. Not reachable without the indexer. |
| 100 | src/kernels/cuda/native_qsa_indexer.cu:35, 231, 259 | `D = 128, R = 4, ROT = 64` | ST | Indexer-only. |
| 101 | src/kernels/cuda/qsa_select.cu:17-19, 161-165, 467-475, 1081, 1115, 1220 | `IDX_DIM 128`, `IDX_HEADS 4`, `R 4`; requires `cap >= qsa_selection_width(kTopkMaxCells, s)` | ST | Indexer-only. With the indexer absent nothing here runs, and nothing else produces the `ids` that `qsa_decode_attn` consumes. |
| 102 | src/kernels/cuda/qsa_decode_attn.cu | The kernel gathers K/V through `ids` and `pools.page_table`, so it is selection-agnostic | RT | An identity `ids` would attend over every cell. No other dense decode kernel exists. |
| 103 | src/kernels/cuda/kv_q4.cu:176, kv_q8.cu | `head_dim == 256` | CT-ok | Matches Qwen3.6. |
| 104 | src/kernels/cuda/kv_stream.cu:23 | `RT = 1024` resolve block | CT-ok | Streaming resolve granularity. Reads `ids`. |
| 105 | src/prefill/kernels.cu:752-761, 1016, 1020 | `split_q_kernel`, `gate_attn_kernel`: `T * 24 * 256`, `24 * 512` | CT | Prompt-path q/gate split and gate. 16 heads for Qwen3.6. |
| 106 | src/prefill/prefill.cpp:649-650, 883-884 | `T * 12288` (= 24 * 512) for `Qf`; `T * 512` for `Kc/Vc` (2*256, unchanged); `T * ZV` for `q` and attn; `T * 128`, `T * 512` for `idx_raw`, `q_idx` | CT | `12288` and `ZV = 6144` must change. `512` for K/V stays. The two indexer buffers are dead without the indexer. |

## H. Experts, router, shared expert

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 107 | include/strata/kernels/cpu/expert.hpp:34-36, 40-47 | `H = 2560`, `FF = 640`, `NE = 512`, `BLOB = 1,382,400`, `ROW_GU`, `SC_GU` | CT | Canonical Q2_0 blob geometry. Canonical only. Used by the rows below. |
| 108 | src/kernels/cpu/expert_layout.cpp:284 | `native_fmt(gt, dt, H, FF, f, err)` | **CT** | **The one compile-time leak in the native-pack path.** The layout loader builds each layer's `NativeFmt` with 2560/640 and then checks `f.bytes == blob`. A 2048/512 pack fails the "blob is X B but its formats make Y" check. |
| 109 | src/kernels/cpu/expert_layout.cpp:233-244, 271 | `native_experts.txt` header "(n_expert N)" overrides `n_expert`; absent file means canonical pack, total = `n_layers*n_expert*BLOB` | RT | |
| 110 | include/strata/kernels/cpu/expert_layout.hpp:20, 32, 35-37 | `n_expert = NE` default, `max_blob = BLOB` default | CT-ok | Overwritten for native packs. |
| 111 | include/strata/kernels/cpu/native_expert.hpp:17-19; src/kernels/cpu/native_expert.cpp:64 | `kNativeActBytes = 4096`, `kNativeHBytes = 1024` upper bounds, checked at layout time | CT-ok | 2048-wide Q8_K is 2336 B; 512-wide Q8_K is 584 B. Both fit. |
| 112 | src/kernels/cpu/native_expert.cpp:40-60 | `native_fmt(gu_type, d_type, n_embd, n_ff)` is runtime; requires whole ggml blocks | RT | 2048 and 512 are multiples of 256 and 32. K-quants and Q8_K activations work for n_ff = 512 (unlike 640). |
| 113 | src/core/expert_source.cpp:1863-1868 (`expert_pool_dispatch`) | Refuses `n_embd != H` ("compiled for a 2560-wide activation") | CT | Canonical single-token path only. |
| 114 | src/core/expert_source.cpp:1869-1875 | `expert_pool_dispatch` refuses a native pack outright ("single-token expert path does not take a native pack") | **ST** | **Native-pack experts run only in verify windows.** generate.cpp:~2180 also requires `--spec T >= 2` and `--prefill`. Every decoded token therefore goes through verify.cpp. |
| 115 | src/core/expert_source.cpp:367-376, 2328; include/strata/core/expert_source.hpp:241 | Canonical size checks with `BLOB`; `ExpertDispatch::n_expert = cpu::NE`; hit path uses `cpu::H` | CT | Canonical only. `n_expert` is overwritten at generate.cpp:3683. |
| 116 | src/core/expert_source.cpp:2475-2500 (`check_experts_gguf`) | Native experts are read from `blk.N.ffn_{gate,up,down}_exps.weight` as three separate tensors, each `[cols, rows, n_expert]` | ST | llama.cpp may fuse gate and up into `ffn_gate_up_exps` (`create_tensor_gate_up_exps`, llama-model.cpp:3253). The engine accepts only the separate form. |
| 117 | src/core/expert_source.cpp:1979-1980 | `MAXT * 10 <= kMaxWindowEntries (128)` | CT-ok | 8 * 10 = 80; with k = 8 it is 64. |
| 118 | src/core/expert_cache.cpp, include/strata/core/expert_cache.hpp | Slots, residency `[n_layers * n_expert]`, profile file `(n_layers, n_expert)` header, `n_layers/n_expert <= 65535` | RT | All sizes passed in. The profile header is checked against `(n_layers, n_expert)` (cache.cpp:43). |
| 119 | src/kernels/cuda/iq_kernels.cu:1076-1140, 2543 | Native GPU expert kernels read `L.n_embd` and `L.n_ff` per layer; the fast sub-warp paths apply only when `n_ff == 640` (else 8-row fallback) | RT | Runtime guard. |
| 120 | src/kernels/cuda/iq_kernels.cu:309-350, 512-518 | Unsloth UD-Q4_K_XL expert types already supported: Q4_K / Q5_K gate-up, Q5_1 / Q8_0 down | RT | Native GPU expert dequant kernels exist for the Qwen3.6 UD quant family. |
| 121 | src/kernels/cuda/s2_expert_grouped.cu:37-46, 139-195, 301-346 | `H = 2560`, `FF = 640`, `ROW_GU`, `SC_GU`, blob offsets | CT | Canonical Q2_0 VRAM-hit kernels. Used for the MTP drafter's experts (row 74), not for native packs. |
| 122 | src/core/remote_experts.cpp:16-18, 246; src/core/peer_experts.cpp:25-26, 115, 130; src/core/remote_expert_opt.cu:13-14 | `H`, `FF` from `cpu::`, `CAP = MAXT * 10`, `k != 10` refused | CT | Multi-GPU expert tiers. Out of scope for a single card. |
| 123 | src/program/generate.cpp:3721 | `moe_hit_grouped_scratch_bytes(K, g.n_embd, cpu::FF)` | CT-ok | Over-allocates by 640 vs 512. |
| 124 | src/kernels/cuda/router_top10.cu:65-215; include/strata/kernels/router_top10.hpp:21 | Generic top-k softmax router: `n_expert <= 512`, `k <= 32`, runtime `n_expert` and `k` | RT | Handles 256 / 8. |
| 125 | src/kernels/cuda/native_router.cu:49, 54, 73, 97, 111-122 | Fused pinned router: `512` logits, `10` outputs, `REG 16`, hard-coded in the kernel and the validity checks | CT | Used only when `n_expert == 512 && k == 10` (row 32). Otherwise bypassed. |
| 126 | src/kernels/cuda/native_moe.cu:62-63, 79-80 | `native_moe_combine` requires `1 <= k <= 15` | RT | |
| 127 | src/kernels/cuda/shared_expert.cu:191, 237-239, 325-328 | `n_tok 1..8`; Q8_K activation needs `n_ff % 256 == 0` | RT | 640 failed this; 512 passes. Header comments in s_gemv.hpp:122-124 and layer.cpp:419-437 ("Q8_K is structurally impossible") describe Flash-Next only. |
| 128 | src/prefill/kernels.cu:616-657, 928-936 | Prompt `route_kernel<REG>`: `REG 16` for 512, `REG 8` for 256; rank loop `< 10`, writes `ids[t*10 + rank]`, `lane < 10` | CT | A 256 variant exists, but top-10 is baked into the kernel. |
| 129 | src/prefill/kernels.cu:660-692, 947, 951; moe_fused.cu:46-53, 199-203, 343; moe_mmq.cu:60-75, 249 | Canonical Q2_0 blob dequant and fused kernels: `1280 * 640`, `2560 * 160`, `n * 640`, `10 * AB` | CT | Canonical only. |
| 130 | src/prefill/moe_fused_iq.cu:59 | `GU_ROWS_K = 2560, D_ROWS_K = 640` | CT | Native-pack fused prompt MoE. Used as K of gate/up and of down. |
| 131 | src/prefill/prefill.cpp:90-91 | `N = 2560, HC = 4, D = N*HC, LR = 320, K = 10, NE = 512; C = 10240, ZV = 6144, HV = 48` | CT | File-wide constants. |
| 132 | src/prefill/prefill.cpp:737 | `Prefill::init` refuses unless `g.n_embd == N && g.hc == HC && g.hc_lr == LR && ss.k == K` | CT / ST | Hard refusal; every chunked-prefill launch fails on Qwen3.6. |
| 133 | src/prefill/prefill.cpp:681-685, 704-721, 896-907, 1294-1322, 1391 | Prompt MoE buffers: `T*K*1280`, `T*K*640`, `q8_bytes(.., 640)`, `1280 * 2560`, `2560 * 640`, `RP = 48` (1303) | CT | `1280 = 2*n_ff`, `640 = n_ff`, `2560 = n_embd`. |
| 134 | src/prefill/prefill.cpp:1724 and loops | Iterates `g.n_layers` and `is_qsa_layer` | RT | |
| 135 | src/prefill/prefill.cpp:854-856, 1000-1026, 1375-1380 | `grs`, `gated`, `inj`, `xq`, `ident` sized from `HC`, `D`, `LR` | CT | Hyper-connection buffers in prefill. |
| 136 | include/strata/prefill/kernels.hpp:15-16, 40, 52, 54, 58, 79 | Declarations documented as `2560 / 4 / 320`, `48 v heads`, `512, top-10`, `[1280, 2560]`, `640`, `[T, 24, 512]` | CT | Comments describing the constants above. |

## I. Hyper-connection (gated residual) kernels

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 137 | src/kernels/cuda/fused_gr.cu:20-28, 253-254, 320-321, 1164-1166 | `N = 2560, HC = 4, LR = 320, D = 10240`; `DOWN_BLOCKS = LR/8`, `UP_BLOCKS = N/32`, `UPM_BLOCKS = N/16`; `fused_gr_supported()` returns `n_embd == N && hc == HC && hc_lr == LR` | CT | Default decode and verify GR kernels. |
| 138 | src/core/layer.cpp:1195, 1319-1342, verify.cpp:336 | `fused = g_fused_gr && ... fused_gr_supported(...)`; verify refuses without it | CT / ST | Row 142. |
| 139 | src/kernels/cuda/gr.cu, native_gr_norm.cu, native_gr_postops.cu | Generic `GrShapes{n_embd, hc, hc_lr}` unfused path | RT | The generic GR is runtime-shaped, but its semantics (down/up/inject mixing) are the hyper-connection. |
| 140 | src/kernels/cuda/cvec.cu:11-17; verify_kernels.cu:256-291 (`broadcast_streams`, `add_streams_broadcast`) | Control-vector apply on `hc*n_embd`; stream broadcast | RT | `MAXK = 16` supports `n_embd <= 4096`. |

## J. Verify window (`verify.cpp`) and the speculation protocol

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 141 | src/core/verify.cpp:331-334 and layer.cpp:476-486 (`layer_verify_compatible`) | The verify window needs native BF16 projections, fused GR, fused native GDN, split-K decode attention, block top-k selection, **native QSA indexer** | ST | Verify is the only decode path for native packs (row 114). It cannot be built on a layer that has no fused GR and no indexer. |
| 142 | src/core/verify.cpp:336-340 | Refuses unless `fused_gr_supported && ss.k == 10 && ssm_state_size == 128 && ssm_d_conv == 4` ("geometry differs from the artifact's") | CT / ST | |
| 143 | src/core/verify.cpp:356-434 | Buffers `R_ (T*HC*N)`, `inj_`, `lo_ (T*hc_lr)`, `xn_ (T*HC*N)`, `idx_raw_L_`, `qidx_`, `tail_snap_` | RT / ST | Sized from `g.hc`, `g.hc_lr`, `idx_*`. |
| 144 | src/core/verify.cpp:550-559 | `IQ = idx_q_heads`, `ID = idx_key_dim`, `NE = n_expert`, `TS = (idx_block-1)*ID` | RT | |
| 145 | src/core/verify.cpp:757-761, 772-870, 1271-1311 | Attention step: `need(v, "indexer.k_proj.weight")`, `indexer.q_proj`, `indexer.q_norm`, `indexer.k_norm`; indexer append, block scores, top-k per window row; commit restores the indexer tail | ST | Same as row 44, for the multi-token path. |
| 146 | src/core/verify.cpp:916 | Batched native router only when `NE == 512 && K == 10` | RT | Guarded. |
| 147 | src/core/verify.cpp:1123-1125 | Head mix needs `output_hc_{norm,down,up}.weight` | ST | Row 23. |
| 148 | src/core/verify.cpp:347-349 | `n_vocab_ = output.weight.ne1` | RT | |
| 149 | include/strata/kernels/verify_kernels.hpp:21 | `kVerifyMaxT = 8` | CT-ok | Draft window cap. |
| 150 | src/kernels/cuda/verify_kernels.cu:366, 397-401, 506 | Grid `48 * 8`; `kVerifyMaxT * 10 <= kResidentPlanMax` | CT-ok | The 48 is a block count, not a layer count. |

## K. Vocabulary, sampler, metadata keys

| # | file:line | Assumption | Class | Note |
|---|---|---|---|---|
| 151 | src/kernels/cuda/sampler.cu:102-700 | `n_vocab` is a kernel argument everywhere; the split top-k merge handles `n_vocab <= 262,144` (607) | RT | 248,320 fits. |
| 152 | src/program/generate.cpp:479-481 | Default EOS ids `{248044, 248046}` | CT-ok | Same tokenizer; override with `--eos-ids`. |
| 153 | include/strata/artifact/gguf_reader.hpp:581-597 | All keys read use the `qwen4exp.` prefix | CT | Row 18. The Qwen3.6 GGUF uses `qwen35moe.*`. Also needed and not read anywhere: `qwen35moe.ssm.{conv_kernel,inner_size,state_size,time_step_rank,group_count}`, `.expert_feed_forward_length`, `.expert_shared_feed_forward_length`, `.full_attention_interval`, `.rope.dimension_sections`, `.nextn_predict_layers`. |
| 154 | frontend/ | Tokenizer, template, tool-call parser, vision | n/a | Model-agnostic per the plan. Not part of this inventory. |

## Tensor names the loader expects (engine side) against llama.cpp qwen35moe

Engine names come from layout.cpp:50-106, layer.cpp, native_dense.cpp:31-33 and expert_source.cpp:2491.
llama.cpp names come from `third_party/llama.cpp/src/models/qwen35moe.cpp:43-138` and `src/llama-arch.cpp:435-587`
(pinned commit 3cf0325).

**One GDN layer, `blk.N.`**

| Engine expects | Shape / form | llama.cpp qwen35moe | Match |
|---|---|---|---|
| `hc_attn_norm.weight`, `hc_attn_down.weight`, `hc_attn_up.weight`, `hc_attn_inject.weight` | `[hc_dim]` F32; BF16 | none | **Engine-only (hyper-connection)** |
| `hc_ffn_norm.weight`, `hc_ffn_down.weight`, `hc_ffn_up.weight`, `hc_ffn_inject.weight` | same | none | **Engine-only** |
| (none) | | `attn_norm.weight` `[n_embd]` | **llama.cpp-only** |
| (none) | | `post_attention_norm.weight` `[n_embd]` (pre-FFN norm; there is no `ffn_norm`) | **llama.cpp-only** |
| `attn_qkv.weight` | `[n_embd, conv_channels]` | `attn_qkv.weight` `[n_embd, 2*key_dim + value_dim]` | yes |
| `attn_gate.weight` | `[n_embd, value_dim]` | `attn_gate.weight` | yes |
| `ssm_conv1d.weight` | `[d_conv, conv_channels]` F32 | `ssm_conv1d.weight` | yes |
| `ssm_alpha.weight`, `ssm_beta.weight` | `[n_embd, v_heads]` BF16 | same names, `{n_embd, n_v_heads}` | yes, dtype to confirm |
| `ssm_a` | `[v_heads]` F32 | `ssm_a` (no `.weight`) | yes |
| `ssm_dt.bias` | `[v_heads]` F32 | `ssm_dt.bias` | yes |
| `ssm_norm.weight` | `[128]` F32 | `ssm_norm.weight` `{head_v_dim}` | yes |
| `ssm_out.weight` | `[value_dim, n_embd]` | `ssm_out.weight` | yes |
| `ffn_gate_inp.weight` | `[n_embd, n_expert]` BF16 | same | yes, dtype to confirm |
| `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight` | `[n_embd, n_ff, n_expert]` x2, `[n_ff, n_embd, n_expert]` | same, or fused `ffn_gate_up_exps.weight` `[n_embd, 2*n_ff, n_expert]` | yes if unfused |
| `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, `ffn_down_shexp.weight` | `[n_embd, n_ff]`, `[n_ff, n_embd]` | same | yes |
| `ffn_gate_inp_shexp.weight` | `[n_embd]` engine form BF16 | `ffn_gate_inp_shexp.weight` `{n_embd}` | yes, dtype to confirm |

**One attention layer, `blk.N.` (layers 3, 7, ..., 39)**

| Engine expects | Shape / form | llama.cpp qwen35moe | Match |
|---|---|---|---|
| the six `hc_*` tensors and two `hc_*_norm` as above | | none | **Engine-only** |
| (none) | | `attn_norm.weight`, `post_attention_norm.weight` | **llama.cpp-only** |
| `attn_q.weight` | `[n_embd, 2*n_head*head_dim]` (q then gate per head, interleaved) | `attn_q.weight` `{n_embd, head_dim*n_head*2}` | yes |
| `attn_k.weight`, `attn_v.weight` | `[n_embd, n_kv*head_dim]` | same | yes, unless the GGUF fuses them into `attn_qkv.weight` (`create_tensor_qkv`) |
| `attn_output.weight` | `[n_head*head_dim, n_embd]` | `attn_output.weight` | yes |
| `attn_q_norm.weight`, `attn_k_norm.weight` | `[head_dim]` F32 | same | yes |
| `indexer.q_proj.weight`, `indexer.k_proj.weight`, `indexer.q_norm.weight`, `indexer.k_norm.weight` | BF16 / F32 | none | **Engine-only (QSA indexer)** |
| the same five MoE tensors, three shexp tensors, `ffn_gate_inp_shexp` as a GDN layer | | same | as above |

**Global tensors**

| Engine expects | llama.cpp qwen35moe |
|---|---|
| `token_embd.weight` `[n_embd, n_vocab]` | same |
| `output.weight` `[n_embd, n_vocab]` (native head; mtp.cpp:428 and verify.cpp:347 read `ne1`) | `output.weight` (optional; falls back to `token_embd.weight`) |
| `output_hc_norm.weight`, `output_hc_down.weight`, `output_hc_up.weight` | `output_norm.weight` `{n_embd}` |
| `blk.1.ple_{key,value,norm_key,norm_query,norm_conv,conv1d}.weight` and `per_layer_token_embd.weight` | none |
| MTP: `<rt_dir>/dense.txt` names `fc_embedding`, `fc_hidden`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `self_attn.*`, `mlp.*`, three `*hyper_connection*` groups | `blk.40.nextn.eh_proj.weight` `{2*n_embd, n_embd}`, `blk.40.nextn.enorm.weight`, `blk.40.nextn.hnorm.weight`, optional `blk.40.nextn.{embed_tokens,shared_head_head,shared_head_norm}.weight`, plus a full attention + MoE block under `blk.40.` (attn_norm, post_attention_norm, attn_q/k/v/output, attn_q_norm, attn_k_norm, ffn_gate_inp, ffn_*_exps, ffn_*_shexp, ffn_gate_inp_shexp) |

## Summary of which subsystems are runtime-parameterized

1. **Already parameterized:** the GDN decode kernels (state 128, head counts and head pairing by `h % h_k`, grids from `h_v`), the generic router, the shared expert (and 512 now satisfies its Q8_K rule), the native CPU and GPU expert kernels (`NativeFmt` and `L.n_embd`/`L.n_ff` per layer, UD-Q4_K_XL types already present), the expert cache, the weight loader, the sampler, `output.weight` and vocabulary size, and the drafter's dense-attention KV.
2. **The geometry struct is never populated from the model.** Only `expert_count` and `expert_used_count` (and rope keys) are read, under a `qwen4exp.` prefix, and only with `--native`. `n_embd`, `n_layers`, head counts, `ssm.*` and `n_ff` stay at Flash-Next defaults. `check_architecture` refuses any other architecture string.
3. **The gated residual is structural.** Every block, the output head, the verify window, the drafter and prefill assume `hc` streams with down/up/inject mixing. The fused kernels are hard-coded to 2560/4/320, and `hc = 1` has no plain-norm meaning. The plain `attn_norm`, `post_attention_norm` and `output_norm` tensors are never referenced.
4. **Dense attention is structural.** `qsa_layer`, `verify.cpp`, the KV state size and the selection width all assume the indexer. The decode and prompt attention kernels are compile-time for group 12 (Qwen3.6 is 8) and the short-context kernel is compile-time for 24 heads. The attention kernel reads through `ids`, so it is selection-agnostic, but no producer of dense `ids` exists.
5. **GDN output gate differs numerically:** Qwen3.6 applies SiLU(z) where the engine applies sigmoid(z) in four kernels (gdn.cu, fused_gdn.cu, native_gdn_preprocess.cu, prefill/kernels.cu). This is a semantic change, not a size change.
6. **Native-pack decode is verify-window only.** The single-token expert dispatch refuses native packs, and the verify window demands fused GR, K = 10, and the native indexer. Any hc = 1 or dense-attention work has to land in verify.cpp as well as layer.cpp.
7. **Prefill is the most compile-time-heavy subsystem.** `Prefill::init` refuses non-2560/4/320/10, and GDN, gate, split-q, route and MoE buffers all use literal constants (HV 48, 24 heads, 1280, 640, top-10).
8. **One leak inside otherwise-runtime expert code:** `expert_layout.cpp:284` passes compile-time `H, FF` to `native_fmt`, so a 2048/512 pack fails the blob-size check.
9. **PLE is optional in kernels but mandatory in the CLI.** `--native` refuses to start without `--ple-gguf`, and a table implies `blk.1.ple_*` tensors. Qwen3.6 has neither. A small session buffer (about 360 KB) is allocated regardless.
10. **MTP drafter is separate from the GGUF** (Q8_0 `rt_dir` from `tools/mtp_rt.py`, canonical Q2_0 experts at 2560/640, three hyper-connection groups, two fc projections). It gets the canonical default geometry, not the target's. The `tools/` packers are not in this tree, so the pack and MTP tensor names are defined by scripts that were not inspected.
