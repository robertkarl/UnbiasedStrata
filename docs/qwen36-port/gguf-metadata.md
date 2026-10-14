# Qwen3.6-35B-A3B GGUF: metadata and tensors as shipped

Dumped 2026-10-11 with `tools/gguf_reader.py` (no gguf-py) from two files:

- **Plain**: `/Users/robertkarl/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` (Unsloth `UD-IQ3_XXS`, 733 tensors, `general.file_type` 23, 13.2 GB). The file the port is built against.
- **MTP**: `/Users/robertkarl/models/qwen3.6/MTP/Qwen3.6-35B-A3B-UD-IQ4_XS-MTP.gguf` (753 tensors, 41 blocks). Same architecture plus the nextn block. Shown only where it differs.

GGUF dims below are **ne order** (fastest first). The numpy / `ref/load.py` shape is the reverse, `(out, in[, experts])`.
"llama.cpp reads it" cites the pinned submodule (`third_party/llama.cpp`, commit 3cf0325).

## Metadata

| key | value | llama.cpp reads it |
| --- | --- | --- |
| `general.architecture` | `qwen35moe` | `llama-arch.cpp:42` |
| `qwen35moe.block_count` | 40 (MTP file: 41) | `llama-model.cpp:1259`; trunk layers = block_count - nextn |
| `qwen35moe.nextn_predict_layers` | **absent** in the plain file; 1 in the MTP file | `llama-model.cpp:1261`, optional, default 0 |
| `qwen35moe.context_length` | 262144 | `llama-model.cpp:1254` |
| `qwen35moe.embedding_length` | 2048 | `:1255` |
| `qwen35moe.attention.head_count` / `head_count_kv` | 16 / 2 | `:1328`, `:1336` |
| `qwen35moe.attention.key_length` / `value_length` | 256 / 256 | `:1372`, `:1375` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | 1e-6 (stored as f32 9.99999997e-07) | `qwen35moe.cpp:7`, required |
| `qwen35moe.rope.dimension_count` | 64 (= 0.25 * 256) | `:1380` |
| `qwen35moe.rope.dimension_sections` | `[11, 11, 10, 0]` (sum 32 = 64 / 2) | `qwen35moe.cpp:9`, required, 4 entries |
| `qwen35moe.rope.freq_base` | 1e7 | `:1347` |
| `qwen35moe.expert_count` / `expert_used_count` | 256 / 8 | `:1263`, `:1265` |
| `qwen35moe.expert_feed_forward_length` | 512 | `qwen35moe.cpp:5` |
| `qwen35moe.expert_shared_feed_forward_length` | 512 | `qwen35moe.cpp:6` |
| `qwen35moe.ssm.conv_kernel` | 4 | `qwen35moe.cpp:12` |
| `qwen35moe.ssm.state_size` | 128 (head dim of k and v) | `:14` |
| `qwen35moe.ssm.group_count` | 16 (k heads) | `:16` |
| `qwen35moe.ssm.time_step_rank` | 32 (v heads) | `:15` |
| `qwen35moe.ssm.inner_size` | 4096 (= 32 * 128, the value dim) | `:13` |
| `qwen35moe.full_attention_interval` | 4 | `:22`; layer `il` is recurrent when `(il+1) % 4 != 0` |
| `tokenizer.ggml.model` / `pre` | `gpt2` / `qwen35` | vocab, 248320 tokens, 247587 merges |
| `tokenizer.ggml.{bos,eos,padding}_token_id` | 248044 / 248046 / 248055 | `add_bos_token` false |
| `tokenizer.chat_template` | 8424 characters | frontend, not llama.cpp's graph |
| `general.*`, `quantize.imatrix.*` | name, license, Unsloth provenance, imatrix file | not used by the graph |

Keys that are **not** in the file and **not** read by the `qwen35moe` loader: `expert_weights_norm`, `expert_weights_scale`,
`attention.recurrent_layers` (read if present, `qwen35moe.cpp:20`), `feed_forward_length`, `expert_group_count`, rope scaling.
`HParams` defaults then apply: `expert_weights_scale` stays 0.0 (`llama-hparams.h:124`, the key is never loaded for this architecture),
so no scaling; `norm_w` is hard-coded `true` at the call site (`qwen35moe.cpp:506`), so the top-8 weights always renormalize to sum 1.
A GGUF cannot change either; a stray `qwen35moe.expert_weights_scale` key is ignored (checked with `--expert-weights-scale` on the fixture).

## Tensors (plain file, 733)

| name | ne | dtype in the IQ3_XXS file | count |
| --- | --- | --- | --- |
| `token_embd.weight` | 2048, 248320 | Q6_K | 1 |
| `output.weight` | 2048, 248320 | Q6_K (not tied) | 1 |
| `output_norm.weight` | 2048 | F32 | 1 |
| `blk.N.attn_norm.weight` | 2048 | F32 | 40 |
| `blk.N.post_attention_norm.weight` | 2048 | F32 | 40 |
| `blk.N.ffn_gate_inp.weight` (router) | 2048, 256 | F32 | 40 |
| `blk.N.ffn_gate_exps.weight` | 2048, 512, 256 | IQ2_S | 40 |
| `blk.N.ffn_up_exps.weight` | 2048, 512, 256 | IQ2_S | 40 |
| `blk.N.ffn_down_exps.weight` | 512, 2048, 256 | IQ3_XXS (37), IQ4_XS (3: layers 34, 38, 39) | 40 |
| `blk.N.ffn_gate_inp_shexp.weight` | **2048** (1-D) | F32 | 40 |
| `blk.N.ffn_{gate,up}_shexp.weight` | 2048, 512 | Q6_K | 80 |
| `blk.N.ffn_down_shexp.weight` | 512, 2048 | Q6_K | 40 |
| GDN layers (30: 0-2, 4-6, ..., 36-38): | | | |
| `blk.N.attn_qkv.weight` | 2048, 8192 (= 2*2048 + 4096) | Q6_K | 30 |
| `blk.N.attn_gate.weight` (z) | 2048, 4096 | Q6_K | 30 |
| `blk.N.ssm_conv1d.weight` | 4, 8192 | F32 | 30 |
| `blk.N.ssm_alpha.weight`, `ssm_beta.weight` | 2048, 32 | F32 | 60 |
| `blk.N.ssm_a` (no `.weight` suffix) | 32 | F32 | 30 |
| `blk.N.ssm_dt.bias` | 32 | F32 | 30 |
| `blk.N.ssm_norm.weight` | 128 | F32 | 30 |
| `blk.N.ssm_out.weight` | 4096, 2048 | Q6_K | 30 |
| Attention layers (10: 3, 7, ..., 39): | | | |
| `blk.N.attn_q.weight` (query and gate, interleaved per head) | 2048, 8192 (= 16 * 2 * 256) | Q6_K | 10 |
| `blk.N.attn_k.weight`, `attn_v.weight` | 2048, 512 (= 2 * 256) | Q6_K | 20 |
| `blk.N.attn_output.weight` | 4096, 2048 | Q6_K | 10 |
| `blk.N.attn_q_norm.weight`, `attn_k_norm.weight` | 256 | F32 | 20 |

Values worth knowing (layer 0 and 3): `ssm_a` is **negative**, -72.3 to -0.019 (it is `-exp(A_log)`, so `gate = softplus(alpha + dt_bias) * ssm_a` is already a log decay);
`ssm_dt.bias` -5.3 to 15.6; every RMS-norm weight is stored multiplicatively (`attn_norm` mean 1.03, `output_norm` mean 2.63), so no "+1" is applied anywhere;
`ssm_conv1d` is within +-0.41; `ffn_gate_inp_shexp` is a 2048-vector with mean 0.

## The nextn block (MTP file only)

`blk.40` is the draft block, 20 tensors: the same ones an attention layer plus MoE carries (`attn_norm`, `post_attention_norm`, `attn_q/k/v/output`,
`attn_q_norm/k_norm`, `ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_gate_inp_shexp`, `ffn_{gate,up,down}_shexp`) plus
`nextn.eh_proj.weight` (4096, 2048: one projection over `[enorm(token embedding), hnorm(hidden)]` concatenated, embedding first),
`nextn.enorm.weight`, `nextn.hnorm.weight`, `nextn.shared_head_norm.weight` (each 2048, F32).
There is **no** `nextn.embed_tokens` and **no** `nextn.shared_head_head`: the block embeds with `token_embd.weight` and the draft head is the main `output.weight`
(`qwen35moe.cpp:583`, `:733`); the head norm is `shared_head_norm` when present, else `output_norm`.
In that file the nextn router `blk.40.ffn_gate_inp.weight` and `ffn_gate_inp_shexp.weight` are **BF16** while the trunk's are F32; the
experts are Q3_K gate/up and Q4_K down (trunk: IQ3_S / IQ4_XS).

## What was unexpected

- **The plain GGUF has no nextn block at all.** `nextn_predict_layers` is absent and `block_count` is 40. The MTP head exists only in the separate `-MTP.gguf` builds, which carry 41 blocks.
- **Gate and up are not fused** (`ffn_gate_exps` + `ffn_up_exps`), and attention layers are **not** fused (`attn_q`, `attn_k`, `attn_v`). `attn_qkv` exists, but only in the 30 GDN layers where it feeds the conv, next to a separate `attn_gate` (z). llama.cpp accepts the fused forms (`llama-model.cpp:3253-3300`, `ffn_gate_up_exps` with gate rows first, `attn_qkv` with Q|K|V rows), and the fixture and oracle support both, but no shipped file uses them.
- **`expert_weights_norm` and `expert_weights_scale` are not in the file and cannot matter**: the architecture hard-codes renormalization and never loads a scale (see above).
- **The router and the GDN `ssm_alpha` / `ssm_beta` are F32**, not BF16. `ref/load.py`'s `FORM` table in `iq_pack.py` must not assume the Flash-Next BF16 forms for them.
- **`attn_q` is 2x wide**: per head 256 query values then 256 gate values, interleaved, which is how `attn_output_gate` is stored. There is no separate gate tensor.
- **`ssm_a` carries no `.weight` suffix** and `ssm_dt` is named `ssm_dt.bias`.
- **`ffn_gate_inp_shexp` is 1-D** (a vector, producing one scalar per token), not a 2048x1 matrix.
- **Per-layer quantization is mixed** (down experts: IQ3_XXS in 37 layers and IQ4_XS in 3; the MTP block uses a different family altogether), so any pack format must carry a type per layer and per role, as `native_experts.txt` already does.
- `ref/load.py` did not know IQ2_S, IQ3_XXS, IQ3_S (gate/up/down of this file); it is extended in the same commit that adds the oracle.
