[Codebase docs](../README.md) › [Tools](README.md) › Reference model (`ref/`)

# The numpy reference model (`ref/`)

## Purpose

`ref/` is a CPU-only, numpy implementation of Qwen3.6-35B-A3B (llama.cpp architecture `qwen35moe`) used as an
**oracle**: it runs a teacher-forced forward pass and writes every intermediate tensor under the name llama.cpp's
graph gives it, so the same tensors can be compared with llama.cpp's own graph and with the engine's CUDA kernels.
It was written for the port from the pinned llama.cpp sources only (upstream Strata never published its own
oracle; [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) section 1). It has no
Flash-Next model; `ref/load.py` is model-agnostic.

| File | Lines | Role |
| --- | --- | --- |
| [`ref/load.py`](../../../ref/load.py) | 217 | `Gguf`: lazy FP32 view of a (sharded) GGUF, per-expert and per-row dequantization |
| [`ref/qwen36.py`](../../../ref/qwen36.py) | 683 | `Model`: the forward pass, the nextn (MTP) block, `Dump`, the CLI |

## `ref/load.py`

```python
from ref.load import Gguf
g = Gguf("model.gguf")              # a shard, a directory of shards, or a list
w = g.tensor("blk.0.attn_gate.weight")         # (out, in) float32
e = g.expert("blk.0.ffn_gate_exps.weight", 7)  # (out, in) of ONE expert
r = g.rows("token_embd.weight", [1, 2, 3])     # rows of a 2-D tensor
```

- `_discover_shards` (`:69`): a directory must contain exactly one shard set or one `.gguf`; a
  `-NNNNN-of-MMMMM.gguf` file finds its siblings and requires all of them.
- `Gguf` (`:113`): opens each shard with `tools/gguf_reader.GGUFFile`, maps names to shards (duplicates refused),
  `self.g` is shard 1 (metadata). `tensor()` (`:149`) returns the whole tensor reshaped to the reverse of the GGUF
  shape; `expert()` (`:191`) reads only one expert's contiguous run of a fused `[in, out, experts]` tensor;
  `rows()` (`:161`) reads individual rows (written for Flash-Next's 27 GB PLE table).
- `_dequant_flat` (`:46`): F32/F16/BF16 by hand, `Q2_0` through `tools/gguf_writer.dequantize_q2_0`, everything
  else through gguf-py `quants.dequantize`. `GEOM` (`:38-43`) lists the 23 supported types, including the i-quants
  added for Qwen3.6's UD files (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, ...).
- `ref/load.py` imports `gguf` lazily and does not locate it; `ref/qwen36.py` calls `tools/_paths.add_gguf_py()`
  first.

## `ref/qwen36.py`

### Structure

| Piece | Lines | What it is |
| --- | --- | --- |
| `HParams`, `read_hparams` | `:66-147` | What llama.cpp derives from the metadata: layers (trunk + `nextn_predict_layers`), heads, `head_dim`, `n_rot`, rope sections and base, eps, experts, SSM sizes, `full_attention_interval`; `is_recr(il)` = recurrent unless `(il+1) % interval == 0`, never for the nextn block. Fixed behaviours: `expert_weights_norm = True`, `expert_weights_scale = 0.0` (the key is never read for this architecture). |
| op helpers | `:151-254` | `rms_norm` (mean in double), `l2_norm_gdn` (`x / sqrt(sum x^2 + eps)`), `silu`, `sigmoid`, `softplus` (threshold 20), `softmax`, `mm`, `quantize_act` (ggml's Q8_K / Q8_0 activation rounding), `rope_multi` (IMRoPE, NeoX pairs `(p, p + n_rot/2)`, text positions `t = h = w = pos`, `e = 0`) |
| `Dump` | `:258-282` | writes `<stem>.f32` and `manifest.txt`, numbering repeated names `#2`, `#3`; a no-op without a directory |
| `LayerState`, `State` | `:286-297` | per-layer GDN conv state `(conv_dim, K-1)` and SSM state `(H_v, S_v, S_k)`, or attention K/V; `n_past` |
| `Model.__init__`, `opts` | `:300-321` | loads `Gguf`, the hparams and the option table (below) |
| `Model.gdn` | `:368-439` | Gated DeltaNet layer: projections, `beta = sigmoid`, `g = softplus(alpha + dt_bias) * ssm_a`, causal conv over `[state | qkv]` then SiLU, q/k l2 norm, the per-token delta-rule recurrence (decay, delta, rank-1 update, readout `/ sqrt(S)`), then `rms_norm(o) * ssm_norm * SiLU(z)` and `ssm_out` |
| `Model.attention` | `:441-501` | full attention: joint q+gate projection (`[q | gate]` per head) or fused `attn_qkv`, q/k RMS norm, IMRoPE, causal GQA softmax over carried KV with `kv head = h // (n_head / n_head_kv)`, sigmoid output gate, `attn_output`. With `mtp=True` emits graph_mtp's `mtp_*` names only. |
| `Model.moe` | `:503-561` | softmax router over all experts, stable descending argsort, top-k, weights renormalized with the sum clamped at `6.1035e-5`, SwiGLU experts (fused `ffn_gate_up_exps` supported), summed in top-k order; plus the shared expert gated by `sigmoid(ffn_gate_inp_shexp . x)` |
| `Model.forward` | `:564-595` | embedding, per layer `attn_norm` -> mixer -> residual -> `post_attention_norm` -> MoE -> residual; then `output_norm` (`h_nextn`), last-token `result_output` (and `result_output_all` with `all_logits`) |
| `Model.mtp_forward` | `:598-634` | the nextn block: `concat(rms_norm(embed) * enorm, rms_norm(h) * hnorm)`, `eh_proj`, attention + MoE block, `shared_head_norm` (else `output_norm`), head `nextn.shared_head_head` else `output.weight` else `token_embd` |
| `main` | `:637-679` | the CLI |

Weights: tensors up to 4 M elements are cached; larger dense matrices are dequantized on every use; experts are
always read one at a time (`expert_ffn_weights`, `:347`), so a 40-layer pass over a real quantized file fits in a
few GB of RAM.

### Options (`Model.opts`)

Each faithful reading has a named **rival**, the reading a port could plausibly get wrong. Flipping one is how the
negative controls are produced.

| Key | Faithful | Rival | What it checks |
| --- | --- | --- | --- |
| `gdn_gate` | `silu` | `sigmoid` | GDN output gate (sigmoid is Flash-Next's engine) |
| `head_pairing` | `tiled` (`h % H_k`) | `interleaved` (`h // (H_v/H_k)`) | v-head to k-head pairing |
| `decay` | `before` | `after` | decay before or after the rank-1 update |
| `conv` | `causal` | `flipped` | conv window direction |
| `attn_gate` | `True` | `False` | attention sigmoid output gate |
| `router_norm` | `None` (renormalize) | `False` | top-k weight renormalization |
| `rope_pairs` | `neox` | `adjacent` | RoPE pairing |
| `gqa_map` | `grouped` | `interleaved` | query head to KV head map |
| `act_quant` | `none` | `q8` | not a rival: a noise model of ggml's activation quantization |
| `force_inputs` | `None` | dir | start each layer from a reference dump's `l_out-(il-1)` |
| `force_topk` | `None` | `{layer: ids}` | use another run's expert choices |

### Command line

```
python3 ref/qwen36.py MODEL.gguf (--tokens 1,2,3 | --text hello) [--dump DIR] [--split CHUNK]
        [--opt KEY=VALUE ...] [--act-quant none|q8] [--force-topk DIR] [--force-inputs DIR]
        [--mtp] [--all-logits]
```

`--text` maps characters to their code points (valid for the synthetic 128-byte vocabulary only). `--split`
feeds the tokens in chunks, carrying state, and dumps only the last chunk. `--mtp` then runs the nextn block on
`(tokens[1:], h[:-1])` at positions 1.. and dumps to `DIR_mtp` (the pairing llama.cpp's speculative code builds).
`--all-logits` adds `result_output_all`, the head at every position, used by the runner tests. It prints the
last-token argmax, max and sum.

## How it is used

1. **Against llama.cpp on synthetic F32 files** ([qwen36-verification.md](qwen36-verification.md)):
   `tools/qwen36_crosscheck.sh` runs the oracle and `qwen36-dump` on the same fixtures and token sequences and
   compares every named tensor. Recorded result: 2,155 comparisons over 116 seam families, worst absolute error
   1.1e-4, router indices identical; each of the 8 rival readings fails at the layer that makes it
   ([oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md)).
2. **On the real files**: dequantization is checked bit-identical against ggml (`qwen36_check_dequant.py`); seam
   tolerances for the engine's parity tests come from runs with `--act-quant q8`, `--force-topk` and
   `--force-inputs` measured by `qwen36_spread.py` ("Tolerance policy" in oracle-crosscheck.md).
3. **As the reference for the engine's CUDA tests**: CMake fixture steps write synthetic GGUFs and oracle dumps
   that C++ tests read seam by seam:

| Fixture producer | Oracle invocation | Consumer test |
| --- | --- | --- |
| `cmake/qwen36_oracle_dumps.cmake` | `k.gguf` -> `o_f32/`, `o_q8/` (`--act-quant q8`) | `qwen36_seam_parity` (`src/kernels/qwen36_seam_parity.cpp`) |
| `cmake/qwen36_attn_fixtures.cmake` | `g8`, `g12`, `g2` at head_dim 256, 40 tokens | `qsa_dense_parity --fixtures` |
| `cmake/qwen36_block_fixtures.cmake` | `toy.gguf`, 8 tokens -> `o8/` | `qwen36_block_test` |
| `tests/python/qwen36_prefill_fixture.py` | `o`, `o_q8`, `neg_*` rivals | `prefill_qwen36_test` (`gdn3`, `full4`) |
| `tests/python/qwen36_mtp_fixture.py` | `--mtp`, `o`, `o_q8` | `qwen36_mtp_test` |
| `tests/python/qwen36_runner_fixture.py` | `--all-logits`, `o8`, `o11`, `o40` (+ `_q8`) | `qwen36_verify_test`, `qwen36_runner_check.py` |

4. **For tool decisions**: `qwen36_router_bf16_flips.py` reads the oracle's `attn_post_norm-N` dumps to measure the
   effect of BF16 routers.

## Model-specific notes

- Only `qwen35moe` is implemented (`read_hparams` asserts the architecture).
- Scope stated in the module docstring: one sequence, causal, no sliding window, no YaRN, F32 throughout; a
  quantized file is dequantized first, so this is "what the weights say", not a bit-exact quantized path.
- Router ties: `argsort` is stable; ggml's tie order is unspecified (oracle-crosscheck.md "Not covered").

## Tests

`python3 -I tools/test_qwen36_oracle.py` (24 unittest methods, numpy and gguf-py only). See
[python-tests.md](python-tests.md).

## Gotchas

- `--force-topk` reshapes each layer's dump to `(len(tokens), -1)`, so it assumes the reference dump covers all the
  tokens (no `--split`).
- `--text` is only meaningful with the synthetic vocabulary.
- Dump names follow llama.cpp's execution order, not build order (in the nextn block `norm-4` is the embedding
  norm); see oracle-crosscheck.md "Quirks of the comparison".

## See also

- [qwen36-verification.md](qwen36-verification.md), [python-tests.md](python-tests.md), [mtp-tools.md](mtp-tools.md)
- [docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md) (formulas with llama.cpp
  file:line), [docs/qwen36-port/step3-gdn.md](../../qwen36-port/step3-gdn.md),
  [docs/qwen36-port/step3-attn.md](../../qwen36-port/step3-attn.md),
  [docs/qwen36-port/step3-block.md](../../qwen36-port/step3-block.md),
  [docs/qwen36-port/step3-prefill.md](../../qwen36-port/step3-prefill.md)
