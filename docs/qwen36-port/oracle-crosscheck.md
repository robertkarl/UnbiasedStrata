# Qwen3.6 oracle cross-check against the pinned llama.cpp graph

Run 2026-10-11 on this branch (`qwen36-oracle`), CPU only, F32 synthetic fixtures, against `third_party/llama.cpp` at 3cf0325.
`ref/qwen36.py` (the numpy oracle) and the submodule's own graph agree on **every named tensor of every case**: 2,155 comparisons over 116 seam
families, worst absolute error **1.1e-4** (on a tensor whose values reach 3.8), the router's integer outputs identical. The printed stats of
`llama-eval-callback` agree on all 209 named nodes it prints. A reading that is wrong in any of the 8 places a port could plausibly get wrong
fails the comparison at the layer that makes it. Dequantization of the real file's types is bit-identical to ggml's (below).

## Reproduce

One command builds the submodule's `libllama` and `llama-eval-callback` (CPU, outside the repo), builds the dumper, writes the fixtures, runs
every case and exits 1 on any failed seam. About 4 seconds once built; the first build takes a few minutes.

```
export STRATA_GGUF_PY=$PWD/third_party/llama.cpp/gguf-py     # optional: tools/_paths.py finds it there anyway
tools/qwen36_crosscheck.sh /path/to/scratch [REAL-MTP.gguf]  # 2nd argument: a real qwen35moe file, for the --like-real case
python3 -I tools/test_qwen36_oracle.py                       # 21 invariant tests, no llama.cpp build needed
```

The same steps by hand, for one case (`$B` is the build directory, `$W` a scratch directory):

```
cmake -S third_party/llama.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build $B --target llama-eval-callback -j8
c++ -std=c++17 -O1 tools/qwen36_dump_llama.cpp -o $B/qwen36-dump -Ithird_party/llama.cpp/include \
    -Ithird_party/llama.cpp/ggml/include -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin

python3 tools/qwen36_synth_gguf.py $W/base.gguf                    # add --like-real REAL.gguf / --fused-gate-up / --fused-qkv / --no-nextn
T=104,101,108,108,111,32,119,111,114,108,100                         # "hello world": the fixture's vocabulary is the 128 ASCII bytes
python3 ref/qwen36.py $W/base.gguf --tokens $T --dump $W/o --mtp     # oracle dumps (+ $W/o_mtp)
$B/qwen36-dump $W/base.gguf $W/d $T 0 mtp                            # llama.cpp's graph, every named tensor in full (+ $W/d_mtp)
python3 tools/qwen36_compare_dumps.py $W/o $W/d                      # per-seam max |difference|; exit 1 on a failure

$B/bin/llama-eval-callback -m $W/base.gguf -p "hello world" -ctk f32 -ctv f32 -fa off -c 64 -t 1 -ngl 0 > $W/eval.txt 2>&1
python3 tools/qwen36_check_eval_callback.py $W/o $W/eval.txt --require-tail
```

`-ctk f32 -ctv f32 -fa off` matter: the default F16 KV cache would put half-precision rounding into attention, and `qwen36-dump` sets the same.
Both tools use one thread, so the only differences left are float32 summation order.

## Cases (all in `tools/qwen36_crosscheck.sh`)

| case | fixture | tokens | what it adds |
| --- | --- | --- | --- |
| `base_11` | default, seed 0 | 11 ("hello world") | the baseline; also the nextn block |
| `base_40` | default | 40 | longer GDN history, attention over 40 keys |
| `base_ar` | default | 40, one token per decode call | llama.cpp's autoregressive GDN path: state and KV carried through 40 calls; only the last call is dumped |
| `base_c7` | default | 40, chunks of 7 | state carried across 5 calls of 7, then 5 |
| `seed7` | seed 7 | 11 | different weights, so a coincidence in seed 0 cannot hide a bug; nextn |
| `fused_gu` | `--fused-gate-up --seed 3` | 11 | `ffn_gate_up_exps` (gate rows then up rows) for trunk and nextn; nextn |
| `fused_qkv` | `--fused-qkv --no-nextn --seed 4` | 11 | `attn_qkv` in the attention layer (Q, K, V rows in that order) |
| `gqa4_2` | `--heads 4,2 --seed 11` | 40 | **non-degenerate GQA**: 4 query heads over 2 KV heads, so `kv = h // (n_head/n_head_kv)` is tested (every other fixture has one KV head, where any mapping passes); nextn |
| `stray_scl` | `--expert-weights-scale 2.5 --seed 5` | 11 | the key is in the file; llama.cpp ignores it, so must the oracle; nextn |
| `like_real` | `--like-real <MTP build>` | 40 | rope base 1e7, real eps, interval, head ratios, and the exact tensor-name set of the real file (32 of 32 patterns); nextn |

Geometry of the default fixture: hidden 64; layers 0-2 Gated DeltaNet, layer 3 full attention; GDN 4 k-heads / 8 v-heads of 16; conv kernel 4;
attention 2 heads / 1 KV head of 32 with 8 rotated dims (25 %) and IMRoPE sections [2,1,1,0]; 8 experts, top 2, ff 32, shared ff 32; vocabulary 128;
one nextn block at index 4.

## Results

All 10 trunk cases: `0 failed` of 211 seams (215 with fused gate/up, 212 with fused qkv). The 6 nextn cases: `0 failed` of 50 (51 fused). Per seam family
(max over all layers and the first nine cases, 2,155 comparisons; the `gqa4_2` case adds 261 more, all within tolerance; "max abs value" is the largest magnitude of the llama.cpp tensor):

#### Trunk (`qwen35moe` graph, 9 cases)

| seam (all layers) | comparisons | max abs error | max abs value |
| --- | --- | --- | --- |
| `model.input_embed` | 9 | 0.0e+00 | 3.763 |
| `norm` | 126 | 1.1e-04 | 3.937 |
| `attn_norm` | 36 | 1.7e-05 | 5.397 |
| `conv_states` | 27 | 8.6e-06 | 3.773 |
| `conv_states_reshaped` | 27 | 8.6e-06 | 3.773 |
| `linear_attn_qkv_mixed` | 27 | 8.5e-06 | 4.543 |
| `qkv_mixed_transposed` | 27 | 8.5e-06 | 4.543 |
| `z` | 27 | 7.6e-06 | 4.58 |
| `beta` | 27 | 5.9e-06 | 6.075 |
| `beta_sigmoid` | 27 | 1.5e-06 | 0.9977 |
| `alpha` | 27 | 9.2e-06 | 3.173 |
| `a_softplus` | 27 | 2.4e-06 | 3.359 |
| `gate` | 27 | 8.5e-06 | 38.85 |
| `conv_input` | 27 | 8.6e-06 | 4.543 |
| `conv_state_last` | 27 | 4.8e-06 | 3.944 |
| `conv_output_raw` | 27 | 1.0e-05 | 9.484 |
| `conv_output_silu` | 27 | 1.1e-05 | 9.483 |
| `q_conv` | 27 | 1.1e-05 | 9.483 |
| `k_conv` | 27 | 9.4e-06 | 7.217 |
| `q_conv_predelta` | 27 | 2.3e-06 | 0.9935 |
| `k_conv_predelta` | 27 | 2.9e-06 | 0.9874 |
| `v_conv_predelta` | 27 | 1.1e-05 | 6.957 |
| `state_predelta` | 27 | 4.6e-06 | 2.4 |
| `new_state` | 27 | 3.9e-06 | 3.382 |
| `attn_output` | 36 | 4.1e-06 | 1.878 |
| `final_output` | 27 | 5.3e-05 | 8.531 |
| `linear_attn_out` | 27 | 1.8e-05 | 3.891 |
| `attn_residual` | 36 | 2.0e-05 | 9.001 |
| `attn_post_norm` | 36 | 1.5e-05 | 5.079 |
| `ffn_moe_logits` | 36 | 2.3e-05 | 12.4 |
| `ffn_moe_probs` | 36 | 6.8e-06 | 0.9995 |
| `ffn_moe_argsort` | 36 | identical | 0 |
| `ffn_moe_topk` | 36 | identical | 0 |
| `ffn_moe_weights` | 36 | 6.8e-06 | 0.9995 |
| `ffn_moe_weights_sum` | 36 | 2.3e-06 | 0.9999 |
| `ffn_moe_weights_sum_clamped` | 36 | 2.3e-06 | 0.9999 |
| `ffn_moe_weights_norm` | 36 | 8.3e-06 | 0.9997 |
| `ffn_moe_gate` | 36 | 1.4e-05 | 4.662 |
| `ffn_moe_up` | 36 | 1.2e-05 | 4.383 |
| `ffn_moe_swiglu` | 36 | 2.4e-05 | 10.1 |
| `ffn_moe_down` | 36 | 1.3e-05 | 5.36 |
| `ffn_moe_weighted` | 36 | 1.2e-05 | 3.552 |
| `ffn_moe_out` | 36 | 1.4e-05 | 3.635 |
| `ffn_up` | 36 | 1.5e-05 | 4.215 |
| `ffn_gate` | 36 | 1.1e-05 | 4.541 |
| `ffn_swiglu` | 36 | 2.9e-05 | 5.875 |
| `ffn_shexp` | 36 | 1.6e-05 | 3.419 |
| `shared_expert_gate` | 36 | 9.8e-06 | 5.545 |
| `shared_expert_gate_sigmoid` | 36 | 2.2e-06 | 0.9961 |
| `ffn_shexp_gated` | 36 | 6.8e-06 | 3.12 |
| `ffn_out` | 36 | 1.2e-05 | 4.451 |
| `l_out` | 36 | 2.3e-05 | 8.791 |
| `Qcur_full` | 9 | 1.4e-05 | 4.189 |
| `Kcur` | 18 | 1.1e-05 | 3.824 |
| `Vcur` | 18 | 1.4e-05 | 4.189 |
| `Qcur_reshaped` | 9 | 1.2e-05 | 3.753 |
| `Qcur_normed` | 9 | 2.2e-05 | 5.186 |
| `Kcur_normed` | 9 | 8.7e-06 | 3.824 |
| `Qcur` | 9 | 2.2e-05 | 5.186 |
| `kq` | 9 | 8.8e-05 | 22.06 |
| `kq_soft_max` | 9 | 3.0e-06 | 1 |
| `kqv` | 9 | 1.1e-05 | 3.075 |
| `attn_pregate` | 9 | 1.1e-05 | 3.075 |
| `gate_reshaped` | 9 | 1.4e-05 | 4.189 |
| `gate_sigmoid` | 9 | 3.3e-06 | 0.9768 |
| `attn_gated` | 9 | 8.2e-06 | 2.669 |
| `h_nextn` | 9 | 1.7e-05 | 3.835 |
| `result_norm` | 9 | 3.6e-06 | 3.259 |
| `result_output` | 9 | 7.3e-06 | 7.799 |
| `ffn_moe_gate_up` | 4 | 6.3e-06 | 3.595 |
| `wqkv` | 1 | 6.7e-06 | 3.492 |

#### Nextn block (`graph_mtp`, 5 cases that carry a nextn block)

| seam (all layers) | comparisons | max abs error | max abs value |
| --- | --- | --- | --- |
| `mtp_tok_embd` | 5 | 0.0e+00 | 3.372 |
| `norm` | 35 | 1.7e-05 | 3.956 |
| `mtp_enorm` | 5 | 4.8e-07 | 3.57 |
| `mtp_hnorm` | 5 | 1.5e-05 | 4.633 |
| `mtp_concat` | 5 | 1.5e-05 | 4.633 |
| `mtp_eh_proj` | 5 | 8.2e-06 | 3.742 |
| `mtp_attn_norm` | 5 | 8.7e-06 | 4.298 |
| `mtp_Qcur_full` | 5 | 8.8e-06 | 4.519 |
| `mtp_Qcur_normed` | 5 | 1.1e-05 | 4.886 |
| `mtp_gate` | 5 | 7.5e-06 | 4.387 |
| `mtp_Kcur_normed` | 5 | 6.6e-06 | 3.528 |
| `mtp_Vcur` | 5 | 8.6e-06 | 4 |
| `kq` | 5 | 6.1e-05 | 23.47 |
| `kq_soft_max` | 5 | 2.1e-06 | 1 |
| `kqv` | 5 | 4.2e-06 | 3.701 |
| `mtp_attn_pregate` | 5 | 4.2e-06 | 3.701 |
| `mtp_attn_out` | 5 | 3.8e-06 | 1.818 |
| `mtp_attn_residual` | 5 | 9.3e-06 | 4.383 |
| `mtp_attn_post_norm` | 5 | 7.5e-06 | 4.184 |
| `ffn_moe_logits` | 5 | 1.6e-05 | 11.01 |
| `ffn_moe_probs` | 5 | 2.7e-06 | 0.9975 |
| `ffn_moe_argsort` | 5 | identical | 0 |
| `ffn_moe_topk` | 5 | identical | 0 |
| `ffn_moe_weights` | 5 | 2.7e-06 | 0.9975 |
| `ffn_moe_weights_sum` | 5 | 1.5e-06 | 0.9993 |
| `ffn_moe_weights_sum_clamped` | 5 | 1.5e-06 | 0.9993 |
| `ffn_moe_weights_norm` | 5 | 2.8e-06 | 0.9982 |
| `ffn_moe_gate` | 5 | 7.7e-06 | 4.102 |
| `ffn_moe_up` | 5 | 7.6e-06 | 3.967 |
| `ffn_moe_swiglu` | 5 | 1.3e-05 | 6.547 |
| `ffn_moe_down` | 5 | 1.2e-05 | 4.218 |
| `ffn_moe_weighted` | 5 | 5.1e-06 | 2.526 |
| `mtp_ffn_moe_out` | 5 | 5.9e-06 | 2.647 |
| `ffn_up` | 5 | 8.7e-06 | 3.683 |
| `ffn_gate` | 5 | 7.2e-06 | 4.162 |
| `ffn_swiglu` | 5 | 1.5e-05 | 5.923 |
| `mtp_ffn_shexp` | 5 | 8.3e-06 | 2.811 |
| `mtp_shared_expert_gate_sigmoid` | 5 | 2.1e-06 | 0.9887 |
| `mtp_ffn_shexp_gated` | 5 | 4.7e-06 | 2.501 |
| `mtp_ffn_out` | 5 | 6.8e-06 | 3.119 |
| `mtp_post_ffn` | 5 | 1.6e-05 | 4.947 |
| `h_nextn` | 5 | 1.4e-05 | 3.562 |
| `mtp_shared_head_norm` | 5 | 1.4e-05 | 3.562 |
| `result_output` | 5 | 2.1e-05 | 7.916 |
| `ffn_moe_gate_up` | 1 | 3.7e-06 | 3.81 |

The largest absolute error is 1.1e-4 on `norm-2#2`, the RMS norm of a GDN layer's recurrence output. Those values are small, so the norm amplifies the
float32 summation-order difference of the recurrence by about 1 / rms; relative to the tensor's largest value it is 2.8e-5. The router's
`ffn_moe_argsort` and `ffn_moe_topk` are bit-identical in every case.

### `llama-eval-callback` printed stats

`tools/qwen36_check_eval_callback.py` reconstructs which indices the tool prints (first and last 3 of every dimension, `%12.4f`) and its float32 `sum`:
209 named nodes through `result_output`, 0 failed; worst printed value 6.2e-5 apart (that is the 4-decimal rounding, not a difference), worst `sum`
3.9e-7 of the tensor's L1 mass. `kq` and `kq_soft_max` are skipped there because llama.cpp pads their KV axis to its cache size (256); the full-tensor
comparison crops the padding and requires it to be exactly zero for `kq_soft_max`.

### Negative controls: the comparison fails when the oracle reads the graph wrongly

Each row runs the oracle with one `Model.opts` entry flipped to its rival (`ref/qwen36.py --opt KEY=VALUE`) against the same llama.cpp dump of `base_11`.
All 8 fail, at the layer and seam that makes the mistake; `tools/test_qwen36_oracle.py` requires every one to move its seam by more than 5 % relative L1.

| rival reading | first failing seam | seams failing / 211 | change at that seam (relative L1) | change in logits |
| --- | --- | --- | --- | --- |
| GDN output gate `sigmoid(z)` (the Flash-Next engine) instead of `SiLU(z)` | `final_output-0` | 179 | 1.07 | 0.92 |
| v-head h paired with k-head `h // 2` instead of `h % 4` | `new_state-0` | 182 | 1.03 | 0.84 |
| decay applied after the rank-1 update instead of before | `new_state-0` | 182 | 0.16 | 0.30 |
| conv window applied in the opposite direction | `conv_output_raw-0` | 189 | 1.23 | 1.05 |
| attention output gate skipped | `attn_gated-3` | 32 | 1.03 | 0.12 |
| top-k weights not renormalized | `ffn_moe_weighted-0` | 152 | 0.14 | 0.12 |
| query head h reads KV head `h % n_kv` instead of `h // (n_head/n_kv)` (on `gqa4_2`; invisible with one KV head) | `kq-3` | 36 | 0.74 (at `kqv-3`) | 0.13 |
| RoPE on adjacent pairs (2p, 2p+1) instead of (p, p + n_rot/2) | `Qcur-3` | 37 | 0.20 (at `kqv-3`) | 0.04 |

## Formulas pinned (file:line in `third_party/llama.cpp`)

**Gated DeltaNet** (`qwen35moe.cpp:359-492`; `qwen35moe.cpp:20-26` decides which layers are recurrent). CPU always takes the fused op
(`llama-context.cpp:232-233`), whose scalar loop is the reference; the chunked and autoregressive graphs in `delta-net-base.cpp:16-371` compute the same thing.

| step | formula | source |
| --- | --- | --- |
| beta | `sigmoid(ssm_beta x)` | `qwen35moe.cpp:382-387` |
| log decay | `g = softplus(ssm_alpha x + ssm_dt.bias) * ssm_a`, `ssm_a < 0` in the file; softplus is `x > 20 ? x : log1p(exp(x))` | `qwen35moe.cpp:389-398`, `ggml-cpu/unary-ops.cpp:80-82` |
| conv | per channel, `y[t] = sum_i w[c][i] * in[c][t+i]` over `[state (K-1) \| qkv]`, then `silu`; new state = last K-1 inputs | `qwen35moe.cpp:402-421`, `delta-net-base.cpp:449-525`, `ggml-cpu/ops.cpp:9703-9754` |
| split | channels `[0,2048)` q, `[2048,4096)` k (16 heads of 128), `[4096,8192)` v (32 heads of 128); head h is a contiguous 128-run | `qwen35moe.cpp:424-444` |
| q, k norm | `x / sqrt(sum(x^2) + eps)`, eps 1e-6, per head: `rms_norm(x, eps/n) * 1/sqrt(n)` | `models.h:14-18`, `qwen35moe.cpp:451-454` |
| head pairing | v-head h uses k-head `h % 16` (tiling, not blocks) | `qwen35moe.cpp:462-466`, `ggml-cpu/ops.cpp:10971-10972` |
| state | `S[j][i]`, j = v index, i = k index, per v-head, `ne [S_k, S_v, H_v]`, zero at sequence start | `ggml-cpu/ops.cpp:10999`, `qwen35moe.cpp:411-413` |
| step | `S *= exp(g)`; `d = (v - S k) * beta`; `S += d (x) k`; `o = (S q) / sqrt(S_v)` | `ggml-cpu/ops.cpp:11012, 11016-11020, 11023-11025, 11028-11032` |
| output | `rms_norm(o) * ssm_norm * SiLU(z)`, per head; then `ssm_out` | `qwen35moe.cpp:267-276, 475-486` |

**Full attention** (`qwen35moe.cpp:278-357`): `attn_q` is `[n_head, 2, head_dim]` per token, so query is the first `head_dim` of each head's `2*head_dim` slice and the gate the second
(`:299-317`); RMS norm on q and k per head (`:305-311`); IMRoPE on the first 64 of 256 dims, pairs `(p, p+32)` (`ggml-cpu/ops.cpp:6205-6213`), theta per sector
from `t,h,w,e` positions = `pos,pos,pos,0` for text (`llama-graph.cpp:131-141`, `ops.cpp:6026-6036`); scale `1/sqrt(256)` (`:340`); causal softmax; GQA groups are
contiguous (`kv head = h / (n_head / n_head_kv)`); output `* sigmoid(gate)` then `attn_output` (`:347-354`).

**Router and experts** (`qwen35moe.cpp:494-548`, `llama-graph.cpp:1993-2355`): logits `ffn_gate_inp x`; `softmax` over all 256 (`:2043`); `argsort_top_k` of the probabilities, descending (`:2109`);
weights are the unbiased probabilities of the chosen experts (`:2123`); **renormalized to sum 1** with the sum clamped at 6.1035e-5 (`:2134-2148`), because `qwen35moe.cpp:506`
passes `norm_w = true`; `expert_weights_scale` is `0.0` (`llama-hparams.h:124`, never loaded for this architecture), so no scale (`:2149`). Experts: `silu(gate x) * (up x)` (`:2246`),
`down`, times weight, summed in top-k order (`:2339-2345`). Shared expert: same SwiGLU with `ffn_{gate,up,down}_shexp`, times `sigmoid(ffn_gate_inp_shexp . x)` (a scalar per token), added to the routed sum
(`qwen35moe.cpp:516-545`).

**Nextn / MTP** (`qwen35moe.cpp:550-741`, a real graph in this commit, run by llama.cpp's `LLAMA_CONTEXT_TYPE_MTP` context): `concat(rms_norm(embed(token)) * enorm, rms_norm(h) * hnorm)` (embedding first),
`eh_proj`, then an attention block (`attn_norm`, joint q+gate, q/k norm, IMRoPE, causal over the MTP block's own KV, sigmoid gate, `attn_output`, residual), `post_attention_norm`, the same MoE
plus gated shared expert, residual, `shared_head_norm` (else `output_norm`), and the main `output.weight` (there is no `nextn.shared_head_head` in shipped files). The pairing the speculative code builds
(`common/speculative.cpp`, "shift the tgt embeddings to the right by one position"): token k with the trunk hidden state of position k-1, at position k; the cross-check does the same (`tokens[1:]`, `h[:-1]`, positions 1..).
The trunk's `h_nextn` is the output-normed hidden state of every input token (`qwen35moe.cpp:230-233`).

## Quirks of the comparison that a later reader will trip over

- **Repeated names.** `cb()` renames the tensor, so `Kcur-3` and `Vcur-3` each exist twice (projection, then the reshaped or rope'd tensor), and every `build_norm` sets `norm-<layer>` on its unweighted result,
  so a layer has up to four. The dumps number repeats `#2`, `#3` in **execution order**, which is the graph's dependency order, not build order: in the nextn block `ggml_concat(e_norm, h_norm)` pulls
  `e_norm` first, so `norm-4` is the embedding norm. The oracle emits in that order. A later `cb` on the same tensor overwrites an earlier name (`v_conv` is only ever visible as `v_conv_predelta`, `post_moe` as `l_out`).
- **Views read before their copy.** `conv_state_update-N` is a view of the cache that llama.cpp evaluates before the copy that writes it; it shows the old state. The new state is `conv_state_last-N`; the comparison ignores the former.
- **`kq` is padded** to the KV cache size (256 here); `result_norm` and `result_output` hold the last token only because `llama_batch_get_one` requests logits for the last position; `h_nextn` holds all tokens.
- **`llama-eval-callback` loses the end of its own output.** Its asynchronous logger drops its queue at exit, so the tail (sometimes through `result_output`) is cut off in a fraction of runs. The checker discards an unterminated block and
  `--require-tail` fails unless the printout reaches `result_output`; `qwen36_crosscheck.sh` retries up to 8 times. `tools/qwen36_dump_llama.cpp` writes files and is not affected.
- **Tokenizer.** The fixture's vocabulary is the 128 ASCII byte tokens (GPT-2 byte-level BPE with a single dead merge), so `-p "text"` yields token id = ASCII code and `llama-eval-callback` can be driven by text.
- **Stray metadata.** A `qwen35moe.expert_weights_scale` key changes nothing (case `stray_scl`); `expert_weights_norm` is not read either. `llama.cpp` warns about "unused tensor blk.4.*" when the nextn block is present and `--mtp` is not requested; harmless.

## ref/load.py dequantization against ggml (real file)

`tools/qwen36_check_dequant.py MODEL.gguf $B/qwen36-dequant` (the helper, `tools/qwen36_dequant_ggml.cpp`, calls ggml's own `to_float` on the raw bytes) on
`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`: **max |difference| 0.0 (bit-identical)** for `token_embd.weight` rows (Q6_K), `output.weight` and a dense Q6_K block, whole experts of
`blk.0.ffn_gate_exps` (IQ2_S), `blk.0.ffn_down_exps` (IQ3_XXS) and `blk.34.ffn_down_exps` (IQ4_XS), two experts each. Independently, `model.input_embed` from the real llama.cpp graph
(a `get_rows` of the Q6_K table, exact dequantization) equals the oracle's rows with max error 0.0 over the 4 smoke-test tokens. So every real-file difference below comes from activation
quantization and summation order, not from the weights.

## Tolerance policy for parity tests on the real file

The oracle is F32 weights x F32 activations. llama.cpp and the CUDA paths round activations to Q8 (`Q8_K` blocks of 256 for K-quant and IQ weights, `Q8_0` blocks of 32 for Q4_0/Q8_0/IQ4_NL)
before each quantized matmul; F32 weights (router, `ssm_alpha`/`ssm_beta`, conv) are not affected. On the real file that moves seams by percents and flips near-tied expert picks
(12 tokens: 1,895 of 3,840 top-k slots differ position-wise once errors have compounded). Therefore:

1. **Teacher-force expert selection** in any real-file parity test: `ref/qwen36.py --force-topk DIR` takes the choices from a dump's `ffn_moe_topk-<layer>.f32`
   (the CUDA test feeds the oracle's own, or the engine's). Exact top-k agreement is checked only on the F32 fixtures (`tools/qwen36_crosscheck.sh`, where it is bit-identical in every case).
2. **Compare per layer, not accumulated**: `--force-inputs DIR` starts every layer from the reference's `l_out-(il-1)`, so each seam's error is that layer's own. Accumulated error over 40
   layers is 5-30x larger (table below, first block) and is not a usable tolerance.
3. **Use these per-seam-class relative tolerances** (`||oracle - reference|| / ||reference||`, 12 tokens, 40 layers, topk and inputs forced). Column "F32 oracle" is the plain oracle; "Q8 oracle" is `--act-quant q8`:

| seam class | F32 oracle median / max over layers | Q8 oracle median / max |
| --- | --- | --- |
| `attn_norm` (input is the forced residual) | 5e-8 / 7e-8 | 5e-8 / 7e-8 |
| `linear_attn_qkv_mixed` (GDN in-projection) | 1.4e-2 / 1.8e-2 | 6.9e-7 / 7.5e-7 |
| `linear_attn_out` (GDN output) | 3.8e-2 / 6.6e-2 | 7.8e-5 / 3.7e-4 |
| `attn_output` (full attention, 10 layers) | 4.2e-2 / 6.3e-2 | 1.1e-4 / 6.7e-4 |
| `ffn_moe_out` | 2.9e-2 / 5.3e-2 | 2.3e-3 / 5.3e-3 |
| `ffn_out` (routed + shared) | 3.0e-2 / 4.7e-2 | 2.1e-3 / 4.5e-3 |
| `l_out` (layer output) | 6.9e-3 / 2.7e-2 | 1.8e-4 / 1.4e-3 |
| `result_output` (logits, last token) | 1.9e-2 | 4.0e-7 |

   Against an F32-activation engine path, use the F32 column with a 2x margin (for example `qkv_mixed` 4e-2, `gdn_out`/`attn_out` 1.3e-1, `moe_out` 1.1e-1, `l_out` 5e-2, logits 4e-2). Against a path that
   rounds activations to Q8 the way ggml does, use the Q8 column with a 4x margin.
4. **Q8 mode is cheap and closes the gap**: `quantize_act` (about 25 lines) reproduces ggml's `quantize_row_q8_K_ref` / `quantize_row_q8_0_ref` and leaves 1e-6 to 1e-3 instead of 1e-2 to 1e-1; at layer 0
   the GDN in-projection goes from 7.6e-3 to 6e-7. What remains in `ffn_moe_out` (2e-3) is the IQ2_S / IQ3_XXS integer dot product against float summation. For the last three rows of that
   table the Q8 oracle is a better reference than the F32 one whenever the engine under test quantizes activations.
5. The accumulated spread, for reference (`tools/qwen36_spread.py`, no forcing at all, medians over layers; F32 / Q8): `attn_norm` 1.3e-1 / 1.1e-1, `gdn_out` 1.9e-1 / 1.6e-1, `moe_out` 3.4e-1 / 2.8e-1,
   `l_out` 5.6e-2 / 4.5e-2, logits 9.8e-2 / 5.9e-2; with only the top-k forced: `l_out` 2.1e-2 / 1.3e-2, logits 3.1e-2 / 3.2e-2. Greedy next-token and the top-5 ids still agree. Q8 mode does not close this one: chaotic
   amplification through 40 layers, plus the attention layers (layer 3 alone takes `l_out-2` from 7e-3 to `attn_output-3` 2e-2), dominate, which is why the per-layer protocol above is the one to use.

## Not covered

- **Quantized numerics.** Everything above is F32 weights. On the real file (below) llama.cpp quantizes activations to Q8 for the matmuls, so seams agree to about 1e-2 relative, and near-tied router scores flip.
- Real-file smoke, 4 token ids (`9707,11,1879,0`) through `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, oracle against `qwen36-dump` (8 threads): the seams computed from F32 weights
  (`beta`, `alpha`, `a_softplus`, `gate`) agree to 2e-5 abs and every projection with a quantized weight to about 1e-2 relative (layer 0 `linear_attn_qkv_mixed` 0.20 of 29.8); the last-token logits
  correlate 0.993, argmax (271) and the top 5 ids are identical, and the router picks different experts in 6 of 32 (token, slot) choices in layer 0, 5 in layer 3 and 12-23 in layers 30-39. That is a plausibility check that the
  oracle reads the real tensors in the real conventions (dequantization of IQ2_S / IQ3_XXS / IQ4_XS / Q6_K, 1-D `ffn_gate_inp_shexp`, negative `ssm_a`), not a parity claim.
- YaRN / rope scaling (the file has none), sliding windows (none), multimodal position ids (text-only: `t = h = w = pos`, `e = 0`), batches of more than one sequence, GPU kernels, F16 KV.
- Ties in the router: `argsort` here is stable descending; ggml's tie order is unspecified. Random fixtures have no ties; the engine's router kernel needs its own tie test (see `router_top10_parity.cpp`).
