[Codebase docs](../README.md) › [Core](README.md) › Layer forward pass

# One layer's decode forward pass

## Purpose

`layer.hpp` / `layer.cpp` compose the parity-tested kernels into one transformer block for **one token**: the mixer
(Gated DeltaNet or attention), the residual handling (hyper-connections or a plain residual), the router, the shared
expert and the combination with the routed experts that someone else computed. These functions are what the
session's graphs capture ([session.md](session.md), [cuda-graphs.md](cuda-graphs.md)); the T-token verify window
re-implements the same arithmetic per row ([verify.md](verify.md)).

Files covered: [include/strata/core/layer.hpp](../../../include/strata/core/layer.hpp),
[src/core/layer.cpp](../../../src/core/layer.cpp). (Note: much of `layer.cpp` is written with several statements per
physical line, so line numbers below point at the start of a function.)

## The block, as captured

```
block_layer_pre[l]   (layer.cpp:1333)
   hc > 1:  [l == 1: PLE block on R]  gr_read(attn) -> mixer -> gr_write(attn) -> gr_read(ffn)
   hc = 1:  rms_norm(R)*attn_norm     -> mixer -> R += out    -> rms_norm(R)*post_attention_norm
            -> moe_route (router GEMV, top-k, DOORBELL publish + ring)
            -> moe_shared (shared expert; "early", default on)
   ---- host or device wait: routed experts fill `parts` (k x n_embd) ----
block_layer_post[l]  (layer.cpp:1485)
            -> moe_combine_parts(parts, weights, shared) -> gr_write(ffn) | R += out  [-> control vector]
```

`block_layer` (`layer.hpp:615`) is `pre` followed by `post`, used by the uncaptured `session_token`. The split
exists because `moe_combine` multiplies `parts` by **this** layer's router weights, so `parts` must be this layer's
experts; the header's "THE MOE, SPLIT AT THE ROUTER" section records the bug (L100) where a one-layer-delayed hand-off
produced fluent but wrong tokens.

Stage numbering (used by `half`, `stage_prefix` and the stage timers): 0 read (attn), 1 mixer, 2 write (attn),
3 read (ffn), 4 router (+ early shared expert), 5 combine, 6 write (ffn). `half == 1` runs 0-2, `half == 2` runs
3-4, `stage_prefix == k` runs stages `0..k-1`; these exist only for measurement captures.

## Mixer 1: Gated DeltaNet (`gdn_layer`, `layer.hpp:99`, `layer.cpp:224`)

Reference order (`ref/gdn.py::gdn_decode`, quoted in the header; `ref/` in this checkout holds only `load.py` and
`qwen36.py`):

1. Activation images: Q8_K and Q8_0 of `mixed`, only when a canonical (non-native) projection needs them.
2. `qkv = attn_qkv @ x` (`ssm_conv_channels` wide: q | k | v, 128-wide heads).
3. Conv step over the rolling `conv_state`, then SiLU on the **whole** conv output (before the split).
4. L2-norm q and k (not v). Legacy path pre-scales q by `1/sqrt(S)`; native path applies it after the readout.
5. `alpha`, `beta` from the BF16 `ssm_alpha` / `ssm_beta` projections; `beta = sigmoid`, `gate = softplus(alpha + dt) * ssm_a`.
6. Recurrent step on `state` (`S x h_v x S`, transposed relative to the reference).
7. `z = attn_gate @ x`; `y = rms_norm(o) * ssm_norm * act(z)` with `act = g.gdn_gate()` (sigmoid / SiLU).
8. `out = ssm_out @ y`.

With the native contract (`native_gdn_enabled()`, `native_bf16_projections`, `S == 128`, `d_conv == 4`) steps 3-4
become one kernel (`fused_gdn_conv_l2`), 5 another (`fused_gdn_ab`), and 6-7 one (`fused_gdn_step_norm`); toggled by
`layer_set_fused_gdn`. Buffers: `GdnBuffers` (`layer.hpp:59`), one shared scratch set; `state` and `conv_state`
are re-pointed per layer by the session (`gdn_point_at`).

## Mixer 2: attention (`qsa_layer`, `layer.hpp:386`, `layer.cpp:901`)

Per token at position `pos` (with `pos_base` the sequence's first cell position):

1. Activation images (Q8_K/Q8_0, BF16 for the indexer when needed).
2. **Step staging**: `qsa_step_fill` computes the per-token counts into the state's pinned `host_step` / `host_pos`,
   which are copied to `st.step` / `st.pos_dev` (by a kernel reading mapped memory when `publish_kernel` is on, else
   two `cudaMemcpyAsync`). The per-token values are data, never kernel arguments, so the layer is capturable.
3. Indexer raw key (Flash-Next only): `indexer.k_proj @ x`, appended before any norm.
4. `k`, `v` projections; RMS-norm + RoPE on k (native or NeoX table form).
5. Append K/V to the cache in the state's format (fp16, int8, q4_0, k8v4; see [session.md](session.md)), with the
   Walsh-Hadamard rotation (`fwht256_inplace_cuda`) for rotated formats. Indexer append + pooling per block.
6. `q_full = attn_q @ x` (`2 * head_dim` per head: q then gate, split by halves), copy q out, norm + RoPE.
7. Indexer query (Flash-Next only), norm + RoPE.
8. Attention:
   * **Dense (no indexer, Qwen3.6):** `qsa_decode_attn_step` (split-K, reads the pools directly) over every cell;
     `ids == nullptr` is the identity. The gather path and the short flash adapter are refused for dense.
   * **Sparse QSA (Flash-Next):** block scores + radix top-k over blocks (`fast_select`, default) or row scores +
     `topk_512_step`; `qsa_kv_resolve` makes streamed blocks resident; then split-K attention (`fast_attn`,
     default) or gather + `qsa_attend_step`.
9. Undo the rotation on the output if rotated; output gate `attn * sigmoid(gate)`; `out = attn_output @ attn32`.

Capacities (`max_blocks`, `cap`) come from the state, not from this token, so grid sizes baked into a graph stay
valid as the sequence grows. Buffers: `QsaBuffers` (`layer.hpp:344`), shared by all attention layers; state:
`QsaState` (`layer.hpp:206`), one per attention layer.

## The MoE half

| Function | Line | What it does |
| --- | --- | --- |
| `moe_route` | `layer.hpp:169`, `layer.cpp:356` | BF16 router GEMV (`project_bf16`, row-split), then top-k: `native_router_top10` only for the 512 x top-10 shape, otherwise the generic `router_top10` (softmax over all experts, stable top-k, renormalised). Then the doorbell: `doorbell_publish` copies `x`, `ids`, `weights` to mapped memory and rings `d_seq` (one kernel; or three memcpys + `doorbell_ring` with publish-kernel off) |
| `moe_route_window` | `layer.hpp:172` | T tokens at once for the verify window: one multi-column BF16 GEMV + one top-k launch, bitwise per token; falls back to per-token `moe_route` for the native router or when `STRATA_ROUTE_PER_TOKEN` is set |
| `moe_shared` | `layer.hpp:186`, `layer.cpp:419` | The shared expert (gate/up/down, mixed quant types, native or canonical) times its scalar gate `ffn_gate_inp_shexp` (must be engine form BF16), into `b.shared` |
| `moe_combine_parts` | `layer.hpp:188` | `out = sum_i weights[i] * parts[i] + shared` (native or canonical combine kernel) |
| `moe_finish` / `moe_layer` | `layer.hpp:180`, `:140` | shared + combine; route + finish |

`layer_set_shared_early(true)` (default; `--shared-late` turns it off) runs `moe_shared` at the end of `pre[l]`,
after the ring, so the GPU computes it while the host runs the CPU experts; `post[l]` then only combines.

Where the routed experts come from is not decided here: `parts` is filled by the CPU pool, the VRAM-resident expert
kernels and other tiers through the session's `PoolFn` / `HitFn` callbacks or the token graph's device hit path
([session.md](session.md); implementations in [experts/](experts/README.md)).

## Residual handling

**Hyper-connections (`hc = 4`, Flash-Next).** `R` is `hc x n_embd`. `gr_read` (norm, low-rank down/up, injection)
produces the mixer input `mixed` and a per-stream `inject`; `gr_write` adds `block_out` back into the streams.
The six `hc_*` weights per half are checked to be F32 (norm) and BF16 (down, up, inject). With
`layer_set_fused_gr(true)` the read is one fused kernel that also applies the previous half's pending write
(`FusedGrArgs.apply`), so `R` lags by one write between halves; it is materialised before the layer-1 PLE and after
the last layer (`block_layer_post`, `layer == n_layers - 1`).

**Plain block (`hc = 1`, Qwen3.6).** `plain_block_pre` / `plain_block_post` (`layer.cpp:1258`, anonymous
namespace) implement llama.cpp's `qwen35moe` pre-norm block:
`x = rms_norm(R) * attn_norm -> mixer -> R += out; x = rms_norm(R) * post_attention_norm -> MoE -> R += out`.
The helpers `plain_norm_rows` and `plain_residual_rows` (`layer.hpp:511`, `:513`) take a row count so the verify
window (T rows) and single-token path (1 row) share kernels. There is no pending write. The norm weights must be
F32 of `n_embd` elements (`plain_norm_weight`). A supplied PLE run is refused at `hc = 1`.

**Control vectors.** After the FFN write, `cvec_apply` adds a control vector to `R` for covered layers (both block
types; `--control-vector-scaled`).

## PLE (Flash-Next layer 1)

`PleRun` (`layer.hpp:568`) bundles the host-mapped n-gram table, weights, the token window, the device history
and scratch. The host-side parts - n-gram hash and the 16-row table gather - **cannot** be inside a captured graph
(they would be frozen at capture time), so the driver calls `ple_issue_token` / `ple_finish_token` (or
`ple_stage_token`, both at once) once per token before the graphs (`layer.cpp:1456-1478`). Inside `pre[1]`,
`ple_block` updates `R` in place and `ple_history_advance` shifts the row-fastest history.

## The two ends of a token

* `embed_row` (`layer.hpp:527`, `layer.cpp:1077`): `NativeEmbed::gather_one` when installed; otherwise a column
  dequant of the canonical S2/S4/S8 `token_embd.weight` that must stay bit-identical to `s_gemv_q8k_split`'s decode.
* `lm_head_mix` (`layer.cpp:1132`): `hc = 1`: `rms_norm(R) * output_norm`; `hc > 1`: `gr_read` with
  `output_hc_*` and no injection. `lm_head` (`layer.cpp:1167`) then quantizes to Q8_K and runs `output.weight`
  canonically; the native alternative is `NativeHead::run` ([dense-and-head.md](dense-and-head.md)).

## The doorbell (`struct Doorbell`, `layer.hpp:409`)

Five mapped-pinned regions (`doorbell_init`, `layer.cpp:1067`): `x_f` (n_embd), `ids` (k), `weights` (k), `seq` and
`flag` (one `uint32` each), each with a host and a device pointer. The device publishes and rings `seq` inside the
graph; the host polls `h_seq`. In the token graph the host answers by writing `h_flag`, which a device kernel
(`doorbell_wait`) spins on. `doorbell_reset` zeroes both counters.

## Process-wide switches (set before capture)

| Setter | Default | CLI (generate.cpp) |
| --- | --- | --- |
| `layer_set_native_bf16` | off | `--native-bf16`, `--native` |
| `layer_set_native_flash_attn_short` | off | `--native-flash-attn-short` (context <= 256, diagnostic) |
| `layer_set_shared_early` | on | `--shared-late` turns off |
| `layer_set_fused_gr` | off | on with GR native MMVF unless `--no-fused-gr`, `--gpu-stages` or layer dumps |
| `layer_set_fast_attn` | on | `--no-fast-attn` |
| `layer_set_publish_kernel` | on | `--no-publish-kernel` |
| `layer_set_fused_gdn` | on | `--no-fused-gdn` |
| `layer_set_fast_select` | on | `--no-fast-select` |
| `qsa_set_kv_*` | fp16, no streaming | `--kv`, `--kv-resident` (see [session.md](session.md)) |

`layer_verify_compatible(g, why)` (`layer.cpp:477`) says whether the current switches are the configuration the
verify window reproduces bit for bit: native BF16 on, fused GR (only at `hc > 1`), fused native GDN, split-K
attention, and with an indexer block selection and the native indexer.

Environment: `STRATA_ROUTE_PER_TOKEN`, `STRATA_ROUTE_PROJ_PER_TOKEN` (window routing A/B), `STRATA_KV_RING_OFF`,
`STRATA_KV_MAIN_OFF` (KV streaming A/B).

## Debug and measurement

* Per-stage timing (`stage_timing_enable`, `stage_timing_report`, `stage_mark_begin/end`; `layer.hpp:332-341`):
  event pairs per (layer, slot), only meaningful under `--no-capture` because event records inside a capture are
  dropped. Slots 16-17 are for marks around graph launches.
* `BlockBuffers::dump` (`layer.hpp:440`): a host buffer the halves copy their outputs into (the "C1 oracle"), issued
  inside the capture so it replays per layer.

## Tests

* `qwen36_block_test` (`tests/cuda/qwen36_block_test.cpp`): the plain block through the engine's own functions on the
  synthetic Qwen3.6 fixture against the oracle's dumps, plus the row-count bitwise property; needs CUDA and numpy,
  skip 77. Notes: [docs/qwen36-port/step3-block.md](../../qwen36-port/step3-block.md).
* Kernel parity tests in `src/kernels/*_parity.cpp` (`gr_parity`, `qsa_parity`, `shared_expert_parity`,
  `ple_parity`, ...) cover the components; see the kernels area.

## Gotchas

* Anything per-token must reach a captured layer as **data in a fixed buffer** (`host_step`, `host_pos`, the
  doorbell), never as a kernel argument or a stack address.
* `post[l]` reads `bb.mixed` / `bb.inject` exactly as `pre[l]` left them; nothing that touches the block buffers
  (in particular `pre[l+1]`) may run between them.
* The `qsa_kv_format` helper `exit(1)`s if a K8V4 state reaches the KV block movers (`layer.hpp:302`).

## See also

* [session.md](session.md), [cuda-graphs.md](cuda-graphs.md), [verify.md](verify.md), [model-geometry.md](model-geometry.md).
* [docs/qwen36-port/step3-gdn.md](../../qwen36-port/step3-gdn.md), [step3-attn.md](../../qwen36-port/step3-attn.md), [step3-experts.md](../../qwen36-port/step3-experts.md).
