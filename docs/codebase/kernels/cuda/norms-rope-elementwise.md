[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Norms, RoPE, elementwise, PLE

# Hyper-connections, norms, RoPE, elementwise glue, control vectors and PLE

## Purpose

This page covers the kernels around the mixers and the MoE:

- the hyper-connection (HC) residual of Flash-Next: `gr_read` collapses the `hc` = 4 residual streams into the
  mixer input and an injection, and `gr_write` adds the block output back into every stream;
- RoPE: host-built float64 tables with the scaling folded in, the analytic pinned kernel, the vision path's M-RoPE
  position table, and the rope-scaling configuration;
- elementwise glue: gates, conversions, SiLU, weighted RMSNorm, the embedding gather, the doorbells and
  mapped-memory copies that hand work to the CPU expert pool;
- control vectors on the residual stream;
- the PLE (per-layer n-gram embedding) block of Flash-Next.

Files: `src/kernels/cuda/gr.cu`, `fused_gr.cu`, `native_gr_norm.cu`, `native_gr_postops.cu`, `rope.cu`,
`native_rope.cu`, `elementwise.cu`, `cvec.cu`, `ple.cu`, `native_ple_postops.cu`; headers `gr.hpp`, `fused_gr.hpp`,
`native_gr_norm.hpp`, `native_gr_postops.hpp`, `rope.hpp`, `rope_scaling.hpp`, `mrope.hpp`, `native_rope.hpp`,
`elementwise.hpp`, `cvec.hpp`, `ple.hpp`, `native_ple_postops.hpp`, `ngram.hpp`.

## Hyper-connection read and write (Flash-Next only)

Geometry `GrShapes` (`gr.hpp:42-46`): n_embd 2560, hc 4, hc_lr 320. The weights are BF16 in manifest layout with no
permutation (`gr.hpp:81-84`): `w_down[k * hc*n_embd + i]`, `w_up[i * hc_lr + k]`, `w_inject` (hc x hc*n_embd, null
for the final mixer). `w_norm` is stored as `(1 + w)` by the converter.

```
xn       = R * w_norm * rsqrt(mean_c(R_c^2) + eps)          per stream c
lo       = silu((xn . w_down[k]) / hc)                      division by hc inside the silu
mixed[d] = mean_c( xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo) )
inject   = xn . w_inject                                    (absent for the final mixer)
gr_write: R_out[c,d] = R[c,d] + block_out[d] * 2 * sigmoid(inject[c] / hc)
```

| Function | Path |
| --- | --- |
| `gr_read` (`gr.cu:344`) | Legacy: `gr_norm_kernel` (hc blocks) then `gr_down_kernel` (hc_lr blocks, warp-strided reduction) then `gr_gate_kernel` (warp per output row) then `gr_mean_kernel` then `gr_inject_kernel` (one block of 32 x hc). Activations are BF16-rounded by default, or F32 with `gr_set_fp32_activations(true)` (`--gr-fp32-activations`). With `gr_set_native_mmvf(true)` (`--gr-native-mmvf`) the read is `native_gr_rms_norm_weighted`, `bf16_gemv_fp32_mmvf` (down), `native_gr_down_silu`, MMVF (up), `native_gr_pre_gated`, MMVF (inject) |
| `gr_write` (`gr.cu:413`) | `gr_write_kernel`, or `native_gr_post` with native MMVF |
| `gr_workspace_init` (`gr.cu:304`) | carves the caller's device buffer (xn, xq, lq, gated, lo) and records its size; `gr_read` checks the size |
| `fused_gr_read` (`fused_gr.cu:1168`) | two kernels: `gr_down_kernel` (41 blocks: 40 x 8 down rows plus the inject rows) applies the previous half's pending write on the fly, then norm, down, inject; `gr_up_kernel` (80 blocks of 32 columns x 4 streams) writes R in place and computes `mixed`. FP32 activations, BF16 weights. The geometry is hard-coded (`fused_gr_supported`) |
| `fused_gr_read_multi` (`fused_gr.cu:1056`) | up to `kFusedGrMaxT` = 8 tokens sharing the weights (verify window); every token's output is bitwise `fused_gr_read`. Variants: plain, split (norm per token and stream) and staged (the down projection's activations staged by `cp.async` on sm_80+, plain copies below). The down kernel's dynamic shared memory (`n_tok x TILEV` floats, TILEV 1280 on sm_75, else 2560) is opted in per device, and tokens are sliced when it does not fit (`down_chunk`, `:687`) |
| `fused_gr_check` (`:1336`), `fused_gr_variant` (`:1326`) | run once per card (by `Verifier::init`): runs split and staged against plain on random data and keeps the newest one that agrees bit for bit; prints the choice |

Pinned adapters (`--use_fast_math`): `native_gr_rms_norm_weighted` (`native_gr_norm.cu:85`; 256 threads below
n_cols 1,024, else 1,024, pinned XOR tree), `native_gr_down_silu` (`native_gr_postops.cu:91`, scale 1/hc then
SiLU), `native_gr_pre_gated` (`:97`; layer reads use the fused `DSV4_HC_PRE_GATED` FMA order; the final mixer uses
individually rounded products, matching the oracle's `il = -1` graph), and `native_gr_post` (`:107`; scale, sigmoid,
x2, `DSV4_HC_POST`).

The engine uses `fused_gr_read` / `_multi` when `--gr-native-mmvf` is on and no per-stage timing or dumps are
requested (`src/program/generate.cpp:2624-2625`). `--native SHARD1` turns it on.

## RoPE

### Table path (`rope.hpp`, `rope.cu`)

- `build_rope_table(n_rot, theta, max_pos, cos, sin)` and `build_rope_table(n_rot, RopeScaling, ...)` (`rope.cu:48`,
  `:61`) fill `[max_pos][n_rot/2]` cos and sin on the host in float64, in the reference's order. With
  `RopeScalingType::None` the scaled builder is bit-identical to the plain one.
- `rope_neox_apply` (`rope.cu:117`): one thread per row (128 threads per block). NEOX pairs `(i, i + n_rot/2)`, partial
  rotation (64 of 256 dims, the rest copied). `pos` is a device array (graph capture). The pair arithmetic is
  `rope_neox_pair` (`rope.hpp:42`, host and device), shared with the indexer's pooling kernel.

### Analytic path (`native_rope.hpp`, `native_rope.cu`)

`native_rope_apply` (`native_rope.cu:118`, `--native-rope`): ggml `rope_multi` / `rope_yarn` for text IMRoPE.
Rows of F32, head_dim 128 or 256, `n_rot` 64; grid `(ceil(head_dim/2 / 128), rows)`, 128 threads. `theta_scale` =
`powf(freq_base, -2 / n_rot)` is computed on the host to match pinned host `powf`. The kernel computes
`pos * theta_scale^pair` with fast-math `powf/cosf/sinf` through `rope_scaled_angle`, unless the opt-in table is
active. Bad arguments throw.

### Rope scaling (`rope_scaling.hpp`)

`RopeScaling` holds `none | linear | yarn` with `freq_base` (1e7), `factor`, `freq_scale_in`, `orig_ctx` (262,144),
`ext_factor`, `attn_factor`, `beta_fast` (32) and `beta_slow` (1), all transcribed from ggml. `kernel_args(n_rot)`
gives the float constants for the analytic kernels (the identity for `none`). `rope_scaled_angle` is ggml's
`rope_yarn`, with the ramp taking a pair index. `rope_scaling_invalid` is the single validity rule.
`rope_scaling_set` (`rope.cu:45`) is called once at start-up before any table is built or graph captured, because K is
cached post-RoPE and a mid-run change would mix two scalings. CLI: `--rope-scaling`, `--rope-scale`,
`--rope-freq-base`, `--rope-freq-scale`, `--yarn-*`.

### M-RoPE and the angle table (`mrope.hpp`, implemented in `native_rope.cu:74-114`)

- `mrope_table_set(device_table)` (per device): null (the default) means the position passed is the rotary position.
  When set, it is an int32 `[cells][3]` (t, h, w) table and the position is read as a cell index. Pair i rotates by
  `tab[cell * 3 + i % 3]` (the sections are 11/11/10/0). Every rope kernel takes it as an argument, so it must be set
  before capture. `generate.cpp:2636-2652` sets it when `--vision` is on.
- `rope_table_set` / `rope_table_for` (`:103`, `:111`): with `STRATA_ROPE_TABLE=1`, the analytic kernels read the
  session's float64-derived cos/sin table instead of computing fast-math angles. The header records 0.0014 rad of
  error at 32K and about 0.02 at 262K for the fast-math path. Off by default, because it moves outputs.

## Elementwise glue (`elementwise.hpp`, `elementwise.cu`)

256-thread elementwise kernels unless noted. Calls with a null stream synchronise.

| Function | What |
| --- | --- |
| `embedding_gather` (`:133`) | decode one canonical (S2/S4/S8) embedding row with separately rounded multiply and add (`-ffp-contract=off` on HIP) |
| `gdn_gate` (`:142`) | `softplus(alpha + dt) * ssm_a` |
| `scale_inplace`, `add_inplace`, `silu_inplace` | `add_inplace` is also Qwen3.6's plain residual add (`src/core/layer.cpp:1237-1240`) |
| `f32_to_f16_bulk`, `f32_to_bf16_bulk` | conversions through [bits-helpers.md](bits-helpers.md) |
| `rms_norm_weighted` (`:386`) | `x / sqrt(mean(x^2) + eps) * w`, 4 warps per block, a warp per row; QSA's q/k/indexer norm (legacy path) |
| `doorbell_publish` / `_value` / `_res` (`:353`, `:345`, `:361`) | one 1,024-thread block copies x, ids and weights into mapped host memory, then `__threadfence_system`, then increments the mapped sequence number. This tells the CPU expert pool to start on the misses. It increments rather than stores so captured graphs ring a new value each replay (`elementwise.hpp:90-93`). `_value` stores a given value (HIP, `STRATA_DOORBELL_STORE=1`). `_res` also publishes residency |
| `doorbell_ring`, `doorbell_wait` (`:379`, `:220`) | a one-thread kernel. `doorbell_wait` spins (`strata_spin_pause`) until the host flag equals the ring, then fences. An event record inside a capture is silently dropped, which is why these are kernels |
| `copy_from_mapped`, `copy_rows_from_mapped`, `copy_i32_from_mapped`, `scatter_rows_f32` | float4 copies from mapped pinned memory, kept on the compute queue instead of copy-engine memcpy nodes |

## Control vectors (`cvec.hpp`, `cvec.cu`)

The engine's side of llama.cpp's `llama_adapter_cvec` with `--cvec-mode`. After layer l's FFN write, every residual
stream h of every token becomes `h - s_l (h . v_l) v_l` (project) or `h + d_l` (add), for l in [first, last]
(`cvec.hpp:5-16`). `cvec_upload` / `cvec_replicate` put the tables on each device. `cvec_set_enabled` flips a device
flag between requests (it synchronises), so captured graphs are the same with the vector on or off. `cvec_apply`
(`cvec.cu:154`) runs one 256-thread block per (stream, token). With `write` it first applies the pending
`R += bo * 2 sigmoid(inj / hc)`. With the flag off the result is bit-identical to no vector. It applies to Qwen3.6's
plain block too (hc = 1, `src/core/layer.cpp:1326-1327`). CLI: `--control-vector`, `--control-vector-scaled`, `--control-vector-layer-range`, `--cvec-mode`, `--cvec-dir`
(see [Program CLI flags](../../program/cli-flags.md)).

## PLE block (Flash-Next only)

The n-gram embedding layer (`ple.layers = [1]`). The host side (`ngram.hpp`: the 64-bit XOR n-gram hash
`ngram_rows`, the IQ4_NL / FP8 row decoders and `PleTable`, the SSD reader) is implemented in `src/kernels/ngram.cpp`
and `src/ngram/ple_reader.cpp` and is not CUDA (see [CPU kernels: n-gram](../cpu/ngram.md)). The GPU half (`ple.hpp:3-13`) computes:

```
key    = grouped_norm(ple_key @ emb)        value = ple_value @ emb
s[c]   = sum_d(key[c] * query[c]) / sqrt(n_embd)        query = grouped_norm(hidden)
gate   = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
gated  = value * gate[c];  normalized = grouped_norm(gated)          <- the conv input
conv   = 4-tap, dilation 3 conv over the history of normalized rows (row-fastest [9][10240])
out    = hidden + gated + silu(conv)
```

| Function | Notes |
| --- | --- |
| `ple_block` (`ple.cu:228`) | Key projection in one of three forms: `key_bf16` (`bf16_gemv_fp32_mmvf`, IQ model files); a native GGUF key (`native_quantize_q8_1` + `native_mmvq`, types Q2_0 / IQ3_XXS / IQ4_XS / Q8_0, `--native-ple-key`); or S2 (`quantize_q8_0` + `s2_gemv_q8`). Value is `bf16_gemv_fp32_mmvf` with `--native-bf16-extra`, otherwise an internal BF16 GEMV. Then the legacy post-ops (`gnorm_kernel`, `gate_kernel`, `bcast_kernel`, `conv_kernel`, `add3_kernel`; RMS sums of f32 products in double, as ggml's CPU does) or `native_ple_postops`. No allocation and no sync; scratch from `ple_block_scratch_bytes` |
| `ple_block_projected` (`ple.cu:369`) | the same after the projections |
| `ple_history_advance` (`ple.cu:204`) | shift the 9-row normalized history and append; one thread per channel |
| `native_ple_postops` (`native_ple_postops.cu:175`, `--native-ple-postops`) | pinned arithmetic after the projections: `native_gr_rms_norm_weighted` norms, a 512-thread gate per stream, broadcast, conv + residual; fixed geometry 2560 x 4, history 9 |
| `native_ple_postops_batch` (`:205`) | T tokens for the prompt path, element for element T single calls with the history advanced |

## Control and data flow

```
Flash-Next layer: fused_gr_read(attn) -> mixer -> [pending write folded into next read]
                  fused_gr_read(ffn)  -> router -> doorbell_publish -> ... -> combine
                  gr_write / folded write -> cvec_apply (if covered) ; layer 1: ple_block -> ple_history_advance
Qwen3.6 layer:    native_qsa_rms_norm_weighted(attn_norm) -> mixer -> add_inplace
                  native_qsa_rms_norm_weighted(post_attention_norm) -> MoE -> add_inplace -> cvec_apply
```

Callers: `src/core/layer.cpp` (`gr_read` / `fused_gr_read` at `:1161`, `:1419-1444`; `ple_block` at `:1369-1482`;
`cvec_apply` at `:1327`, `:1498-1504`), `src/core/verify.cpp` (`fused_gr_read_multi`, `ple_block`, `cvec_apply`),
`src/core/mtp.cpp`, `src/prefill/prefill.cpp` (`native_ple_postops_batch`).

## Configuration

- CLI: `--gr-fp32-activations`, `--gr-native-mmvf`, `--no-fused-gr`, `--native-rope`, `--native-ple-postops`,
  `--native-ple-key`, `--native-bf16-extra`, `--rope-scaling` and related flags, `--vision`,
  `--control-vector*`, `--cvec-mode`, `--no-ple`, `--ple-io`.
- Environment: `STRATA_HC_SPLIT` (unset or `2` = staged if the check passes, `1` = at most split, `0` = plain),
  `STRATA_GR_V3=1` (another two-kernel multi read with another summation order, opt-in), `STRATA_GR_DOWN_MAX4=1`
  (4-token accumulators for launches of <= 4 tokens, same bits), `STRATA_ROPE_TABLE=1`; HIP only:
  `STRATA_GR_FAST`, `STRATA_GR_SPLIT`, `STRATA_DOORBELL_STORE`.

## Model-specific behaviour

- Flash-Next uses the HC read/write and PLE. Qwen3.6 has neither: its residual is plain (`native_qsa_rms_norm_weighted`
  + `add_inplace`), and `fused_gr.cu` hard-codes 2560 / 4 / 320.
- Both models use NEOX partial RoPE with n_rot 64 of 256 and freq_base 1e7, with IMRoPE sections 11/11/10/0 for the
  vision path.

## Tests

`gr_parity` (legacy, fused, multi and variants against each other, with `STRATA_HC_SPLIT` variants), `cvec_parity`,
`rope_parity` (tables against a float64 transcription for none / linear / YaRN; the native kernel), `elementwise_parity`
(including the `gdn_l2_norm` vs `rms_norm_weighted` distinction), `ple_parity` (hash, IQ4_NL table, block against a
captured ggml graph; needs the real artifacts), `ple_fp8_parity`, `ple_q5_parity`. Oracle-linked (sources not in
this repository): `native_gr_norm_parity`, `native_gr_postops_parity`, `native_rope_parity`,
`native_ple_postops_parity`.

## Gotchas

- `GrWorkspace` must be device memory. A host buffer produces an illegal access with a truncated address
  (`gr.hpp:51-54`).
- `inj_prev` and `inject_out` of `FusedGrArgs` must be different buffers (`fused_gr.hpp:16-17`).
- `fused_gr_variant` on a card that has not been checked is the plain read unless `STRATA_HC_SPLIT` names a variant.
- `ple.hpp:15-20`: the conv input is `normalized` (the grouped norm of the gated values), not the gated values as
  `ref/ngram.py`'s docstring says.
- `ngram.hpp:204-214` describes Windows read-ahead behaviour. This fork builds for Linux, where the advice belongs in
  `GgufFile::open`.

## See also

- [gdn.md](gdn.md), [attention-qsa.md](attention-qsa.md), [gemv-and-quant.md](gemv-and-quant.md),
  [bits-helpers.md](bits-helpers.md)
- [Core](../../core/README.md), [Program CLI flags](../../program/cli-flags.md)
- `docs/upstream/DETAILS.md` (n-gram table I/O, vision)
