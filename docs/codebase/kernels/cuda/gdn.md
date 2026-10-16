[Codebase docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Gated DeltaNet

# Gated DeltaNet (GDN) kernels

## Purpose

The GDN ("linear attention") layers are most of both models' layers: Flash-Next has a GDN layer wherever
`layer % 4 != 3`, and Qwen3.6 has 30 of its 40. These kernels compute everything between the layer's projections
(`attn_qkv`, `attn_gate` (z), `ssm_alpha`, `ssm_beta`, `ssm_out`): the 4-tap causal convolution with SiLU, the L2
norm of q and k, the gate and beta, the delta-rule recurrence on the (S x S) state per value head, and the gated
RMS output norm. The projections themselves are GEMVs ([gemv-and-quant.md](gemv-and-quant.md),
[moe-and-experts.md](moe-and-experts.md)).

Files:
- `src/kernels/cuda/gdn.cu`, `include/strata/kernels/gdn.hpp`: the reference (legacy) kernels, `GdnShapes`,
  `GdnGate`.
- `src/kernels/cuda/fused_gdn.cu`, `include/strata/kernels/fused_gdn.hpp`: the fused decode kernels (the default
  with the native contract).
- `src/kernels/cuda/native_gdn.cu`, `native_gdn_preprocess.cu` and headers: the pinned llama.cpp adapters.
- `src/kernels/cuda/verify_kernels.cu` (GDN part), `include/strata/kernels/verify_kernels.hpp`: the multi-token
  verify and commit kernels.

The prompt path's GDN kernels are in `src/prefill/kernels.cu` (not covered here; see
[Prefill](../../prefill/README.md)).

## The math

`include/strata/kernels/gdn.hpp:19-26` and `fused_gdn.hpp:10-12`. For value head h with key head
`idx[h] = h % h_k` (modulo, not interleave; `gdn.hpp:67-68`):

```
g      = exp(gate[h])                    gate = softplus(alpha + dt) * ssm_a   (ssm_a < 0, so g is in (0, 1))
beta   = sigmoid(beta_raw[h])
S     <- g * S + k (beta * (v - g * S^T k))^T                 (decay before the rank-1 update)
o      = S^T q / sqrt(128)                                    (fused/native: scale on the readout)
y      = rms_norm(o) * ssm_norm * act(z)                      act = sigmoid (Flash-Next) | SiLU (Qwen3.6)
```

Each `(j, h)` column of the state is independent, so one thread or warp can own a column for the whole step with no
barrier inside the recurrence (`gdn.hpp:28-30`).

### State and history layouts

- Recurrent state: `(S, h_v, S)` = `state[(i * h_v + h) * S + j]`, with j fastest so that a warp walking j is
  coalesced. This differs from `ref/gdn.py`'s `(S, S, H_v)` and is stated in `gdn.hpp:13-17`.
- Conv weight: GGML-native `kW[c * d_conv + i]` (ne0 = d_conv fast), with tap 0 reading the oldest history row.
- Conv history: the legacy `gdn_conv_step` keeps `(d_conv - 1, channels)`, oldest first (`gdn.hpp:74`). The fused,
  native and verify kernels keep `[channel][3]` (`fused_gdn.hpp:21`, `native_gdn_preprocess.hpp:13`). A session uses
  one family throughout (selected in `src/core/layer.cpp:248-256`).

### Geometry

`GdnShapes` (`gdn.hpp:39-43`): `S` (state size, 128), `h_k` (16 for both models), `h_v` (48 Flash-Next, 32 Qwen3.6);
`h_v % h_k == 0` is the one rule. Conv channels are `2 * h_k * S + h_v * S` (10,240 Flash-Next, 8,192 Qwen3.6). The
fused and native kernels hard-code S = 128.

### `GdnGate`

`enum class GdnGate { Sigmoid, Silu }` (`gdn.hpp:53`). Flash-Next (`qwen4exp`) multiplies by `sigmoid(z)` and Qwen3.6
(`qwen35moe`) by `SiLU(z) = z * sigmoid(z)`. The layer passes `ModelGeometry::gdn_gate()`, never a global, so one
process can serve both models. Every output-norm kernel takes it: `gdn_out_norm`, `fused_gdn_step_norm`,
`native_gdn_out_norm` and `gdn_step_norm_multi`. The sigmoid arithmetic is bitwise what it was before the parameter
existed. The oracle's negative control puts the two readings 1.07 relative L1 apart (`gdn.hpp:45-52`).

## Main entry points

### Reference kernels (`gdn.cu`)

| Function | Kernel / launch | Notes |
| --- | --- | --- |
| `gdn_step` (`gdn.cu:157`) | `gdn_step_kernel` (`:43`), grid `(S/32, h_v)`, 32 threads | one thread per `(h, j)` column; q must arrive already scaled by `1/sqrt(S)` (`gdn.hpp:60-61`). `MAX_H = 1` head per block: the comment at `:13-29` records that 8 heads per block left 24 of 48 SMs idle (0.28 ms per block) |
| `gdn_conv_step` (`:178`) | `gdn_conv_kernel`, 256 threads | `ggml_ssm_conv` with the carried state prepended; updates the state |
| `gdn_l2_norm` (`:193`) | `gdn_l2_kernel`, one warp per row | `x / sqrt(sum(x^2) + eps)` (sum, not mean; `gdn.hpp:88-92`) |
| `gdn_beta_gate` (`:226`) | 128 threads | `beta = sigmoid(beta)`; the layer must call it (`gdn_step` does not) |
| `gdn_out_norm` (`:239`) | `gdn_out_norm_kernel`, one warp per head | per-head RMS, `ssm_norm`, `act(z)` |
| `gdn_gate` | in `elementwise.cu:142` | `softplus(alpha + dt) * ssm_a`, softplus with ggml's `x > 20` branch |

### Fused kernels (`fused_gdn.cu`), the default with `--native-gdn` + `--native-bf16`

| Function | Kernel / launch | Replaces |
| --- | --- | --- |
| `fused_gdn_conv_l2` (`fused_gdn.cu:129`) | `gdn_conv_l2_kernel` (`:75`), one 128-thread block per 128-channel head | conv + SiLU (`__expf`) + L2 norm of the first `qk_heads` (= 2 h_k: q then k) heads; 3 launches |
| `fused_gdn_ab` (`:140`) | `gdn_ab_kernel` (`:97`), one warp per row, 8 rows per 256-thread block | the two BF16 (h_v x n_embd) projections against the F32 activation (8 BF16 per `uint4` load, FMA chain) with the softplus gate and sigmoid beta epilogues; 2 MMVF + 2 elementwise launches |
| `fused_gdn_step_norm` (`:152`) | `gdn_step_norm_kernel` (`:16`), grid `h_v`, block `(128, 4)` | One block owns a whole head: 512 threads = (column, row group of 32), each thread holding 32 state rows in registers. The 128 columns of a row are one contiguous 512 B load. The kernel does the step and the output norm with `act`. The header (`fused_gdn.hpp:3-8`) contrasts it with the pinned step's strided loads (4 useful bytes per 32 B sector, 47.6 us per layer) |

### Pinned adapters (`native_gdn*.cu`, compiled with `--use_fast_math`)

| Function | Notes |
| --- | --- |
| `native_gdn_step` (`native_gdn.cu:96`) | llama.cpp `gated_delta_net` arithmetic with Strata's state layout: grid `(h_v, 1, S/4)`, block `(32, 4)`, each warp one column with 4 rows per lane in registers. q must not be pre-scaled; the readout is scaled. Throws on bad spans |
| `native_gdn_conv_silu` (`native_gdn_preprocess.cu:144`) | conv (with ggml's zero-bias add) and `expf` SiLU; writes both the raw and the SiLU output |
| `native_gdn_l2_norm` (`:162`) | `scale(rms_norm(x, eps / 128), 1 / sqrt(128))`, which is ggml's L2 norm expressed as its CUDA graph computes it |
| `native_gdn_beta_gate` (`:168`), `native_gdn_gate` (`:174`) | sigmoid; softplus with threshold 20 |
| `native_gdn_out_norm` (`:187`) | fused RMSNorm x gamma, then `sigmoid * mul` or `silu * mul` |

`native_gdn_set_enabled` / `native_gdn_enabled` (`native_gdn.cu:93-94`) is the switch the layer reads.

### Verify window (`verify_kernels.cu`)

The recurrent state cannot simply be overwritten by the next window, so the GDN verify kernels come in two halves
(`verify_kernels.hpp:4-12`). The verify half runs all T tokens in order from the stored state, writes only per-token
outputs and leaves the state untouched. The commit half, launched once acceptance is known, replays the first
`*n_keep` tokens and writes the state. No snapshot of the state (113 MB at Flash-Next's geometry, per the header) is
kept, and rejected drafts never touch it.

| Function | Kernel / launch | Bitwise equal to |
| --- | --- | --- |
| `gdn_conv_l2_multi` (`verify_kernels.cu:441`) | grid `(channels/128, n_tok)`, 128 threads; reads history, does not write it | `fused_gdn_conv_l2` per token |
| `gdn_conv_commit` (`:452`) | 256 threads | history <- last 3 of `[history, qkv_0 .. qkv_{n_keep-1}]` |
| `gdn_ab_multi` (`:458`) | 64 threads, `MAX_T` 4 or 8 | `fused_gdn_ab` per column |
| `gdn_step_norm_multi` (`:475`) | verify: `gdn_step_norm_multi_kernel` (`:160`), grid `h_v`, block `(128, 4)`; commit without outputs: `gdn_step_commit_kernel` (`:118`), grid `(h_v, 4)`, block `(32, 4)` | `fused_gdn_step_norm`, including `act` |

`n_keep == nullptr` means verify (the state is read only). A non-null `n_keep` means commit. `STRATA_GDN_COMMIT_SPLIT=0`
keeps the full kernel for a commit that writes no outputs (`t_out_begin >= n_tok`). T is at most `kVerifyMaxT` = 8.

## Control and data flow

Decode (`src/core/layer.cpp:248-330`):

```
attn_qkv GEMV -> fused_gdn_conv_l2                       (fused_pre: native GDN + native BF16 + d_conv 4 + S 128)
                 | native_gdn_conv_silu -> native_gdn_l2_norm x2
                 | gdn_conv_step -> gdn_l2_norm x2
alpha/beta    -> fused_gdn_ab | BF16 projections -> native_gdn_beta_gate + native_gdn_gate | gdn_beta_gate + gdn_gate
attn_gate GEMV (z)
step + norm   -> fused_gdn_step_norm(act = g.gdn_gate())   (fused: native GDN, S 128)
                 | native_gdn_step -> native_gdn_out_norm | gdn_step -> gdn_out_norm
ssm_out GEMV
```

The verify window calls `gdn_*_multi` and the commit (`src/core/verify.cpp:791`, `:796`, `:1398`, `:1953`). Callers:
`layer.cpp`, `verify.cpp`. The kernels depend only on the CUDA runtime.

## Configuration

- CLI: `--native-gdn` (`native_gdn_set_enabled`), `--native-bf16` (needed for `fused_gdn_ab`), `--no-fused-gdn`
  (engine switch `layer_set_fused_gdn`). `--native SHARD1` turns on the first two.
- Environment: `STRATA_GDN_COMMIT_SPLIT` (default on).
- Build: the native files compile with `--use_fast_math` (`CMakeLists.txt:399-412`).

## Model-specific behaviour

| | Flash-Next | Qwen3.6 |
| --- | --- | --- |
| h_k / h_v (values per key head) | 16 / 48 (3) | 16 / 32 (2) |
| conv channels, value dim | 10,240, 6,144 | 8,192, 4,096 |
| output gate | `GdnGate::Sigmoid` | `GdnGate::Silu` |

The Qwen3.6 port added the `GdnGate` argument to `gdn_out_norm`, `fused_gdn_step_norm`, `native_gdn_out_norm` and
`gdn_step_norm_multi`, and made the kernels generic in `h_v` (`git diff 12a7825 HEAD`).

## Tests

- `gdn_parity` (`src/kernels/gdn_parity.cpp`): legacy, fused, native and verify kernels at 32 v-heads with SiLU and
  48 with sigmoid, with the other gate as the negative case.
- `qwen36_seam_parity`: decode kernels against the numpy oracle's seams (needs the generated fixtures; exit code 77
  skips).
- Oracle-linked (sources not in this repository): `native_gdn_parity`, `native_gdn_preprocess_parity`,
  `native_gdn_pipeline_parity`.

## Gotchas

- The two L2-norm forms differ: `gdn_l2_norm` divides by `sqrt(sum + eps)`, while elementwise's `rms_norm_weighted`
  divides by `sqrt(mean + eps)` and multiplies by w. The two differ by `sqrt(S)` = 11.3x, and the tests assert they
  are distinguishable (`gdn.hpp:88-92`, `elementwise.hpp:73-77`).
- `gdn_step` expects q already scaled by `1/sqrt(S)`. `native_gdn_step` and `fused_gdn_step_norm` expect it unscaled
  and scale the readout.
- `gdn.hpp`'s header comment says "48 layers are GDN, 12 are QSA". `docs/QWEN36-35B-A3B.md` section 2 gives
  Flash-Next 48 layers with attention every 4th (36 GDN + 12 QSA). The header line appears stale.
- `fused_gdn_conv_l2` needs `channels % 128 == 0`, and the fused path needs `d_conv == 4` and S = 128 (the layer
  checks both before choosing it).

## See also

- [attention-qsa.md](attention-qsa.md), [sampling-and-verify.md](sampling-and-verify.md) (the other verify
  helpers), [norms-rope-elementwise.md](norms-rope-elementwise.md) (`gdn_gate`, `scale_inplace`)
- `docs/qwen36-port/step3-gdn.md`, `docs/QWEN36-35B-A3B.md` section 3.2, `docs/qwen36-port/oracle-crosscheck.md`
