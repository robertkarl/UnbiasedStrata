# Qwen3.6 port, Phase 2 step 3: the prompt path at Qwen3.6's geometry

Working notes of the prefill item of step 3 (2026-10-11, branch `qwen36-step3-prefill`, branched from `qwen36-step2` at
8b7525e).  The design is `docs/QWEN36-35B-A3B.md` section 3.6 (the table of literal sites) and section 6, step 3, the
prefill bullet; the rules are `step3-common.md`.  Every item says what was decided, where it lives, how it is checked,
and the numbers from both cards.

## 1. What changed, site by site

`src/prefill/*` is the whole scope.  The study's table (3.6) had eight literal sites; what each became:

| Study's site | What it was | What it is now |
| --- | --- | --- |
| `prefill.cpp` constants `N, HC, D, LR, K, NE, C, ZV, HV` | file-wide `constexpr` (2560 / 4 / 10240 / 320 / 10 / 512 / 10240 / 6144 / 48) | a `Geom` struct built from the `ModelGeometry` (`Geom::of`): n_embd, hc, hc*n_embd, hc_lr, k, n_expert, n_ff (and 2 n_ff, the shared width), conv channels, value dim, value heads, key heads, heads, KV heads, head dim, `attn_q`'s width, n_head*256, n_head_kv*256, `indexer`, `silu`.  Every count (`gdn_set_bytes`, `qsa_set_bytes`, `moe_bufs`, `moe_set_bytes`, `bytes_needed`) and every launch in `carve` and `run_impl` reads it.  The old names stay as local aliases inside each function so the arithmetic reads as before |
| `Prefill::init`'s refusal (`n_embd == 2560 && hc == 4 && hc_lr == 320 && k == 10`) | one geometry | per architecture: `qwen4exp` keeps exactly that check; `qwen35moe` needs `hc == 1`; both need `ss.k == n_expert_used`, `k` in 1..32 and at most `n_expert`, a GDN geometry the kernels are built for (`gdn_geometry_supported`: 48/16/10240 or 32/16/8192, state 128, conv 4), head_dim 256, 2 KV heads, n_rot 64, n_embd a multiple of 256, n_ff of 64.  Each refusal names what it saw |
| `Qf` as `T * 12288`, `q` / `attn` as `T * ZV`, `idx_raw`, `q_idx` | 24 heads x 512; the indexer's two buffers always | `T * G.qf` (n_head * 512), `T * G.av` (n_head * 256); the indexer's buffers, the selection scores and the ids rows only with an indexer (`G.indexer`).  Dense attention carries no ids: a null is the identity selection (below) |
| MoE buffers `T*K*1280`, `T*K*640`, `1280 * 2560`, `2560 * 640`; `mmq::fits(gt, 1280)`, `matrix_bytes(dt, N, 640)` | 2 n_ff and n_ff as 1280 / 640 | `G.ff2`, `G.ff`, `G.ffs`; `mmq_plan()` takes each layer's widths from the native layout (`fmt[l].n_embd`, `fmt[l].n_ff`; the canonical Q2_0 pack's 2560 / 640 otherwise), so the plan is per pack, not per build.  The peer paths (`set_peer`, `bind_stage_helper`, the peer products in `run_impl`) use the same members; `PeerPrefill`'s event ring and host bounds are sized once from `NE_MAX = 512` |
| `grs`, `gated`, `inj`, `xq`, `ident` from `HC`, `D`, `LR` (`:854-856, :1000-1026, :1375-1380`) | the hyper-connection's buffers and the GR read / write around every half | carved only at `hc > 1`.  At `hc == 1` the half is `x = rms_norm(R, attn_norm) -> mixer -> R += out; x = rms_norm(R, post_attention_norm) -> MoE -> R += out` through two new kernels, `rms_norm_images` (the FP32 norm with its BF16 and FP16 images, gr_norm_kernel's arithmetic on one stream) and `residual_add`.  The GR path is untouched (`gr_*` kernels, the F-2 fusion, `STRATA_GR_UNFUSED`); the embedding broadcast becomes a copy.  `draft_kv` (the MTP drafter's K/V through this path) is Flash-Next's graph and returns false for `qwen35moe` (step 4 brings the nextn block) |
| GDN kernels on `HV`, `HK`, `C`, launched `<<<HV, ...>>>`; `VPK = 3` | file constants | templates on `<HV, HK, C>` (`gdn_gates_kernel<HV>`, the three conv kernels and the l2 norm on `<C>`, `gdn_rec_kernel<HV, HK, C, SILU>`, `gdn_rec_cols_kernel`, `gdn_rec_cols_pipe_kernel`, `gdn_rec_kh_kernel<HV, HK, C>` with `VPK = HV / HK` and a `static_assert(HV % HK == 0 && VPK <= RG)`, `gdn_out_norm_kernel<HV, SILU>`, `gdn_keyhead_ok<HV, HK, C>`), dispatched by `gdn_dispatch` from a `GdnGeom {hv, hk, c, silu}` at exactly the two geometries; any other stops the process (the kernels' convention) after `init` has refused it by name.  The output gate is `gdn_gate<SILU>(z)`: `z * sigmoid(z)` for Qwen3.6, `sigmoid(z)` for Flash-Next, chosen from `g.arch` (`Geom::silu`) |
| `route_kernel<REG>` with `rank < 10`, `ids[t*10 + rank]`, `lane < 10` | top-10 baked | `k` a kernel parameter (the warp kernel at 512 and 256 experts, `router_top10` for any other count); `moe_combine` takes the width and `k`; the SwiGLU kernels take n_ff; `swiglu_pair` runs over `n * n_ff` |
| `split_q_kernel`, `gate_attn_kernel` over `T * 24 * 256` and `h * 512` | 24 heads | `n_head` a parameter (head_dim 256, `attn_q` as `[n_head, 2, 256]` per token in both models) |
| `moe_fused_iq.cu:59` `GU_ROWS_K 2560 / D_ROWS_K 640`; the `1280 / 2560` weight-row counts; `dm[row * 2560]`; `out + row * (10 * AB)` | constants | `NativeGeom` carries `n_embd` and `n_ff`; the kernel's stage count, weight-row blocks per tile, activation row stride, H block count and output stride come from them (loop bounds, not unrolled constants); `experts_native` refuses a width that does not tile the work item (n_embd a multiple of 256, 2 n_ff and n_embd multiples of the item's 64 WW rows).  `moe_fused.cu` is NOT touched: it is the Strata Q2_0 blob's kernel and the blob format itself is the Flash-Next shape (gate/up codes `[1280][640 B]`, down `[2560][160 B]`); the canonical pack is refused for any other geometry since step 2, so no Qwen3.6 chunk reaches it (`fused_l` needs `!lay.native`).  The study's "likewise" for that file does not apply |
| the layer loop at `:1724`, `RP = 48` at `:1303` | "check" | the loop runs `g.n_layers` (it always did); `RP = 48` is the peer-streaming ring's slot count (`STRATA_PF_PEER_RING`), not a head count: unchanged |

Also changed: `kv_append_kernel` keeps `t * 512 + kvh * 256 + d` and the grid `(T, 2, 8)`: 2 KV heads x 256 in both
models (init refuses anything else); `rope()` keeps its 64 rotated dims and the pairs `(p, p + 32)` (n_rot 64, the
IMRoPE sections and the 1e7 base are the same in both models; `rope_scaling()` is the process config).  The
`on_seam` hook on `Prefill` (null by default, nothing else changes) is how the harness reads the layer loop by
llama.cpp's seam names and teacher-forces every layer's input.  `STRATA_PREFILL_ALLOC_TRACE=1` names the allocation
that did not fit (it found the harness's own bug below).

### The attention step

The study (3.3, risk 3) has the prompt attention through `qsa_prompt_attn_batch` on sm_80+ and the batched decode
kernel below cc 70, both at `G = 12` with a split-K scratch that grows with the context.  The attention branch
(`qwen36-step3-attn`) templated `G` at 2 / 8 / 12, made a null `ids` the identity selection and bounded the scratch
at 128 blocks per KV head; its three kernel commits are cherry-picked here (section 4).  The prefill's dense path
is then: the shapes zero the three indexer widths together (`shapes_of`: that is the kernels' `qsa_dense` signal;
`qsa.cu`'s guard refuses a shape with only some of them zero), the capacity is `qsa_selection_cap` (every cell of
the primary state), no ids are carved or passed, `qsa_attn_group` is asked before any launch and its refusal is
`run`'s error string (a 16-head model asked for G = 12 is a message, not a mis-indexed run), the step records'
width is `n_kv` (`qsa_step_fill` under the dense shape).  The indexer tensors are not required of a dense layer.
No kernel of the attention branch was rewritten here.

## 2. The harness

### 2.1 Fixtures (`tests/python/qwen36_prefill_fixture.py`)

The oracle's own fixture is 64 wide with heads of 16 and 32; the prompt kernels are built for the real widths
(the GDN templates at 128-wide heads, the attention kernels at head_dim 256, the fused experts at 256-wide tiles),
so the harness has its own: the synthetic model of `tools/qwen36_synth_gguf.py` (its weights, its 128-token
vocabulary) at the real per-layer geometry, few layers, few experts:

| Case | Layers | Geometry | Forms |
| --- | --- | --- | --- |
| `gdn3` | 3, all Gated DeltaNet (interval 4: no layer satisfies `(il + 1) % 4 == 0`) | n_embd 2048, GDN 16 / 32 heads of 128 (conv 8192), attention 16 / 2 of 256, n_rot 64, sections 11/11/10/0, base 1e7, 16 experts top-8 of 512, shared 512 | projections, experts, shared expert and head Q8_0 (what the engine serves natively); `ffn_gate_inp`, `ssm_alpha`, `ssm_beta`, `ffn_gate_inp_shexp` F32 rounded to BF16-exact values (the engine's forms, exact); norms, conv, dt bias, ssm_a, `token_embd` F32 |
| `full4` | 4: three DeltaNet, layer 3 full attention | the same | the same |

The router runs at half the toy's gain (`ffn_gate_inp * 0.5`): at the toy's spread the top-8 of 16 carry 99 % of
the softmax mass and the renormalisation rival moves `ffn_out` by 1.3 %, invisible to a 5 % gate; at this gain it
moves it by 5.7 %.  For each case the script runs `ref/qwen36.py` on 40 tokens (the ASCII codes of
`"the quick brown fox jumps over the lazy "`) three ways: the faithful reading (`o`), `--act-quant q8` (`o_q8`), and
the rival readings `neg_gate` (`gdn_gate=sigmoid`), `neg_pair` (`head_pairing=interleaved`), `neg_norm`
(`router_norm=False`) and, on `full4`, `neg_attngate` (`attn_gate=False`).  The oracle dequantizes Q8_0 with
gguf-py, bit-identical to ggml (oracle-crosscheck.md), so every difference the harness sees is the engine's
arithmetic.  `gdn3` is a file `read_model_geometry` refuses by design (block_count must be a multiple of the
interval): the harness takes `full4`'s geometry with `gdn3`'s layer count, which is what the oracle ran it as.
4m55s on VM 221 for both cases (the oracle in numpy); 269 and 351 MB.

### 2.2 The test (`tests/cuda/prefill_qwen36_test.cpp`; CTest `prefill_qwen36_fixtures`, `prefill_qwen36_test`, `prefill_qwen36_attn_test`)

The fixture's tensors go into a `WeightTable` row by row (`WeightTable::set`, the one `[shared]` change: Q8_0 as
`native_data`, the BF16 set as `Bf16InF32`, the rest F32), the experts into an in-memory `ExpertSource` of
`[gate | up | down]` blobs (unpinned: the prompt path's stager copies them) under a `native_experts.txt` the test
writes, the session from `session_init` at 64 cells, the embedding rows through `embd_rows`.  Then:

1. `Prefill::init` refusals: top-10 on the top-8 file ("top-10 routing over 16 experts (the file says top-8 ...)"),
   and 32 value heads / 8 key heads (VPK 4; "no GDN kernels for 32 value heads / 8 key heads / 6144 conv channels
   (built for 48/16/10240 and 32/16/8192 ...)").  Both must be refused with those words.
2. One chunk of 40 and chunks of 7 (the oracle's `base_c7` case), every layer's input teacher-forced at the `l_in`
   seam to the oracle's `l_out` of the layer before (oracle-crosscheck.md, "Tolerance policy", item 2: per layer,
   not accumulated), each seam held to the faithful dumps as the relative L1 of the sums over the chunks:
   `attn_norm`, `linear_attn_out` / `attn_output`, `attn_post_norm`, `ffn_out`, `l_out`, and `ffn_moe_topk`.
3. The router's top-k: a rank swap is allowed only where the oracle's own softmax puts the two experts within
   2e-3 of each other (the prompt path's router multiplies BF16-rounded activations, as the study notes under
   `STRATA_PREFILL_BF16X2`; the decode path's is FP32), any other disagreement fails.
4. The negative controls: the engine, forced from the faithful dumps, against the rival readings' dumps at the seam
   that makes each, more than 5 % apart.
5. For the record: the same seams against the Q8 oracle, and the accumulated error without forcing.

The gates (`gate_for`): `attn_norm` 1e-4 (FP32 norm), `attn_post_norm` 2e-3 (the mixer's error passes through
the norm of R + out), `linear_attn_out` / `attn_output` 1e-2 (FP16 GEMMs over FP16-dequantized Q8_0 weights, BF16
alpha / beta), `ffn_out` 2e-2 (the int8 MMQ products: q8_1 activations and H), `l_out` 1e-2.  They are the measured
numbers below with a 4x to 10x margin; the negatives sit at 0.06 to 0.92.

### 2.3 The GDN kernels alone (`src/prefill/gdn_rec_parity.cu`)

Kept its bit-for-bit check of the two recurrence variants, now at both geometries; added a double-precision host
transcription of the recurrence (decay before the rank-1 update, v-head h with k-head h % HK, the gated RMS-normed
output) that the engine's own `gdn_recurrence` and `gdn_conv` launches are held to (through their `GdnGeom`), the
rival readings computed by the same reference (the other model's gate, the interleaved pairing h // VPK) required
to sit more than 5 % away, and the geometry refusals.  Links `strata_prefill`.

### 2.4 The fused native experts (`tests/cuda/prefill_fused_iq_test.cpp`)

The reference part runs at both geometries (the widths are variables set per pass); the timing part keeps the pack
it was measured on.  Exit 77 below sm_80 (the 1080 Ti: the prompt MoE takes MMQ or the FP16 path there, below).

## 3. Numbers

### 3.1 VM 221, RTX 4090 (sm_89), 2026-10-11

Build: `/models/work/build-step3-prefill.sh configure && ... all` (the common file's configure line at sm_89).

`./build/gdn_rec_parity`: 0 failures.  Both geometries, T = 1 .. 4099: the two recurrence variants the same bits
(output, state, FP16 output).  Engine vs reference (T = 1, 7, 33, 257): y16 1.76e-4 to 1.79e-4, state 4e-8 to
1.0e-7 relative L1 at both 48/16/10240 (sigmoid) and 32/16/8192 (SiLU).  Rivals: the other gate 0.86 to 1.10 from
the right reading (the engine 0.87 to 1.10 from the rival), the interleaved pairing 1.24 to 1.33.  `gdn_conv`: h
5.2e-8 to 5.4e-8, the history exact.  Refused: 32/16/10240, 32/8/8192, 48/16/8192.

`./build/prefill_fused_iq_test --no-timing`: passed; every pair (IQ2_S / IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S /
IQ4_XS gate/up over Q2_0 / IQ4_NL down) within 1.45e-2 to 1.49e-2 rel RMS of the FP32 reference at 2560 / 640
top-10 and 1.46e-2 to 1.49e-2 at 2048 / 512 top-8.  `prefill_fused_moe_test`, `gemm_bf16_parity`: pass (the
Flash-Next geometry unchanged).

`./build/prefill_qwen36_test /models/work/qwen36-step3-prefill-fx gdn3` and `full4`: pass, both chunkings.
Relative L1 against the F32 oracle with forced layer inputs (one chunk of 40; chunks of 7 within 5 % of these):

| seam | layer 0 | layer 1 | layer 2 | layer 3 (attention) | gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_norm` | 2.4e-8 | 2.2e-8 | 3.1e-8 | 2.7e-8 | 1e-4 |
| `linear_attn_out` / `attn_output` | 8.7e-4 | 8.5e-4 | 8.3e-4 | 5.9e-4 | 1e-2 |
| `attn_post_norm` | 4.4e-4 | 3.6e-4 | 3.0e-4 | 8.2e-5 | 2e-3 |
| `ffn_out` | 1.0e-2 | 8.2e-3 | 9.2e-3 | 7.9e-3 | 2e-2 |
| `l_out` | 3.7e-3 | 2.6e-3 | 2.5e-3 | 2.1e-3 | 1e-2 |

The router: 7 rank swaps of 960 (40 x 8 x 3) slots on `gdn3` (one more in layer 3 on `full4`), every one a pair
within 2e-5 to 1.3e-3 of each other in the oracle's softmax (an adjacent-rank swap); none elsewhere.  Negatives:
`neg_gate` 0.922, `neg_pair` 0.690, `neg_norm` 0.057 at the seams that make them, `neg_attngate` 0.541 at
`attn_output-3`.  Accumulated without forcing, `l_out` 3.7e-3 (layer 0) to 8.7e-3 (layer 3).  Against the Q8
oracle the GDN seams are worse (1.4e-2: the GDN projections run FP16 GEMMs, not Q8 products, so the Q8 oracle is
the wrong reference for them) and `ffn_out` is not better (1.2e-2 to 2.4e-2): the MMQ products' q8_1 rounding of
the activations and of H is not ggml's Q8_K rounding of the K-quant path the Q8 oracle models.  The F32 column is
the one the gates use.

Where the 1e-2 of `ffn_out` comes from: Q8_0 weights x q8_1 activations (a scale per 32) on int8 tensor cores,
SwiGLU in FP32, H to q8_1 again, the down product the same.  The fused test measures the same path at 1.45e-2 rel
RMS against an FP32 reference over random weights; the oracle's own real-file numbers for `ffn_moe_out` are
2.9e-2 (F32) / 2.3e-3 (Q8).

### 3.2 VM 211, GTX 1080 Ti (sm_61), 2026-10-11

Under the card's lock (taken 13:08 after the experts agent released it; held 182 s): `/home/rk/run211-prefill.sh`,
the tree at `/home/rk/qwen36-step3-prefill/` (configure as the common file says at sm_61 with
`STRATA_EXPERIMENTAL_SM60=ON`, the five test targets built in 168 s), the fixtures and dumps copied from VM 221 to
`/home/rk/qwen36-step3-prefill-fx/`; the log is `/home/rk/logs/qwen36-step3-prefill-211.log`.

- `gdn_rec_parity`: 0 failures; the same bits for both variants at both geometries; engine vs reference y16
  1.76e-4 to 1.79e-4, state 4e-8 to 1.0e-7; the rivals 0.86 to 1.33 away; `gdn_conv` 5.2e-8 to 5.4e-8; the three
  geometry refusals. The numbers equal the 4090's to the printed digits.
- `gemm_bf16_parity`: 0 failures (Pascal's fp32 widening path).
- `prefill_fused_iq_test`, `prefill_fused_moe_test`: exit 77 ("the fused kernels need sm_80 or newer"), as expected.
- `prefill_qwen36_test gdn3` and `full4`: pass, both chunkings.  Against the F32 oracle with forced inputs:
  `attn_norm` 2.2e-8 to 3.1e-8, `linear_attn_out` 8.3e-4 to 8.6e-4, `attn_output-3` 5.9e-4, `attn_post_norm`
  8.2e-5 to 4.4e-4, `ffn_out` 7.9e-3 to 1.05e-2, `l_out` 2.1e-3 to 3.7e-3; the router 5 (gdn3) / 7 (full4)
  near-tie swaps, none elsewhere; negatives 0.922 / 0.690 / 0.057 / 0.541; accumulated `l_out` 3.7e-3 to 9.1e-3.
  The same as the 4090 to two digits: the prompt MoE took the same path on both cards.
- Which prompt-MoE path runs on Pascal: MMQ.  `STRATA_PREFILL_TIMING=1` shows the MMQ phases (gather, dequant =
  the gather marks, gemm gate/up 1 ms); and `STRATA_PREFILL_MMQ=0` (the FP16 GEMM over dequantized weights) gives
  different numbers (accumulated `l_out` 1.4e-3 / 3.0e-3 / 4.1e-3 instead of 3.7e-3 / 6.0e-3 / 8.2e-3; against the
  Q8 oracle `ffn_out-0` 2.25e-2 instead of 2.36e-2), so the default run was not that path.  Q8_0 x q8_1 through
  llama.cpp's MMQ on sm_61 is dp4a, the study's risk 2 wants it enabled there, and it is.  The FP16 path is the
  closer one to the F32 oracle (no int8 rounding of the activations), which is what the Q8 column says too.
- 40 tokens through 3 GDN layers (and 4 with attention): a 60 to 100 ms GPU timeline per run, 36 to 40 ms of it
  host staging of the unpinned expert blobs; no throughput claim from a 40-token harness (step 5 measures).

## 4. Git

Commits on `qwen36-step3-prefill` past 8b7525e, in order: the kernels (templated GDN, runtime widths), the fused
native experts' widths, the prefill geometry and plain block, `gdn_rec_parity` at both geometries, `[shared]
WeightTable::set`, the fused test's second geometry, the three cherry-picks from `qwen36-step3-attn` (4a24760 and
65a4f2b whole, 856daa0's three kernel files without its parity-test hunk, which belongs to that branch), the dense
attention wiring with the harness and its fixtures, the harness's chunk accounting.

What the integrator should know:

- `src/kernels/cuda/qsa_decode_attn.cu`, `qsa_prompt_attn.cu`, `qsa.cu` and the two headers are the attention
  branch's versions at 856daa0; the decode GDN kernels and `gdn_parity.cpp` are the GDN branch's at 7954497;
  rebasing onto the integration branch makes those cherry-picks empty.
- The GDN branch's 7954497 (`kernels::GdnGate`, the decode kernels' gate argument) and 4f9aeb9 (`[shared]`
  `ModelGeometry::gdn_gate()`) are cherry-picked here too: `GdnGeom::gate` is that enum, required with no default,
  and `Geom::of` takes it from `g.gdn_gate()`, so the prompt and decode paths agree by construction (drop the two
  at integration).  The block branch's `layout.cpp` change (`[shared]` 69d52c9) is not needed by the prompt path:
  `Prefill::run` reads the two norms through `need()` and refuses a non-F32 one by name.
- `WeightTable::set` is the only change outside `src/prefill`, `tests/cuda/prefill_*`, `tests/python`,
  `docs/qwen36-port` and the CMake lines that register the tests.

## 5. Not done, and surprises

- **The dense ids and scratch on the serve path** are the attention branch's bound (128 blocks per KV head, no ids
  buffer); `qsa_set_bytes` counts no ids rows for a dense model and the scratch at the bounded size.  A 256K
  context on the serve path is step 5's measurement, not this harness's (64 cells).
- **The prompt MoE on Pascal**: `moe_fused*.cu` need sm_80 (their tests exit 77 below it); the unfused path is MMQ
  where `mmq::fits` on sm_61 (dp4a) and the FP16 GEMM path otherwise, both at the new geometry (`mmq_plan` from the
  layout's widths, `dq_gu` / `dq_d` from `Geom`).  Which one the 1080 Ti takes for Q8_0 is recorded in 3.2.
- **The harness's own bug**: `ss.qsa_states[ss.qsa_primary()]` for a model with no attention layer at all (`gdn3`)
  reads past a zero-length array; the prefill now guards every read of the primary state with `ss.qsa_alloc`
  (`primary_cells`).  Found by `STRATA_PREFILL_ALLOC_TRACE` as three multi-exabyte `cudaMalloc`s.
- **Teacher forcing across a non-blocking stream**: a pageable `cudaMemcpy` into R returns before its DMA has
  landed, and the next kernel on the prompt stream read half-old rows (2.5e-3 at `attn_norm-1`); the harness waits
  for the device after the copy.  Not an engine issue (the engine never copies host rows into R mid-layer).
- **The study's table** is right about every site; two of its "check" items were not geometry (`RP = 48`, the
  peer ring; the layer loop already ran `g.n_layers`), and `moe_fused.cu` cannot take the new geometry because its
  blob format is the geometry (section 1).
- **The study's harness line** ("chunked-prefill-then-decode against token-by-token decode") is met with the oracle
  standing in for the decode: the decode mixers are the other agents' and the oracle is the reference both are held
  to; the prefill is compared at every position of the chunked and the single-chunk run.
