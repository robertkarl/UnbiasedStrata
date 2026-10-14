# Qwen3.6 port, Phase 2 step 3: the block without hyper-connections

Working notes, kept as the step was done (2026-10-11, branch `qwen36-step3-block`, from `qwen36-step2` at 8b7525e).
The design is `docs/QWEN36-35B-A3B.md` section 3.5 and section 6 (step 3, fourth bullet); the plan is
`docs/PLAN-qwen36-35b-a3b.md` section 5 step 3. Every item says what was decided, where it lives, how it is
checked, and the numbers from both cards.

## 1. What changed, in one paragraph

`ModelGeometry::hc == 1` (set by the `qwen35moe` architecture string in step 2) now selects a second block path:
llama.cpp's plain pre-norm residual block, in the single-token functions (`block_layer_pre`, `block_layer_post`,
`lm_head_mix`) and in the verify window (`Verifier::record_window`'s `pre`, `post` and head), through two row-count
helpers that the single-token path calls with one row and the window with its T rows. The hyper-connection kernels
are never entered for this model: the GR read on one stream is not a plain norm (its bottleneck gates the normed
stream with a sigmoid and its norm weight is stored as `1 + w`), which the harness shows as a negative case. The
Flash-Next path (`hc = 4`) is untouched: every `hc == 1` test is a branch in front of the existing code, no GR
call site moved, and the no-geometry `layer_verify_compatible` keeps its Flash-Next meaning.

    x = rms_norm(R) * attn_norm             -> mixer (GDN | attention) -> R += out
    x = rms_norm(R) * post_attention_norm   -> MoE (routed top-k, normalised, + shared expert) -> R += out
    head: rms_norm(R) * output_norm -> output.weight

## 2. Decisions

| Decision | Where | Why |
| --- | --- | --- |
| Two helpers, `plain_norm_rows(R, w, mixed, n_embd, eps, n_tok, stream)` and `plain_residual_rows(R, block_out, n_embd, n_tok, stream)`, take a ROW COUNT | `include/strata/core/layer.hpp` ("the plain block"), `src/core/layer.cpp` | the single-token path (1 row) and the verify window (T rows) call the same kernels, so the window's row arithmetic equals the single-token path by construction; the harness asserts it bitwise |
| The norm is `native_qsa_rms_norm_weighted` (one block per row, fixed reduction order, `rsqrtf`, `(scale * x) * gamma`); the add is `add_inplace` | `layer.cpp` | the engine's pinned-arithmetic norm kernel, already used on q/k and in the drafter; ggml's `rms_norm` then `mul` in that order |
| No pending write at hc = 1: `R += out` is applied where the reference adds it (stage 2 and `post`), never folded into the next read | `layer.cpp` (`plain_block_pre` / `plain_block_post`), `verify.cpp` | the fold exists to fuse the GR write into the GR read; with no GR there is nothing to fuse, and an in-place add keeps `R` materialised for the control vector, the hand-off and the head |
| The stages and the `half` / `stage_prefix` selection are the GR path's (0 read, 1 mixer, 2 write, 3 read, 4 route) | `plain_block_pre` | `session_replay_stages`, the per-stage timing and the measurement splits apply unchanged; the harness drives stage 0 and half 2 separately |
| The norm weights are read as `const float*` after a form check (F32, `n_embd` elements) | `plain_norm_weight` in `layer.cpp`, `plain_w` in `verify.cpp`, `lm_head_mix` | Qwen3.6's norms are F32 in the file and the pack keeps them F32; a BF16 row read as f32 walks 2x the tensor inside the arena and yields a finite wrong norm (the `layout.cpp` failure) |
| `check_one` skips the `hc_*` rows at hc = 1 and requires `attn_norm` / `post_attention_norm` (F32, `n_embd`) instead; at hc > 1 the two plain names are skipped | `src/core/layout.cpp` | row 22 of the inventory; the indexer rows are the attention branch's |
| `layer_verify_compatible(const ModelGeometry&, std::string&)`: at hc = 1 the fused GR read is not required; without an indexer the block selection and the native indexer are not required | `layer.cpp`, `layer.hpp` | the window's compatibility check named kernels this model has no use for; the no-argument form stays Flash-Next's |
| `verify_geometry_supported(g, k, why)` replaces the window's `ss.k == 10` refusal: `k == g.n_expert_used` (8 from the file), the fused GR shape only at hc > 1, state 128 / conv 4 | `include/strata/core/verify.hpp`, `verify.cpp` | row 142; a free function so the negative case (top-10 on an 8-expert file) is a unit test and the message names both numbers |
| The window's head at hc = 1: `output_norm` over the T rows in one launch when the native head is loaded; the canonical head goes through `lm_head`, which branches the same way | `verify.cpp` (`record_window`, the head block) | the per-row arithmetic is the single-token `lm_head_mix`'s |
| A PLE run with hc = 1 is refused (`block_layer_pre`, `record_window`) | `layer.cpp`, `verify.cpp` | the model has no PLE (`has_ple = false`); a wired one would be a different model |
| `session_zero`'s first token needs no change | `src/core/session.cpp:141-148` | the loop is over `g.hc` (row 26): one copy at hc = 1 |
| `broadcast_streams` / `add_streams_broadcast` need no change | `src/kernels/cuda/verify_kernels.cu:256-291` | runtime `hc` (row 140): the identity copy at hc = 1 |

## 3. The harness: `tests/cuda/qwen36_block_test.cpp`

Maya's layer composition shape on the synthetic fixture (`tools/qwen36_synth_gguf.py`: hidden 64, layers 0-2 GDN,
layer 3 attention, 8 experts top-2, F32 everywhere) against the oracle's dumps of 8 tokens
(`cmake/qwen36_block_fixtures.cmake` runs `ref/qwen36.py --dump`; the fixture test `qwen36_block_fixtures` depends on
step 2's `qwen36_synth_fixtures`). CTest: `qwen36_block_test` (exit 77: no CUDA device or no numpy).

**Through the engine's own functions.** The test writes a pack (index.txt + dense.bin) holding, per layer, the two
norms (F32, index kind 2), the router `ffn_gate_inp` (BF16, kind 4, the engine's contract) and the shared expert's
scalar gate, plus `output_norm`; `WeightTable::load` reads it like any pack. Then, per layer and token, with the
mixer's output and the experts' outputs TEACHER-FORCED from the dumps:

| seam | engine call | tolerance (relative L2) |
| --- | --- | --- |
| `attn_norm-N` | `block_layer_pre(..., stage_prefix = 1)` on `R = l_out-(N-1)` (`model.input_embed` at N = 0) | 1e-6 |
| `attn_residual-N` | `plain_residual_rows` with `block_out = linear_attn_out-N` (GDN) or `attn_output-N` (attention) | 1e-6 |
| `attn_post_norm-N` | `block_layer_pre(..., half = 2)` on `R = attn_residual-N` (shared expert off: its weights are not in the pack) | 1e-6 |
| `ffn_moe_topk-N`, `ffn_moe_weights_norm-N` | the same call's router (`moe_route`: BF16 weight, F32 activations, `router_top10` at 8 / 2) | picks identical; weights 1e-2 reported |
| `ffn_out-N`, `l_out-N` | `block_layer_post` with `parts = ffn_moe_down-N`, `weights = ffn_moe_weights_norm-N`, `shared = ffn_shexp_gated-N` | 1e-6 |
| `result_norm` | `lm_head_mix` on `R = l_out-3[last]` | 1e-6 |
| `result_output` | the engine's `result_norm` through a host F32 projection of `output.weight` | 1e-5 |

**The window.** `plain_norm_rows` and `plain_residual_rows` over the 8 rows against the one-row calls per row
(bitwise, `memcmp`), `native_moe_combine_multi` against `native_moe_combine` per row (bitwise), and the 8-token
sequence carried through the four layers from the embeddings (`R` is never reloaded from the dumps) against
`attn_norm-N`, `attn_residual-N`, `attn_post_norm-N`, `ffn_out-N`, `l_out-N` at every position, then `h_nextn`
(every row) and `result_norm` (row 7), all at 1e-6.

**Negative cases** (each must differ from the oracle by more than 5 % relative L1): the GR read at hc = 1 with
identity bottleneck weights (`GrShapes{64, 1, 64}`, `w_down = w_up = I`, `w_norm = attn_norm`) against
`attn_norm-0`; the post-attention norm skipped (`attn_residual-0` as the MoE input) and unweighted; top-10 on the
8-expert file refused by `verify_geometry_supported` with both numbers in the message; a layer whose `attn_norm`
is not in the pack refused by name; `layer_verify_compatible` at hc = 1 passing without the fused GR read while
the Flash-Next geometry still demands it.

**The mixers are reported, not asserted.** The fixture's projections are F32 and the engine has no F32 projection
kernel (the mixers take quantized planes or native GGUF types), so `block_layer_pre(..., stage_prefix = 2)` is
attempted and its own error printed (`blk.N.attn_qkv.weight is missing`; the attention layer's RoPE table of 0 rows).
The mixer seams (`conv_output_silu-N`, `new_state-N`, `final_output-N`, `linear_attn_out-N`; `kq_soft_max-3`,
`kqv-3`, `attn_gated-3`, `attn_output-3`) wait on the GDN and attention branches and on step 4's pack, which is
where the engine first runs a Qwen3.6 mixer end to end.

**Why 1e-6 and not the policy's 5e-8.** `oracle-crosscheck.md`'s norm figure is numpy F32 against numpy F32 on a
forced input; the CUDA norm uses `rsqrtf` (2 ulp, 2.4e-7) under `--use_fast_math`, so one CUDA ulp is above that
figure. The measured numbers are below.

## 4. Build and test commands

On VM 221 (the 4090, sm_89; the working copy is synced with the worktree's empty `third_party/llama.cpp` excluded
and the pinned submodule copied from the step-2 tree):

    rsync -a --delete --exclude .git --exclude '/build/' --exclude '/build-mac/' --exclude '/logs/' \
          --exclude '/third_party/llama.cpp/' ~/Code/UnbiasedStrata-step3-block/ rk@192.168.50.151:/models/work/qwen36-step3-block/
    /models/work/build-step3-block.sh configure      # the common file's configure line, logged
    /models/work/build-step3-block.sh target qwen36_block_test
    /models/work/build-step3-block.sh ctest 'qwen36'
    /models/work/build-step3-block.sh all             # everything else still builds (the gate binary links the engine)

On VM 211 (the 1080 Ti, sm_61) under the `/home/rk/VM-BUSY` lock, the same with `-DSTRATA_EXPERIMENTAL_SM60=ON
-DCMAKE_CUDA_ARCHITECTURES=61` and the working copy in `/home/rk/qwen36-step3-block/`.

## 5. Results

### VM 221, the RTX 4090 (sm_89), 2026-10-11

`./build/qwen36_block_test build/qwen36_synth build/qwen36_block/o8` (the fixtures written by hand, below), all 30
checks pass:

| seam (single-token path through the engine) | measured, worst over 4 layers x 8 tokens | tolerance |
| --- | ---: | ---: |
| `attn_norm-N` (`block_layer_pre` stage 0) | 1.13e-7 | 1e-6 |
| `attn_residual-N` | 0.0 | 1e-6 |
| `attn_post_norm-N` (`block_layer_pre` half 2) | 1.23e-7 | 1e-6 |
| `ffn_moe_topk-N` (the BF16 router) | 0 of 64 slots differ | identical |
| `ffn_moe_weights_norm-N` (the BF16 router) | 9.82e-3 | 1e-2, reported |
| `ffn_out-N` (`block_layer_post`) | 7.02e-8 | 1e-6 |
| `l_out-N` (`block_layer_post`) | 4.82e-8 | 1e-6 |
| `result_norm` (`lm_head_mix`) | 0.0 | 1e-6 |
| `result_output` (host projection of the engine's `result_norm`) | 1.63e-7 | 1e-5 |

| the window (8 rows) | measured | tolerance |
| --- | ---: | ---: |
| `plain_norm_rows`, `plain_residual_rows`, `native_moe_combine_multi` against the one-row calls | bitwise | bitwise |
| `attn_norm-N` / `attn_residual-N` / `attn_post_norm-N` / `ffn_out-N` | 6.98e-8 / 2.61e-8 / 7.78e-8 / 2.52e-8 | 1e-6 |
| `l_out-N` per position, `R` carried from the embeddings (layers 0..3) | 1.99e-8 / 2.22e-8 / 2.65e-8 / 2.81e-8 | 1e-6 |
| `h_nextn` (all rows) / `result_norm` (row 7) | 4.75e-8 / 3.25e-8 | 1e-6 |

| negative case | relative L1 against the oracle | required |
| --- | ---: | ---: |
| the GR read at hc = 1, identity bottleneck, against `attn_norm-0` | 0.423 | > 0.05 |
| the post-attention norm skipped | 0.276 | > 0.05 |
| the post-attention norm without its weight | 0.180 | > 0.05 |
| top-10 on the 8-expert file | refused: "the session routes top-10 but the file routes top-2 of 8 experts" | refused by name |
| the fixture's GDN state 16 | refused: "the T-token GDN kernels are built for state 128 and conv 4" | refused by name |
| `blk.4.attn_norm.weight` absent | refused by name | refused by name |

The mixer seams, reported: layers 0-2 `blk.N.attn_qkv.weight is missing`, layer 3 `qsa_layer: position 0 is
outside the RoPE table (0 rows)` (no attention state is built for a mixer that cannot run on F32 weights).

The 9.8e-3 on the router's renormalised weights is the BF16 contract on an F32 weight (2^-9 per element over a
64-long dot product); the picks are identical on this fixture, whose logits are spread for that purpose. The
real file's flip rate is the router branch's measurement (study 3.4).

**Two things found on the way.** (1) `cmake/qwen36_synth_fixtures.cmake` (step 2's) fails on this cmake 3.28
under policy CMP0007 (`list(GET)` on `"toy.gguf;"` with its empty flag element), so `ctest -R qwen36` cannot
reach `qwen36_block_test` until the step-2 branch fixes it; the fixtures were written by hand
(`python3 -I tools/qwen36_synth_gguf.py build/qwen36_synth/toy.gguf`, then the block fixture script with `-P`)
and the test run directly. (2) `read_model_geometry` leaves `n_vocab` at the Flash-Next default (248,320) on the
synthetic file; the engine takes the vocabulary from `output.weight`'s `ne1` (verify.cpp, mtp.cpp), and so does
the test, but a reader of `ModelGeometry::n_vocab` should know it is not the file's.

### VM 211, the GTX 1080 Ti (sm_61, Pascal), 2026-10-11, under `/home/rk/VM-BUSY`

The same tree (`/home/rk/qwen36-step3-block/`, the submodule copied from the step-2 tree), configured with
`-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`, `cmake --build build -j 8 --target qwen36_block_test`
(the engine library and the kernels compile for sm_61), the fixtures by hand as above, then
`./build/qwen36_block_test build/qwen36_synth build/qwen36_block/o8`: all 30 checks pass with **the same numbers as
the 4090 to the printed digit** (1.13e-7, 1.23e-7, 7.02e-8, 4.82e-8, 1.63e-7, 9.82e-3; the window's 1.99e-8 ..
2.81e-8; the negatives 0.423 / 0.276 / 0.180; the router's 64 picks identical). The kernels the plain block adds
to the path are a block-per-row reduction and an elementwise add: nothing Pascal lacks. The lock was held for
the sync, the build, the run and released (about 12 minutes).

## 6. The hunks, for the integrator (against 8b7525e)

| file | hunks (old -> new line ranges) | what |
| --- | --- | --- |
| `include/strata/core/layer.hpp` | 479,8 -> 479,35 | `layer_verify_compatible(g, why)`, the plain-block section, `plain_norm_rows`, `plain_residual_rows` |
| `src/core/layer.cpp` | 473,17 -> 473,22 | `layer_verify_compatible` with the geometry; the no-argument form calls it with the defaults |
| | 1087,6 -> 1092,20 | `lm_head_mix`: the `hc == 1` branch (`output_norm`) in front of the GR mix |
| | 1167,7 -> 1186,115 | the two helpers, `plain_norm_weight`, `plain_block_pre`, `plain_block_post`; `block_layer_pre` dispatches on `g.hc == 1` as its first statement |
| | 1316,7 -> 1443,9 | `block_layer_post` dispatches on `g.hc == 1` as its first statement |
| `src/core/layout.cpp` ([shared]) | 2,6 -> 2,7; 81,6 -> 82,9; 96,6 -> 100,10; 108,6 -> 116,10 | `<cstring>`; the `hc_*` 2-D rows skipped at hc = 1; the two plain norms in the 1-D table, skipped at hc > 1 |
| `include/strata/core/verify.hpp` | 42,6 -> 42,13 | `verify_geometry_supported` |
| `src/core/verify.cpp` | 125,6 -> 125,30 | `verify_geometry_supported` (after the first anonymous namespace) |
| | 329,13 -> 353,12 | `init`: the two refusals |
| | 557,7 -> 580,23 | `record_window`: the PLE refusal, `eps`, `plain_w` |
| | 622,7 -> 661,14; 631,8 -> 677,8 | `pre`: the `hc_*` lookups only at hc > 1, `pending` only at hc > 1 |
| | 693,9 -> 739,16 | `pre`: the first read (`attn_norm` or the GR read) |
| | 899,7 -> 952,15 | `pre`: the residual add and `post_attention_norm`, or the second GR read |
| | 998,7 -> 1059,7 | `fuse_head_gr` only at hc > 1 |
| | 1091,7 -> 1152,15 | `post`: the plain write, else the last-layer materialisation |
| | 1120,10 -> 1189,24 | the head: `output_hc_*` only at hc > 1; the plain head over the T rows |
| `CMakeLists.txt` | 490,6 -> 490,22 | `qwen36_block_test`, `qwen36_block_fixtures` |
| new | `tests/cuda/qwen36_block_test.cpp`, `cmake/qwen36_block_fixtures.cmake`, this file | |

The attention branch's regions (`qsa_layer` ~945-975 at the base, the window's attention step ~757-870, the state
carve) and the GDN branch's (the decode GDN kernels, their launch sites) are not touched: the plain block calls
`qsa_layer` / `gdn_layer` through their existing entry points, and the window's mixer steps are left as they are
between my two hunks at 739 and 952.

## 7. What waits on the sibling branches and step 4

- The mixer seams (section 3) and `block_layer_pre` stage 1 on this model: the GDN gate (sigmoid today, SiLU on the
  GDN branch) and the dense attention (the indexer still runs in `qsa_layer`; the attention branch removes it).
  The window's attention step still builds the indexer's selection and `max_blocks_` from `idx_block` and demands
  the `indexer.*` tensors; that is the attention branch's hunk in `record_window` and in `layer_verify_compatible`'s
  `has_indexer()` conditions are already in place for it.
- The real-file run of the window (the engine's first Qwen3.6 token) needs step 4's pack: the native projections,
  the F32 norms kept F32 (`attn_norm`, `post_attention_norm`, `output_norm` as index kind 2), the router and the
  shexp gate as BF16, and the `qwen35moe_not_yet` gate in `generate.cpp` lifted.
- `check_one`'s indexer rows (`indexer.q_proj`, `indexer.k_proj`, `indexer.*_norm`) are still required on attention
  layers at hc = 1; the attention branch owns them.

