# Qwen3.6 port, Phase 2 step 3: dense attention replacing QSA

Working notes of the `attn` area (branch `qwen36-step3-attn`, from `qwen36-step2` at 8b7525e, 2026-10-11). The
design is `docs/QWEN36-35B-A3B.md` section 3.3 (six items) and section 6 step 3, second bullet. Every item says
what was decided, where it lives, how it is checked, and the numbers from both cards.

## 1. What changed, item by item

| 3.3 item | Decision | Where |
| --- | --- | --- |
| 1, the producer | A **null `ids` is the identity selection**: the decode and prompt kernels read cell `j = j` for `j < step[kStepWidth]`. The engine's dense paths pass `nullptr` and carve no ids buffer (prefill at 2,048 queries x 262,144 cells would carry 2 GiB of ids). The explicit form `qsa_identity_ids(ids, steps, n_q, cap, stream)` exists for a caller that must hand a buffer on (a dump, the streaming resolver) and for the test, which holds the two forms bit for bit equal | `src/kernels/cuda/qsa_decode_attn.cu`, `qsa_prompt_attn.cu`; `qsa_layer` and the verify window's attention step pass null |
| 1, the indexer skipped | `g.has_indexer()` (step 2: false for `qwen35moe`) gates the raw key projection, the append, the indexer query, the block scores and the top-k in `qsa_layer` and in `Verifier`'s attention step; the four `indexer.*` tensors are required only with an indexer | `src/core/layer.cpp` (c4793cd), `src/core/verify.cpp` (bc2fb91), each one commit |
| 2, G = 12 | `G` is a template parameter of every attention kernel, instantiated at **2, 8, 12**; `qsa_attn_group(s, why, n)` picks it from the geometry and returns 0 with a message naming the heads for any other ratio; `qsa_attn_group_check(s, G, why, n)` refuses a G that does not divide the geometry (G = 12 on 16 / 2 is a message, never a mis-indexed run); `qsa_prompt_attn_batch` returns false for an unknown ratio and the decode kernel then refuses by name | `qsa_decode_attn.cu`, `qsa_prompt_attn.cu`, `include/strata/kernels/qsa_decode_attn.hpp` |
| 3, the selection width and the capacity | **One builder**, `strata::core::qsa_shapes(g)` (`include/strata/core/qsa_shapes.hpp`, header-only), used by `layer.cpp`, `verify.cpp`, `mtp.cpp`, `session.cpp` and `prefill.cpp` (the last two as `[shared]` commits): for a geometry without an indexer it zeroes `idx_n_head`, `idx_dim` **and** `idx_top_k`. The dense signal is `qsa_dense(s)` = the three zero together; `qsa_shapes_valid(s, why, n)` (what every kernel's guard checks, non-exiting for the test) refuses a half-set shape - two zeros with the 2,048 budget kept, which the first hand-written builders in verify and mtp made and the kernels' guard then exited on at the first append. `qsa_selection_width(n_kv, s)` returns `n_kv` when dense, so `qsa_step_fill` writes `width = n_kv`; **`qsa_selection_cap(max_cells, s)`** is the capacity every buffer is sized by: `max_cells` when dense, `qsa_selection_width(kTopkMaxCells, s)` = 2,051 otherwise. The indexer entry points refuse a dense shape by name (`validate_indexer`). Not converted: `generate.cpp:7469` (a snapshot-hash shape that sets only `n_head_kv`/`head_dim`/`idx_dim`; not edited on this branch) | `qsa_shapes.hpp`, `qsa.hpp`, `qsa.cu`, `layer.cpp`, `verify.cpp`, `mtp.cpp`, `session.cpp`, `prefill.cpp` |
| 4, the state without the indexer | `kv_plan` gives a dense state `pooled_rows = 2` (the drafter's ring) and `idx_dim` is 0, so the three indexer regions are 0 bytes (section 3). The buffers carve no `cell_scores`, `ids`, `k_scratch`, `v_scratch` for the dense model: at 262,144 cells those are 1 + 1 + 512 + 512 MiB (the gather scratch was `cap x n_head_kv x 256 x 2 B` each) | `layer.cpp`: `kv_plan`, `qsa_state_bytes/init`, `qsa_buffers_bytes/init` |
| 5, the split-K scratch | The grid is capped at **128 blocks per KV head**; a block walks its span of the selection in 64-cell sub-chunks with the running (m, l, acc) rescale. The span is computed **on the device from the token's width**, not from the capacity, so a selection up to 8,192 cells still spreads one block per 64 cells (decode at a short depth keeps its parallelism). **The bound applies to dense shapes only**: a sparse (Flash-Next) shape keeps `ceil(cap / 64)` blocks at any capacity, because the MTP drafter's ring runs this kernel at its `--mtp-window` (32,768 by default: 512 blocks, `mtp.cpp:249`) and keeps its previous grid, scratch (12 MiB at 24 heads) and summation order. On sm_75+ one looped kernel serves every capacity (the 4090 measures it equal to the old one at 2,051 cells). Below sm_75 (`STRATA_EXPERIMENTAL_SM60`, PR #540's kernel) the looped form costs 16-28 registers on sm_61 (80-92 at G 12 against 63-64; a register cap spilled and cost more), so every sparse shape, and a dense capacity under 8,192 cells, runs `attn_chunk_kernel_pre75_single`, PR #540's body as it was: every Flash-Next state (the QSA layers at cap 2,051, the drafter's ring at its window) is the old kernel to the instruction, and the looped `attn_chunk_kernel_pre75` runs only for the dense model's long-context states. `qsa_decode_attn_scratch_floats(cap, s) = min(ceil(cap / 64), 128) x n_head x 258 + 64`: **528,448 floats = 2.02 MiB per query row at 16 heads** for any cap >= 8,192 (was 64.5 MiB at 262,144); 16 MiB for the 8-row verify window, 64 MiB for prefill at `attn_batch` 32 | `qsa_decode_attn.cu` (`kMaxChunks`, `chunks_for`, both chunk kernels, `attn_merge_kernel<G>`) |
| 6, KV streaming | `qsa_kv_streaming_applies(g, why)` (false for a no-indexer geometry, with the sentence that says why: every block resident every token is the whole KV over PCIe per token); `qsa_state_init` refuses a streamed dense state with that message as the backstop. The CLI check belongs in `generate.cpp`, which this branch does not edit (the step-2 branch owns it): the integrator wires `--kv-resident > 0 && !qsa_kv_streaming_applies(g, why)` into the start-up refusal | `include/strata/core/layer.hpp`, `layer.cpp` |

The formats carry over unchanged: the fp16 / INT8 / Q4_0 / K8V4 pools, the Hadamard rotation (`kv_rot`: q and the
output rotated in `qsa_layer` and in verify exactly as before), the page table (shuffled in every test), the
RoPE tables (the test builds them with `build_rope_table` at base 1e7 and holds `rope_neox_apply` to the oracle).

The prompt path on Pascal is the batched decode kernel (`qsa_prompt_attn_batch` returns false below cc 70), so
G = 8 and the scratch bound above are the prefill attention there; on the 4090 the MMA kernels run at G = 8 (and
2, 12). The MMA kernels pad the 16-row tile with zero rows for any G <= 16 exactly as 12 did; one real bug came
out of G = 2: the two `mma` kernels wrote accumulator rows 0..7 unconditionally, which at G < 8 overwrote the next
KV head's output (the 4 / 2 fixture read 0.99 relative L2). The writes are guarded by G now (856daa0).

## 2. The dense state's layout, exactly

`qsa_state_init` carves one 16-byte-aligned cursor (`layer.cpp`); for a no-indexer geometry (`pooled_rows = 2`,
`idx_dim = 0`, `idx_block = 4`, mode 0, no ring) the regions are, in order:

| region | bytes | note |
| --- | --- | --- |
| K/V pools | `kv_pool_bytes(s, pages)` + 64: fp16 `pages x 2 x page_size x 256 x 2 x 2`; INT8 `pages x page_size x 1,056 + 64`; Q4_0 `pages x page_size x 576 + 64`; K8V4 `rows x 256 + rows x 8 + rows x 144 + 64` | `pages = ceil(max_cells / 4)`; 1,024 MiB fp16 / 528 INT8 / 288 Q4_0 / 408 K8V4 at 262,144 cells |
| `page_table` | `pages x 4` | 256 KiB at 262,144 |
| residency map | none | mode 1 is refused for the dense model |
| `idx_tail` | `3 x idx_dim x 4` = **0** | pointer taken, nothing carved |
| `idx_dead` | `idx_dim x 4` = **0** | |
| `idx_pooled` | `pooled_rows x idx_dim x 4` = **2 x 0 x 4 = 0** | `idx_pooled_rows = 2` as the drafter's ring |
| `idx_block_pos` | 16 (one int32, aligned) | kept: `QsaIndexerBuffers` is still a struct of four pointers |
| `cos_tab`, `sin_tab` | `max_cells x 32 x 4` each, in the primary state only; the others borrow | 32 MiB each at 262,144 (`n_rot` 64) |
| `step` | 16 | `kStepCount` int32 |
| `attention_status` | 16 | |
| `pos_dev` | `n_head x 4` -> 64 | |
| trailer | +256 | `qsa_state_bytes` adds 256 after the alignment |

So the indexer's 32 MiB per attention layer at 262,144 cells (`65,538 x 128 x 4`) is gone; a dense state is the
pools, the page table, the rope tables (primary only) and 112 bytes. `qsa_state_zero` memsets the 0-byte regions
(a no-op). The buffers (`qsa_buffers_bytes`, one set per session): `x_q8k`, `x_q8_0`, `x_bf16`, `q_full`, `qcur`,
`kcur`, `vcur`, `idx_raw` (0), `q_idx` (0), **no** `cell_scores` / `ids` / `k_scratch` / `v_scratch`, `attn`,
`attn16`, `attn32`, `attn_q8k`, `attn_scratch` (2.02 MiB at 16 heads, any `max_cells` >= 8,192).

`conversation_state.cpp` / `conversation_snapshot.cpp` (the session save and restore) still address the indexer
regions by `qsa_real_shapes().idx_block`; with `idx_dim = 0` their sizes are 0. Not exercised here (step 4's area).

## 3. How it is checked

`src/kernels/qsa_dense_parity.cpp` (CTest `qsa_dense_parity`, after the fixture test `qwen36_attn_fixtures`, which
runs `cmake/qwen36_attn_fixtures.cmake`: `tools/qwen36_synth_gguf.py --heads 16,2|24,2|4,2 --head-dim 256
--no-nextn` and `ref/qwen36.py --text "dense attention over sixteen heads, G=8!" --dump` on each; 40 tokens; about
1 s). The `--head-dim` option is the one `[shared]` change (7368da2): the toy fixture is head_dim 32 and the
engine's kernels are 256, so an engine-level comparison against the oracle needs a fixture at 256 (`n_rot` 64,
the real IMRoPE sections in proportion, base 1e7).

1. **Host only.** The one builder on a `qwen35moe` geometry gives a dense shape the guard accepts, on Flash-Next's defaults the sparse real shape; the half-set shape (two zeros, budget 2,048) is neither dense nor valid (the negative). The chunk count by shape: dense 2,051 -> 33, 8,192 / 262,144 -> 128; sparse 32,768 -> 512 with the old scratch formula. The group from the geometry (16/2 -> 8, 24/2 -> 12, 4/2 -> 2, 26/2 refused naming 26 and 13);
   **G = 12 asked of 16 heads refused by name**; width and cap (sparse 3,000 -> 2,051 / 2,051; dense 3,000 ->
   3,000 / `max_cells`); `qsa_step_fill` width; `chunks` (2,051 -> 33, 8,192 and 262,144 -> 128); the scratch
   at 2,051 x 24 heads equals the old formula and at 262,144 x 16 heads is 528,448 floats.
2. **The decode kernel against a host reference in double over the dequantized pools**, every KV format (fp16,
   INT8, Q4_0, K8V4; the pools filled through the engine's own append kernels behind a shuffled page table,
   read back and dequantized exactly as the kernel's loads do), at G = 8 (16 / 2) and G = 12 (24 / 2), widths
   300, 3,000 and 20,000 (the last one takes the multi-sub-chunk path: 313 sub-chunks over 128 blocks), three
   query rows per case with widths w, w - 1, w - 77 and their own steps; **null ids == explicit identity ids bit
   for bit**; the prompt kernel on the same rows where the host takes it (sm_70+). G = 2 fp16 at 3,000.
   **Negative:** the sparse width 2,051 handed to the kernel on the 3,000-cell context (what the old
   `qsa_selection_width` would have written for Qwen3.6) differs from the dense reference by 0.65-0.69 relative
   L1.
3. **The oracle's fixtures.** For g8, g12, g2: `Qcur_reshaped-3` and `Kcur-3` through the engine's
   `rms_norm_weighted` and `rope_neox_apply` (table from `build_rope_table`) against `Qcur-3` and `Kcur-3#2`;
   the softmax of the engine's q/k (host, double) against `kq_soft_max-3`; the attention over the engine's fp32
   q/k/v (host, double) against `kqv-3`; then the **fp16 KV through `kv_append_step` and the batched decode kernel
   with null ids, 40 queries at widths 1..40**, against `kqv-3`; `qsa_gate_apply_f32` on `Qcur_full-3` against
   `attn_gated-3`. **Negatives, the oracle's controls:** the gate skipped (relative L1 0.995-0.998 at
   `attn_gated-3`; the oracle's table says 1.03 on its own fixture), RoPE on adjacent pairs from the oracle's
   normed q/k (0.17-0.20 at `kqv-3`; the oracle: 0.20), the GQA map `h % n_kv` (0.69-0.72; the oracle: 0.74).
4. `src/kernels/qsa_attn_bench.cpp` (CTest `qsa_attn_bench`, 2,051 and 8,192 cells at 16 / 2): the kernel's
   time per call on random pools, explicit ids only, so it builds against the pre-step-3 kernel as well (the
   "before" column below is that build: `8b7525e`'s `qsa.hpp`, `qsa_decode_attn.{hpp,cu}`, `qsa_prompt_attn.cu`
   dropped into a copy of this tree).

Tolerances and what was measured (both cards):

| check | tolerance | 4090 (sm_89) | 1080 Ti (sm_61) |
| --- | --- | --- | --- |
| decode kernel vs host reference, dequantized pools, all formats, G 8 / 12 / 2 | 3e-5 rel L2 | 2.0e-7 .. 3.9e-7 | 2.0e-7 .. 3.9e-7 (the `pre75` kernel runs there) |
| prompt kernel vs host reference (fp16 / INT8 / Q4_0 / K8V4), G 8 / 12 / 2 | 2e-3 | 4.1e-7 .. 6.3e-7; K8V4 1.6e-4 | the host keeps the decode kernel (cc < 70): skipped |
| engine norm + RoPE vs `Qcur-3`, `Kcur-3#2` | 2e-6 | 8.1e-8 .. 9.8e-8 | 8.1e-8 .. 9.8e-8 |
| host softmax of the engine's q/k vs `kq_soft_max-3` | 1e-5 | 1.6e-7 .. 1.7e-7 | 1.6e-7 .. 1.7e-7 |
| host attention over the engine's fp32 q/k/v vs `kqv-3` | 1e-5 | 1.5e-7 .. 1.7e-7 | 1.5e-7 .. 1.7e-7 |
| decode kernel over the **fp16** KV vs `kqv-3`; gated vs `attn_gated-3` | 1.5e-3 | 2.2e-4 .. 2.3e-4 | 2.2e-4 .. 2.3e-4 |

The 2.2e-4 is the fp16 KV pool (K and V rounded to 11 bits; the oracle's are F32): the same q/k/v in fp32 give
1.5e-7, so the kernel's own arithmetic is at the 1e-7 level and the format is the whole gap. The policy's
`attn_out` class (`oracle-crosscheck.md`: 1.1e-4 median, 6.7e-4 max for the Q8 oracle, 4x margin) is the right
class for it and 1.5e-3 is inside that margin.

The rest of the attention suite on the branch: `qsa_parity`, `qsa_prompt_attn_parity`, `kv_hybrid_parity`,
`kv_stream_parity`, `qsa_topk_parity`, `qsa_topk_active_parity` (the Flash-Next geometry, G = 12: the
instantiation Flash-Next keeps). `kv_hybrid_parity`'s case [5/5] now skips below sm_70 instead of failing (the
prompt kernel refuses cc < 70 by design and prefill takes the batched decode kernel, which case [4/4] holds to
the same reference).

## 4. The numbers

`qsa_attn_bench`, one query row, identity selection of `cells` cells, median of the reps, microseconds per call
(the chunk kernel plus the merge). "before" is the pre-step-3 kernel (G = 12 only: it refuses 16 heads).

**RTX 4090 (sm_89), VM 221**, `--reps 200` (24 / 2) and `100` (16 / 2):

| cells | 24/2 fp16 before -> after | 24/2 INT8 | 24/2 Q4_0 | 16/2 fp16 | 16/2 INT8 | 16/2 Q4_0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,051 | 32.8 -> 32.7 | 33.8 -> 34.5 | 34.6 -> 34.8 | 29.3 | 29.4 | 30.3 |
| 8,192 | 57.2 -> 57.0 | 58.2 -> 57.3 | 59.2 -> 59.2 | 48.0 | 47.4 | 49.2 |
| 32,768 | 215.7 -> 148.4 | 205.0 -> 143.2 | 212.6 -> 150.7 | 130.9 | 110.5 | 115.7 |

(A rerun of the final build: fp16 32.8 / 56.3 / 138.0, Q4_0 34.6 / 58.3 / 147.5 at 2,051 / 8,192 / 32,768; the
sm_89 build runs the one looped kernel at every capacity.)
| 131,072 | | | | 540.7 | 539.6 | 467.9 |
| 262,144 | | | | 1,032 | 1,051 | 1,053 |

At 2,051 cells on Flash-Next's path the kernel is the same grid and the same arithmetic (33 blocks of one
sub-chunk; the test holds the scratch formula equal there) and the time is the same within the run-to-run
spread (32.7 vs 32.8 us; INT8 34.5 vs 33.8 is the spread of a 34 us launch, the Q4_0 pair is 34.8 vs 34.6). Above
8,192 the bounded grid is faster (fewer, longer blocks: 148 vs 216 us at 32,768) and the scratch stays at 2.0 /
3.0 MiB instead of 12. At 262,144 cells, 16 heads, one row costs 1.03 ms on the 4090 whatever the format: the
fp16 pool streams at 520 GB/s and the quantized pools are bound by the dequantization in the value loop, not by
bytes (INT8 263 GB/s, Q4_0 143 GB/s over fewer bytes, the same time). Ten attention layers at 256K are therefore
about 10 ms of a token on this card.

**GTX 1080 Ti (sm_61), VM 211** (the `attn_chunk_kernel_pre75` path of `STRATA_EXPERIMENTAL_SM60`), `--reps 100`
(24 / 2) and `50` (16 / 2):

| cells | 24/2 fp16 before -> after | 24/2 INT8 | 24/2 Q4_0 | 16/2 fp16 | 16/2 INT8 | 16/2 Q4_0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,051 | 41.0-42.0 -> 41.0-41.9 | 38.9 -> 38.9-39.9 | 41.0-42.0 -> 41.0 | 43.0 | 33.8 | 34.8 |
| 8,192 | 141.3-142.3 -> 140.3-141.3 | 141.3 -> 140.3-141.3 | 146.4-147.5 -> 143.4-144.4 | 140.4 | 121.9 | 135.2 |
| 32,768 | 493.6-494.6 -> 494.4-494.6 | 496.6 -> 496.6-497.4 | 505.9-506.9 -> 497.6-497.7 | 378.9 | 330.8 | 373.8 |
| 131,072 | | | | 1,290 | 1,126 | 1,287 |
| 262,144 | | | | 2,354 | 2,117 | 2,605 |

The before / after pairs are interleaved A/B runs (rounds 2 and 3 of three; round 1 is the warm-up). The 24/2
shape is sparse, so after e7c9c8d both sides are the same grid (one block per 64 cells at every capacity, the
same 12.09 MiB scratch at 32,768) and the same single-chunk kernel body: the pairs agree within the 1 us
run-to-run spread at every capacity and format, which is the drafter's case (its ring at 32,768) as much as the
QSA layers' (2,051). The 16/2 columns are the dense model's states (cap >= 131,072 in practice), which take the
looped kernel on Pascal: on Q4_0 it is 10-15% slower than the single-chunk body would be at short depths (the
looped body's extra registers cost the Q4_0 value loop a block per SM; a `__launch_bounds__(THREADS, 4)` cap
spilled and made every format slower, 44.0 / 44.0 / 47.1 us at 2,051 in that build), about 0.15 ms of a token
at 8K depth, which the A2 floor cannot see; at long depths it is ahead of the unbounded grid (465 vs 504 us at
32,768 in the 24/2 trial that measured the looped kernel on a sparse shape).

At 262,144 cells, 16 heads, one row is 2.1 ms INT8 / 2.4 ms fp16 / 2.6 ms Q4_0 on the 1080 Ti: 21-26 ms of a
token for the ten attention layers at the full context (the fp16 pool streams at 228 GB/s of the card's 484;
Q4_0 is bound by the nibble decode in the value loop at 58 GB/s over a quarter of the bytes). At 8K depth the ten
layers are 1.2-1.4 ms. The verify window multiplies the per-row figure by its rows (8), so the window's attention
at 256K is about 17-21 ms on this card; A2's 10 tok/s at 32K depth leaves 100 ms per token, of which the
attention is 3.3-3.8 ms.

## 5. Commands

4090 (`/models/work/qwen36-step3-attn`, `build-step3-attn.sh` = the configure of `step3-common.md` with
`-DCMAKE_CUDA_ARCHITECTURES=89`, 162 s for the full tree):

    export STRATA_GGUF_PY=/models/work/qwen36-step3-attn/third_party/llama.cpp/gguf-py
    cd build && ctest --output-on-failure -R "qwen36_attn_fixtures|qsa_dense_parity|qsa_parity|qsa_prompt_attn_parity|kv_hybrid_parity|kv_stream_parity|qsa_topk"
    ./qsa_dense_parity --fixtures qwen36_attn
    for kv in fp16 int8 q4_0; do ./qsa_attn_bench --heads 24,2 --kv $kv --cells 2051,8192,32768 --reps 200; done
    for kv in fp16 int8 q4_0; do ./qsa_attn_bench --heads 16,2 --kv $kv --cells 2051,8192,32768,131072,262144 --reps 100; done
    # the "before" column: the same bench built in a copy of the tree with 8b7525e's four attention files
    /models/work/qwen36-step3-attn-base/build/qsa_attn_bench --heads 24,2 --kv fp16 --cells 2051,8192,32768 --reps 200

1080 Ti (`/home/rk/qwen36-step3-attn`, `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`, under
`/home/rk/VM-BUSY`): the same, by `/home/rk/pascal-run-attn.sh` (logs in `/home/rk/logs/qwen36-step3-attn-*.log`).

## 6. Not done, surprises, and where the study was off

- `generate.cpp` is not edited (the step-2 branch owns it): the `--kv-resident` refusal for a dense model is a
  predicate (`qsa_kv_streaming_applies`) plus the backstop in `qsa_state_init`; the integrator adds the one call.
- `prefill.cpp` is not edited (the prefill agent's): its three sites take `qsa_selection_cap`, a null `sel_ids`
  for the dense model, and `attn_scratch` at the new bound (section 1).
- The Volta (`launch70`, m8n8k4) and Turing prompt kernels are templated on G but no such card was on hand; the
  4090 runs `launch_i8` / `launch<>`, Pascal runs none of them. Flash-Next's G = 12 on those paths is unchanged
  code apart from the template parameter.
- No test drives `Verifier` (the verify window) on a `qwen35moe` geometry yet; its attention step is checked by inspection and by the builder test above. That is step 4's streaming-runner test (the window against the oracle's `l_out-N` per position).
- `layer_verify_compatible` still demands the native QSA indexer flag for the verify window; it is a global
  (on by default) and not a geometry question, so it is left for the block agent's verify rewrite.
- Pascal's looped kernel is 10-15% slower than the single-chunk one on the Q4_0 pool at short depths (section 4);
  the single-chunk body is kept for capacities under 8,192 so Flash-Next's path is untouched, and the dense model
  pays it only below about 8K of depth, where the attention is a small part of the token.
- The study's 3.3 item 5 sizes the scratch "(cap/64) x n_head x 258 + 64 ... 64.5 MiB per query row": right.
  Its proposed fixes were "a larger chunk per block, or an online-softmax merge that does not keep a partial per
  64-cell chunk": the kernel does the second inside the block (the running rescale) while keeping the 64-cell
  sub-chunk as the unit of work, so the short-context grid and arithmetic are untouched. The bound came out at
  2.02 MiB, not the 2 MiB-per-row-at-128-chunks guess rounded either way: 528,448 floats.
- Item 4 says the drafter's ring passes "pooled_rows = 2"; what makes the dense state small is `idx_dim = 0`
  (the tail and the dead key are sized by it too), the 2 rows alone would still have carved `3 x 128 x 4 + 128 x 4
  + 2 x 128 x 4` bytes. Both are in place.
- Nothing in the study mentions `qsa.cu`'s geometry guard (`validate`: `idx_dim <= 0 || idx_n_head <= 0` is
  "geometry is not set up", and `n_rot > idx_dim` is refused): every KV append would have exited on the dense
  shape. The guard now distinguishes a dense shape (all three indexer widths 0) from a half-set-up one.
- The oracle's negative controls reproduce on the head_dim-256 fixtures at the same order (gate 1.0, adjacent
  RoPE 0.17-0.20, GQA map 0.69-0.72); the oracle's own table was on its head_dim-32 fixture.
