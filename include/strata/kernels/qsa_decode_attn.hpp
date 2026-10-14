// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (min(ceil(cap / 64), 128), n_head_kv):  s = q.k * scale for G heads x 64 cells per sub-chunk, the
//                                                  running max m and sum l, acc = p . V  ->  one partial per block
//     grid (n_head):                               merge the partials with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`), Q4_0 and K8V4.  The grid is sized by
// the capacity; the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
//
// Qwen3.6 (docs/QWEN36-35B-A3B.md 3.3): `G`, the query heads per KV head, is a template parameter (2, 8, 12;
// `qsa_attn_group` picks it from the geometry and refuses any other ratio by name); a NULL `ids` is the identity
// selection, cell j = j for j < step[kStepWidth], which is dense attention without an ids buffer (prefill at
// 2,048 queries x 262,144 cells would otherwise carry 2 GiB of ids); and the number of blocks per KV head is
// capped at 128 FOR DENSE SHAPES so the split-K partials are bounded at 2.02 MiB per query row (16 heads) instead
// of growing to 64.5 MiB at 262,144 cells - each block walks its share of the selection in 64-cell sub-chunks with
// an online rescale, computed from the token's width on the device, so a short selection keeps one block per 64
// cells.  A sparse shape keeps one block per 64 cells at any capacity: the QSA layers (cap 2,051) and the MTP
// drafter's ring (its window, 32,768 by default) run the previous grid, scratch and summation order.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    const int32_t* page_table = nullptr;
};

/// The query heads per KV head the kernels are instantiated for, from the geometry: 2, 8 or 12, else 0 with the
/// refusal in `why` (naming the heads).  The entry points below call this and exit(1) on 0.
int qsa_attn_group(const QsaShapes& s, char* why, size_t why_len);
/// Whether `G` is a built group AND the geometry is `n_head == G * n_head_kv`: a 16-head model asked for G = 12
/// is refused here by name, never run mis-indexed (the negative test of docs/QWEN36-35B-A3B.md section 6, step 3).
bool qsa_attn_group_check(const QsaShapes& s, int G, char* why, size_t why_len);

/// Blocks per KV head for `cap` cells: `ceil(cap / 64)` for a sparse shape (the kernel before Qwen3.6, at any
/// capacity: the MTP drafter's ring runs it at its window), `min(ceil(cap / 64), 128)` for a dense one.  The scratch
/// holds one partial per block.
int64_t qsa_decode_attn_chunks(int64_t cap, const QsaShapes& s);

/// Scratch floats per query row for `cap` selected cells: `chunks(cap, s) * n_head * 258 + 64` (partial
/// accumulators, maxima and sums).  Dense: bounded at 528,448 floats (2.02 MiB) at 16 heads for any cap >= 8,192.
/// Sparse: the previous formula, unchanged.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

/// `ids` (cap,) i32 or NULL for the identity selection (every cell up to step[kStepWidth]).
void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap] or NULL
/// (the identity for every query), steps [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the
/// single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

/// The explicit form of the identity selection: `ids[q * cap + j] = j` for `j < steps[q][kStepWidth]`, for `n_q`
/// rows.  Capturable (the widths are read on the device).  The engine's dense paths pass NULL instead; this exists
/// for a caller that must hand a buffer to something else (the KV-streaming resolver, a dump) and for the parity
/// test, which holds the two forms bit for bit equal.
void qsa_identity_ids(int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, void* stream);

}  // namespace strata::kernels
