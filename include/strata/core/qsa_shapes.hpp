// include/strata/core/qsa_shapes.hpp - THE ONE builder of a `QsaShapes` from a `ModelGeometry`.
//
// Five sites used to transcribe the same six assignments by hand (layer.cpp, verify.cpp, mtp.cpp, session.cpp and
// prefill.cpp), and the Qwen3.6 port showed what that costs: the geometry says "no indexer" with two zeros
// (`idx_q_heads`, `idx_key_dim`) while the kernels' dense predicate (`qsa_dense`) wants the selection budget
// `idx_top_k` zero as well - a builder that copies the two and keeps `qsa_real_shapes()`'s 2,048 makes a shape
// that is neither sparse nor dense, which `qsa.cu`'s guard refuses as "not set up" at the first KV append.
// Header-only so the parity test can build the shape without linking the engine.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/kernels/qsa.hpp"

namespace strata::core {

/// The kernels' geometry for this model: the attention widths from the file, `n_rot`, `idx_block`, `page_size` and
/// (with an indexer) `idx_top_k` from the kernel contracts in `qsa_real_shapes()`; without an indexer the three
/// indexer widths are zero together, which is what `qsa_dense()` and the kernels' guard test.
inline strata::kernels::QsaShapes qsa_shapes(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    if (!g.has_indexer()) { s.idx_n_head = 0; s.idx_dim = 0; s.idx_top_k = 0; }
    return s;
}

}  // namespace strata::core
