// src/kernels/qsa_dense_parity.cpp - dense attention (Qwen3.6-35B-A3B, docs/QWEN36-35B-A3B.md 3.3) through the
// engine's split-K decode kernel and the tensor-core prompt kernel: the query-head group G as a template
// parameter (2, 8, 12), a null `ids` as the identity selection, the bounded split-K scratch, and the sparse
// selection width refused for the dense model.  GPU, synthetic, plus the oracle's fixtures when given.
//
//   qsa_dense_parity [--fixtures DIR] [--selftest]
//
// DIR is what cmake/qwen36_attn_fixtures.cmake writes: <name>.gguf and <name>-dump/ (ref/qwen36.py's seams on 40
// tokens) for g8 (16 / 2), g12 (24 / 2) and g2 (4 / 2) at head_dim 256.  The dumped `Qcur_reshaped-3`, `Kcur-3`,
// `Vcur-3#2` and `Qcur_full-3` go through the engine's own RMS norm, RoPE table, fp16 KV append, decode attention
// and output gate, and are held to the oracle's `Qcur-3`, `Kcur-3#2`, `kq_soft_max-3`, `kqv-3`, `attn_gated-3`.
//
// The negative-test discipline (docs/qwen36-port/upstream-tooling.md section 3): each rival reading must differ by
// more than 5% relative L1 before the right one is judged - the gate skipped (the oracle's 1.03 at attn_gated-3),
// RoPE on adjacent pairs instead of NEOX halves (0.20 at kqv-3), the GQA map `h % n_kv` (0.74 at kqv-3), G = 12 on
// 16 heads (refused by name), and the sparse width 2,051 on a 3,000-cell context (every cell past 2,051 unread).
//
// Exit 0 pass, 1 fail, 77 skip (no CUDA device, or --fixtures given and the fixtures are missing).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/model_geometry.hpp"
#include "strata/core/qsa_shapes.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;
void verdict(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  %-78s %s%s%s\n", name.c_str(), ok ? "ok" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++g_fail;
}
std::string sci(double v) { char b[32]; std::snprintf(b, sizeof b, "%.2e", v); return b; }

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
template <typename T> T* dalloc(size_t n) {
    T* d = nullptr;
    ck(cudaMalloc(&d, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(d, 0, n * sizeof(T) + 256), "memset");
    return d;
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// ||a - b|| / ||b|| (L2) and sum|a - b| / sum|b| (L1), NaN-safe: a non-finite value is an infinite error
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        n += (double) (a[i] - b[i]) * (a[i] - b[i]);
        d += (double) b[i] * b[i];
    }
    return d > 0 ? std::sqrt(n / d) : (n > 0 ? INFINITY : 0.0);
}
double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        n += std::fabs((double) a[i] - b[i]);
        d += std::fabs((double) b[i]);
    }
    return d > 0 ? n / d : (n > 0 ? INFINITY : 0.0);
}
bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// ---- the host reference: causal-free attention of one query over `cells` dequantized K/V rows, in double.
//   q [n_head][HD], K/V [cell][kv][HD]; the GQA map is grouped (h / G) unless `mod_map`; probabilities out if asked
constexpr int HD = 256;
void ref_attend(const std::vector<float>& q, const std::vector<double>& K, const std::vector<double>& V, int64_t cells,
                int64_t width, int64_t n_head, int64_t n_kv, std::vector<float>& out, std::vector<double>* probs,
                bool mod_map = false) {
    const int64_t G = n_head / n_kv;
    const double scale = 1.0 / std::sqrt((double) HD);
    out.assign((size_t) n_head * HD, 0.f);
    if (probs) probs->assign((size_t) n_head * cells, 0.0);
    std::vector<double> sc((size_t) width);
    for (int64_t h = 0; h < n_head; ++h) {
        const int64_t kv = mod_map ? h % n_kv : h / G;
        double m = -INFINITY;
        for (int64_t j = 0; j < width; ++j) {
            double s = 0;
            const double* kr = K.data() + ((size_t) j * n_kv + kv) * HD;
            for (int d = 0; d < HD; ++d) s += kr[d] * (double) q[(size_t) h * HD + d];
            sc[j] = s * scale;
            m = std::max(m, sc[j]);
        }
        double l = 0;
        for (int64_t j = 0; j < width; ++j) { sc[j] = std::exp(sc[j] - m); l += sc[j]; }
        std::vector<double> acc(HD, 0.0);
        for (int64_t j = 0; j < width; ++j) {
            const double p = sc[j] / l;
            if (probs) (*probs)[(size_t) h * cells + j] = p;
            const double* vr = V.data() + ((size_t) j * n_kv + kv) * HD;
            for (int d = 0; d < HD; ++d) acc[d] += p * vr[d];
        }
        for (int d = 0; d < HD; ++d) out[(size_t) h * HD + d] = (float) acc[d];
    }
}

// ---- NEOX RoPE on the host, first n_rot dims, pairs (i, i + n_rot/2) in float64 (the engine's table is float64 too),
//      or the rival adjacent-pair reading (2i, 2i+1)
void host_rope(std::vector<float>& x, int64_t rows, int n_rot, double theta, const std::vector<int>& pos, bool adjacent) {
    for (int64_t r = 0; r < rows; ++r) {
        float* v = x.data() + (size_t) r * HD;
        for (int i = 0; i < n_rot / 2; ++i) {
            const double f = std::pow(theta, -2.0 * i / n_rot);
            const double a = pos[(size_t) r] * f, c = std::cos(a), s = std::sin(a);
            const int ia = adjacent ? 2 * i : i, ib = adjacent ? 2 * i + 1 : i + n_rot / 2;
            const double x0 = v[ia], x1 = v[ib];
            v[ia] = (float) (x0 * c - x1 * s);
            v[ib] = (float) (x0 * s + x1 * c);
        }
    }
}

// ---- Walsh-Hadamard of a 256-row, orthonormal (scale 1/16), self-inverse: kv_q4.hpp's rotation
void fwht256_host(double* x) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += 2 * len)
            for (int j = i; j < i + len; ++j) { const double a = x[j], b = x[j + len]; x[j] = a + b; x[j + len] = a - b; }
    for (int i = 0; i < 256; ++i) x[i] *= 1.0 / 16.0;
}

// ---- a paged KV state of `cells` capacity in one of the four formats, filled through the engine's own append kernels
struct Pools {
    int mode = 0;   // 0 fp16, 1 int8, 2 q4_0, 3 k8v4
    int64_t n_kv = 0, pages = 0, rows = 0;
    std::vector<int32_t> table;   // logical page -> physical (shuffled)
    k::QsaAttnPools p;
    uint16_t *k_pool = nullptr, *v_pool = nullptr, *k_scale = nullptr, *v_scale = nullptr;
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint8_t *k_q4 = nullptr, *v_q4 = nullptr;
    int32_t* d_table = nullptr;
    bool rot_k = false, rot_v = false, rot_q = false;   // what the format rotates (the kernel's output follows V)
    std::vector<double> K, V;   // dequantized, [cell][kv][HD], as the kernel will read them (rotated where the format does)
};

Pools make_pools(int mode, int64_t cells, const k::QsaShapes& s, std::mt19937& rng, const std::vector<float>& kcur,
                 const std::vector<float>& vcur) {   // kcur/vcur [cell][kv][HD] f32, unrotated
    Pools P;
    P.mode = mode;
    P.n_kv = s.n_head_kv;
    P.pages = (cells + s.page_size - 1) / s.page_size;
    P.rows = P.pages * s.n_head_kv * s.page_size;
    P.table.resize((size_t) P.pages);
    for (int64_t i = 0; i < P.pages; ++i) P.table[i] = (int32_t) i;
    std::shuffle(P.table.begin(), P.table.end(), rng);
    P.d_table = up(P.table);
    P.p.page_table = P.d_table;
    const uint64_t q4row = k::kv_q4_bytes_per_head(HD);
    if (mode == 0) { P.k_pool = dalloc<uint16_t>(P.rows * HD); P.v_pool = dalloc<uint16_t>(P.rows * HD); P.p.k_pool = P.k_pool; P.p.v_pool = P.v_pool; }
    else if (mode == 1) {
        P.k_q = dalloc<int8_t>(P.rows * HD); P.v_q = dalloc<int8_t>(P.rows * HD);
        P.k_scale = dalloc<uint16_t>(P.rows * (HD / k::KV_Q8_GROUP)); P.v_scale = dalloc<uint16_t>(P.rows * (HD / k::KV_Q8_GROUP));
        P.p.k_q = P.k_q; P.p.v_q = P.v_q; P.p.k_scale = P.k_scale; P.p.v_scale = P.v_scale;
    } else if (mode == 2) {
        P.k_q4 = dalloc<uint8_t>(P.rows * q4row); P.v_q4 = dalloc<uint8_t>(P.rows * q4row);
        P.p.k_q4 = P.k_q4; P.p.v_q4 = P.v_q4;
        P.rot_k = P.rot_v = P.rot_q = true;
    } else {
        P.k_q = dalloc<int8_t>(P.rows * HD); P.k_scale = dalloc<uint16_t>(P.rows * (HD / k::KV_Q8_GROUP));
        P.v_q4 = dalloc<uint8_t>(P.rows * q4row);
        P.p.k_q = P.k_q; P.p.k_scale = P.k_scale; P.p.v_q4 = P.v_q4;
        P.rot_v = true;
    }
    // the cells through the append kernels, one step buffer per cell, V (and K) rotated as the engine does first
    std::vector<int32_t> steps((size_t) cells * k::kStepCount);
    for (int64_t c = 0; c < cells; ++c) k::qsa_step_fill(steps.data() + c * k::kStepCount, c, s);
    int32_t* d_steps = up(steps);
    float* d_k = up(kcur);
    float* d_v = up(vcur);
    if (P.rot_k) k::fwht256_inplace_cuda(d_k, cells * s.n_head_kv, nullptr);
    if (P.rot_v) k::fwht256_inplace_cuda(d_v, cells * s.n_head_kv, nullptr);
    for (int64_t c = 0; c < cells; ++c) {
        const int32_t* st = d_steps + c * k::kStepCount;
        const float* kc = d_k + c * s.n_head_kv * HD;
        const float* vc = d_v + c * s.n_head_kv * HD;
        if (mode == 0) k::kv_append_step(P.k_pool, P.v_pool, P.d_table, st, kc, vc, s, nullptr);
        else if (mode == 1) k::kv_append_q8_step(P.k_q, P.v_q, P.k_scale, P.v_scale, P.d_table, st, kc, vc, s, nullptr);
        else if (mode == 2) k::kv_append_q4_step(P.k_q4, P.v_q4, P.d_table, st, kc, vc, s, nullptr);
        else {
            k::kv_append_q8_step(P.k_q, P.k_q, P.k_scale, P.k_scale, P.d_table, st, kc, kc, s, nullptr);
            k::kv_append_q4_step(P.v_q4, P.v_q4, P.d_table, st, vc, vc, s, nullptr);
        }
    }
    ck(cudaDeviceSynchronize(), "append");
    // read the pools back and dequantize exactly as the kernel's loads do
    auto row_of = [&](int64_t cell, int64_t kv) { return ((int64_t) P.table[(size_t) (cell / s.page_size)] * s.n_head_kv + kv) * s.page_size + cell % s.page_size; };
    P.K.assign((size_t) cells * s.n_head_kv * HD, 0.0);
    P.V.assign((size_t) cells * s.n_head_kv * HD, 0.0);
    auto deq_f16 = [&](const uint16_t* pool, std::vector<double>& out) {
        const auto h = down(pool, (size_t) P.rows * HD);
        for (int64_t c = 0; c < cells; ++c) for (int64_t kv = 0; kv < s.n_head_kv; ++kv) for (int d = 0; d < HD; ++d)
            out[((size_t) c * s.n_head_kv + kv) * HD + d] = k::f32_from_f16(h[(size_t) row_of(c, kv) * HD + d]);
    };
    auto deq_q8 = [&](const int8_t* codes, const uint16_t* scales, std::vector<double>& out) {
        const auto hc = down(codes, (size_t) P.rows * HD);
        const auto hs = down(scales, (size_t) P.rows * (HD / k::KV_Q8_GROUP));
        for (int64_t c = 0; c < cells; ++c) for (int64_t kv = 0; kv < s.n_head_kv; ++kv) for (int d = 0; d < HD; ++d) {
            const int64_t r = row_of(c, kv);
            out[((size_t) c * s.n_head_kv + kv) * HD + d] = (double) hc[(size_t) r * HD + d] * k::f32_from_f16(hs[(size_t) r * (HD / k::KV_Q8_GROUP) + d / k::KV_Q8_GROUP]);
        }
    };
    auto deq_q4 = [&](const uint8_t* pool, std::vector<double>& out) {
        const auto h = down(pool, (size_t) P.rows * q4row);
        for (int64_t c = 0; c < cells; ++c) for (int64_t kv = 0; kv < s.n_head_kv; ++kv) for (int d = 0; d < HD; ++d) {
            const uint8_t* blk = h.data() + (size_t) row_of(c, kv) * q4row + (d / k::QK4_0) * sizeof(k::block_q4_0);
            uint16_t sc; std::memcpy(&sc, blk, 2);
            const int j = d % k::QK4_0;
            const int code = j < 16 ? (blk[2 + j] & 0x0F) : (blk[2 + j - 16] >> 4);
            out[((size_t) c * s.n_head_kv + kv) * HD + d] = (double) (code - 8) * k::f32_from_f16(sc);
        }
    };
    if (mode == 0) { deq_f16(P.k_pool, P.K); deq_f16(P.v_pool, P.V); }
    else if (mode == 1) { deq_q8(P.k_q, P.k_scale, P.K); deq_q8(P.v_q, P.v_scale, P.V); }
    else if (mode == 2) { deq_q4(P.k_q4, P.K); deq_q4(P.v_q4, P.V); }
    else { deq_q8(P.k_q, P.k_scale, P.K); deq_q4(P.v_q4, P.V); }
    cudaFree(d_steps); cudaFree(d_k); cudaFree(d_v);
    return P;
}
void free_pools(Pools& P) {
    for (void* p : {(void*) P.k_pool, (void*) P.v_pool, (void*) P.k_scale, (void*) P.v_scale, (void*) P.k_q, (void*) P.v_q,
                    (void*) P.k_q4, (void*) P.v_q4, (void*) P.d_table})
        if (p) cudaFree(p);
}
const char* mode_name(int m) { return m == 0 ? "fp16" : m == 1 ? "int8" : m == 2 ? "q4_0" : "k8v4"; }

k::QsaShapes dense_shapes(int64_t n_head, int64_t n_kv) {
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = n_head; s.n_head_kv = n_kv;
    s.idx_n_head = 0; s.idx_dim = 0; s.idx_top_k = 0;   // what layer.cpp's qsa_shapes() does for a no-indexer geometry
    return s;
}

// ================= 1. the group, the width, the capacity, the scratch (host only) =================
void section_dispatch() {
    std::printf("\n-- the query-head group, the dense width and capacity, the bounded scratch\n");
    char why[256];
    const k::QsaShapes s16 = dense_shapes(16, 2), s24 = dense_shapes(24, 2), s4 = dense_shapes(4, 2), s26 = dense_shapes(26, 2);
    verdict("16 / 2 is group 8 (Qwen3.6)", k::qsa_attn_group(s16, why, sizeof why) == 8);
    verdict("24 / 2 is group 12 (Flash-Next)", k::qsa_attn_group(s24, why, sizeof why) == 12);
    verdict("4 / 2 is group 2 (the fixture)", k::qsa_attn_group(s4, why, sizeof why) == 2);
    why[0] = 0;
    const bool r26 = k::qsa_attn_group(s26, why, sizeof why) == 0;
    verdict("26 / 2 (13 per KV head) is refused, naming the heads", r26 && std::strstr(why, "26") && std::strstr(why, "13"), why);
    why[0] = 0;
    const bool r12 = !k::qsa_attn_group_check(s16, 12, why, sizeof why);
    verdict("NEGATIVE: G = 12 asked of 16 heads over 2 is refused by name, not run", r12 && std::strstr(why, "16") && std::strstr(why, "12"), why);
    verdict("G = 8 on 16 / 2 and G = 12 on 24 / 2 pass the check", k::qsa_attn_group_check(s16, 8, why, sizeof why) && k::qsa_attn_group_check(s24, 12, why, sizeof why));
    k::QsaShapes sparse = k::qsa_real_shapes();
    verdict("sparse (Flash-Next) width at 3,000 cells is 2,051 and its cap 2,051",
            k::qsa_selection_width(3000, sparse) == 2051 && k::qsa_selection_cap(262144, sparse) == 2051);
    verdict("dense width at 3,000 cells is 3,000 and the cap is max_cells",
            k::qsa_selection_width(3000, s16) == 3000 && k::qsa_selection_cap(262144, s16) == 262144 && k::qsa_dense(s16) && !k::qsa_dense(sparse));
    int32_t st[k::kStepCount];
    k::qsa_step_fill(st, 2999, s16);
    verdict("qsa_step_fill writes width = n_kv for the dense shape", st[k::kStepWidth] == 3000 && st[k::kStepNKv] == 3000);
    k::qsa_step_fill(st, 2999, sparse);
    verdict("... and 2,051 for the sparse one", st[k::kStepWidth] == 2051);
    verdict("dense chunks: 2,051 cells -> 33 blocks (the old grid), 8,192 -> 128, 262,144 -> 128",
            k::qsa_decode_attn_chunks(2051, s16) == 33 && k::qsa_decode_attn_chunks(8192, s16) == 128 && k::qsa_decode_attn_chunks(262144, s16) == 128);
    verdict("sparse chunks are the old grid at any capacity: 2,051 -> 33, 32,768 (the MTP drafter's window) -> 512",
            k::qsa_decode_attn_chunks(2051, sparse) == 33 && k::qsa_decode_attn_chunks(32768, sparse) == 512 &&
            k::qsa_decode_attn_scratch_floats(32768, sparse) == 512ull * 24 * 258 + 64);
    // the one builder (strata/core/qsa_shapes.hpp) from a qwen35moe geometry, and the half-set shape it prevents
    {
        strata::core::ModelGeometry g;
        g.arch = strata::core::ModelArch::Qwen35Moe; g.hc = 1; g.hc_lr = 0; g.has_ple = false;
        g.n_head = 16; g.n_head_kv = 2; g.head_dim = 256; g.idx_q_heads = 0; g.idx_key_dim = 0;
        g.ssm_v_heads = 32; g.n_expert = 256; g.n_expert_used = 8; g.n_ff = 512; g.n_layers = 40; g.derive();
        const k::QsaShapes built = strata::core::qsa_shapes(g);
        verdict("qsa_shapes(qwen35moe geometry) is dense: idx_n_head, idx_dim, idx_top_k all 0, n_rot 64, idx_block 4",
                k::qsa_dense(built) && built.idx_top_k == 0 && built.n_rot == 64 && built.idx_block == 4 && built.n_head == 16);
        why[0] = 0;
        verdict("... and the kernels' guard accepts it", k::qsa_shapes_valid(built, why, sizeof why), why);
        strata::core::ModelGeometry f;   // Flash-Next's defaults: the sparse shape, unchanged
        const k::QsaShapes fn = strata::core::qsa_shapes(f);
        verdict("qsa_shapes(qwen4exp geometry) is the sparse real shape (24 / 2, idx 4 x 128, top-k 2048)",
                !k::qsa_dense(fn) && fn.idx_n_head == 4 && fn.idx_dim == 128 && fn.idx_top_k == 2048 && k::qsa_shapes_valid(fn, why, sizeof why));
        k::QsaShapes half = k::qsa_real_shapes();   // what a hand-written builder made: two zeros, the budget kept
        half.n_head = 16; half.idx_n_head = 0; half.idx_dim = 0;
        why[0] = 0;
        const bool refused = !k::qsa_shapes_valid(half, why, sizeof why);
        verdict("NEGATIVE: the half-set shape (idx_n_head 0, idx_dim 0, idx_top_k 2048) is neither dense nor valid",
                !k::qsa_dense(half) && refused && std::strstr(why, "half set up"), why);
    }
    const uint64_t old2051 = 33ull * 24 * 258 + 64, new262k = 128ull * 16 * 258 + 64;
    verdict("scratch at cap 2,051 x 24 heads is the old formula (33 x 24 x 258 + 64 floats)",
            k::qsa_decode_attn_scratch_floats(2051, s24) == old2051);
    verdict("scratch at cap 262,144 x 16 heads is 128 x 16 x 258 + 64 floats = 528,448 (2.02 MiB), not 64.5 MiB",
            k::qsa_decode_attn_scratch_floats(262144, s16) == new262k && new262k == 528448,
            std::to_string(new262k * 4 / 1024) + " KiB");
}

// ================= 2. the decode kernel against the host reference, every KV format, three widths =================
struct Timer { std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); double s() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } };

void section_random(int64_t n_head, int64_t n_kv, int mode, int64_t width, std::mt19937& rng, bool prompt_too) {
    const k::QsaShapes s = dense_shapes(n_head, n_kv);
    const int64_t cells = width;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> kcur((size_t) cells * n_kv * HD), vcur((size_t) cells * n_kv * HD);
    for (auto& x : kcur) x = nd(rng) * 0.7f;
    for (auto& x : vcur) x = nd(rng);
    Pools P = make_pools(mode, cells, s, rng, kcur, vcur);
    // three query rows at widths w, w-1, w-77 (each its own step), q rotated when the format rotates K
    const int64_t n_q = 3;
    const int64_t widths[3] = {width, width - 1, std::max<int64_t>(width - 77, 1)};
    std::vector<float> q((size_t) n_q * n_head * HD);
    for (auto& x : q) x = nd(rng) * 1.5f;
    float* d_q = up(q);
    if (P.rot_q) k::fwht256_inplace_cuda(d_q, n_q * n_head, nullptr);
    const auto q_dev = down(d_q, q.size());   // the (possibly rotated) q the kernel reads, for the reference
    std::vector<int32_t> steps((size_t) n_q * k::kStepCount);
    for (int64_t i = 0; i < n_q; ++i) k::qsa_step_fill(steps.data() + i * k::kStepCount, widths[i] - 1, s);
    int32_t* d_steps = up(steps);
    const int64_t cap = cells;
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* d_scr = dalloc<float>(scr * n_q);
    float* d_attn = dalloc<float>((size_t) n_q * n_head * HD);
    const std::string tag = std::to_string(n_head) + "/" + std::to_string(n_kv) + " " + mode_name(mode) + " w=" + std::to_string(width);
    // the kernel with a NULL ids (the dense identity)
    k::qsa_decode_attn_batch(d_q, P.p, nullptr, d_steps, cap, s, d_scr, d_attn, n_q, nullptr);
    ck(cudaDeviceSynchronize(), "decode");
    const auto out_null = down(d_attn, (size_t) n_q * n_head * HD);
    // the same with the explicit identity ids
    int32_t* d_ids = dalloc<int32_t>((size_t) n_q * cap);
    k::qsa_identity_ids(d_ids, d_steps, n_q, cap, nullptr);
    k::qsa_decode_attn_batch(d_q, P.p, d_ids, d_steps, cap, s, d_scr, d_attn, n_q, nullptr);
    ck(cudaDeviceSynchronize(), "decode ids");
    const auto out_ids = down(d_attn, (size_t) n_q * n_head * HD);
    verdict(tag + ": null ids == explicit identity ids, bit for bit", bitwise(out_null, out_ids));
    {
        const auto ids_h = down(d_ids, (size_t) n_q * cap);
        bool ok = true;
        for (int64_t i = 0; i < n_q && ok; ++i) for (int64_t j = 0; j < widths[i] && ok; ++j) ok = ids_h[(size_t) i * cap + j] == j;
        verdict(tag + ": qsa_identity_ids writes 0..width-1 per row", ok);
    }
    // the host reference per row
    Timer tm;
    double worst = 0;
    std::vector<float> ref_row;
    for (int64_t i = 0; i < n_q; ++i) {
        std::vector<float> qi(q_dev.begin() + i * n_head * HD, q_dev.begin() + (i + 1) * n_head * HD);
        ref_attend(qi, P.K, P.V, cells, widths[i], n_head, n_kv, ref_row, nullptr);
        std::vector<float> got(out_null.begin() + i * n_head * HD, out_null.begin() + (i + 1) * n_head * HD);
        worst = std::max(worst, rel_l2(got, ref_row));
        if (i == 0 && width == 3000) {
            // NEGATIVE: the sparse width (2,051) on this context reads 949 cells fewer - what the old
            // qsa_selection_width would have handed the kernel for Qwen3.6
            std::vector<int32_t> sp(steps.begin(), steps.begin() + k::kStepCount);
            sp[k::kStepWidth] = 2051;
            int32_t* d_sp = up(sp);
            k::qsa_decode_attn_batch(d_q, P.p, nullptr, d_sp, cap, s, d_scr, d_attn, 1, nullptr);
            ck(cudaDeviceSynchronize(), "sparse width");
            const auto narrow = down(d_attn, (size_t) n_head * HD);
            const double l1 = rel_l1(narrow, ref_row);
            verdict(tag + ": NEGATIVE: the sparse width 2,051 differs from the dense 3,000 by > 5% rel L1", l1 > 0.05, sci(l1));
            cudaFree(d_sp);
        }
    }
    const double tol = 3e-5;   // fp32 kernel with __expf against the double reference over the SAME dequantized rows
    verdict(tag + ": decode kernel vs host reference over the dequantized pools, rel L2 <= " + sci(tol), worst <= tol,
            sci(worst) + " (" + std::to_string((int) (tm.s() * 1000)) + " ms ref)");
    // the prompt (tensor-core) kernel at the same rows: sm_70+ only, the host returns false below it
    if (prompt_too) {
        float* d_pa = dalloc<float>((size_t) n_q * n_head * HD);
        if (k::qsa_prompt_attn_batch(d_q, P.p, nullptr, d_steps, cap, s, d_pa, n_q, nullptr)) {
            ck(cudaDeviceSynchronize(), "prompt");
            const auto pa = down(d_pa, (size_t) n_q * n_head * HD);
            k::qsa_prompt_attn_batch(d_q, P.p, d_ids, d_steps, cap, s, d_pa, n_q, nullptr);
            ck(cudaDeviceSynchronize(), "prompt ids");
            const auto pa_ids = down(d_pa, (size_t) n_q * n_head * HD);
            verdict(tag + ": prompt kernel: null ids == explicit identity ids, bit for bit", bitwise(pa, pa_ids));
            double w2 = 0;
            for (int64_t i = 0; i < n_q; ++i) {
                std::vector<float> qi(q_dev.begin() + i * n_head * HD, q_dev.begin() + (i + 1) * n_head * HD);
                ref_attend(qi, P.K, P.V, cells, widths[i], n_head, n_kv, ref_row, nullptr);
                std::vector<float> got(pa.begin() + i * n_head * HD, pa.begin() + (i + 1) * n_head * HD);
                w2 = std::max(w2, rel_l2(got, ref_row));
            }
            const double ptol = 2e-3;   // fp16 hi/lo split operands with fp32 accumulation (qsa_prompt_attn.hpp)
            verdict(tag + ": prompt kernel vs host reference, rel L2 <= " + sci(ptol), w2 <= ptol, sci(w2));
        } else {
            std::printf("  %-78s %s\n", (tag + ": prompt kernel").c_str(), "skipped (the host keeps the decode kernel on this card / format)");
        }
        cudaFree(d_pa);
    }
    cudaFree(d_q); cudaFree(d_steps); cudaFree(d_scr); cudaFree(d_attn); cudaFree(d_ids);
    free_pools(P);
}

// ================= 3. the oracle's fixtures =================
std::vector<float> load_f32(const fs::path& p, size_t expect) {
    std::vector<float> v;
    FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.string().c_str()); std::exit(77); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) { std::fprintf(stderr, "short read %s\n", p.string().c_str()); std::exit(1); }
    std::fclose(f);
    if (v.size() != expect) { std::fprintf(stderr, "%s holds %zu floats, expected %zu\n", p.string().c_str(), v.size(), expect); std::exit(1); }
    return v;
}

void section_fixture(const fs::path& dir, const std::string& name) {
    std::printf("\n-- fixture %s: the oracle's attention layer (blk.3) through the engine's norm, RoPE, fp16 KV, decode kernel, gate\n", name.c_str());
    const fs::path gguf = dir / (name + ".gguf"), dump = dir / (name + "-dump");
    strata::core::ModelGeometry g;
    std::string err;
    if (!strata::core::read_model_geometry(gguf.string(), g, err)) { std::fprintf(stderr, "%s: %s\n", gguf.string().c_str(), err.c_str()); std::exit(1); }
    const int64_t NH = g.n_head, NKV = g.n_head_kv, T = 40, L = 3;
    if (g.head_dim != HD) { std::fprintf(stderr, "%s: head_dim %lld, the kernel is 256\n", name.c_str(), (long long) g.head_dim); std::exit(1); }
    k::QsaShapes s = dense_shapes(NH, NKV);
    s.n_rot = g.n_rot;
    verdict(name + ": geometry " + std::to_string(NH) + " / " + std::to_string(NKV) + " at head_dim 256, n_rot " + std::to_string(g.n_rot) + ", no indexer",
            !g.has_indexer() && g.idx_q_heads == 0 && k::qsa_dense(s));
    const strata::GgufModel model = strata::GgufModel::open(gguf.string());
    auto tensor = [&](const std::string& nm, size_t n) {
        size_t sh = 0;
        const strata::TensorInfo* t = model.find(nm, &sh);
        if (!t || t->elements() != n || t->type != 0) { std::fprintf(stderr, "%s: %s missing or not F32[%zu]\n", name.c_str(), nm.c_str(), n); std::exit(1); }
        std::vector<float> v(n);
        std::memcpy(v.data(), model.shard(sh).tensor_data(*t), n * 4);
        return v;
    };
    const auto wqn = tensor("blk.3.attn_q_norm.weight", HD), wkn = tensor("blk.3.attn_k_norm.weight", HD);
    // the oracle's file names: <name>-<layer>.f32, a repeated name numbered <name>-<layer>#2.f32 (oracle-crosscheck.md)
    auto seam = [&](const std::string& nm, size_t n) {
        const size_t hash = nm.find('#');
        const std::string base = nm.substr(0, hash), rep = hash == std::string::npos ? "" : nm.substr(hash);
        return load_f32(dump / (base + "-" + std::to_string(L) + rep + ".f32"), n);
    };
    const auto q_pre = seam("Qcur_reshaped", T * NH * HD);        // [t][h][d], before the norm
    const auto k_pre = seam("Kcur", T * NKV * HD);                 // [t][kv*HD + d], the projection
    const auto v_in = seam("Vcur#2", T * NKV * HD);                // [t][kv][d]
    const auto q_full = seam("Qcur_full", T * NH * 2 * HD);        // [t][h][q | gate]
    const auto q_ref = seam("Qcur", T * NH * HD);                  // normed and roped
    const auto k_ref = seam("Kcur#2", T * NKV * HD);
    const auto p_ref = seam("kq_soft_max", NH * T * T);            // [h][t][j]
    const auto kqv_ref = seam("kqv", NH * T * HD);                 // [h][t][d]
    const auto gated_ref = seam("attn_gated", T * NH * HD);        // [t][h*HD + d]
    const auto q_normed_ref = seam("Qcur_normed", T * NH * HD);
    const auto k_normed_ref = seam("Kcur_normed", T * NKV * HD);
    // ---- norm + RoPE on the device: the engine's rms_norm_weighted and its float64-built table
    float* d_q = up(q_pre);
    float* d_k = up(k_pre);
    k::rms_norm_weighted(d_q, up(wqn), T * NH, HD, (float) g.rms_eps, nullptr);
    k::rms_norm_weighted(d_k, up(wkn), T * NKV, HD, (float) g.rms_eps, nullptr);
    std::vector<float> cos_t((size_t) T * (g.n_rot / 2)), sin_t(cos_t.size());
    k::build_rope_table((int) g.n_rot, g.rope_base, (int) T, cos_t.data(), sin_t.data());
    const float* d_cos = up(cos_t);
    const float* d_sin = up(sin_t);
    std::vector<int> pos_q((size_t) T * NH), pos_k((size_t) T * NKV);
    for (int64_t t = 0; t < T; ++t) { for (int64_t h = 0; h < NH; ++h) pos_q[t * NH + h] = (int) t; for (int64_t h = 0; h < NKV; ++h) pos_k[t * NKV + h] = (int) t; }
    k::rope_neox_apply(d_q, d_q, T * NH, HD, (int) g.n_rot, d_cos, d_sin, up(pos_q), nullptr);
    k::rope_neox_apply(d_k, d_k, T * NKV, HD, (int) g.n_rot, d_cos, d_sin, up(pos_k), nullptr);
    ck(cudaDeviceSynchronize(), "norm rope");
    const auto q_eng = down(d_q, q_pre.size()), k_eng = down(d_k, k_pre.size());
    const double tol_rope = 2e-6;
    verdict(name + ": engine norm + NEOX RoPE of q == oracle Qcur-3, rel L2 <= " + sci(tol_rope), rel_l2(q_eng, q_ref) <= tol_rope, sci(rel_l2(q_eng, q_ref)));
    verdict(name + ": engine norm + NEOX RoPE of k == oracle Kcur-3#2, rel L2 <= " + sci(tol_rope), rel_l2(k_eng, k_ref) <= tol_rope, sci(rel_l2(k_eng, k_ref)));
    // ---- the fp16 KV through kv_append_step, the T queries through the batched decode kernel (null ids)
    std::mt19937 rng(99);
    std::vector<float> kc(k_eng), vc(v_in);
    Pools P = make_pools(0, T, s, rng, kc, vc);
    std::vector<int32_t> steps((size_t) T * k::kStepCount);
    for (int64_t t = 0; t < T; ++t) k::qsa_step_fill(steps.data() + t * k::kStepCount, t, s);
    int32_t* d_steps = up(steps);
    const int64_t cap = T;
    float* d_scr = dalloc<float>(k::qsa_decode_attn_scratch_floats(cap, s) * T);
    float* d_attn = dalloc<float>((size_t) T * NH * HD);
    k::qsa_decode_attn_batch(d_q, P.p, nullptr, d_steps, cap, s, d_scr, d_attn, T, nullptr);
    ck(cudaDeviceSynchronize(), "decode");
    const auto attn = down(d_attn, (size_t) T * NH * HD);   // [t][h][d]
    std::vector<float> kqv_t((size_t) NH * T * HD);          // to the oracle's [h][t][d]
    for (int64_t t = 0; t < T; ++t) for (int64_t h = 0; h < NH; ++h)
        std::memcpy(kqv_t.data() + ((size_t) h * T + t) * HD, attn.data() + ((size_t) t * NH + h) * HD, HD * 4);
    const double e_kqv = rel_l2(kqv_t, kqv_ref);
    // the same attention in double over the engine's fp32 q/k/v (no fp16 rounding): how much of e_kqv is the KV format
    std::vector<double> K32(k_eng.begin(), k_eng.end()), V32(v_in.begin(), v_in.end());
    std::vector<float> kqv_f32((size_t) NH * T * HD), probs_f32((size_t) NH * T * T, 0.f), row;
    std::vector<double> pr;
    for (int64_t t = 0; t < T; ++t) {
        std::vector<float> qi(q_eng.begin() + t * NH * HD, q_eng.begin() + (t + 1) * NH * HD);
        ref_attend(qi, K32, V32, T, t + 1, NH, NKV, row, &pr);
        for (int64_t h = 0; h < NH; ++h) {
            std::memcpy(kqv_f32.data() + ((size_t) h * T + t) * HD, row.data() + (size_t) h * HD, HD * 4);
            for (int64_t j = 0; j < T; ++j) probs_f32[((size_t) h * T + t) * T + j] = (float) pr[(size_t) h * T + j];
        }
    }
    const double e_p = rel_l2(probs_f32, p_ref), e_f32 = rel_l2(kqv_f32, kqv_ref);
    verdict(name + ": softmax over the engine's q/k (host, double) == oracle kq_soft_max-3, rel L2 <= 1e-5", e_p <= 1e-5, sci(e_p));
    verdict(name + ": attention over the engine's fp32 q/k/v (host, double) == oracle kqv-3, rel L2 <= 1e-5", e_f32 <= 1e-5, sci(e_f32));
    const double tol_kqv = 1.5e-3;   // the fp16 KV pool: K and V rounded to 11 bits (the oracle's are F32)
    verdict(name + ": decode kernel over the fp16 KV == oracle kqv-3, rel L2 <= " + sci(tol_kqv), e_kqv <= tol_kqv, sci(e_kqv));
    // ---- the gate: attn * sigmoid(second half of each head's q_full), qsa_gate_apply_f32
    float* d_qf = up(q_full);
    float* d_gated = dalloc<float>((size_t) T * NH * HD);
    for (int64_t t = 0; t < T; ++t)
        k::qsa_gate_apply_f32(d_attn + t * NH * HD, d_qf + t * NH * 2 * HD, s, d_gated + t * NH * HD, nullptr);
    ck(cudaDeviceSynchronize(), "gate");
    const auto gated = down(d_gated, (size_t) T * NH * HD);
    const double e_g = rel_l2(gated, gated_ref);
    verdict(name + ": gated output == oracle attn_gated-3, rel L2 <= " + sci(tol_kqv), e_g <= tol_kqv, sci(e_g));
    // ---- NEGATIVE: the gate skipped (the oracle's control: 1.03 relative L1)
    const double n_gate = rel_l1(attn, gated_ref);
    verdict(name + ": NEGATIVE: the gate skipped differs from attn_gated-3 by > 5% rel L1", n_gate > 0.05, sci(n_gate));
    // ---- NEGATIVE: RoPE on adjacent pairs (2i, 2i+1), from the oracle's own normed q/k (control: 0.20 at kqv-3)
    {
        std::vector<float> qa(q_normed_ref), ka(k_normed_ref);
        host_rope(qa, T * NH, (int) g.n_rot, g.rope_base, pos_q, true);
        host_rope(ka, T * NKV, (int) g.n_rot, g.rope_base, pos_k, true);
        std::vector<float> qn(q_normed_ref), kn(k_normed_ref);   // the right reading on the host, as a control of the control
        host_rope(qn, T * NH, (int) g.n_rot, g.rope_base, pos_q, false);
        host_rope(kn, T * NKV, (int) g.n_rot, g.rope_base, pos_k, false);
        verdict(name + ": host NEOX RoPE agrees with the oracle (the rival below is the only change), rel L2 <= 1e-5",
                rel_l2(qn, q_ref) <= 1e-5 && rel_l2(kn, k_ref) <= 1e-5, sci(std::max(rel_l2(qn, q_ref), rel_l2(kn, k_ref))));
        std::vector<double> Ka(ka.begin(), ka.end());
        std::vector<float> kqv_a((size_t) NH * T * HD);
        for (int64_t t = 0; t < T; ++t) {
            std::vector<float> qi(qa.begin() + t * NH * HD, qa.begin() + (t + 1) * NH * HD);
            ref_attend(qi, Ka, V32, T, t + 1, NH, NKV, row, nullptr);
            for (int64_t h = 0; h < NH; ++h) std::memcpy(kqv_a.data() + ((size_t) h * T + t) * HD, row.data() + (size_t) h * HD, HD * 4);
        }
        const double n_rope = rel_l1(kqv_a, kqv_ref);
        verdict(name + ": NEGATIVE: adjacent-pair RoPE differs from kqv-3 by > 5% rel L1", n_rope > 0.05, sci(n_rope));
    }
    // ---- NEGATIVE: the GQA map h % n_kv instead of h / G (visible only with more than one KV head; control 0.74)
    if (NKV > 1 && NH / NKV > 1) {
        std::vector<float> kqv_m((size_t) NH * T * HD);
        for (int64_t t = 0; t < T; ++t) {
            std::vector<float> qi(q_eng.begin() + t * NH * HD, q_eng.begin() + (t + 1) * NH * HD);
            ref_attend(qi, K32, V32, T, t + 1, NH, NKV, row, nullptr, true);
            for (int64_t h = 0; h < NH; ++h) std::memcpy(kqv_m.data() + ((size_t) h * T + t) * HD, row.data() + (size_t) h * HD, HD * 4);
        }
        const double n_map = rel_l1(kqv_m, kqv_ref);
        verdict(name + ": NEGATIVE: the GQA map h % n_kv differs from kqv-3 by > 5% rel L1", n_map > 0.05, sci(n_map));
    }
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_steps); cudaFree(d_scr); cudaFree(d_attn); cudaFree(d_qf); cudaFree(d_gated);
    free_pools(P);
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixtures;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fixtures" && i + 1 < argc) fixtures = argv[++i];
        else if (a == "--selftest") {}
        else { std::fprintf(stderr, "usage: qsa_dense_parity [--fixtures DIR] [--selftest]\n"); return 2; }
    }
    std::printf("qsa_dense_parity: dense attention (Qwen3.6) through the split-K decode kernel and the prompt kernel\n");
    section_dispatch();
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("SKIP: no CUDA device\n"); return 77; }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("\n-- device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::mt19937 rng(2024);
    std::printf("\n-- the decode kernel vs the host reference: G = 8 (16 / 2) and G = 12 (24 / 2), four KV formats, widths 300 / 3,000 / 20,000\n");
    for (const int64_t heads : {16, 24})
        for (const int mode : {0, 1, 2, 3})
            for (const int64_t w : {300, 3000, 20000})
                section_random(heads, 2, mode, w, rng, w == 3000);
    std::printf("\n-- G = 2 (4 / 2), fp16, 3,000 cells\n");
    section_random(4, 2, 0, 3000, rng, true);
    if (!fixtures.empty()) {
        const fs::path dir(fixtures);
        if (fs::exists(dir / "SKIP") || !fs::exists(dir / "g8.gguf")) {
            std::printf("SKIP: the fixtures in %s were not written (python3 with numpy and gguf-py)\n", fixtures.c_str());
            return 77;
        }
        for (const char* n : {"g8", "g12", "g2"}) section_fixture(dir, n);
    } else {
        std::printf("\n(no --fixtures DIR: the oracle's seams were not checked)\n");
    }
    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAIL" : "qsa_dense_parity OK", g_fail);
    return g_fail ? 1 : 0;
}
