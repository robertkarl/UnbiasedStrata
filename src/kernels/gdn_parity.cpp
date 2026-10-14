// src/kernels/gdn_parity.cpp - P2.S2's test for the GDN recurrence, the conv and the two norms; Qwen3.6 step 3
// re-pins it at 32 value heads with the SiLU output gate.
//
// `ref/gdn.py` carries eleven PROPERTY checks, several of which exist because a rival reading of the SOURCE
// produces well-formed output.  The four that the kernels here can be wrong about are re-pinned below, each
// asserted OBSERVABLE against a deliberately-wrong computation before the kernel is judged:
//
//   1. MODULO head pairing, h % h_k, not INTERLEAVE.  With H_k=2 and H_v=4 the two give [0,1,0,1] and
//      [0,0,1,1]: they differ at h=1 and h=2.  Both produce a full-rank state of the right shape.
//   2. Decay applied BEFORE the rank-1 update.  Applying it after is a one-line move that keeps every shape
//      and every magnitude.
//   3. The conv's SLIDE DIRECTION.  `kernel[0]` must read the OLDEST state row and the new input must land in
//      the LAST state row; reversing either is invisible on a symmetric state, so the state rows here are
//      labelled so each position is identifiable.
//   4. `l2_norm`'s `+ eps` is an absolute floor on the SQUARED NORM, not on the mean.  The mean reading
//      differs by sqrt(S) = 11.3x at S = 128.  `ref/gdn.py` PROPERTY 1 instruments the derivation, so it is
//      not taken on trust there either.
//
// and the fifth, the one numerical difference between the two models this engine serves (docs/QWEN36-35B-A3B.md
// 3.2, geometry-inventory.md row 39):
//
//   5. THE OUTPUT GATE.  Flash-Next (`qwen4exp`, 16 k-heads / 48 v-heads) closes the layer with
//      `rms_norm(o) * ssm_norm * sigmoid(z)`; Qwen3.6 (`qwen35moe`, 16 / 32) with `... * SiLU(z)`.  Both are
//      well-formed and the same size; the oracle's negative control puts them 1.07 relative L1 apart at
//      `final_output-0`.  Every output kernel takes the choice as `GdnGate`; here each geometry is run with ITS
//      gate against a reference of its gate, and then the kernel is run with the OTHER gate and required to
//      differ from that reference by more than 5% - the wrong reading fails through the kernel itself, not
//      only on the host.
//
// It also checks the STATE LAYOUT, which is the one place these kernels intentionally differ from the
// reference: (S, h_v, S) with j fastest instead of (S, S, h_v).  A layout mix-up is silent, so the state is
// filled with a value that encodes its own coordinates.
//
// Geometries: the recurrence and the output kernels run at BOTH head counts, 16/48 (VPK 3) and 16/32 (VPK 2),
// through all four implementations - the legacy `gdn_step` + `gdn_out_norm`, the pinned `native_gdn_step` +
// `native_gdn_out_norm`, the fused `fused_gdn_step_norm`, and the verify window's `gdn_step_norm_multi` (which
// must be bitwise the fused kernel per token).
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::GdnGate;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
float gate_act(GdnGate g, float x) { return g == GdnGate::Silu ? silu_f(x) : sigmoid_f(x); }
const char* gate_name(GdnGate g) { return strata::kernels::gdn_gate_name(g); }
GdnGate other(GdnGate g) { return g == GdnGate::Silu ? GdnGate::Sigmoid : GdnGate::Silu; }

/// The reference's `gdn_step`, with switches for the rival readings.
///
/// `state` is passed in the REFERENCE layout (S, S, h_v) - i.e. (i, j, h) - so that this function is a
/// transcription of `ref/gdn.py` and not of the kernel.  `q` is taken AS GIVEN: the legacy kernel wants it
/// pre-scaled by 1/sqrt(S), the native and fused kernels scale the readout themselves, so the caller passes the
/// scaled q here and the unscaled q to those kernels.
void ref_step(std::vector<double>& state, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
              int S, int h_k, int h_v, std::vector<float>& o, bool modulo_heads, bool decay_first) {
    auto at = [&](int i, int j, int h) -> double& { return state[(size_t) (i * S + j) * h_v + h]; };
    std::vector<float> oo((size_t) h_v * S, 0.0f);
    for (int h = 0; h < h_v; ++h) {
        const int src = modulo_heads ? (h % h_k) : (h / (h_v / h_k));
        const double dec = std::exp((double) gate[(size_t) h]);
        const double b = (double) beta[(size_t) h];
        for (int j = 0; j < S; ++j) {
            if (decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double sk = 0;
            for (int i = 0; i < S; ++i) sk += at(i, j, h) * (double) k[(size_t) (src * S + i)];
            const double d = ((double) v[(size_t) (h * S + j)] - sk) * b;
            for (int i = 0; i < S; ++i) at(i, j, h) += (double) k[(size_t) (src * S + i)] * d;
            if (!decay_first) for (int i = 0; i < S; ++i) at(i, j, h) *= dec;
            double dot = 0;
            for (int i = 0; i < S; ++i) dot += at(i, j, h) * (double) q[(size_t) (src * S + i)];
            oo[(size_t) (h * S + j)] = (float) dot;
        }
    }
    o = oo;
}

/// `y = rms_norm(o, eps) * ssm_norm * act(z)`, one RMS per head (row), as `ref/gdn.py` / `ref/qwen36.py` write it.
void ref_out_norm(const std::vector<float>& o, const std::vector<float>& z, const std::vector<float>& sn, int h_v,
                  int S, float eps, GdnGate act, std::vector<float>& y) {
    y.assign((size_t) h_v * S, 0.0f);
    for (int h = 0; h < h_v; ++h) {
        double acc = 0;
        for (int i = 0; i < S; ++i) acc += (double) o[(size_t) (h * S + i)] * (double) o[(size_t) (h * S + i)];
        const float inv = (float) (1.0 / std::sqrt(acc / S + (double) eps));
        for (int i = 0; i < S; ++i)
            y[(size_t) (h * S + i)] = o[(size_t) (h * S + i)] * inv * sn[(size_t) i] * gate_act(act, z[(size_t) (h * S + i)]);
    }
}

/// `ref/gdn.py::conv_step` in its own (d_conv, C) row-major orientation.
void ref_conv(std::vector<float>& cs, const std::vector<float>& x, const std::vector<float>& kern, int C,
              int dc, std::vector<float>& out) {
    out.assign((size_t) C, 0.0f);
    for (int c = 0; c < C; ++c) {
        double acc = 0;
        for (int i = 0; i < dc - 1; ++i) acc += (double) cs[(size_t) (i * C + c)] * (double) kern[(size_t) (i * C + c)];
        acc += (double) x[(size_t) c] * (double) kern[(size_t) ((dc - 1) * C + c)];
        out[(size_t) c] = (float) acc;
        for (int i = 0; i < dc - 2; ++i) cs[(size_t) (i * C + c)] = cs[(size_t) ((i + 1) * C + c)];
        cs[(size_t) ((dc - 2) * C + c)] = x[(size_t) c];
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// The device state (S, h_v, S), j fastest, read back into the reference layout and compared.
double state_rel(const std::vector<double>& want, const std::vector<float>& got_dev, int S, int h_v) {
    double d = 0, m = 0;
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
            for (int h = 0; h < h_v; ++h) {
                const double w = want[(size_t) (i * S + j) * h_v + h];
                d += std::fabs(w - (double) got_dev[(size_t) (i * h_v + h) * S + j]);
                m += std::fabs(w);
            }
    return d / (m > 1e-30 ? m : 1e-30);
}

template <class T>
T* dev(const std::vector<T>& h, const char* what) {
    T* p = nullptr;
    check(cudaMalloc(&p, h.size() * sizeof(T) + 16), what);
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    return p;
}
template <class T>
std::vector<T> host(const T* d, size_t n, const char* what) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), what);
    return h;
}

int g_bad = 0;
void observable(const char* what, double rel) {
    std::printf("  %-58s %-4s (%.2f%% apart)\n", what, rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
    if (!(rel > 0.05)) ++g_bad;
}
void within(const char* what, double rel, double tol) {
    std::printf("  %-58s rel %.3e%s\n", what, rel, rel <= tol ? "" : "   *** over tolerance ***");
    if (!(rel <= tol)) ++g_bad;
}

/// Sections 1 and 5 for one geometry: the recurrence through every implementation, the state layout, the output
/// gate with its negative case.  `model_gate` is the gate the model with this head count uses.
void recurrence_and_gate(int S, int h_k, int h_v, GdnGate model_gate, std::mt19937& rng, cudaStream_t stream) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::printf("\n== geometry S %d, h_k %d, h_v %d (VPK %d), model gate %s\n", S, h_k, h_v, h_v / h_k,
                gate_name(model_gate));

    // The state encodes its own coordinates, so a layout mix-up cannot pass: reading (i,j,h) where the
    // buffer holds (i,h,j) gives a DIFFERENT number, not a wrong-but-plausible one.
    std::vector<double> st_ref((size_t) S * S * h_v);
    std::vector<float> st_dev((size_t) S * h_v * S);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
            for (int h = 0; h < h_v; ++h) {
                const float val = (float) (0.001 * (i + 2 * j + 3 * h) - 1.0);
                st_ref[(size_t) (i * S + j) * h_v + h] = val;
                st_dev[(size_t) (i * h_v + h) * S + j] = val;   // (S, h_v, S), j fastest
            }

    std::vector<float> q((size_t) h_k * S), k((size_t) h_k * S), v((size_t) h_v * S), gate((size_t) h_v),
        beta((size_t) h_v), z((size_t) h_v * S), sn((size_t) S);
    for (auto& x : q) x = gauss(rng) * 0.1f;
    for (auto& x : k) x = gauss(rng) * 0.1f;
    for (auto& x : v) x = gauss(rng);
    for (auto& x : gate) x = -(2.0f + 4.0f * (float) (rng() % 100) / 100.0f);   // dec in (0.0025, 0.135]
    for (auto& x : beta) x = (float) (rng() % 100) / 100.0f;
    for (auto& x : z) x = gauss(rng);
    for (auto& x : sn) x = 1.0f + 0.1f * gauss(rng);
    const float eps = 1e-6f;
    const float rs = 1.0f / std::sqrt((float) S);
    std::vector<float> q_scaled = q;                     // the legacy kernel's contract: q pre-scaled
    for (auto& x : q_scaled) x *= rs;

    std::vector<float> want_o;                           // o = (S q) / sqrt(S) - what every implementation emits
    std::vector<double> st_mod = st_ref;
    ref_step(st_mod, q_scaled, k, v, gate, beta, S, h_k, h_v, want_o, true, true);
    std::vector<float> want_y, want_y_wrong;
    ref_out_norm(want_o, z, sn, h_v, S, eps, model_gate, want_y);
    ref_out_norm(want_o, z, sn, h_v, S, eps, other(model_gate), want_y_wrong);

    // ---- the traps, observable before any kernel is judged
    {
        std::vector<double> a = st_ref;
        std::vector<float> oo;
        ref_step(a, q_scaled, k, v, gate, beta, S, h_k, h_v, oo, false, true);
        observable("modulo vs interleave head pairing", rel_l1(want_o, oo));
    }
    {
        std::vector<double> a = st_ref;
        std::vector<float> oo;
        ref_step(a, q_scaled, k, v, gate, beta, S, h_k, h_v, oo, true, false);
        // The two readings differ by exactly `k (x) (beta * v * (1 - dec))`, so the fixture is only able
        // to see the trap when `dec` is far from 1 and `beta` is far from 0.  The first version used
        // gate in [-0.99, 0] - dec in [0.37, 1] - and managed 3.25%, which is a threshold the wrong
        // reading nearly clears.  Deep decay is also the realistic regime: `ssm_a = -exp(A_log) < 0` and
        // softplus is positive, so gate is comfortably negative in the real model.
        observable("decay before vs after the update", rel_l1(want_o, oo));
    }
    {
        char what[96];
        std::snprintf(what, sizeof what, "%s vs %s on the output gate (host)", gate_name(model_gate),
                      gate_name(other(model_gate)));
        observable(what, rel_l1(want_y, want_y_wrong));
    }

    // ---- device buffers
    float* d_st = dev(st_dev, "st");
    float* d_q = dev(q, "q");
    float* d_qs = dev(q_scaled, "qs");
    float* d_k = dev(k, "k");
    float* d_v = dev(v, "v");
    float* d_g = dev(gate, "g");
    float* d_b = dev(beta, "b");
    float* d_z = dev(z, "z");
    float* d_sn = dev(sn, "sn");
    float *d_o = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_o, (size_t) h_v * S * 4), "o");
    check(cudaMalloc(&d_y, (size_t) h_v * S * 4), "y");
    const strata::kernels::GdnShapes sh{S, h_k, h_v};
    auto reset_state = [&] { check(cudaMemcpy(d_st, st_dev.data(), st_dev.size() * 4, cudaMemcpyHostToDevice), "rst"); };

    // The kernel accumulates in f32 and the reference in f64, and the state is a RECURRENCE over 128 terms,
    // so 1e-5 is the arithmetic's floor here and not a structural allowance.
    const double tol = 1e-5;

    // ---- 1. legacy: gdn_step, then gdn_out_norm
    {
        reset_state();
        strata::kernels::gdn_step(d_st, d_qs, d_k, d_v, d_g, d_b, d_o, sh, nullptr);
        within("legacy gdn_step: o vs reference", rel_l1(want_o, host(d_o, want_o.size(), "o")), tol);
        within("legacy gdn_step: state (read via (S,h_v,S)) vs reference",
               state_rel(st_mod, host(d_st, st_dev.size(), "st"), S, h_v), tol);
        strata::kernels::gdn_out_norm(d_o, d_z, d_sn, d_y, h_v, S, eps, model_gate, nullptr);
        within("legacy gdn_out_norm: y vs reference", rel_l1(want_y, host(d_y, want_y.size(), "y")), tol);
        strata::kernels::gdn_out_norm(d_o, d_z, d_sn, d_y, h_v, S, eps, other(model_gate), nullptr);
        observable("legacy gdn_out_norm with the WRONG gate differs (negative)",
                   rel_l1(want_y, host(d_y, want_y.size(), "y")));
    }
    // ---- 2. pinned native: native_gdn_step (q unscaled, readout scaled), then native_gdn_out_norm
    if (S == 128) {
        reset_state();
        strata::kernels::native_gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_o, sh, stream);
        check(cudaStreamSynchronize(stream), "native step");
        within("native_gdn_step: o vs reference", rel_l1(want_o, host(d_o, want_o.size(), "o")), tol);
        within("native_gdn_step: state vs reference", state_rel(st_mod, host(d_st, st_dev.size(), "st"), S, h_v), tol);
        strata::kernels::native_gdn_out_norm(d_o, d_z, d_sn, d_y, h_v, S, eps, model_gate, stream);
        check(cudaStreamSynchronize(stream), "native norm");
        within("native_gdn_out_norm: y vs reference", rel_l1(want_y, host(d_y, want_y.size(), "y")), tol);
        strata::kernels::native_gdn_out_norm(d_o, d_z, d_sn, d_y, h_v, S, eps, other(model_gate), stream);
        check(cudaStreamSynchronize(stream), "native norm wrong");
        observable("native_gdn_out_norm with the WRONG gate differs (negative)",
                   rel_l1(want_y, host(d_y, want_y.size(), "y")));
    }
    // ---- 3. fused: step + norm in one kernel (the default native decode path)
    std::vector<float> fused_state, fused_y;
    if (S == 128) {
        reset_state();
        strata::kernels::fused_gdn_step_norm(d_st, d_q, d_k, d_v, d_g, d_b, d_z, d_sn, eps, model_gate, d_y, h_k, h_v,
                                             stream);
        check(cudaStreamSynchronize(stream), "fused");
        fused_state = host(d_st, st_dev.size(), "st");
        fused_y = host(d_y, want_y.size(), "y");
        within("fused_gdn_step_norm: state vs reference", state_rel(st_mod, fused_state, S, h_v), tol);
        within("fused_gdn_step_norm: y vs reference", rel_l1(want_y, fused_y), tol);
        reset_state();
        strata::kernels::fused_gdn_step_norm(d_st, d_q, d_k, d_v, d_g, d_b, d_z, d_sn, eps, other(model_gate), d_y, h_k,
                                             h_v, stream);
        check(cudaStreamSynchronize(stream), "fused wrong");
        observable("fused_gdn_step_norm with the WRONG gate differs (negative)",
                   rel_l1(want_y, host(d_y, want_y.size(), "y")));
    }
    // ---- 4. the verify window's kernel, one token: verify (state untouched) then commit (state written), and
    //         both bitwise the fused kernel
    if (S == 128) {
        const int C = 2 * h_k * S + h_v * S;
        std::vector<float> hbuf((size_t) C);
        std::memcpy(hbuf.data(), q.data(), q.size() * 4);
        std::memcpy(hbuf.data() + q.size(), k.data(), k.size() * 4);
        std::memcpy(hbuf.data() + 2 * q.size(), v.data(), v.size() * 4);
        float* d_h = dev(hbuf, "h");
        const std::vector<int32_t> keep1{1};
        int32_t* d_keep = dev(keep1, "keep");
        reset_state();
        strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_sn, eps, model_gate, d_y, h_k, h_v, 1,
                                             nullptr, stream, 0);
        check(cudaStreamSynchronize(stream), "multi verify");
        const std::vector<float> vy = host(d_y, want_y.size(), "y");
        const std::vector<float> vst = host(d_st, st_dev.size(), "st");
        within("gdn_step_norm_multi (verify): y vs reference", rel_l1(want_y, vy), tol);
        const bool untouched = std::memcmp(vst.data(), st_dev.data(), st_dev.size() * 4) == 0;
        std::printf("  %-58s %s\n", "gdn_step_norm_multi (verify) leaves the state untouched", untouched ? "yes" : "*** NO ***");
        if (!untouched) ++g_bad;
        const bool bit_y = std::memcmp(vy.data(), fused_y.data(), vy.size() * 4) == 0;
        std::printf("  %-58s %s\n", "gdn_step_norm_multi y is BITWISE fused_gdn_step_norm", bit_y ? "yes" : "*** NO ***");
        if (!bit_y) ++g_bad;
        strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_sn, eps, model_gate, d_y, h_k, h_v, 1,
                                             d_keep, stream, 0);
        check(cudaStreamSynchronize(stream), "multi commit");
        const std::vector<float> cst = host(d_st, st_dev.size(), "st");
        const bool bit_st = std::memcmp(cst.data(), fused_state.data(), cst.size() * 4) == 0;
        std::printf("  %-58s %s\n", "gdn_step_norm_multi (commit) state is BITWISE the fused state", bit_st ? "yes" : "*** NO ***");
        if (!bit_st) ++g_bad;
        // the split commit kernel (t_out_begin >= n_tok, no outputs): the same state, bitwise
        reset_state();
        strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_sn, eps, model_gate, d_y, h_k, h_v, 1,
                                             d_keep, stream, 1);
        check(cudaStreamSynchronize(stream), "multi commit split");
        const std::vector<float> sst = host(d_st, st_dev.size(), "st");
        const bool bit_split = std::memcmp(sst.data(), fused_state.data(), sst.size() * 4) == 0;
        std::printf("  %-58s %s\n", "gdn_step_norm_multi (split commit) state is BITWISE the fused state", bit_split ? "yes" : "*** NO ***");
        if (!bit_split) ++g_bad;
        reset_state();
        strata::kernels::gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_sn, eps, other(model_gate), d_y, h_k, h_v, 1,
                                             nullptr, stream, 0);
        check(cudaStreamSynchronize(stream), "multi wrong");
        observable("gdn_step_norm_multi with the WRONG gate differs (negative)",
                   rel_l1(want_y, host(d_y, want_y.size(), "y")));
        cudaFree(d_h); cudaFree(d_keep);
    }

    // a zero state stays zero, and a zero gate/beta leaves the state untouched (the reference's
    // PROPERTY 4 setup) - a cheap independent invariant that no layout can fake
    {
        std::vector<float> zero(st_dev.size(), 0.0f), zg((size_t) h_v, 0.0f), zb((size_t) h_v, 0.0f);
        check(cudaMemcpy(d_st, zero.data(), zero.size() * 4, cudaMemcpyHostToDevice), "z0");
        check(cudaMemcpy(d_g, zg.data(), zg.size() * 4, cudaMemcpyHostToDevice), "z1");
        check(cudaMemcpy(d_b, zb.data(), zb.size() * 4, cudaMemcpyHostToDevice), "z2");
        strata::kernels::gdn_step(d_st, d_qs, d_k, d_v, d_g, d_b, d_o, sh, nullptr);
        double nz = 0;
        for (float x : host(d_st, zero.size(), "z3")) nz += std::fabs(x);
        const bool ok = nz == 0.0;
        std::printf("  %-58s %s\n", "gate=beta=0 from a zero state is a no-op", ok ? "yes" : "*** NO ***");
        if (!ok) ++g_bad;
    }
    cudaFree(d_st); cudaFree(d_q); cudaFree(d_qs); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_z); cudaFree(d_sn); cudaFree(d_o); cudaFree(d_y);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: gdn_parity [--selftest]\n"); return 2; }
    }

    std::mt19937 rng(2024);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    cudaStream_t stream = nullptr;
    // A BLOCKING stream, deliberately: the fixtures are reset with pageable `cudaMemcpy`, which may return before the
    // DMA lands, and a non-blocking stream does not order after the legacy stream - the first version of this test
    // raced its own resets and reported the native kernels 7.5x off at one geometry and exact at the other.
    check(cudaStreamCreate(&stream), "stream");

    // ================= 1 + 5. the recurrence and the output gate, at both models' geometries =================
    // Flash-Next: 16 k-heads / 48 v-heads, sigmoid.  Qwen3.6-35B-A3B: 16 / 32, SiLU (the sigmoid run is its negative).
    recurrence_and_gate(128, 16, 48, GdnGate::Sigmoid, rng, stream);
    recurrence_and_gate(128, 16, 32, GdnGate::Silu, rng, stream);
    int bad = g_bad;

    // ================= 2. the conv =================
    {
        const int C = 24, dc = 4;      // small, so the host reference is a plain loop
        // TWO DIFFERENT LAYOUTS MEET HERE, and this is where the first version of this test failed at 0.136.
        // `ref/gdn.py` carries the conv state as (d_conv-1, C) ROW-MAJOR - (i, c) at i*C + c - because it is
        // numpy.  The kernel carries it as (C, d_conv-1) with c fastest - c*(dc-1) + i - because that is
        // ggml's `ne = {d_conv-1, C}` with ne0 fast, and it is a runtime buffer so either is legal.  The
        // mapping between them is written out below rather than assumed, in both directions.
        std::vector<float> cs_ref((size_t) (dc - 1) * C), x((size_t) C), kern((size_t) dc * C);
        for (int i = 0; i < dc - 1; ++i)
            for (int c = 0; c < C; ++c) cs_ref[(size_t) (i * C + c)] = (float) (10 * (i + 1) + c);  // labelled
        for (int c = 0; c < C; ++c) x[(size_t) c] = (float) (100 + c);
        for (size_t i = 0; i < kern.size(); ++i) kern[i] = gauss(rng);

        auto to_dev_layout = [&](const std::vector<float>& r) {
            std::vector<float> d(r.size());
            for (int i = 0; i < dc - 1; ++i)
                for (int c = 0; c < C; ++c) d[(size_t) (c * (dc - 1) + i)] = r[(size_t) (i * C + c)];
            return d;
        };

        std::vector<float> cs = to_dev_layout(cs_ref);
        std::vector<float> want;
        std::vector<float> cs_want_ref = cs_ref;
        ref_conv(cs_want_ref, x, kern, C, dc, want);
        const std::vector<float> cs_want = to_dev_layout(cs_want_ref);

        // THE LAYOUT: the manifest's `ssm_conv1d.weight` has ne = [d_conv, C], so ne0 = d_conv is fast.
        // The kernel takes `kW[c*dc + i]`; the reference's `kern` is (dc, C) row-major.  The transcription
        // below is the ONLY place the two meet.
        std::vector<float> kW((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kW[(size_t) (c * dc + i)] = kern[(size_t) (i * C + c)];
        // and the rival reading, which must differ or the fixture cannot see a transposed weight
        std::vector<float> kW_wrong((size_t) C * dc);
        for (size_t i = 0; i < kern.size(); ++i) kW_wrong[i] = kern[i];
        {
            // apply the WRONG layout by hand: tap i of channel c is kW[c*dc+i] read as kW[i*C+c]
            std::vector<float> cs2 = cs;
            std::vector<float> out_bad((size_t) C, 0.0f);
            for (int c = 0; c < C; ++c) {
                double a = 0;
                for (int i = 0; i < dc - 1; ++i) a += (double) cs2[(size_t) (i * C + c)] * (double) kW_wrong[(size_t) (c * dc + i)];
                a += (double) x[(size_t) c] * (double) kW_wrong[(size_t) (c * dc + dc - 1)];
                out_bad[(size_t) c] = (float) a;
            }
            std::printf("\n== the conv\n");
            observable("conv weight layout is observable", rel_l1(want, out_bad));
        }

        float* d_cs = dev(cs, "cs");
        float* d_x = dev(x, "cx");
        float* d_w = dev(kW, "cw");
        float* d_o = nullptr;
        check(cudaMalloc(&d_o, (size_t) C * 4), "co");
        strata::kernels::gdn_conv_step(d_cs, d_x, d_w, d_o, C, dc, nullptr);
        within("conv out vs reference", rel_l1(want, host(d_o, (size_t) C, "cgo")), 1e-6);
        within("conv state slide (mapped) vs reference", rel_l1(cs_want, host(d_cs, cs.size(), "cgs")), 1e-6);
        cudaFree(d_cs); cudaFree(d_x); cudaFree(d_w); cudaFree(d_o);
    }

    // ================= 3. l2_norm =================
    {
        const int rows = 16, cols = 128;
        std::vector<float> x((size_t) rows * cols), x_ref;
        for (auto& val : x) val = gauss(rng) * 3.0f;
        x_ref = x;
        const float eps = 1e-6f;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_ref[(size_t) (r * cols + i)] * (double) x_ref[(size_t) (r * cols + i)];
            const float inv = (float) (1.0 / std::sqrt(acc + (double) eps));
            for (int i = 0; i < cols; ++i) x_ref[(size_t) (r * cols + i)] *= inv;
        }
        // The rival reading puts THE SAME eps on the MEAN instead of on the squared norm, which at
        // ||x||^2 ~ 1152 and cols = 128 changes the divisor from sqrt(1152) to sqrt(9) - a factor of 11.3, so
        // the trap is large here.  It would be a much smaller correction on a near-zero row, where the floor
        // is the whole answer; that regime is not what this fixture exercises.
        std::vector<float> x_mean = x;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double) x_mean[(size_t) (r * cols + i)] * (double) x_mean[(size_t) (r * cols + i)];
            const float inv = (float) (1.0 / std::sqrt(acc / cols + (double) eps));
            for (int i = 0; i < cols; ++i) x_mean[(size_t) (r * cols + i)] *= inv;
        }
        std::printf("\n== l2_norm\n");
        observable("eps on sum vs on mean is observable", rel_l1(x_ref, x_mean));

        float* d_x = dev(x, "lx");
        strata::kernels::gdn_l2_norm(d_x, rows, cols, eps, nullptr);
        within("l2_norm vs reference", rel_l1(x_ref, host(d_x, x.size(), "lcg")), 1e-6);
        cudaFree(d_x);
    }

    // ================= 4. the closing norm on its own, both gates through both single-purpose kernels =================
    {
        const int S2 = 128, hv2 = 8;
        std::vector<float> o((size_t) hv2 * S2), z((size_t) hv2 * S2), sn((size_t) S2);
        for (auto& val : o) val = gauss(rng);
        for (auto& val : z) val = gauss(rng);
        for (auto& val : sn) val = 1.0f + 0.1f * gauss(rng);
        const float eps = 1e-6f;
        std::vector<float> y_sig, y_silu;
        ref_out_norm(o, z, sn, hv2, S2, eps, GdnGate::Sigmoid, y_sig);
        ref_out_norm(o, z, sn, hv2, S2, eps, GdnGate::Silu, y_silu);
        std::printf("\n== the closing norm, both gates\n");
        observable("sigmoid vs SiLU on the gate is observable", rel_l1(y_sig, y_silu));
        float* d_o = dev(o, "no");
        float* d_z = dev(z, "nz");
        float* d_sn = dev(sn, "ns");
        float* d_y = nullptr;
        check(cudaMalloc(&d_y, y_sig.size() * 4), "ny");
        for (GdnGate gt : {GdnGate::Sigmoid, GdnGate::Silu}) {
            const std::vector<float>& want = gt == GdnGate::Silu ? y_silu : y_sig;
            const std::vector<float>& wrong = gt == GdnGate::Silu ? y_sig : y_silu;
            char what[96];
            strata::kernels::gdn_out_norm(d_o, d_z, d_sn, d_y, hv2, S2, eps, gt, nullptr);
            std::vector<float> got = host(d_y, want.size(), "ncy");
            std::snprintf(what, sizeof what, "gdn_out_norm(%s) vs reference", gate_name(gt));
            within(what, rel_l1(want, got), 1e-6);
            std::snprintf(what, sizeof what, "gdn_out_norm(%s) vs the other gate's reference", gate_name(gt));
            observable(what, rel_l1(wrong, got));
            strata::kernels::native_gdn_out_norm(d_o, d_z, d_sn, d_y, hv2, S2, eps, gt, stream);
            check(cudaStreamSynchronize(stream), "nn");
            got = host(d_y, want.size(), "nny");
            std::snprintf(what, sizeof what, "native_gdn_out_norm(%s) vs reference", gate_name(gt));
            within(what, rel_l1(want, got), 1e-5);
        }
        cudaFree(d_o); cudaFree(d_z); cudaFree(d_sn); cudaFree(d_y);
    }

    bad = g_bad;
    std::printf("\ngdn: %d failures\n", bad);
    cudaStreamDestroy(stream);
    if (bad) return 1;
    if (selftest) std::printf("gdn_parity OK\n");
    return 0;
}
