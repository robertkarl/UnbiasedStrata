// src/kernels/qwen36_seam_parity.cpp - Qwen3.6 step 3 (GDN, router, shared expert): the engine's DECODE kernels
// held to the numpy oracle's seams, by llama.cpp's cb() names, on the synthetic fixture.
//
//   qwen36_seam_parity <dir>     <dir> is what cmake/qwen36_oracle_dumps.cmake wrote: k.gguf, o_f32/, o_q8/
//
// The oracle (ref/qwen36.py) agrees with the pinned llama.cpp graph on every seam (docs/qwen36-port/
// oracle-crosscheck.md), so holding a kernel to its dumps is holding it to llama.cpp.  The fixture is the toy
// geometry at the engine's pinned state size: hidden 64, GDN 2 k-heads / 4 v-heads of 128 (VPK 2, the real model's
// ratio), conv 4, 256 experts top-8 of 32, shared expert 512 wide in Q8_0, the BF16-read weights (ssm_alpha,
// ssm_beta, ffn_gate_inp, ffn_gate_inp_shexp) BF16-exact.  11 tokens, one at a time, the GDN state carried.
//
// Seams, per the study's step 3:
//   GDN        `conv_output_silu-0` (every token), `new_state-0` (after the 11th), `final_output-0` (every token),
//              through every decode implementation: legacy (gdn.cu), pinned native (native_gdn*.cu), fused
//              (fused_gdn.cu), and the verify window's multi-token kernels (verify_kernels.cu), which carry the
//              state across two windows (8 + 3 tokens) the way verify.cpp does.  The fused path also holds
//              `fused_gdn_ab` to `gate-0` / `beta_sigmoid-0` and `fused_gdn_conv_l2` to `q_conv_predelta-0`.
//   router     `ffn_moe_topk-0` exact and `ffn_moe_weights_norm-0` from the oracle's own `ffn_moe_logits-0`,
//              256 experts, top-8, through `router_top10`.
//   shexp      `ffn_shexp_gated-0` through `shared_expert` on the native Q8_0 path (the 1-D BF16 scalar gate),
//              against the Q8 oracle (the activation rounding the engine performs), and `shared_expert_multi`
//              bitwise the per-token call.
//
// Negative cases, through the kernels, each required > 5% relative L1 before the right reading is judged: the
// sigmoid gate on `final_output-0` (the Flash-Next reading; the oracle's control says 1.07), the un-normalised
// router weights on `ffn_moe_weights_norm-0` (0.14), the shared expert's gate skipped on `ffn_shexp_gated-0`.
//
// Tolerances are relative L1 (|a-b| summed over |a| summed): the F32 oracle's per-seam classes of
// oracle-crosscheck.md ("Tolerance policy") where the kernel works in F32, the Q8 oracle with its 4x margin where
// the kernel rounds activations.  Exit 77 when the fixture is absent (no python with numpy at configure time).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::kernels::GdnGate;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int g_bad = 0;
void within(const std::string& what, double rel, double tol) {
    std::printf("  %-70s rel %.3e  (tol %.0e)%s\n", what.c_str(), rel, tol, rel <= tol ? "" : "   *** over tolerance ***");
    if (!(rel <= tol)) ++g_bad;
}
void observable(const std::string& what, double rel) {
    std::printf("  %-70s %-4s (%.2f%% apart)\n", what.c_str(), rel > 0.05 ? "yes" : "*** NO ***", rel * 100);
    if (!(rel > 0.05)) ++g_bad;
}
void yes(const std::string& what, bool ok) {
    std::printf("  %-70s %s\n", what.c_str(), ok ? "yes" : "*** NO ***");
    if (!ok) ++g_bad;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}
double rel_l1(const float* a, const float* b, size_t n) {
    return rel_l1(std::vector<float>(a, a + n), std::vector<float>(b, b + n));
}

std::vector<float> load_f32(const fs::path& dir, const std::string& stem, size_t want) {
    const fs::path p = dir / (stem + ".f32");
    FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.string().c_str()); std::exit(1); }
    std::vector<float> v(want);
    const size_t got = std::fread(v.data(), 4, want, f);
    const bool more = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != want || more) {
        std::fprintf(stderr, "%s: expected %zu floats, file has %s\n", p.string().c_str(), want, more ? "more" : "fewer");
        std::exit(1);
    }
    return v;
}

template <class T>
T* dev(const std::vector<T>& h, const char* what) {
    T* p = nullptr;
    check(cudaMalloc(&p, h.size() * sizeof(T) + 64), what);
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    return p;
}
template <class T>
T* dev_zero(size_t n, const char* what) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T) + 64), what);
    check(cudaMemset(p, 0, n * sizeof(T) + 64), what);
    return p;
}
template <class T>
std::vector<T> host(const T* d, size_t n, const char* what) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), what);
    return h;
}

/// A tensor of the fixture as raw bytes, with its GGUF `ne` and type checked.
struct Tensor {
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    const float* f32() const { return (const float*) bytes.data(); }
    std::vector<float> floats() const { return std::vector<float>(f32(), f32() + bytes.size() / 4); }
};
Tensor tensor(const strata::GgufFile& g, const std::string& name, const std::vector<uint64_t>& ne, uint32_t type) {
    const strata::TensorInfo* t = g.find(name);
    if (!t) { std::fprintf(stderr, "%s: missing\n", name.c_str()); std::exit(1); }
    if (t->shape != ne || t->type != type) {
        std::fprintf(stderr, "%s: type %s ne", name.c_str(), t->type_name());
        for (auto d : t->shape) std::fprintf(stderr, " %llu", (unsigned long long) d);
        std::fprintf(stderr, " is not the fixture this test expects\n");
        std::exit(1);
    }
    Tensor r;
    r.ne = t->shape;
    r.type = t->type;
    r.bytes.assign(g.tensor_data(*t), g.tensor_data(*t) + strata::tensor_payload_bytes(*t));
    return r;
}
std::vector<uint16_t> to_bf16(const std::vector<float>& f) {
    std::vector<uint16_t> b(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        b[i] = strata::kernels::bf16_from_f32(f[i]);
        if (strata::kernels::f32_from_bf16(b[i]) != f[i]) {
            std::fprintf(stderr, "a BF16-read weight is not BF16-exact (the fixture was not written with --bf16-exact)\n");
            std::exit(1);
        }
    }
    return b;
}

// --------------------------------------------------------------------------------------------- geometry
constexpr int N = 64, HK = 2, HV = 4, S = 128, DC = 4;
constexpr int QK = HK * S, V = HV * S, C = 2 * QK + V;
constexpr int E = 256, K = 8, FF_SH = 512;
constexpr int T = 11;
constexpr float EPS = 1e-6f;
constexpr uint32_t GGML_F32 = 0, GGML_Q8_0 = 8;

/// The oracle's `new_state-0`: numpy (Hv, S_v = j, S_k = i) -> the engine's (i, h, j).
std::vector<float> oracle_state_to_engine(const std::vector<float>& o) {
    std::vector<float> s((size_t) S * HV * S);
    for (int h = 0; h < HV; ++h)
        for (int j = 0; j < S; ++j)
            for (int i = 0; i < S; ++i) s[((size_t) i * HV + h) * S + j] = o[((size_t) h * S + j) * S + i];
    return s;
}

struct Fixture {
    fs::path dir, o32, oq8;
    // weights
    std::vector<float> conv_w, ssm_norm, ssm_a, dt, alpha_w, beta_w;
    std::vector<uint16_t> alpha_bf16, beta_bf16, ginp_bf16;
    std::vector<uint8_t> sh_gate, sh_up, sh_down;
    // oracle inputs and seams
    std::vector<float> x_attn_norm, qkv, z, alpha_raw, beta_raw, gate_ref, beta_sig_ref, conv_silu_ref, q_pre_ref,
        k_pre_ref, new_state_ref, final_ref, x_post_norm, logits, topk_ref, wnorm_ref, w_raw_ref, shexp_gated_q8,
        shexp_gated_f32, shexp_ungated_q8;
};

Fixture load(const fs::path& dir) {
    Fixture f;
    f.dir = dir; f.o32 = dir / "o_f32"; f.oq8 = dir / "o_q8";
    strata::GgufFile g((dir / "k.gguf").string());
    f.conv_w = tensor(g, "blk.0.ssm_conv1d.weight", {DC, C}, GGML_F32).floats();      // ne0 = d_conv fastest: kW[c*4 + i]
    f.ssm_norm = tensor(g, "blk.0.ssm_norm.weight", {S}, GGML_F32).floats();
    f.ssm_a = tensor(g, "blk.0.ssm_a", {HV}, GGML_F32).floats();
    f.dt = tensor(g, "blk.0.ssm_dt.bias", {HV}, GGML_F32).floats();
    f.alpha_w = tensor(g, "blk.0.ssm_alpha.weight", {N, HV}, GGML_F32).floats();     // (HV, N) row-major
    f.beta_w = tensor(g, "blk.0.ssm_beta.weight", {N, HV}, GGML_F32).floats();
    f.alpha_bf16 = to_bf16(f.alpha_w);
    f.beta_bf16 = to_bf16(f.beta_w);
    f.ginp_bf16 = to_bf16(tensor(g, "blk.0.ffn_gate_inp_shexp.weight", {N}, GGML_F32).floats());
    f.sh_gate = tensor(g, "blk.0.ffn_gate_shexp.weight", {N, FF_SH}, GGML_Q8_0).bytes;   // FF_SH rows of N
    f.sh_up = tensor(g, "blk.0.ffn_up_shexp.weight", {N, FF_SH}, GGML_Q8_0).bytes;
    f.sh_down = tensor(g, "blk.0.ffn_down_shexp.weight", {FF_SH, N}, GGML_Q8_0).bytes;   // N rows of FF_SH

    f.x_attn_norm = load_f32(f.o32, "attn_norm-0", (size_t) T * N);
    f.qkv = load_f32(f.o32, "linear_attn_qkv_mixed-0", (size_t) T * C);
    f.z = load_f32(f.o32, "z-0", (size_t) T * V);
    f.alpha_raw = load_f32(f.o32, "alpha-0", (size_t) T * HV);
    f.beta_raw = load_f32(f.o32, "beta-0", (size_t) T * HV);
    f.gate_ref = load_f32(f.o32, "gate-0", (size_t) T * HV);
    f.beta_sig_ref = load_f32(f.o32, "beta_sigmoid-0", (size_t) T * HV);
    f.conv_silu_ref = load_f32(f.o32, "conv_output_silu-0", (size_t) T * C);
    f.q_pre_ref = load_f32(f.o32, "q_conv_predelta-0", (size_t) T * QK);
    f.k_pre_ref = load_f32(f.o32, "k_conv_predelta-0", (size_t) T * QK);
    f.new_state_ref = oracle_state_to_engine(load_f32(f.o32, "new_state-0", (size_t) HV * S * S));
    f.final_ref = load_f32(f.o32, "final_output-0", (size_t) T * V);
    f.x_post_norm = load_f32(f.o32, "attn_post_norm-0", (size_t) T * N);
    f.logits = load_f32(f.o32, "ffn_moe_logits-0", (size_t) T * E);
    f.topk_ref = load_f32(f.o32, "ffn_moe_topk-0", (size_t) T * K);
    f.wnorm_ref = load_f32(f.o32, "ffn_moe_weights_norm-0", (size_t) T * K);
    f.w_raw_ref = load_f32(f.o32, "ffn_moe_weights-0", (size_t) T * K);
    f.shexp_gated_q8 = load_f32(f.oq8, "ffn_shexp_gated-0", (size_t) T * N);
    f.shexp_gated_f32 = load_f32(f.o32, "ffn_shexp_gated-0", (size_t) T * N);
    f.shexp_ungated_q8 = load_f32(f.oq8, "ffn_shexp-0", (size_t) T * N);
    return f;
}

// --------------------------------------------------------------------------------------------- GDN
// The per-token scratch every path shares; the state and the conv history are reset per path.
struct GdnDev {
    float *conv_hist, *state, *qkv_t, *conv_out, *h, *alpha, *beta, *gate, *o, *z_t, *y, *w, *norm, *a, *dt, *x_t;
    uint16_t *alpha_bf16, *beta_bf16;
};
GdnDev gdn_dev(const Fixture& f) {
    GdnDev d{};
    d.conv_hist = dev_zero<float>((size_t) C * (DC - 1), "hist");
    d.state = dev_zero<float>((size_t) S * HV * S, "state");
    d.qkv_t = dev_zero<float>(C, "qkv");
    d.conv_out = dev_zero<float>(C, "conv_out");
    d.h = dev_zero<float>(C, "h");
    d.alpha = dev_zero<float>(HV, "alpha");
    d.beta = dev_zero<float>(HV, "beta");
    d.gate = dev_zero<float>(HV, "gate");
    d.o = dev_zero<float>(V, "o");
    d.z_t = dev_zero<float>(V, "z");
    d.y = dev_zero<float>(V, "y");
    d.x_t = dev_zero<float>(N, "x");
    d.w = dev(f.conv_w, "conv_w");
    d.norm = dev(f.ssm_norm, "ssm_norm");
    d.a = dev(f.ssm_a, "ssm_a");
    d.dt = dev(f.dt, "dt");
    d.alpha_bf16 = dev(f.alpha_bf16, "alpha_bf16");
    d.beta_bf16 = dev(f.beta_bf16, "beta_bf16");
    return d;
}
void gdn_reset(const GdnDev& d) {
    check(cudaMemset(d.conv_hist, 0, (size_t) C * (DC - 1) * 4), "reset hist");
    check(cudaMemset(d.state, 0, (size_t) S * HV * S * 4), "reset state");
    check(cudaDeviceSynchronize(), "reset sync");
}
void token_inputs(const Fixture& f, const GdnDev& d, int t) {
    check(cudaMemcpy(d.qkv_t, f.qkv.data() + (size_t) t * C, C * 4, cudaMemcpyHostToDevice), "qkv_t");
    check(cudaMemcpy(d.z_t, f.z.data() + (size_t) t * V, V * 4, cudaMemcpyHostToDevice), "z_t");
    check(cudaMemcpy(d.alpha, f.alpha_raw.data() + (size_t) t * HV, HV * 4, cudaMemcpyHostToDevice), "alpha_t");
    check(cudaMemcpy(d.beta, f.beta_raw.data() + (size_t) t * HV, HV * 4, cudaMemcpyHostToDevice), "beta_t");
    check(cudaMemcpy(d.x_t, f.x_attn_norm.data() + (size_t) t * N, N * 4, cudaMemcpyHostToDevice), "x_t");
    check(cudaDeviceSynchronize(), "inputs sync");
}

/// One decode path over the 11 tokens.  `path`: 0 legacy, 1 native, 2 fused.  Returns the worst per-token relative
/// L1 at `conv_output_silu-0` and `final_output-0`, and the state's at the end; with `act` the gate used.
struct GdnResult { double conv = 0, final_out = 0, state = 0, gate = 0, beta = 0, qk_pre = 0; };
GdnResult run_gdn(const Fixture& f, const GdnDev& d, int path, GdnGate act, cudaStream_t cs) {
    using namespace strata::kernels;
    GdnResult r;
    gdn_reset(d);
    const strata::kernels::GdnShapes sh{S, HK, HV};
    const float rs = 1.0f / std::sqrt((float) S);
    for (int t = 0; t < T; ++t) {
        token_inputs(f, d, t);
        const float* conv_silu_t = f.conv_silu_ref.data() + (size_t) t * C;
        if (path == 0) {                                            // ---- legacy (gdn.cu + elementwise.cu)
            gdn_conv_step(d.conv_hist, d.qkv_t, d.w, d.conv_out, C, DC, nullptr);
            check(cudaMemcpy(d.h, d.conv_out, C * 4, cudaMemcpyDeviceToDevice), "h");
            silu_inplace(d.h, C, nullptr);
            check(cudaDeviceSynchronize(), "legacy conv");
            r.conv = std::fmax(r.conv, rel_l1(conv_silu_t, host(d.h, C, "h").data(), C));
            gdn_l2_norm(d.h, HK, S, EPS, nullptr);
            gdn_l2_norm(d.h + QK, HK, S, EPS, nullptr);
            scale_inplace(d.h, QK, rs, nullptr);                    // the legacy step wants q pre-scaled
            gdn_beta_gate(d.beta, HV, nullptr);
            gdn_gate(d.alpha, d.dt, d.a, d.gate, 1, HV, nullptr);
            gdn_step(d.state, d.h, d.h + QK, d.h + 2 * QK, d.gate, d.beta, d.o, sh, nullptr);
            gdn_out_norm(d.o, d.z_t, d.norm, d.y, HV, S, EPS, act, nullptr);
            check(cudaDeviceSynchronize(), "legacy token");
        } else if (path == 1) {                                     // ---- pinned native, unfused
            native_gdn_conv_silu(d.conv_hist, d.qkv_t, d.w, d.conv_out, d.h, C, DC, cs);
            check(cudaStreamSynchronize(cs), "native conv");
            r.conv = std::fmax(r.conv, rel_l1(conv_silu_t, host(d.h, C, "h").data(), C));
            native_gdn_l2_norm(d.h, HK, S, EPS, cs);
            native_gdn_l2_norm(d.h + QK, HK, S, EPS, cs);
            native_gdn_beta_gate(d.beta, HV, cs);
            native_gdn_gate(d.alpha, d.dt, d.a, d.gate, HV, cs);
            native_gdn_step(d.state, d.h, d.h + QK, d.h + 2 * QK, d.gate, d.beta, d.o, sh, cs);
            native_gdn_out_norm(d.o, d.z_t, d.norm, d.y, HV, S, EPS, act, cs);
            check(cudaStreamSynchronize(cs), "native token");
        } else {                                                    // ---- fused (the default native decode)
            fused_gdn_conv_l2(d.conv_hist, d.qkv_t, d.w, d.h, C, 2 * HK, EPS, cs);
            fused_gdn_ab(d.x_t, d.alpha_bf16, d.beta_bf16, d.dt, d.a, d.gate, d.beta, N, HV, cs);
            check(cudaStreamSynchronize(cs), "fused pre");
            const std::vector<float> h = host(d.h, C, "h");
            // h is q|k L2-normalised (not pre-scaled) and v after SiLU: the v part is conv_output_silu's, the q/k
            // parts are the oracle's q_conv_predelta / k_conv_predelta
            r.conv = std::fmax(r.conv, rel_l1(conv_silu_t + 2 * QK, h.data() + 2 * QK, V));
            r.qk_pre = std::fmax(r.qk_pre, rel_l1(f.q_pre_ref.data() + (size_t) t * QK, h.data(), QK));
            r.qk_pre = std::fmax(r.qk_pre, rel_l1(f.k_pre_ref.data() + (size_t) t * QK, h.data() + QK, QK));
            r.gate = std::fmax(r.gate, rel_l1(f.gate_ref.data() + (size_t) t * HV, host(d.gate, HV, "gate").data(), HV));
            r.beta = std::fmax(r.beta, rel_l1(f.beta_sig_ref.data() + (size_t) t * HV, host(d.beta, HV, "beta").data(), HV));
            fused_gdn_step_norm(d.state, d.h, d.h + QK, d.h + 2 * QK, d.gate, d.beta, d.z_t, d.norm, EPS, act, d.y, HK, HV, cs);
            check(cudaStreamSynchronize(cs), "fused token");
        }
        r.final_out = std::fmax(r.final_out, rel_l1(f.final_ref.data() + (size_t) t * V, host(d.y, V, "y").data(), V));
    }
    r.state = rel_l1(f.new_state_ref, host(d.state, (size_t) S * HV * S, "state"));
    return r;
}

/// The verify window's kernels: two windows (8 + 3 tokens) as verify.cpp runs them - the multi kernels over the
/// window from the untouched state, then the commit of every token (all accepted here).
GdnResult run_verify_window(const Fixture& f, const GdnDev& d, GdnGate act, cudaStream_t cs) {
    using namespace strata::kernels;
    GdnResult r;
    gdn_reset(d);
    float* qkv_all = dev(f.qkv, "qkv_all");
    float* x_all = dev(f.x_attn_norm, "x_all");
    float* z_all = dev(f.z, "z_all");
    float* h_all = dev_zero<float>((size_t) T * C, "h_all");
    float* gate_all = dev_zero<float>((size_t) T * HV, "gate_all");
    float* beta_all = dev_zero<float>((size_t) T * HV, "beta_all");
    float* y_all = dev_zero<float>((size_t) T * V, "y_all");
    for (int t0 = 0; t0 < T; t0 += kVerifyMaxT) {
        const int n = std::min(kVerifyMaxT, T - t0);
        const std::vector<int32_t> keep{n};
        int32_t* d_keep = dev(keep, "keep");
        gdn_conv_l2_multi(d.conv_hist, qkv_all + (size_t) t0 * C, d.w, h_all + (size_t) t0 * C, C, 2 * HK, EPS, n, cs, 0);
        gdn_ab_multi(x_all + (size_t) t0 * N, d.alpha_bf16, d.beta_bf16, d.dt, d.a, gate_all + (size_t) t0 * HV,
                     beta_all + (size_t) t0 * HV, N, HV, n, cs);
        // verify: outputs for every token, the state untouched
        gdn_step_norm_multi(d.state, h_all + (size_t) t0 * C, C, gate_all + (size_t) t0 * HV, beta_all + (size_t) t0 * HV,
                            z_all + (size_t) t0 * V, d.norm, EPS, act, y_all + (size_t) t0 * V, HK, HV, n, nullptr, cs, 0);
        check(cudaStreamSynchronize(cs), "verify window");
        for (int t = t0; t < t0 + n; ++t) {
            const std::vector<float> h = host(h_all + (size_t) t * C, C, "h");
            r.conv = std::fmax(r.conv, rel_l1(f.conv_silu_ref.data() + (size_t) t * C + 2 * QK, h.data() + 2 * QK, V));
            r.qk_pre = std::fmax(r.qk_pre, rel_l1(f.q_pre_ref.data() + (size_t) t * QK, h.data(), QK));
            r.qk_pre = std::fmax(r.qk_pre, rel_l1(f.k_pre_ref.data() + (size_t) t * QK, h.data() + QK, QK));
            r.gate = std::fmax(r.gate, rel_l1(f.gate_ref.data() + (size_t) t * HV, host(gate_all + (size_t) t * HV, HV, "g").data(), HV));
            r.beta = std::fmax(r.beta, rel_l1(f.beta_sig_ref.data() + (size_t) t * HV, host(beta_all + (size_t) t * HV, HV, "b").data(), HV));
            r.final_out = std::fmax(r.final_out, rel_l1(f.final_ref.data() + (size_t) t * V, host(y_all + (size_t) t * V, V, "y").data(), V));
        }
        // commit: all n tokens accepted, conv history and state advance
        gdn_conv_commit(d.conv_hist, qkv_all + (size_t) t0 * C, C, d_keep, cs);
        gdn_step_norm_multi(d.state, h_all + (size_t) t0 * C, C, gate_all + (size_t) t0 * HV, beta_all + (size_t) t0 * HV,
                            z_all + (size_t) t0 * V, d.norm, EPS, act, d.y, HK, HV, n, d_keep, cs, n);
        check(cudaStreamSynchronize(cs), "commit window");
        cudaFree(d_keep);
    }
    r.state = rel_l1(f.new_state_ref, host(d.state, (size_t) S * HV * S, "state"));
    cudaFree(qkv_all); cudaFree(x_all); cudaFree(z_all); cudaFree(h_all); cudaFree(gate_all); cudaFree(beta_all); cudaFree(y_all);
    return r;
}

void gdn_section(const Fixture& f, cudaStream_t cs) {
    std::printf("\n== GDN layer 0: %d k-heads / %d v-heads of %d (VPK %d), conv %d, %d tokens one at a time\n", HK, HV, S,
                HV / HK, DC, T);
    const GdnDev d = gdn_dev(f);
    // tolerances: conv (one 4-tap f32 dot per channel) 1e-6; the oracle-crosscheck classes for the rest, F32 column
    // with the 2x margin folded in: gate/beta (a 64-term dot against a BF16-exact weight) 1e-6, q/k L2 1e-6,
    // final_output 8e-5 (the "gdn_out" class), the recurrent state after 11 tokens 2e-5.
    const char* names[] = {"legacy (gdn.cu)", "native (native_gdn*.cu)", "fused (fused_gdn.cu)"};
    for (int path = 0; path < 3; ++path) {
        const GdnResult r = run_gdn(f, d, path, GdnGate::Silu, cs);
        const std::string p = names[path];
        within(p + ": conv_output_silu-0 (worst token)", r.conv, 1e-6);
        if (path == 2) {
            within(p + ": q_conv_predelta-0 / k_conv_predelta-0 (fused_gdn_conv_l2)", r.qk_pre, 1e-6);
            within(p + ": gate-0 (fused_gdn_ab softplus epilogue)", r.gate, 1e-6);
            within(p + ": beta_sigmoid-0 (fused_gdn_ab sigmoid epilogue)", r.beta, 1e-6);
        }
        within(p + ": final_output-0 (worst token, SiLU gate)", r.final_out, 8e-5);
        within(p + ": new_state-0 after 11 tokens", r.state, 2e-5);
    }
    {
        const GdnResult r = run_verify_window(f, d, GdnGate::Silu, cs);
        const std::string p = "verify window (verify_kernels.cu, 8 + 3 tokens)";
        within(p + ": conv_output_silu-0 v part / q,k predelta", std::fmax(r.conv, r.qk_pre), 1e-6);
        within(p + ": gate-0 / beta_sigmoid-0 (gdn_ab_multi)", std::fmax(r.gate, r.beta), 1e-6);
        within(p + ": final_output-0 (worst token, SiLU gate)", r.final_out, 8e-5);
        within(p + ": new_state-0 after both commits", r.state, 2e-5);
    }
    // the negative: the Flash-Next gate through the kernels (the oracle's control: 1.07 at final_output-0)
    {
        const GdnResult r = run_gdn(f, d, 2, GdnGate::Sigmoid, cs);
        observable("sigmoid gate (the qwen4exp reading) moves final_output-0 (negative)", r.final_out);
        const GdnResult rv = run_verify_window(f, d, GdnGate::Sigmoid, cs);
        observable("sigmoid gate through the verify window moves final_output-0 (negative)", rv.final_out);
    }
}

// --------------------------------------------------------------------------------------------- router
void router_section(const Fixture& f) {
    std::printf("\n== router layer 0: %d experts, top-%d, from the oracle's ffn_moe_logits-0 (%d tokens)\n", E, K, T);
    float* d_l = dev(f.logits, "logits");
    int* d_ids = dev_zero<int>((size_t) T * K, "ids");
    float* d_w = dev_zero<float>((size_t) T * K, "w");
    strata::kernels::router_top10(d_l, T, E, K, d_ids, d_w, nullptr);
    const std::vector<int> ids = host(d_ids, (size_t) T * K, "ids");
    const std::vector<float> w = host(d_w, (size_t) T * K, "w");
    int id_bad = 0;
    for (size_t i = 0; i < ids.size(); ++i) id_bad += ids[i] != (int) f.topk_ref[i];
    yes("ffn_moe_topk-0: every (token, slot) identical (" + std::to_string(id_bad) + " differ)", id_bad == 0);
    within("ffn_moe_weights_norm-0 (softmax, gather, renormalise)", rel_l1(f.wnorm_ref, w), 1e-5);
    observable("un-normalised weights (ffn_moe_weights-0) differ from the normalised (negative)", rel_l1(f.wnorm_ref, f.w_raw_ref));
    // every token's weights sum to 1 (the clamp at 2^-14 is unreachable: top-8 of 256 sums to at least 8/256)
    double worst = 0;
    for (int t = 0; t < T; ++t) {
        double s = 0;
        for (int i = 0; i < K; ++i) s += w[(size_t) t * K + i];
        worst = std::fmax(worst, std::fabs(s - 1.0));
    }
    within("weights sum to 1 per token (|sum-1|)", worst, 1e-6);
    cudaFree(d_l); cudaFree(d_ids); cudaFree(d_w);
}

// --------------------------------------------------------------------------------------------- shared expert
void shexp_section(const Fixture& f, cudaStream_t cs) {
    using namespace strata::kernels;
    std::printf("\n== shared expert layer 0: %d wide, Q8_0 weights on the native path, 1-D BF16 scalar gate (%d tokens)\n", FF_SH, T);
    uint8_t* d_gate = dev(f.sh_gate, "sh_gate");
    uint8_t* d_up = dev(f.sh_up, "sh_up");
    uint8_t* d_down = dev(f.sh_down, "sh_down");
    uint16_t* d_ginp = dev(f.ginp_bf16, "ginp");
    float* d_x = dev(f.x_post_norm, "x_post_norm");
    uint16_t* d_xb = dev_zero<uint16_t>((size_t) T * N, "x_bf16");
    f32_to_bf16_bulk(d_x, d_xb, (int64_t) T * N, cs);
    float* d_scratch = dev_zero<float>(shared_expert_scratch_bytes(FF_SH) / 4 + 16, "scratch");
    const size_t q81 = native_q8_1_bytes(FF_SH > N ? FF_SH : N, kVerifyMaxT);
    uint8_t* d_q81 = dev_zero<uint8_t>(q81, "q8_1");
    float* d_out = dev_zero<float>((size_t) T * N, "out");
    NativeSharedWeights nw;
    nw.gate_type = (int) GGML_Q8_0; nw.gate_data = d_gate;
    nw.up_type = (int) GGML_Q8_0; nw.up_data = d_up;
    nw.down_type = (int) GGML_Q8_0; nw.down_data = d_down;
    nw.q8_1 = d_q81;
    const SForm dummy{2, -1, 64, Codebook::Affine, false, 0};   // unused: all three projections are native
    shared_expert_set_native_bf16(true);
    for (int t = 0; t < T; ++t)
        shared_expert(nullptr, nullptr, d_xb + (size_t) t * N, dummy, nullptr, nullptr, nullptr, dummy, nullptr, nullptr,
                      nullptr, dummy, nullptr, nullptr, nullptr, d_ginp, d_scratch, d_out + (size_t) t * N, N, FF_SH, 32,
                      cs, d_x + (size_t) t * N, &nw);
    check(cudaStreamSynchronize(cs), "shexp");
    const std::vector<float> got = host(d_out, (size_t) T * N, "out");
    // the Q8 oracle is the reference for a path that rounds activations to Q8_1; its class is ffn_out 2.1e-3 /
    // 4.5e-3 on the real file where the EXPERT weights are IQ2/IQ3; here every weight is Q8_0 (exact integer
    // products on both sides) so the only difference left is summation order
    within("ffn_shexp_gated-0 vs the Q8 oracle (--act-quant q8)", rel_l1(f.shexp_gated_q8, got), 1e-5);
    std::printf("  %-70s rel %.3e  (the activation rounding itself; not a gate)\n", "ffn_shexp_gated-0 vs the F32 oracle",
                rel_l1(f.shexp_gated_f32, got));
    observable("the scalar gate skipped (ffn_shexp-0) differs from the gated (negative)", rel_l1(f.shexp_gated_q8, f.shexp_ungated_q8));
    // the verify window's multi-token call: every token bitwise the per-token call
    {
        float* d_g = dev_zero<float>((size_t) kVerifyMaxT * FF_SH, "g");
        float* d_u = dev_zero<float>((size_t) kVerifyMaxT * FF_SH, "u");
        float* d_s = dev_zero<float>(kVerifyMaxT, "s");
        float* d_out2 = dev_zero<float>((size_t) T * N, "out2");
        for (int t0 = 0; t0 < T; t0 += kVerifyMaxT) {
            const int n = std::min(kVerifyMaxT, T - t0);
            shared_expert_multi(n, d_x + (size_t) t0 * N, d_xb + (size_t) t0 * N, nw, d_ginp, d_g, d_u, d_s,
                                d_out2 + (size_t) t0 * N, N, FF_SH, cs);
        }
        check(cudaStreamSynchronize(cs), "shexp multi");
        const std::vector<float> got2 = host(d_out2, (size_t) T * N, "out2");
        const double rel = rel_l1(got, got2);
        const bool bitwise = std::memcmp(got.data(), got2.data(), got.size() * 4) == 0;
        std::printf("  %-70s %s (rel %.3e)\n", "shared_expert_multi (8 + 3 tokens) vs per-token shared_expert", bitwise ? "bitwise" : "NOT bitwise", rel);
        within("shared_expert_multi vs per-token (the multi-column MMVQ's own rounding)", rel, 1e-5);
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_s); cudaFree(d_out2);
    }
    shared_expert_set_native_bf16(false);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_down); cudaFree(d_ginp); cudaFree(d_x); cudaFree(d_xb);
    cudaFree(d_scratch); cudaFree(d_q81); cudaFree(d_out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: qwen36_seam_parity <dir with k.gguf, o_f32/, o_q8/ (cmake/qwen36_oracle_dumps.cmake)>\n");
        return 2;
    }
    const fs::path dir = argv[1];
    if (fs::exists(dir / "SKIP") || !fs::exists(dir / "k.gguf") || !fs::exists(dir / "o_f32" / "final_output-0.f32")) {
        std::printf("qwen36_seam_parity: fixture not present in %s (no python3 with numpy and gguf-py?): skipped\n",
                    dir.string().c_str());
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("qwen36_seam_parity: no CUDA device: skipped\n");
        return 77;
    }
    const Fixture f = load(dir);
    cudaStream_t cs = nullptr;
    check(cudaStreamCreate(&cs), "stream");   // blocking: ordered after the synchronous memcpys that stage each token
    gdn_section(f, cs);
    router_section(f);
    shexp_section(f, cs);
    std::printf("\nqwen36_seam_parity: %d failures\n", g_bad);
    cudaStreamDestroy(cs);
    return g_bad ? 1 : 0;
}
