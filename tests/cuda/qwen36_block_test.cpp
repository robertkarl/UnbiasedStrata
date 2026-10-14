// tests/cuda/qwen36_block_test.cpp - the plain block (hc = 1, Qwen3.6) composed by the engine, held to the oracle.
//
//   qwen36_block_test <synth dir> <dump dir>
//
// <synth dir> holds tools/qwen36_synth_gguf.py's toy.gguf (CMake's qwen36_synth_fixtures); <dump dir> the oracle's
// cb()-named dumps of that file on 8 tokens (cmake/qwen36_block_fixtures.cmake runs ref/qwen36.py).  Maya's layer
// composition shape (docs/QWEN36-35B-A3B.md section 6, step 3, fourth bullet), for every layer of the fixture
// (0-2 GDN + MoE, 3 attention + MoE) and the head:
//
//   1. the single-token path THROUGH THE ENGINE'S OWN FUNCTIONS on a WeightTable loaded from a pack the test writes
//      (the F32 norms, the BF16 router, the BF16 shared-expert gate): `block_layer_pre` stage 0 (`attn_norm-N`), the
//      residual add (`attn_residual-N`), `block_layer_pre` half 2 (`attn_post_norm-N`, the router's `ffn_moe_topk-N`
//      and `ffn_moe_weights_norm-N`), `block_layer_post` (`ffn_out-N`, `l_out-N`), `lm_head_mix` (`result_norm`) and
//      the logits (`result_output`, the engine's normed vector through a host F32 projection of `output.weight`);
//   2. the verify window's row arithmetic: the T = 8 row calls (`plain_norm_rows`, `plain_residual_rows`,
//      `native_moe_combine_multi`) bitwise the one-row calls per row, and the 8-token sequence carried through the
//      four layers against `attn_norm-N`, `attn_residual-N`, `attn_post_norm-N`, `ffn_out-N`, `l_out-N`, `h_nextn`
//      per position;
//   3. the negative cases (upstream-tooling.md section 3): the GR read at hc = 1 with identity bottleneck weights
//      is NOT the plain norm (> 5 % relative L1), the post-attention norm skipped (and unweighted) is not the normed
//      activation, top-10 on the 8-expert file is refused by name, a missing norm is refused by name.
//
// THE MIXERS AND THE EXPERTS ARE TEACHER-FORCED from the dumps (`linear_attn_out-N`, `attn_output-N`, the per-expert
// `ffn_moe_down-N`, `ffn_shexp_gated-N`): the fixture's projections are F32 and the engine has no F32 projection
// kernel (the GDN and attention mixers are the sibling step-3 branches', the quantized pack is step 4's), so the
// block's own seams are asserted here and the mixer seams are REPORTED (the engine's own error on the attempt).
//
// Tolerances (relative L2, ||engine - oracle|| / ||oracle||): the norms 1e-6 (`rsqrtf` is 2 ulp, the oracle's own
// numpy-vs-numpy figure of 5e-8 in oracle-crosscheck.md is below one CUDA ulp), the adds and the MoE composition
// 1e-6 (F32 with FMA against numpy's sequential adds), the logits 1e-5 (a 64-long F32 dot product), the router's
// renormalised weights 1e-2 reported (the engine's router weight is BF16 by contract; the picks must be identical).
// Exit 77 without a CUDA device or when the fixtures were not written (no numpy), 1 on a failure.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/model_geometry.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::BlockBuffers;
using strata::core::ModelGeometry;
using strata::core::MoEBuffers;
using strata::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-104s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof b, "%.2e", v); return b; }
#define CUDA_OK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-qwen36-block-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

// ---- the oracle's dumps: <stem>.f32 in C order of the numpy array, shapes from manifest.txt (`stem kind ne0,ne1,..`)
struct Dumps {
    fs::path dir;
    std::map<std::string, std::vector<int64_t>> ne;   // GGUF order: ne[0] fastest
    bool open(const fs::path& d) {
        dir = d;
        std::ifstream m(d / "manifest.txt");
        if (!m) return false;
        std::string line;
        while (std::getline(m, line)) {
            std::istringstream is(line);
            std::string stem, kind, dims;
            if (!(is >> stem >> kind >> dims)) continue;
            std::vector<int64_t> v;
            std::istringstream ds(dims);
            std::string tok;
            while (std::getline(ds, tok, ',')) v.push_back(std::stoll(tok));
            ne[stem] = v;
        }
        return !ne.empty();
    }
    bool has(const std::string& stem) const { return ne.count(stem) != 0; }
    template <class T> std::vector<T> read(const std::string& stem) const {
        std::ifstream f(dir / (stem + ".f32"), std::ios::binary);
        if (!f) { std::fprintf(stderr, "qwen36_block_test: missing dump %s\n", stem.c_str()); std::exit(1); }
        f.seekg(0, std::ios::end);
        const std::streamoff n = f.tellg();
        f.seekg(0);
        std::vector<T> v((size_t) n / sizeof(T));
        f.read((char*) v.data(), n);
        return v;
    }
    std::vector<float> f(const std::string& stem) const { return read<float>(stem); }
    /// the "i32" dumps (`ffn_moe_topk-N`) are written as float32 like every other (the oracle's Dump casts to F32)
    std::vector<int32_t> i(const std::string& stem) const {
        const std::vector<float> v = read<float>(stem);
        std::vector<int32_t> r(v.size());
        for (size_t k = 0; k < v.size(); ++k) r[k] = (int32_t) std::lround(v[k]);
        return r;
    }
};

// ---- metrics (NaN-safe: a non-finite value is a failure, not a small number)
double rel_l2(const float* a, const float* b, size_t n) {
    double d = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        d += (double) (a[i] - b[i]) * (a[i] - b[i]);
        m += (double) b[i] * b[i];
    }
    return m > 0 ? std::sqrt(d / m) : (d > 0 ? INFINITY : 0.0);
}
double rel_l1(const float* a, const float* b, size_t n) {
    double d = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        d += std::fabs((double) a[i] - b[i]);
        m += std::fabs((double) b[i]);
    }
    return m > 0 ? d / m : (d > 0 ? INFINITY : 0.0);
}

// ---- device helpers
float* dev(size_t n) { float* p = nullptr; CUDA_OK(cudaMalloc(&p, n * 4 + 256)); CUDA_OK(cudaMemset(p, 0, n * 4 + 256)); return p; }
void up(float* d, const float* h, size_t n) { CUDA_OK(cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice)); }
std::vector<float> down(const float* d, size_t n) { std::vector<float> v(n); CUDA_OK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }
std::vector<int32_t> down_i(const int32_t* d, size_t n) { std::vector<int32_t> v(n); CUDA_OK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }

uint16_t bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// ---- the pack the test writes: the tensors the plain block and the router read, in the engine's forms
//      (index row: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook
//       has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind; kind 2 = F32, kind 4 = raw BF16)
struct PackRow { std::string name; int kind; std::vector<uint8_t> bytes; int64_t ne0, ne1; };
void write_pack(const fs::path& dir, const std::vector<PackRow>& rows) {
    fs::create_directories(dir);
    std::vector<uint8_t> blob;
    std::string idx;
    uint64_t at = 0;
    for (const PackRow& r : rows) {
        at = (at + 255) & ~255ull;
        blob.resize(at + r.bytes.size());
        std::memcpy(blob.data() + at, r.bytes.data(), r.bytes.size());
        idx += r.name + " 0 " + std::to_string(r.kind) + " " + std::to_string(at) + " " + std::to_string(r.bytes.size()) +
               " " + std::to_string(at) + " " + std::to_string(r.bytes.size()) + " " + std::to_string(r.ne0) + " " +
               std::to_string(r.ne1) + " 0 0 1 0 0 0 0 0 0 0\n";
        at += r.bytes.size();
    }
    at = (at + 255) & ~255ull;
    std::ofstream(dir / "index.txt", std::ios::binary)
        << "# align 256 pool " << at << " tensors " << rows.size() << "\n" << idx;
    std::ofstream(dir / "dense.bin", std::ios::binary).write((const char*) blob.data(), (std::streamsize) blob.size());
}
PackRow f32_row(const std::string& name, const float* p, int64_t ne0, int64_t ne1) {
    PackRow r{name, 2, {}, ne0, ne1};
    r.bytes.resize((size_t) ne0 * (ne1 > 0 ? ne1 : 1) * 4);
    std::memcpy(r.bytes.data(), p, r.bytes.size());
    return r;
}
PackRow bf16_row(const std::string& name, const float* p, int64_t ne0, int64_t ne1) {
    PackRow r{name, 4, {}, ne0, ne1};
    const size_t n = (size_t) ne0 * (ne1 > 0 ? ne1 : 1);
    r.bytes.resize(n * 2);
    for (size_t i = 0; i < n; ++i) { const uint16_t h = bf16_rne(p[i]); std::memcpy(r.bytes.data() + 2 * i, &h, 2); }
    return r;
}

const float* tensor_f32(const strata::GgufFile& f, const std::string& name) {
    const strata::TensorInfo* t = f.find(name);
    if (t == nullptr || t->type != 0) { std::fprintf(stderr, "qwen36_block_test: %s is not an F32 tensor of the fixture\n", name.c_str()); std::exit(1); }
    return (const float*) f.tensor_data(*t);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: qwen36_block_test <synth dir> <dump dir>\n"); return 2; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("qwen36_block_test: no CUDA device, skipped\n");
        return 77;
    }
    const fs::path synth = argv[1], dumpdir = argv[2];
    if (fs::exists(synth / "SKIP") || fs::exists(dumpdir / "SKIP") || !fs::exists(dumpdir / "manifest.txt")) {
        std::printf("qwen36_block_test: the fixtures were not written (no numpy / gguf-py), skipped\n");
        return 77;
    }
    const std::string gguf = (synth / "toy.gguf").string();
    ModelGeometry g;
    std::string err;
    if (!strata::core::read_model_geometry(gguf, g, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Dumps d;
    if (!d.open(dumpdir)) { std::fprintf(stderr, "qwen36_block_test: cannot read %s/manifest.txt\n", argv[2]); return 1; }
    const strata::GgufFile file(gguf);
    const int64_t N = g.n_embd, K = g.n_expert_used, L = g.n_layers, E = g.n_expert;
    // the vocabulary from the head tensor itself: `read_model_geometry` leaves `n_vocab` at its default (the engine
    // takes it from `output.weight`'s ne1 as well, verify.cpp)
    const strata::TensorInfo* wo_info = file.find("output.weight");
    if (wo_info == nullptr || wo_info->shape.size() != 2) { std::fprintf(stderr, "qwen36_block_test: no 2-D output.weight\n"); return 1; }
    const int64_t V = (int64_t) wo_info->shape[1];
    const std::vector<float> x0 = d.f("model.input_embed");
    const int T = (int) (x0.size() / (size_t) N);
    std::printf("qwen36_block_test: %s (hc %lld, n_embd %lld, %lld layers, %lld experts top-%lld, eps %g), %d tokens\n",
                gguf.c_str(), (long long) g.hc, (long long) N, (long long) L, (long long) E, (long long) K, g.rms_eps, T);
    check(g.hc == 1 && g.arch == strata::core::ModelArch::Qwen35Moe, "the fixture is qwen35moe at hc = 1");
    check(T >= 2 && T <= strata::kernels::kVerifyMaxT, "the dump holds a verify window's worth of tokens (2..8)");
    const float eps = (float) g.rms_eps;
    cudaStream_t cs = nullptr;
    CUDA_OK(cudaStreamCreate(&cs));

    // ---- the pack: per layer the two norms (F32) and the router (BF16, the engine's contract), plus the shared
    //      expert's scalar gate (BF16) and the head norm; the mixer and expert projections are NOT in it (see the header)
    TempDir tmp;
    std::vector<PackRow> rows;
    for (int64_t l = 0; l < L; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        rows.push_back(f32_row(p + "attn_norm.weight", tensor_f32(file, p + "attn_norm.weight"), N, 0));
        rows.push_back(f32_row(p + "post_attention_norm.weight", tensor_f32(file, p + "post_attention_norm.weight"), N, 0));
        rows.push_back(bf16_row(p + "ffn_gate_inp.weight", tensor_f32(file, p + "ffn_gate_inp.weight"), N, E));
        rows.push_back(bf16_row(p + "ffn_gate_inp_shexp.weight", tensor_f32(file, p + "ffn_gate_inp_shexp.weight"), N, 0));
    }
    rows.push_back(f32_row("output_norm.weight", tensor_f32(file, "output_norm.weight"), N, 0));
    write_pack(tmp.path / "pack", rows);
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes((tmp.path / "pack").string(), pool, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    void* arena = nullptr;
    CUDA_OK(cudaMalloc(&arena, pool));
    WeightTable wt;
    if (!wt.load((tmp.path / "pack").string(), arena, pool, err)) { std::fprintf(stderr, "pack load: %s\n", err.c_str()); return 1; }
    check(wt.find("blk.0.attn_norm.weight") != nullptr && wt.find("blk.0.attn_norm.weight")->kind == strata::core::WeightKind::F32,
          "the pack loads: blk.0.attn_norm.weight is F32");

    // ---- the engine's buffers, as a session carves them
    void* block_arena = nullptr;
    CUDA_OK(cudaMalloc(&block_arena, strata::core::block_buffers_bytes(g)));
    BlockBuffers bb;
    strata::core::block_buffers_init(g, block_arena, bb);
    void* moe_arena = nullptr;
    CUDA_OK(cudaMalloc(&moe_arena, strata::core::moe_buffers_bytes(g, K)));
    MoEBuffers mb;
    strata::core::moe_buffers_init(g, K, moe_arena, mb);
    float* parts = dev((size_t) K * N);
    strata::core::GdnBuffers gb;        // empty: the mixers are not run (reported), see the header
    strata::core::QsaState qst;
    strata::core::QsaBuffers qb;
    strata::core::layer_set_native_bf16(true);   // the decode configuration the verify window reproduces

    auto sync = [&] { CUDA_OK(cudaStreamSynchronize(cs)); };
    const char* mixer_seam[2] = {"linear_attn_out", "attn_output"};

    // ================================ 1. the single-token path through the engine ================================
    std::printf("1. the single-token path: block_layer_pre / block_layer_post / lm_head_mix per layer and token\n");
    double worst_norm = 0, worst_post = 0, worst_res = 0, worst_ffn = 0, worst_lout = 0, worst_w = 0;
    int flips = 0;
    for (int64_t l = 0; l < L; ++l) {
        const std::string sl = "-" + std::to_string(l);
        const std::vector<float> x_in = l == 0 ? x0 : d.f("l_out-" + std::to_string(l - 1));
        const std::vector<float> an = d.f("attn_norm" + sl), mix = d.f(std::string(mixer_seam[strata::core::is_qsa_layer(g, l)]) + sl),
                                 res = d.f("attn_residual" + sl), pn = d.f("attn_post_norm" + sl),
                                 wn = d.f("ffn_moe_weights_norm" + sl), downs = d.f("ffn_moe_down" + sl),
                                 shg = d.f("ffn_shexp_gated" + sl), ffn = d.f("ffn_out" + sl), lo = d.f("l_out" + sl);
        const std::vector<int32_t> topk = d.i("ffn_moe_topk" + sl);
        for (int t = 0; t < T; ++t) {
            // stage 0: x = rms_norm(R) * attn_norm
            up(bb.R, x_in.data() + (size_t) t * N, (size_t) N);
            if (!strata::core::block_layer_pre(wt, g, l, t, 0, gb, qst, qb, mb, K, bb, cs, err, nullptr, nullptr, 0, 1)) {
                check(false, "block_layer_pre stage 0: " + err); continue;
            }
            sync();
            worst_norm = std::max(worst_norm, rel_l2(down(bb.mixed, (size_t) N).data(), an.data() + (size_t) t * N, (size_t) N));
            // the residual add with the mixer's output teacher-forced
            up(bb.block_out, mix.data() + (size_t) t * N, (size_t) N);
            strata::core::plain_residual_rows(bb.R, bb.block_out, N, 1, cs);
            sync();
            worst_res = std::max(worst_res, rel_l2(down(bb.R, (size_t) N).data(), res.data() + (size_t) t * N, (size_t) N));
            // half 2: x = rms_norm(R) * post_attention_norm, then the router (no shared expert: its weights are not here)
            up(bb.R, res.data() + (size_t) t * N, (size_t) N);
            strata::core::layer_set_shared_early(false);
            if (!strata::core::block_layer_pre(wt, g, l, t, 0, gb, qst, qb, mb, K, bb, cs, err, nullptr, nullptr, 2, 0)) {
                check(false, "block_layer_pre half 2: " + err); continue;
            }
            sync();
            worst_post = std::max(worst_post, rel_l2(down(bb.mixed, (size_t) N).data(), pn.data() + (size_t) t * N, (size_t) N));
            const std::vector<int32_t> ids = down_i(mb.ids, (size_t) K);
            for (int64_t j = 0; j < K; ++j) flips += ids[(size_t) j] != topk[(size_t) t * K + j];
            worst_w = std::max(worst_w, rel_l2(down(mb.weights, (size_t) K).data(), wn.data() + (size_t) t * K, (size_t) K));
            // post: combine the oracle's per-expert outputs with its renormalised weights and shared expert, R += out
            up(mb.weights, wn.data() + (size_t) t * K, (size_t) K);
            up(mb.shared, shg.data() + (size_t) t * N, (size_t) N);
            up(parts, downs.data() + (size_t) t * K * N, (size_t) K * N);
            strata::core::layer_set_shared_early(true);
            if (!strata::core::block_layer_post(wt, g, l, K, mb, bb, parts, cs, err)) { check(false, "block_layer_post: " + err); continue; }
            sync();
            worst_ffn = std::max(worst_ffn, rel_l2(down(bb.block_out, (size_t) N).data(), ffn.data() + (size_t) t * N, (size_t) N));
            worst_lout = std::max(worst_lout, rel_l2(down(bb.R, (size_t) N).data(), lo.data() + (size_t) t * N, (size_t) N));
        }
        // the mixer: reported, not asserted (the attempt's own error says why)
        up(bb.R, x_in.data(), (size_t) N);
        const bool ran = strata::core::block_layer_pre(wt, g, l, 0, 0, gb, qst, qb, mb, K, bb, cs, err, nullptr, nullptr, 0, 2);
        sync();
        std::printf("  layer %lld mixer seam %s: %s\n", (long long) l, mixer_seam[strata::core::is_qsa_layer(g, l)],
                    ran ? ("ran: " + fmt(rel_l2(down(bb.block_out, (size_t) N).data(), mix.data(), (size_t) N)) + " relative L2 (reported)").c_str()
                        : ("not run (" + err + ")").c_str());
    }
    check(worst_norm <= 1e-6, "attn_norm-N, all layers and tokens: " + fmt(worst_norm) + " relative L2 (tolerance 1e-6)");
    check(worst_res <= 1e-6, "attn_residual-N (mixer out forced): " + fmt(worst_res) + " (1e-6)");
    check(worst_post <= 1e-6, "attn_post_norm-N through block_layer_pre half 2: " + fmt(worst_post) + " (1e-6)");
    check(flips == 0, "ffn_moe_topk-N: the engine's BF16 router picks the oracle's experts (" + std::to_string(flips) + " of " +
                          std::to_string(L * T * K) + " slots differ)");
    std::printf("  ffn_moe_weights_norm-N from the BF16 router: %s relative L2 (reported; the contract rounds the weight)\n", fmt(worst_w).c_str());
    check(worst_w <= 1e-2, "ffn_moe_weights_norm-N within 1e-2 of the F32 oracle");
    check(worst_ffn <= 1e-6, "ffn_out-N through block_layer_post (experts forced): " + fmt(worst_ffn) + " (1e-6)");
    check(worst_lout <= 1e-6, "l_out-N through block_layer_post: " + fmt(worst_lout) + " (1e-6)");

    // the head: result_norm through lm_head_mix, the logits through a host projection of the engine's vector
    {
        const std::vector<float> lo = d.f("l_out-" + std::to_string(L - 1)), rn = d.f("result_norm"), ro = d.f("result_output");
        up(bb.R, lo.data() + (size_t) (T - 1) * N, (size_t) N);
        if (!strata::core::lm_head_mix(wt, g, bb, cs, err)) check(false, "lm_head_mix: " + err);
        sync();
        const std::vector<float> h = down(bb.mixed, (size_t) N);
        const double e_rn = rel_l2(h.data(), rn.data(), (size_t) N);
        check(e_rn <= 1e-6, "result_norm through lm_head_mix (output_norm): " + fmt(e_rn) + " (1e-6)");
        const float* wo = tensor_f32(file, "output.weight");   // ne [n_embd, n_vocab]: row v contiguous
        std::vector<float> logits((size_t) V);
        for (int64_t v = 0; v < V; ++v) {
            double acc = 0;
            for (int64_t i = 0; i < N; ++i) acc += (double) h[(size_t) i] * wo[(size_t) v * N + i];
            logits[(size_t) v] = (float) acc;
        }
        const double e_ro = rel_l2(logits.data(), ro.data(), (size_t) V);
        check(e_ro <= 1e-5, "result_output (engine's result_norm x output.weight on the host): " + fmt(e_ro) + " (1e-5)");
    }

    // ================================ 2. the verify window's row arithmetic ================================
    std::printf("2. the verify window: the T-row calls bitwise the one-row calls, the 8-token sequence per position\n");
    {
        float* R = dev((size_t) T * N);
        float* mixed = dev((size_t) T * N);
        float* one = dev((size_t) N);
        float* bo = dev((size_t) T * N);
        float* partsT = dev((size_t) T * K * N);
        float* wT = dev((size_t) T * K);
        float* shT = dev((size_t) T * N);
        float* boT = dev((size_t) T * N);
        bool bit_norm = true, bit_add = true, bit_comb = true;
        double w_norm = 0, w_res = 0, w_post = 0, w_ffn = 0, w_lout = 0;
        up(R, x0.data(), (size_t) T * N);   // the window's R, carried through the layers from the embeddings
        for (int64_t l = 0; l < L; ++l) {
            const std::string sl = "-" + std::to_string(l), p = "blk." + std::to_string(l) + ".";
            const float* wp_attn = (const float*) wt.find(p + "attn_norm.weight")->data;
            const float* wp_post = (const float*) wt.find(p + "post_attention_norm.weight")->data;
            const std::vector<float> an = d.f("attn_norm" + sl), mix = d.f(std::string(mixer_seam[strata::core::is_qsa_layer(g, l)]) + sl),
                                     res = d.f("attn_residual" + sl), pn = d.f("attn_post_norm" + sl),
                                     wn = d.f("ffn_moe_weights_norm" + sl), downs = d.f("ffn_moe_down" + sl),
                                     shg = d.f("ffn_shexp_gated" + sl), ffn = d.f("ffn_out" + sl), lo = d.f("l_out" + sl);
            const std::vector<float> R_in = down(R, (size_t) T * N);
            // the T-row norm against T one-row norms, bitwise, and against the oracle
            strata::core::plain_norm_rows(R, wp_attn, mixed, N, eps, T, cs);
            sync();
            std::vector<float> mT = down(mixed, (size_t) T * N);
            for (int t = 0; t < T; ++t) {
                strata::core::plain_norm_rows(R + (size_t) t * N, wp_attn, one, N, eps, 1, cs);
                sync();
                const std::vector<float> m1 = down(one, (size_t) N);
                bit_norm = bit_norm && std::memcmp(m1.data(), mT.data() + (size_t) t * N, (size_t) N * 4) == 0;
            }
            w_norm = std::max(w_norm, rel_l2(mT.data(), an.data(), (size_t) T * N));
            // R += mixer out (forced), T rows at once against one row at a time
            up(bo, mix.data(), (size_t) T * N);
            strata::core::plain_residual_rows(R, bo, N, T, cs);
            sync();
            const std::vector<float> rT = down(R, (size_t) T * N);
            for (int t = 0; t < T; ++t) {
                up(one, R_in.data() + (size_t) t * N, (size_t) N);
                strata::core::plain_residual_rows(one, bo + (size_t) t * N, N, 1, cs);
                sync();
                const std::vector<float> r1 = down(one, (size_t) N);
                bit_add = bit_add && std::memcmp(r1.data(), rT.data() + (size_t) t * N, (size_t) N * 4) == 0;
            }
            w_res = std::max(w_res, rel_l2(rT.data(), res.data(), (size_t) T * N));
            strata::core::plain_norm_rows(R, wp_post, mixed, N, eps, T, cs);
            sync();
            w_post = std::max(w_post, rel_l2(down(mixed, (size_t) T * N).data(), pn.data(), (size_t) T * N));
            // the combine over T rows (the window's launch) against the one-row combine, and the oracle
            up(partsT, downs.data(), (size_t) T * K * N);
            up(wT, wn.data(), (size_t) T * K);
            up(shT, shg.data(), (size_t) T * N);
            strata::kernels::native_moe_combine_multi(partsT, wT, shT, boT, N, K, T, cs);
            sync();
            const std::vector<float> cT = down(boT, (size_t) T * N);
            for (int t = 0; t < T; ++t) {
                strata::kernels::native_moe_combine(partsT + (size_t) t * K * N, wT + (size_t) t * K, shT + (size_t) t * N, one, N, K, cs);
                sync();
                const std::vector<float> c1 = down(one, (size_t) N);
                bit_comb = bit_comb && std::memcmp(c1.data(), cT.data() + (size_t) t * N, (size_t) N * 4) == 0;
            }
            w_ffn = std::max(w_ffn, rel_l2(cT.data(), ffn.data(), (size_t) T * N));
            strata::core::plain_residual_rows(R, boT, N, T, cs);
            sync();
            const double e_l = rel_l2(down(R, (size_t) T * N).data(), lo.data(), (size_t) T * N);
            w_lout = std::max(w_lout, e_l);
            std::printf("  layer %lld: l_out%s over %d positions %s relative L2 (R carried from the embeddings)\n",
                        (long long) l, sl.c_str(), T, fmt(e_l).c_str());
        }
        check(bit_norm, "plain_norm_rows over T rows is bitwise the one-row call, every row and layer");
        check(bit_add, "plain_residual_rows over T rows is bitwise the one-row call");
        check(bit_comb, "native_moe_combine_multi over T rows is bitwise native_moe_combine per row");
        check(w_norm <= 1e-6, "window attn_norm-N, all positions: " + fmt(w_norm) + " (1e-6)");
        check(w_res <= 1e-6, "window attn_residual-N: " + fmt(w_res) + " (1e-6)");
        check(w_post <= 1e-6, "window attn_post_norm-N: " + fmt(w_post) + " (1e-6)");
        check(w_ffn <= 1e-6, "window ffn_out-N: " + fmt(w_ffn) + " (1e-6)");
        check(w_lout <= 1e-6, "window l_out-N per position, R carried through the layers: " + fmt(w_lout) + " (1e-6)");
        // the head over the window's rows: h_nextn is every token's output-normed state, result_norm the last one's
        const std::vector<float> hn = d.f("h_nextn"), rn = d.f("result_norm");
        strata::core::plain_norm_rows(R, (const float*) wt.find("output_norm.weight")->data, mixed, N, eps, T, cs);
        sync();
        const std::vector<float> hT = down(mixed, (size_t) T * N);
        const double e_h = rel_l2(hT.data(), hn.data(), (size_t) T * N);
        const double e_r = rel_l2(hT.data() + (size_t) (T - 1) * N, rn.data(), (size_t) N);
        check(e_h <= 1e-6, "window h_nextn (output_norm over the T rows): " + fmt(e_h) + " (1e-6)");
        check(e_r <= 1e-6, "window result_norm (row T-1): " + fmt(e_r) + " (1e-6)");
        for (float* p : {R, mixed, one, bo, partsT, wT, shT, boT}) cudaFree(p);
    }

    // ================================ 3. the negative cases ================================
    std::printf("3. negative cases (each wrong reading must differ from the oracle by > 5%% relative L1)\n");
    {
        // the GR read at hc = 1 with identity bottleneck weights: NOT the plain norm.  `gr_read`'s norm weight is
        // stored as (1 + w) and its bottleneck gates the normed stream with a sigmoid, so one stream through the
        // hyper-connection kernels is a different function - the reason the plain block is a second path.
        using strata::kernels::GrShapes;
        const GrShapes gs{N, 1, N};   // hc_lr = n_embd, so the down / up projections can be identities
        strata::kernels::GrWorkspace ws;
        const size_t wb = strata::kernels::gr_workspace_bytes(gs);
        void* wsm = nullptr;
        CUDA_OK(cudaMalloc(&wsm, wb));
        strata::kernels::gr_workspace_init(gs, wsm, ws);
        std::vector<uint16_t> ident((size_t) N * N, 0);
        for (int64_t i = 0; i < N; ++i) ident[(size_t) i * N + i] = bf16_rne(1.0f);
        uint16_t* d_ident = nullptr;
        CUDA_OK(cudaMalloc(&d_ident, ident.size() * 2));
        CUDA_OK(cudaMemcpy(d_ident, ident.data(), ident.size() * 2, cudaMemcpyHostToDevice));
        const std::vector<float> an = d.f("attn_norm-0");
        up(bb.R, x0.data(), (size_t) N);
        strata::kernels::gr_read(bb.R, (const float*) wt.find("blk.0.attn_norm.weight")->data, d_ident, d_ident, nullptr,
                                 eps, gs, ws, bb.mixed, bb.inject, cs);
        sync();
        const double e_gr = rel_l1(down(bb.mixed, (size_t) N).data(), an.data(), (size_t) N);
        check(e_gr > 0.05, "the GR read at hc = 1 with identity bottleneck weights is not attn_norm-0: " + fmt(e_gr) + " relative L1");
        cudaFree(wsm);
        cudaFree(d_ident);
        // the post-attention norm skipped: the residual itself fed to the MoE; and the norm without its weight
        const std::vector<float> res = d.f("attn_residual-0"), pn = d.f("attn_post_norm-0");
        const double e_skip = rel_l1(res.data(), pn.data(), (size_t) T * N);
        check(e_skip > 0.05, "post_attention_norm skipped (attn_residual-0 as the MoE input): " + fmt(e_skip) + " relative L1");
        std::vector<float> ones((size_t) N, 1.0f);
        float* d_ones = dev((size_t) N);
        up(d_ones, ones.data(), (size_t) N);
        float* rT = dev((size_t) T * N);
        float* mT = dev((size_t) T * N);
        up(rT, res.data(), (size_t) T * N);
        strata::core::plain_norm_rows(rT, d_ones, mT, N, eps, T, cs);
        sync();
        const double e_unw = rel_l1(down(mT, (size_t) T * N).data(), pn.data(), (size_t) T * N);
        check(e_unw > 0.05, "post_attention_norm without its weight: " + fmt(e_unw) + " relative L1");
        cudaFree(d_ones); cudaFree(rT); cudaFree(mT);
        // k = 10 on the 8-expert file: refused by name; the file's own k accepted.  The fixture's GDN state is 16, which
        // the T-token GDN kernels (state 128) refuse on their own, so the k checks run on the geometry at state 128.
        std::string why;
        const bool r16 = strata::core::verify_geometry_supported(g, K, why);
        check(!r16 && why.find("state 128") != std::string::npos,
              "verify_geometry_supported refuses the fixture's GDN state 16 by name: " + why);
        ModelGeometry g128 = g;
        g128.ssm_state_size = 128;
        g128.derive();
        why.clear();
        const bool r10 = strata::core::verify_geometry_supported(g128, 10, why);
        check(!r10 && why.find("top-10") != std::string::npos && why.find("top-" + std::to_string(K)) != std::string::npos,
              "verify_geometry_supported refuses k = 10 on the file's top-" + std::to_string(K) + ": " + why);
        why.clear();
        const bool rk = strata::core::verify_geometry_supported(g128, K, why);
        check(rk, "verify_geometry_supported accepts the file's k " + std::to_string(K) + (rk ? "" : ": " + why));
        why.clear();
        check(!strata::core::verify_geometry_supported(g128, 0, why), "verify_geometry_supported refuses k = 0");
        // the plain block refuses a layer whose norm is not in the pack, naming it
        err.clear();
        const bool missing = strata::core::block_layer_pre(wt, g, L, 0, 0, gb, qst, qb, mb, K, bb, cs, err, nullptr, nullptr, 0, 1);
        check(!missing && err.find("blk." + std::to_string(L) + ".attn_norm.weight") != std::string::npos,
              "a missing attn_norm is refused by name: " + err);
        // the verify compatibility at hc = 1 does not demand the fused GR read or the indexer
        strata::core::layer_set_fused_gr(false);
        strata::kernels::native_gdn_set_enabled(true);   // the default decode path's GDN kernels (generate.cpp turns them on)
        why.clear();
        const bool compat = strata::core::layer_verify_compatible(g, why);
        check(compat, "layer_verify_compatible at hc = 1 without the fused GR read: " + (compat ? std::string("ok") : why));
        ModelGeometry fn;   // Flash-Next's defaults: the fused read is still required there
        check(!strata::core::layer_verify_compatible(fn, why) && why.find("hyper-connection") != std::string::npos,
              "layer_verify_compatible at hc = 4 still requires the fused GR read: " + why);
    }

    cudaFree(parts); cudaFree(moe_arena); cudaFree(block_arena); cudaFree(arena);
    cudaStreamDestroy(cs);
    std::printf(g_fail ? "qwen36_block_test: %d FAILED\n" : "qwen36_block_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
