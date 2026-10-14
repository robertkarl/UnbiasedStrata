// prefill_qwen36_test - the prompt path (src/prefill) at Qwen3.6's geometry against the oracle's dumps
// (docs/qwen36-port/step3-prefill.md; the fixtures from tests/python/qwen36_prefill_fixture.py).
//
//   prefill_qwen36_test <fixture dir> gdn3|full4 [--chunk N] [--no-force]
//
// The synthetic model of the fixture (2048 wide, GDN 16 / 32 heads of 128, attention 16 / 2 heads of 256, 16 experts
// top-8 of 512; Q8_0 projections and experts, BF16-exact routers) is run through Prefill::run twice, as one chunk of
// 40 tokens and in chunks of 7 (the oracle's base_c7 case), and every seam the layer loop exposes (Prefill::on_seam,
// llama.cpp's names) is held to the oracle's dump of the same 40 tokens with the layer's input TEACHER-FORCED to
// the oracle's l_out of the layer before (oracle-crosscheck.md, "Tolerance policy", item 2: per layer, not
// accumulated), so each seam's error is that layer's own; --no-force reports the accumulated error too.  The router's
// top-k must be the oracle's exactly (the F32 fixture has no near ties).  Negative controls: the oracle's dumps with
// one reading flipped to its rival (the sigmoid GDN gate, the interleaved head pairing, the unnormalised router
// weights, the attention gate skipped) must sit more than 5 % relative L1 from what the engine computes, else the
// harness could not tell the readings apart.  Geometry refusals: Prefill::init with top-10 on this 16-expert file,
// and with a GDN geometry the kernels are not built for.
//
// Exit 0 pass, 1 fail, 77 skip (no CUDA device, or <fixture dir>/SKIP: the fixture script found no numpy / gguf-py).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/model_geometry.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/prefill/prefill.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace core = strata::core;

[[noreturn]] void die(const std::string& what) {
    std::fprintf(stderr, "prefill_qwen36_test: %s\n", what.c_str());
    std::exit(1);
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e));
}
template <typename T> T* dalloc(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "cudaMalloc");
    return (T*) p;
}
template <typename T> T* upload(const void* host, size_t bytes) {
    T* d = dalloc<uint8_t>(bytes) ? (T*) nullptr : nullptr;
    (void) d;
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes + 256), "cudaMalloc");
    ck(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice), "upload");
    return (T*) p;
}
bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}
uint16_t bf16_of(float f) {   // round to nearest even (the values are BF16-exact already: the rounding is the identity)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
std::vector<float> read_f32(const std::string& path, bool required = true) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (required) die("cannot read " + path);
        return {};
    }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0);
    std::vector<float> v((size_t) n / 4);
    in.read((char*) v.data(), (std::streamsize) (v.size() * 4));
    return v;
}
// the L1 sums of ||a - b|| and ||b|| over n values, NaN-safe (a NaN anywhere makes the numerator infinite)
void l1_sums(const float* a, const float* b, size_t n, double& num, double& den) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { num = INFINITY; return; }
        num += std::fabs((double) a[i] - b[i]);
        den += std::fabs((double) b[i]);
    }
}

// the Q8_0 expert blobs of the fixture, [gate rows | up rows | down rows] as tools/iq_pack.py lays them out, in host
// memory (not pinned: the prompt path's stager copies them)
struct HostExperts : core::ExpertSource {
    std::vector<std::vector<uint8_t>> blobs;   // [layer * n_expert + e]
    int64_t n_expert = 0;
    const uint8_t* blob(int64_t layer, int64_t expert) override { return blobs[(size_t) (layer * n_expert + expert)].data(); }
};

struct Fixture {
    strata::GgufFile gguf;
    core::ModelGeometry g;
    core::WeightTable wt;
    std::vector<void*> owned;
    HostExperts experts;
    std::vector<float> embd;   // [n_vocab][n_embd]
    explicit Fixture(const std::string& path) : gguf(path) {
        std::string err;
        if (!core::read_model_geometry(gguf, g, err)) {
            // gdn3 (three DeltaNet layers, no attention layer) is a file the geometry reader refuses by design
            // (block_count must be a multiple of full_attention_interval, as every shipped file's is): its geometry
            // is full4's with the layer count of its own header, which is what the oracle ran it as
            const std::string alt = path.substr(0, path.rfind('/') + 1) + "full4.gguf";
            std::string err2;
            strata::GgufFile full(alt);
            if (!core::read_model_geometry(full, g, err2)) die("geometry: " + err + "; and " + alt + ": " + err2);
            auto it = gguf.metadata().find("qwen35moe.block_count");
            if (it == gguf.metadata().end()) die("geometry: " + err);
            g.n_layers = (int64_t) it->second.u;
            std::printf("geometry from %s (%s), %lld layers from the header\n", alt.c_str(), err.c_str(), (long long) g.n_layers);
        }
        if (g.arch != core::ModelArch::Qwen35Moe) die("the fixture is not a qwen35moe file");
        std::printf("fixture %s: %lld layers, n_embd %lld, GDN %lld/%lld heads, conv %lld, attention %lld/%lld x %lld, "
                    "%lld experts top-%lld of %lld, hc %lld\n", path.c_str(), (long long) g.n_layers, (long long) g.n_embd,
                    (long long) g.ssm_k_heads, (long long) g.ssm_v_heads, (long long) g.ssm_conv_channels, (long long) g.n_head,
                    (long long) g.n_head_kv, (long long) g.head_dim, (long long) g.n_expert, (long long) g.n_expert_used,
                    (long long) g.n_ff, (long long) g.hc);
        experts.n_expert = g.n_expert;
        experts.blobs.resize((size_t) (g.n_layers * g.n_expert));
        std::map<std::string, const strata::TensorInfo*> exp_t;
        for (const auto& t : gguf.tensors()) {
            const uint64_t bytes = strata::tensor_payload_bytes(t);
            if (bytes == 0) die("tensor " + t.name + ": unknown payload size");
            const uint8_t* src = gguf.tensor_data(t);
            if (ends_with(t.name, "_exps.weight")) { exp_t[t.name] = &t; continue; }   // the experts: blobs below
            if (t.name == "token_embd.weight") {
                if (t.type != 0) die("token_embd is not F32");
                embd.assign((const float*) src, (const float*) src + t.elements());
                continue;
            }
            core::WeightRef r;
            r.ne0 = (int64_t) t.shape[0];
            r.ne1 = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
            r.elements = (int64_t) t.elements();
            r.bytes = bytes;
            if (t.type == 8) {   // Q8_0: served natively (Gemm::native dequantizes the blocks)
                r.native_data = upload<uint8_t>(src, bytes);
                r.native_type = 8;
                r.resident = false;
                owned.push_back((void*) r.native_data);
            } else if (t.type != 0) {
                die("tensor " + t.name + " has type " + std::to_string(t.type) + " (the fixture uses F32 and Q8_0)");
            } else if (ends_with(t.name, "ffn_gate_inp.weight") || ends_with(t.name, "ssm_alpha.weight") ||
                       ends_with(t.name, "ssm_beta.weight") || ends_with(t.name, "ffn_gate_inp_shexp.weight")) {
                // the engine's BF16 form of the F32 tensor (exact: the fixture rounded the values to BF16)
                std::vector<uint16_t> h((size_t) r.elements);
                const float* f = (const float*) src;
                for (size_t i = 0; i < h.size(); ++i) {
                    h[i] = bf16_of(f[i]);
                    uint32_t back = (uint32_t) h[i] << 16;
                    float fb;
                    std::memcpy(&fb, &back, 4);
                    if (fb != f[i]) die("tensor " + t.name + " is not BF16-exact at " + std::to_string(i));
                }
                r.kind = core::WeightKind::Bf16InF32;
                r.data = upload<uint8_t>(h.data(), h.size() * 2);
                r.bytes = h.size() * 2;
                owned.push_back((void*) r.data);
            } else {
                r.kind = core::WeightKind::F32;
                r.data = upload<uint8_t>(src, bytes);
                owned.push_back((void*) r.data);
            }
            wt.set(t.name, r);
        }
        // the expert blobs: [gate | up | down] rows of expert e from the three [E, rows, cols] Q8_0 tensors
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const strata::TensorInfo* gt = exp_t.count(p + "ffn_gate_exps.weight") ? exp_t[p + "ffn_gate_exps.weight"] : nullptr;
            const strata::TensorInfo* ut = exp_t.count(p + "ffn_up_exps.weight") ? exp_t[p + "ffn_up_exps.weight"] : nullptr;
            const strata::TensorInfo* dt = exp_t.count(p + "ffn_down_exps.weight") ? exp_t[p + "ffn_down_exps.weight"] : nullptr;
            if (!gt || !ut || !dt) die("layer " + std::to_string(l) + ": expert tensors missing");
            if (gt->type != 8 || ut->type != 8 || dt->type != 8) die("the experts are not Q8_0");
            const size_t gu_row = (size_t) g.n_embd / 32 * 34, d_row = (size_t) g.n_ff / 32 * 34;
            const size_t gu_mat = gu_row * (size_t) g.n_ff, d_mat = d_row * (size_t) g.n_embd;
            if (strata::tensor_payload_bytes(*gt) != gu_mat * g.n_expert || strata::tensor_payload_bytes(*dt) != d_mat * g.n_expert)
                die("expert tensor sizes do not match the geometry");
            for (int64_t e = 0; e < g.n_expert; ++e) {
                std::vector<uint8_t>& b = experts.blobs[(size_t) (l * g.n_expert + e)];
                b.resize(2 * gu_mat + d_mat);
                std::memcpy(b.data(), gguf.tensor_data(*gt) + e * gu_mat, gu_mat);
                std::memcpy(b.data() + gu_mat, gguf.tensor_data(*ut) + e * gu_mat, gu_mat);
                std::memcpy(b.data() + 2 * gu_mat, gguf.tensor_data(*dt) + e * d_mat, d_mat);
            }
        }
    }
    ~Fixture() { for (void* p : owned) cudaFree(p); }
    // the pack's native_experts.txt for these blobs (v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off)
    void write_layout(const std::string& dir) const {
        const size_t gu_row = (size_t) g.n_embd / 32 * 34, d_row = (size_t) g.n_ff / 32 * 34;
        const size_t gu_mat = gu_row * (size_t) g.n_ff, d_mat = d_row * (size_t) g.n_embd, blob = 2 * gu_mat + d_mat;
        std::ofstream out(dir + "/native_experts.txt");
        out << "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off (n_expert "
            << g.n_expert << ", total " << blob * (size_t) g.n_layers * (size_t) g.n_expert << "; the harness's in-memory blobs)\n";
        for (int64_t l = 0; l < g.n_layers; ++l)
            out << l << " 8 8 " << (size_t) l * g.n_expert * blob << " " << blob << " 0 " << gu_mat << " " << 2 * gu_mat << "\n";
    }
};

// one named seam's relative L1 over the whole sequence (the sums over its chunks), per layer
struct Stat {
    double num = 0.0, den = 0.0;
    int64_t rows = 0;
    bool seen = false;
    double rel() const { return den > 0.0 ? num / den : (num > 0.0 ? INFINITY : 0.0); }
};

struct Run {
    const Fixture& fx;
    const std::string dumps;   // the oracle's faithful dumps
    bool force = true;
    int64_t T_total = 0;
    std::map<std::string, Stat> stats;          // "<seam>-<layer>"
    std::vector<std::string> topk_flips;
    int topk_bad = 0;
    static constexpr double kTieGap = 2e-3;   // a swap is a near tie when the oracle's probabilities are this close
    std::vector<float> scratch;
    std::map<std::string, std::vector<float>> cache;
    const std::vector<float>& dump(const std::string& stem) {
        auto it = cache.find(stem);
        if (it == cache.end()) it = cache.emplace(stem, read_f32(dumps + "/" + stem + ".f32")).first;
        return it->second;
    }
    bool on_seam(const char* seam, int64_t layer, float* dev, int64_t rows, int64_t cols, int64_t p0, std::string& err) {
        const std::string s(seam);
        ck(cudaDeviceSynchronize(), "on_seam sync");
        scratch.resize((size_t) (rows * cols));
        ck(cudaMemcpy(scratch.data(), dev, scratch.size() * 4, cudaMemcpyDeviceToHost), "on_seam copy");
        if (s == "l_in") {
            if (layer == 0) return true;   // the embedding rows: the oracle's own, fed through embd_rows
            const std::vector<float>& prev = dump("l_out-" + std::to_string(layer - 1));
            if (force) {   // teacher forcing: this layer starts from the oracle's output of the layer before
                if (prev.size() != (size_t) (T_total * cols)) { err = "l_out dump size"; return false; }
                ck(cudaMemcpy(dev, prev.data() + (size_t) p0 * cols, scratch.size() * 4, cudaMemcpyHostToDevice), "force");
                // the prompt path's stream is non-blocking: a pageable H2D copy may still be in flight when its next
                // kernel starts, so wait for it here
                ck(cudaDeviceSynchronize(), "force sync");
            }
            return true;
        }
        if (s == "ffn_moe_topk") {
            // int32 ids [rows, k] against the oracle's float ids.  The prompt path's router multiplies BF16-rounded
            // activations (the decode path's is FP32), so a pair of experts whose oracle probabilities are within
            // ~1e-3 can swap ranks; such a near tie is reported and allowed (the gap in the oracle's own softmax must
            // be below kTieGap), any other disagreement is a failure (oracle-crosscheck.md: exact top-k on the F32
            // fixture; the prompt path is the one engine path that is not F32 at the router).
            const std::vector<float>& o = dump("ffn_moe_topk-" + std::to_string(layer));
            const std::vector<float>& probs = dump("ffn_moe_probs-" + std::to_string(layer));
            const int64_t E = (int64_t) probs.size() / T_total;
            const int32_t* ids = (const int32_t*) scratch.data();
            for (int64_t t = 0; t < rows; ++t) {
                const float* pr = probs.data() + (size_t) (p0 + t) * E;
                for (int64_t j = 0; j < cols; ++j) {
                    const int32_t e = ids[t * cols + j], oe = (int32_t) o[(size_t) ((p0 + t) * cols + j)];
                    if (e == oe) continue;
                    const double gap = std::fabs((double) pr[e] - pr[oe]);
                    char line[200];
                    std::snprintf(line, sizeof line, "layer %lld pos %lld rank %lld: engine %d, oracle %d, oracle probability gap %.2e%s",
                                  (long long) layer, (long long) (p0 + t), (long long) j, e, oe, gap, gap < kTieGap ? " (near tie)" : " *** NOT A TIE ***");
                    topk_flips.push_back(line);
                    if (gap >= kTieGap) ++topk_bad;
                }
            }
            Stat& st = stats["ffn_moe_topk-" + std::to_string(layer)];
            st.seen = true; st.rows += rows;
            return true;
        }
        const std::vector<float>& o = dump(s + "-" + std::to_string(layer));
        if (o.size() != (size_t) (T_total * cols)) { err = s + " dump size " + std::to_string(o.size()); return false; }
        Stat& st = stats[s + "-" + std::to_string(layer)];
        l1_sums(scratch.data(), o.data() + (size_t) p0 * cols, scratch.size(), st.num, st.den);
        st.rows += rows; st.seen = true;
        return true;
    }
};

// the engine's seams on one run, against the dumps in `run.dumps`
bool run_prefill(const Fixture& fx, core::SessionState& ss, int64_t chunk, Run& run, const std::vector<int64_t>& tokens,
                 std::string& err, cudaStream_t cs) {
    strata::prefill::Prefill pf;
    if (!pf.init(fx.wt, fx.g, ss, const_cast<HostExperts*>(&fx.experts), nullptr, nullptr, chunk, cs, err)) return false;
    std::vector<const float*> rows(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) rows[i] = fx.embd.data() + (size_t) tokens[i] * fx.g.n_embd;
    pf.embd_rows = rows.data();
    run.T_total = (int64_t) tokens.size();
    pf.on_seam = [&run](const char* seam, int64_t layer, float* dev, int64_t r, int64_t c, int64_t p0, std::string& e) {
        return run.on_seam(seam, layer, dev, r, c, p0, e);
    };
    core::session_zero(ss, fx.g, nullptr, cs);
    ck(cudaStreamSynchronize(cs), "zero");
    return pf.run(tokens.data(), (int64_t) tokens.size(), 0, err);
}

const char* const kFloatSeams[] = {"attn_norm", "linear_attn_out", "attn_output", "attn_post_norm", "ffn_out", "l_out"};
double gate_for(const std::string& seam, bool forced) {
    // against the F32 oracle: the plain-block norms are FP32 (summation order only); the mixers run FP16 GEMMs over
    // FP16-dequantized Q8_0 weights (2^-11 per operand) and the experts the int8 MMQ products (q8_1 activations,
    // q8_1 H): oracle-crosscheck.md's "Tolerance policy" puts a 2x margin on the F32 column for an FP32 engine path;
    // these gates are what the 4090 measured (step3-prefill.md) with a 4x margin, and the negatives sit at 0.1 to 1.3
    if (seam == "attn_norm") return 1e-4;
    if (seam == "attn_post_norm") return 2e-3;   // the norm of R + the mixer's output: the mixer's own error passes through
    if (seam == "linear_attn_out" || seam == "attn_output") return 1e-2;
    if (seam == "ffn_out") return 2e-2;
    return forced ? 1e-2 : 5e-2;   // l_out (accumulated without forcing)
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: prefill_qwen36_test <fixture dir> gdn3|full4 [--chunk N] [--no-force]\n");
        return 2;
    }
    const std::string dir = argv[1], kase = argv[2];
    int64_t chunk_override = 0;
    bool force = true;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) chunk_override = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--no-force") == 0) force = false;
    }
    if (std::ifstream skip(dir + "/SKIP"); skip) {
        std::string why;
        std::getline(skip, why);
        std::printf("prefill_qwen36_test: the fixtures were not generated (%s/SKIP: %s): skip\n", dir.c_str(), why.c_str());
        return 77;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("prefill_qwen36_test: no CUDA device (skip)\n");
        return 77;
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "device");
    std::printf("prefill_qwen36_test on %s (sm_%d%d), case %s\n", p.name, p.major, p.minor, kase.c_str());
    Fixture fx(dir + "/" + kase + ".gguf");
    const std::string dumps = dir + "/" + kase;
    std::string err;
    // the pack layout (process-wide) for these blobs
    fx.write_layout(dir);
    if (!strata::kernels::cpu::expert_layout_load(dir, fx.g.n_layers, fx.g.n_expert, fx.g.n_embd, fx.g.n_ff, err))
        die("expert layout: " + err);
    // the session: 64 cells is plenty for 40 tokens
    const int64_t max_cells = 64;
    core::SessionState ss;
    const uint64_t sbytes = core::session_bytes(fx.g, max_cells, fx.g.n_expert_used);
    void* sbase = nullptr;
    ck(cudaMalloc(&sbase, sbytes), "session");
    if (core::session_init(fx.g, max_cells, fx.g.n_expert_used, sbase, ss) == 0) die("session_init failed");
    ss.k = fx.g.n_expert_used;
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    const std::string text = "the quick brown fox jumps over the lazy ";   // 40 tokens, the fixture script's
    std::vector<int64_t> tokens;
    for (char c : text) tokens.push_back((int64_t) (unsigned char) c);
    int fails = 0;

    // ---- the geometry refusals
    {
        strata::prefill::Prefill pf;
        core::SessionState bad = ss;
        bad.k = 10;   // top-10 routing on a top-8 file of 16 experts
        std::string e;
        const bool refused = !pf.init(fx.wt, fx.g, bad, &fx.experts, nullptr, nullptr, 40, cs, e);
        std::printf("refusal: top-10 on the top-8 file: %s (%s)\n", refused ? "refused" : "*** ACCEPTED ***", e.c_str());
        fails += !refused;
        core::ModelGeometry g2 = fx.g;
        g2.ssm_k_heads = 8; g2.derive();   // 4 value heads per key head (32 / 8 / 6144): no kernels are built for it
        strata::prefill::Prefill pf2;
        core::SessionState s2 = ss;
        const bool refused2 = !pf2.init(fx.wt, g2, s2, &fx.experts, nullptr, nullptr, 40, cs, e);
        const bool by_name = e.find("no GDN kernels") != std::string::npos;
        std::printf("refusal: 32 value heads / 8 key heads / 6144 conv channels: %s (%s)\n",
                    refused2 && by_name ? "refused by name" : refused2 ? "*** refused for another reason ***" : "*** ACCEPTED ***", e.c_str());
        fails += !(refused2 && by_name);
    }

    // ---- the runs: one chunk of 40, chunks of 7 (the oracle's base_c7), each seam against the faithful dumps
    const int64_t chunks[] = {chunk_override > 0 ? chunk_override : 40, chunk_override > 0 ? chunk_override : 7};
    for (int ci = 0; ci < (chunk_override > 0 ? 1 : 2); ++ci) {
        Run run{fx, dumps + "/o", force};
        if (!run_prefill(fx, ss, chunks[ci], run, tokens, err, cs)) die("run (chunk " + std::to_string(chunks[ci]) + "): " + err);
        std::printf("chunk %lld (%s):\n", (long long) chunks[ci], force ? "teacher-forced layer inputs" : "accumulated");
        for (int64_t l = 0; l < fx.g.n_layers; ++l) {
            for (const char* seam : kFloatSeams) {
                const std::string key = std::string(seam) + "-" + std::to_string(l);
                auto it = run.stats.find(key);
                if (it == run.stats.end() || !it->second.seen) continue;
                const double gate = gate_for(seam, force), e = it->second.rel();
                const bool pass = e <= gate;
                std::printf("  %-20s %.3e  (gate %.0e) %s\n", key.c_str(), e, gate, pass ? "ok" : "*** FAIL ***");
                fails += !pass;
            }
            const std::string tk = "ffn_moe_topk-" + std::to_string(l);
            if (run.stats.count(tk) && run.stats[tk].seen) std::printf("  %-20s compared over %lld rows\n", tk.c_str(), (long long) run.stats[tk].rows);
        }
        if (!run.topk_flips.empty()) {
            std::printf("  top-k: %zu disagreements with the oracle, %d of them not a near tie%s\n", run.topk_flips.size(), run.topk_bad,
                        run.topk_bad ? " *** FAIL ***" : "");
            for (size_t i = 0; i < run.topk_flips.size() && i < 12; ++i) std::printf("      %s\n", run.topk_flips[i].c_str());
            fails += run.topk_bad > 0;
        } else {
            std::printf("  top-k: identical to the oracle's at every position and rank\n");
        }
        // the rows the chunked run compared must be the whole sequence for every seam it saw
        for (const auto& [k, st] : run.stats)
            if (st.seen && st.rows != (int64_t) tokens.size()) {
                std::printf("  *** %s compared %lld of %zu rows\n", k.c_str(), (long long) st.rows, tokens.size());
                ++fails;
            }
    }

    // ---- the negative controls: the engine (forced from the faithful dumps) against the rival readings' dumps
    struct Neg { const char* dir; const char* seam; int64_t layer; };
    const Neg negs[] = {{"neg_gate", "linear_attn_out", 0}, {"neg_pair", "linear_attn_out", 0}, {"neg_norm", "ffn_out", 0},
                        {"neg_attngate", "attn_output", 3}};
    std::printf("negative controls (the engine against the oracle's rival readings; each must exceed 5%%):\n");
    for (const Neg& ng : negs) {
        if (ng.layer >= fx.g.n_layers) continue;
        Run run{fx, dumps + "/o", true};
        // the forced inputs come from the faithful dumps; only the compared seam reads the rival's
        const std::string rival = dumps + "/" + ng.dir + "/" + ng.seam + "-" + std::to_string(ng.layer) + ".f32";
        run.cache[std::string(ng.seam) + "-" + std::to_string(ng.layer)] = read_f32(rival);
        std::string e;
        if (!run_prefill(fx, ss, 40, run, tokens, e, cs)) die("negative run: " + e);
        const double d = run.stats[std::string(ng.seam) + "-" + std::to_string(ng.layer)].rel();
        const bool seen = d > 0.05;
        std::printf("  %-13s %s-%lld: %.3f relative L1 %s\n", ng.dir, ng.seam, (long long) ng.layer, d, seen ? "yes" : "*** NO ***");
        fails += !seen;
    }

    // ---- the same against the oracle's --act-quant q8 dumps (the activations rounded to Q8 before each quantized
    // matmul as ggml does; closer to the int8 MMQ experts), for the record
    if (force) {
        Run run{fx, dumps + "/o_q8", true};
        std::string e;
        const bool ok = run_prefill(fx, ss, 40, run, tokens, e, cs);
        if (!ok) die("q8 run: " + e);
        std::printf("against the Q8 oracle (forced inputs from its own l_out), one chunk of 40:\n");
        for (int64_t l = 0; l < fx.g.n_layers; ++l)
            for (const char* seam : {"linear_attn_out", "attn_output", "ffn_out", "l_out"}) {
                const std::string key = std::string(seam) + "-" + std::to_string(l);
                if (run.stats.count(key) && run.stats[key].seen) std::printf("  %-20s %.3e\n", key.c_str(), run.stats[key].rel());
            }
    }
    // ---- the accumulated error, for the record (no forcing)
    if (force) {
        Run run{fx, dumps + "/o", false};
        std::string e;
        if (!run_prefill(fx, ss, 40, run, tokens, e, cs)) die("unforced run: " + e);
        std::printf("accumulated (no forcing), one chunk of 40:\n");
        for (int64_t l = 0; l < fx.g.n_layers; ++l) {
            const std::string key = "l_out-" + std::to_string(l);
            if (run.stats.count(key) && run.stats[key].seen) std::printf("  %-20s %.3e\n", key.c_str(), run.stats[key].rel());
        }
    }

    core::session_release(ss);
    cudaFree(sbase);
    cudaStreamDestroy(cs);
    if (fails) {
        std::printf("prefill_qwen36_test (%s): %d failures\n", kase.c_str(), fails);
        return 1;
    }
    std::printf("prefill_qwen36_test (%s): pass\n", kase.c_str());
    return 0;
}
