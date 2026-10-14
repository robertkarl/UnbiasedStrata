// tests/cuda/qwen36_mtp_test.cpp - the Qwen3.6 drafter (the -MTP file's nextn block through core/mtp.cpp) held to the
// oracle's `--mtp` dumps, seam by seam and position by position (docs/qwen36-port/step4-drafter.md).
//
//   qwen36_mtp_test <fixture dir>
//       tests/python/qwen36_mtp_fixture.py --out <fixture dir>: mtp5.gguf (the prompt-path fixture's full4 trunk plus
//       the nextn block blk.4, Q8_0 projections / experts / head / embedding), the oracle's dumps for 76 tokens (o,
//       o_mtp: faithful; o_q8, o_q8_mtp: --act-quant q8) and the draft pack cut from it (pack/).
//   qwen36_mtp_test --real --gguf FILE --pack DIR --dump D --mtp-dump D_mtp --tokens t,t,...
//       the real file: `tools/qwen36_dump_llama.cpp FILE D t,t,... 0 mtp` wrote the trunk's `l_out-<last>` (every token's
//       final residual) to D and llama.cpp's own MTP graph (token k with the hidden of k-1, at position k) to D_mtp;
//       the drafter's top-1 at every position against llama.cpp's MTP top-1 and against the prompt's next token.
//
// The synthetic case, per position k = 1..75 (token k with the trunk's final residual of k-1, the oracle's pairing):
//   1. `MtpDrafter::debug_row` (the one-row forward outside any graph) hands every seam to the test, which holds it
//      to the Q8 oracle's row k-1 by llama.cpp's cb() name (relative L2 per row, the worst over the positions, against
//      the gates below; the F32 oracle's figure is reported beside it), the router's top-8 identical as a set, the
//      draft equal to the oracle's argmax at every position;
//   2. the negative cases (upstream-tooling.md section 3): concat(hnorm, enorm) (the hidden state first) must move
//      `mtp_eh_proj` by more than 5 % relative L1, hnorm skipped must move `mtp_hnorm`, and the drafter paired with
//      position k (the hidden of k instead of k-1) must move `mtp_hnorm` and `result_output`;
//   3. the captured graph path (`draft_first`): draft 0 equals the uncaptured forward's draft at the same cell;
//   4. the prompt-side pass (`Prefill::draft_kv`, 64 cells through the prompt path's GEMMs) in place of the drafter's
//      own K/V for cells 0..63, then positions 65.. through the forward again, held to the same gates.
//
// Tolerances: the Q8 oracle rounds activations as ggml does before each quantized matmul; the engine's MMVQ and the
// native expert kernels take q8_1 activations (the same rounding, integer dot products), so the per-seam figures are
// oracle-crosscheck.md's Q8 column with its 4x margin (`attn_out` 6.7e-4 -> 3e-3, `moe_out` / `ffn_out` 5.3e-3 -> 2e-2,
// `l_out` 1.4e-3 -> 6e-3), the norms of forced inputs 1e-5 (the CUDA rsqrtf), the fp16 K/V pool 2.2e-4 at the attention
// (step3-attn.md) inside the attn_out class.  Exit 77 without a CUDA device or the fixtures, 1 on a failure.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/model_geometry.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/prefill.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace core = strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-110s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof b, "%.2e", v); return b; }
[[noreturn]] void die(const std::string& what) { std::fprintf(stderr, "qwen36_mtp_test: %s\n", what.c_str()); std::exit(1); }
void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) die(std::string(what) + ": " + cudaGetErrorString(e)); }

template <class T> T* upload(const void* src, size_t bytes) {
    void* d = nullptr;
    ck(cudaMalloc(&d, bytes ? bytes : 1), "cudaMalloc");
    if (bytes) ck(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "upload");
    return (T*) d;
}

std::vector<float> read_f32(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("missing dump " + path.string());
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0);
    std::vector<float> v((size_t) n / 4);
    f.read((char*) v.data(), n);
    return v;
}

// ---- one named seam's errors over the positions: relative L2 per row (the worst), relative L1 over all rows
struct Stat {
    double worst_l2 = 0.0, l1_num = 0.0, l1_den = 0.0;
    int rows = 0;
    void add(const float* e, const float* o, int64_t n) {
        double d2 = 0.0, o2 = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double d = (double) e[i] - o[i];
            d2 += d * d; o2 += (double) o[i] * o[i];
            l1_num += std::fabs(d); l1_den += std::fabs((double) o[i]);
        }
        worst_l2 = std::max(worst_l2, o2 > 0 ? std::sqrt(d2 / o2) : (d2 > 0 ? INFINITY : 0.0));
        ++rows;
    }
    double l1() const { return l1_den > 0 ? l1_num / l1_den : (l1_num > 0 ? INFINITY : 0.0); }
};

// ---- the oracle's dumps of the nextn block: <stem>.f32 in C order; row t of a (.., T) tensor
struct Dumps {
    fs::path dir;
    std::map<std::string, std::vector<float>> cache;
    const std::vector<float>& get(const std::string& stem) {
        auto it = cache.find(stem);
        if (it == cache.end()) it = cache.emplace(stem, read_f32(dir / (stem + ".f32"))).first;
        return it->second;
    }
    bool has(const std::string& stem) const { return fs::exists(dir / (stem + ".f32")); }
};

const char* const kSeams[] = {"mtp_tok_embd", "mtp_enorm", "mtp_hnorm", "mtp_concat", "mtp_eh_proj", "mtp_attn_norm",
                              "mtp_Vcur", "mtp_Kcur_normed", "mtp_Qcur_full", "mtp_Qcur_normed", "mtp_attn_pregate",
                              "mtp_attn_out", "mtp_attn_residual", "mtp_attn_post_norm", "ffn_moe_logits",
                              "ffn_moe_weights_norm", "mtp_ffn_shexp_gated", "mtp_ffn_out", "mtp_post_ffn",
                              "mtp_shared_head_norm", "result_output"};
bool layer_seam(const std::string& s) { return s != "mtp_shared_head_norm" && s != "result_output"; }
// Relative L2 per row against the Q8 oracle.  The first product after the forced inputs (eh_proj) measures 2.3e-4 on
// both cards; every product after it sees an input that differs from the oracle's by that much, and the oracle's own
// q8 rounding of a slightly different input flips about 1 % of the int8 codes by one step (2e-4 x rms/amax x 127), which
// is 2e-3 of the output: the measured 2.0-2.2e-3 at the Q/K/V projections.  So the gates past eh_proj are the measured
// figures (step4-drafter.md, both cards) with a 2-3x margin, inside the policy's Q8 classes with their 4x margin
// (attn_out 6.7e-4 -> 2.7e-3 per forced layer; here four products compound), and the real checks are the router's
// picks and the argmax, which are exact.
double gate_for(const std::string& s) {
    if (s == "mtp_tok_embd") return 1e-6;                                             // an exact dequantization
    if (s == "mtp_enorm" || s == "mtp_hnorm" || s == "mtp_concat") return 1e-5;       // norms of forced inputs
    if (s == "mtp_eh_proj" || s == "mtp_attn_norm") return 1e-3;                     // one q8_1 x Q8_0 product (2.3e-4)
    if (s == "mtp_Vcur" || s == "mtp_Kcur_normed" || s == "mtp_Qcur_full" || s == "mtp_Qcur_normed") return 5e-3;   // 2.1e-3
    if (s == "mtp_attn_pregate") return 5e-3;                                         // 1.6e-3 (the fp16 pool inside)
    if (s == "mtp_attn_out") return 1e-2;                                             // 5.4e-3
    if (s == "mtp_attn_residual" || s == "mtp_attn_post_norm" || s == "ffn_moe_logits") return 5e-3;   // 1.1-1.2e-3
    if (s == "ffn_moe_weights_norm") return 1e-2;                                     // 1.7e-3: near-ties move the softmax
    if (s == "mtp_ffn_shexp_gated" || s == "mtp_ffn_out") return 3e-2;                // 1.1-1.2e-2 (moe_out class)
    if (s == "mtp_post_ffn" || s == "mtp_shared_head_norm") return 1.5e-2;            // 5.3e-3 (l_out class)
    return 3e-2;                                                                      // result_output: the logits
}

// ---- the trunk's weights and experts for Prefill::init (the fixture's 4 Q8_0 layers), as prefill_qwen36_test loads them
struct HostExperts : core::ExpertSource {
    std::vector<std::vector<uint8_t>> blobs;
    int64_t n_expert = 0;
    const uint8_t* blob(int64_t layer, int64_t expert) override { return blobs[(size_t) (layer * n_expert + expert)].data(); }
};
uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
bool ends_with(const std::string& s, const char* suf) { const size_t n = std::strlen(suf); return s.size() >= n && s.compare(s.size() - n, n, suf) == 0; }

struct Trunk {
    core::WeightTable wt;
    std::vector<void*> owned;
    HostExperts experts;
    int64_t n_vocab = 0;
    void load(const strata::GgufFile& gguf, const core::ModelGeometry& g, bool with_experts) {
        experts.n_expert = g.n_expert;
        experts.blobs.resize((size_t) (g.n_layers * g.n_expert));
        std::map<std::string, const strata::TensorInfo*> exp_t;
        for (const auto& t : gguf.tensors()) {
            int64_t blk = -1;
            if (t.name.rfind("blk.", 0) == 0) blk = std::atoll(t.name.c_str() + 4);
            if (blk >= g.n_layers) continue;   // the nextn block: the drafter's pack
            const uint64_t bytes = strata::tensor_payload_bytes(t);
            if (bytes == 0) die("tensor " + t.name + ": unknown payload size");
            const uint8_t* src = gguf.tensor_data(t);
            if (ends_with(t.name, "_exps.weight")) { exp_t[t.name] = &t; continue; }
            core::WeightRef r;
            r.ne0 = (int64_t) t.shape[0];
            r.ne1 = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
            r.elements = (int64_t) t.elements();
            r.bytes = bytes;
            if (t.name == "output.weight") n_vocab = r.ne1;
            if (t.name == "output.weight" || t.name == "token_embd.weight") {   // shape only: the native head / embedding
                r.resident = false;
                r.native_type = (int) t.type;
            } else if (t.type == 8) {   // Q8_0: served natively
                r.native_data = upload<uint8_t>(src, bytes);
                r.native_type = 8;
                r.resident = false;
                owned.push_back((void*) r.native_data);
            } else if (t.type != 0) {
                die("tensor " + t.name + " has type " + std::to_string(t.type) + " (the fixture uses F32 and Q8_0)");
            } else if (ends_with(t.name, "ffn_gate_inp.weight") || ends_with(t.name, "ssm_alpha.weight") ||
                       ends_with(t.name, "ssm_beta.weight") || ends_with(t.name, "ffn_gate_inp_shexp.weight")) {
                std::vector<uint16_t> h((size_t) r.elements);
                const float* f = (const float*) src;
                for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_of(f[i]);
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
        if (!with_experts) return;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const strata::TensorInfo* gt = exp_t.count(p + "ffn_gate_exps.weight") ? exp_t[p + "ffn_gate_exps.weight"] : nullptr;
            const strata::TensorInfo* ut = exp_t.count(p + "ffn_up_exps.weight") ? exp_t[p + "ffn_up_exps.weight"] : nullptr;
            const strata::TensorInfo* dt = exp_t.count(p + "ffn_down_exps.weight") ? exp_t[p + "ffn_down_exps.weight"] : nullptr;
            if (!gt || !ut || !dt) die("layer " + std::to_string(l) + ": expert tensors missing");
            if (gt->type != 8 || ut->type != 8 || dt->type != 8) die("the trunk's experts are not Q8_0");
            const size_t gu_row = (size_t) g.n_embd / 32 * 34, d_row = (size_t) g.n_ff / 32 * 34;
            const size_t gu_mat = gu_row * (size_t) g.n_ff, d_mat = d_row * (size_t) g.n_embd;
            for (int64_t e = 0; e < g.n_expert; ++e) {
                std::vector<uint8_t>& b = experts.blobs[(size_t) (l * g.n_expert + e)];
                b.resize(2 * gu_mat + d_mat);
                std::memcpy(b.data(), gguf.tensor_data(*gt) + e * gu_mat, gu_mat);
                std::memcpy(b.data() + gu_mat, gguf.tensor_data(*ut) + e * gu_mat, gu_mat);
                std::memcpy(b.data() + 2 * gu_mat, gguf.tensor_data(*dt) + e * d_mat, d_mat);
            }
        }
    }
    void write_layout(const core::ModelGeometry& g, const std::string& dir) const {
        const size_t gu_row = (size_t) g.n_embd / 32 * 34, d_row = (size_t) g.n_ff / 32 * 34;
        const size_t gu_mat = gu_row * (size_t) g.n_ff, d_mat = d_row * (size_t) g.n_embd, blob = 2 * gu_mat + d_mat;
        std::ofstream out(dir + "/native_experts.txt");
        out << "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off (n_expert "
            << g.n_expert << ", total " << blob * (size_t) g.n_layers * (size_t) g.n_expert << "; the harness's in-memory blobs)\n";
        for (int64_t l = 0; l < g.n_layers; ++l)
            out << l << " 8 8 " << (size_t) l * g.n_expert * blob << " " << blob << " 0 " << gu_mat << " " << 2 * gu_mat << "\n";
    }
    ~Trunk() { for (void* p : owned) cudaFree(p); }
};

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("strata-qwen36-mtp-test-" + std::to_string((long long) getpid()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

// ---- the drafter's forward at one position, every seam against the oracle's row
struct Compare {
    Dumps *q8 = nullptr, *f32 = nullptr;
    int64_t il = 0;
    std::map<std::string, Stat> st_q8, st_f32;
    std::vector<float> host;
    int topk_bad = 0, topk_rows = 0;
    int row = 0;   // the oracle's row (position - 1)
    std::vector<int32_t> ids;
    std::vector<float> last_logits;   // the row's result_output
    std::string stem(const std::string& s) const { return layer_seam(s) ? s + "-" + std::to_string(il) : s; }
    void on_seam(const char* seam, const float* dev, int64_t n) {
        const std::string s(seam);
        host.resize((size_t) n);
        ck(cudaMemcpy(host.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost), "seam copy");
        if (s == "ffn_moe_topk") {   // int32 ids behind the float pointer: identical as a set
            ids.assign((const int32_t*) host.data(), (const int32_t*) host.data() + n);
            const std::vector<float>& o = q8->get(stem(s));
            std::set<int32_t> es(ids.begin(), ids.end()), os;
            for (int64_t j = 0; j < n; ++j) os.insert((int32_t) std::lround(o[(size_t) (row * n + j)]));
            ++topk_rows;
            if (es != os) ++topk_bad;
            return;
        }
        if (s == "result_output") last_logits = host;
        if (s == "ffn_moe_weights_norm") {   // matched by expert id (the oracle's order is by rank)
            const std::vector<float>& o = q8->get(stem(s));
            const std::vector<float>& oi = q8->get(stem("ffn_moe_topk"));
            std::vector<float> matched((size_t) n, 0.0f);
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < n; ++i)
                    if ((int32_t) std::lround(oi[(size_t) (row * n + i)]) == ids[(size_t) j]) matched[(size_t) j] = o[(size_t) (row * n + i)];
            st_q8[s].add(host.data(), matched.data(), n);
            return;
        }
        for (Dumps* d : {q8, f32}) {
            if (d == nullptr) continue;
            const std::vector<float>& o = d->get(stem(s));
            if (o.size() < (size_t) ((row + 1) * n)) die("dump " + stem(s) + " is too short for row " + std::to_string(row));
            (d == q8 ? st_q8 : st_f32)[s].add(host.data(), o.data() + (size_t) row * n, n);
        }
    }
};

int argmax(const float* v, int64_t n) {
    int b = 0;
    for (int64_t i = 1; i < n; ++i) if (v[i] > v[b]) b = (int) i;
    return b;
}

int run_synthetic(const std::string& dir) {
    if (std::ifstream skip(dir + "/SKIP"); skip) {
        std::string why;
        std::getline(skip, why);
        std::printf("qwen36_mtp_test: the fixtures were not generated (%s/SKIP: %s): skip\n", dir.c_str(), why.c_str());
        return 77;
    }
    const std::string gguf_path = dir + "/mtp5.gguf", pack = dir + "/pack";
    if (!fs::exists(gguf_path) || !fs::exists(pack + "/dense.txt")) { std::printf("qwen36_mtp_test: no fixture in %s: skip\n", dir.c_str()); return 77; }
    strata::GgufFile gguf(gguf_path);
    core::ModelGeometry g;
    std::string err;
    if (!core::read_model_geometry(gguf, g, err)) die("geometry: " + err);
    if (g.arch != core::ModelArch::Qwen35Moe || g.nextn_layers != 1) die("the fixture is not a qwen35moe file with one nextn block");
    std::printf("fixture %s: %lld trunk layers + the nextn block blk.%lld, n_embd %lld, attention %lld/%lld x %lld, %lld experts top-%lld of %lld\n",
                gguf_path.c_str(), (long long) g.n_layers, (long long) g.n_layers, (long long) g.n_embd, (long long) g.n_head,
                (long long) g.n_head_kv, (long long) g.head_dim, (long long) g.n_expert, (long long) g.n_expert_used, (long long) g.n_ff);
    Trunk trunk;
    trunk.load(gguf, g, true);
    const int64_t n_vocab = trunk.n_vocab;
    core::NativeHead head;
    if (!head.load({gguf_path}, g.n_embd, n_vocab, err)) die("native head: " + err);
    core::NativeEmbed embed;
    if (!embed.load({gguf_path}, g.n_embd, n_vocab, err)) die("native embedding: " + err);
    core::set_native_embed(&embed);
    // the tokens and the trunk's final residual rows (the oracle's l_out-<last>: what the verify window hands the drafter)
    Dumps o, o_mtp, q8, q8_mtp;
    o.dir = dir + "/o"; o_mtp.dir = dir + "/o_mtp"; q8.dir = dir + "/o_q8"; q8_mtp.dir = dir + "/o_q8_mtp";
    const std::string text = "the quick brown fox jumps over the lazy dog while the cat sleeps on the mat ";
    std::vector<int32_t> tokens;
    for (char c : text) tokens.push_back((int32_t) (unsigned char) c);
    const int n_tok = (int) tokens.size();
    const int64_t N = g.n_embd;
    // the Q8 oracle's trunk: its nextn dumps (the primary reference) were computed from these rows
    const std::vector<float>& R_all = q8.get("l_out-" + std::to_string(g.n_layers - 1));
    if (R_all.size() != (size_t) n_tok * N) die("l_out dump: " + std::to_string(R_all.size()) + " floats for " + std::to_string(n_tok) + " tokens");
    float* R_dev = upload<float>(R_all.data(), R_all.size() * 4);
    // the session (the drafter borrows the primary attention state's RoPE tables) and the drafter
    const int64_t max_cells = 128, max_t = 4;
    core::SessionState ss;
    void* sbase = nullptr;
    ck(cudaMalloc(&sbase, core::session_bytes(g, max_cells, g.n_expert_used)), "session");
    if (core::session_init(g, max_cells, g.n_expert_used, sbase, ss) == 0) die("session_init failed");
    ss.k = g.n_expert_used;
    core::MtpDrafter mtp;
    if (!mtp.load(pack, g, ss, (int) max_t, err, 0)) die("mtp load: " + err);
    float* window_R = nullptr;
    ck(cudaMalloc((void**) &window_R, (size_t) max_t * N * 4), "window_R");
    if (!mtp.bind(trunk.wt, &head, window_R, err)) die("mtp bind: " + err);
    std::printf("drafter: %s, experts %d/%d blob %lld B, %.0f MiB of VRAM\n", mtp.qwen36() ? "qwen35moe" : "?",
                mtp.expert_gu_type(), mtp.expert_d_type(), (long long) mtp.expert_blob_bytes(), (double) mtp.vram_bytes() / 1048576.0);

    // ---- 1. the forward at every position, seam by seam
    const int n_pos = n_tok - 1;
    Compare cmp;
    cmp.q8 = &q8_mtp; cmp.f32 = nullptr; cmp.il = g.n_layers;
    const std::vector<float>& o_logits = q8_mtp.get("result_output");
    std::vector<int32_t> top1((size_t) n_pos, -1);
    int argmax_bad = 0;
    auto hook = [&](const char* s, const float* d, int64_t n) { cmp.on_seam(s, d, n); };
    for (int k = 1; k <= n_pos; ++k) {
        cmp.row = k - 1;
        int32_t t1 = -1;
        // the engine's convention: RoPE at the cell (the hidden state's position); the oracle rotates at k, one more,
        // which the attention cannot see (RoPE is relative), so the comparison holds to float rounding
        if (!mtp.debug_row(R_dev + (size_t) (k - 1) * N, tokens[(size_t) k], k - 1, k - 1, core::MtpDrafter::DebugOpts{}, hook, &t1, err))
            die("debug_row at position " + std::to_string(k) + ": " + err);
        top1[(size_t) (k - 1)] = t1;
        if (t1 != argmax(o_logits.data() + (size_t) (k - 1) * n_vocab, n_vocab)) {
            // a flip is a near tie when the oracle's gap between its pick and the engine's is within a few times the
            // row's largest per-logit error (the logits class is 1e-2 relative); anything wider is a failure
            const float* o = o_logits.data() + (size_t) (k - 1) * n_vocab;
            const float gap = o[argmax(o, n_vocab)] - o[t1];
            float worst = 0.0f;
            for (int64_t i = 0; i < n_vocab; ++i) worst = std::max(worst, std::fabs(cmp.last_logits[(size_t) i] - o[i]));
            std::printf("  position %d: the draft %d, the Q8 oracle's argmax %d, oracle gap %.3e, largest logit error %.3e%s\n", k, t1,
                        argmax(o, n_vocab), gap, worst, gap <= 3.0f * worst ? " (near tie)" : " *** NOT A TIE ***");
            if (gap > 3.0f * worst) ++argmax_bad;
        }
    }
    // the same positions from the F32 oracle's trunk rows against its own nextn dumps (reported: the F32 oracle does
    // not round activations, so its column is the engine's q8_1 rounding itself, the 1e-2 class)
    Compare cmp32;
    cmp32.q8 = &o_mtp; cmp32.f32 = nullptr; cmp32.il = g.n_layers;
    {
        const std::vector<float>& R32 = o.get("l_out-" + std::to_string(g.n_layers - 1));
        float* R32_dev = upload<float>(R32.data(), R32.size() * 4);
        auto hook32 = [&](const char* s, const float* d, int64_t n) { cmp32.on_seam(s, d, n); };
        for (int k = 1; k <= n_pos; ++k) {
            cmp32.row = k - 1;
            int32_t t1 = -1;
            if (!mtp.debug_row(R32_dev + (size_t) (k - 1) * N, tokens[(size_t) k], k - 1, k - 1, core::MtpDrafter::DebugOpts{}, hook32, &t1, err))
                die("debug_row (F32 rows) at position " + std::to_string(k) + ": " + err);
        }
        cudaFree(R32_dev);
    }
    std::printf("the forward at %d positions (token k with the hidden of k-1), relative L2 per row, worst over the positions:\n", n_pos);
    std::printf("  %-28s %12s %12s %12s\n", "seam", "vs Q8 oracle", "gate", "vs F32 oracle");
    for (const char* s : kSeams) {
        const Stat& a = cmp.st_q8[s];
        const double f = cmp32.st_q8.count(s) ? cmp32.st_q8[s].worst_l2 : 0.0;
        const double gate = gate_for(s);
        std::printf("  %-28s %12s %12s %12s\n", s, fmt(a.worst_l2).c_str(), fmt(gate).c_str(), fmt(f).c_str());
        check(a.rows == n_pos && a.worst_l2 <= gate, std::string(s) + " within " + fmt(gate));
    }
    check(cmp.topk_rows == n_pos && cmp.topk_bad == 0, "ffn_moe_topk: the router's top-8 identical as a set at every position (" + std::to_string(cmp.topk_bad) + " of " + std::to_string(cmp.topk_rows) + " differ)");
    check(argmax_bad == 0, "the draft equals the Q8 oracle's argmax at every position but near ties (" + std::to_string(argmax_bad) + " of " + std::to_string(n_pos) + " differ beyond the logits' error)");
    {
        int f32_bad = 0;
        const std::vector<float>& ol = o_mtp.get("result_output");
        for (int k = 1; k <= n_pos; ++k) if (top1[(size_t) (k - 1)] != argmax(ol.data() + (size_t) (k - 1) * n_vocab, n_vocab)) ++f32_bad;
        std::printf("  (the draft against the F32 oracle's argmax: %d of %d differ, reported)\n", f32_bad, n_pos);
    }

    // ---- 2. the negative cases: each rival reading must sit more than 5 % (relative L1) from the oracle
    auto rival = [&](const char* what, const core::MtpDrafter::DebugOpts& opts, int shift, const char* seam_name, const char* seam2) {
        Compare neg;
        neg.q8 = &q8_mtp; neg.f32 = nullptr; neg.il = g.n_layers;
        for (int k = 1; k <= n_pos - shift; ++k) {
            neg.row = k - 1;
            int32_t t1 = -1;
            auto h = [&](const char* s, const float* d, int64_t n) { neg.on_seam(s, d, n); };
            if (!mtp.debug_row(R_dev + (size_t) (k - 1 + shift) * N, tokens[(size_t) k], k - 1, k - 1, opts, h, &t1, err))
                die(std::string("debug_row (") + what + "): " + err);
        }
        for (const char* sn : {seam_name, seam2}) {
            if (sn == nullptr) continue;
            const double l1 = neg.st_q8[sn].l1();
            check(l1 > 0.05, std::string("negative: ") + what + " moves " + sn + " by " + fmt(l1) + " relative L1 (> 0.05)");
        }
    };
    core::MtpDrafter::DebugOpts hidden_first, skip_hnorm;
    hidden_first.hidden_first = true;
    skip_hnorm.skip_hnorm = true;
    rival("concat(hnorm, enorm), the hidden state first", hidden_first, 0, "mtp_eh_proj", "result_output");
    rival("hnorm skipped", skip_hnorm, 0, "mtp_hnorm", "mtp_eh_proj");
    rival("paired with position k instead of k-1", core::MtpDrafter::DebugOpts{}, 1, "mtp_hnorm", "result_output");

    // ---- 3. the captured graph (draft_first: the round graph's catch-up and full layer at cell 0) agrees with the
    //         uncaptured forward at the same cell
    {
        std::vector<int32_t> drafts((size_t) max_t, -1);
        if (!mtp.draft_first((int) max_t, R_dev, tokens[1], 0, drafts.data(), err)) die("draft_first: " + err);
        std::printf("  draft_first at cell 0: drafts %d %d %d; the forward's draft at position 1: %d\n", drafts[0], drafts[1], drafts[2], top1[0]);
        check(drafts[0] == top1[0], "the captured round graph's draft 0 equals the uncaptured forward's at cell 0");
    }

    // ---- 4. the prompt-side pass: cells 0..63 through Prefill::draft_kv, then the forward at the positions after
    {
        TempDir tmp;
        trunk.write_layout(g, tmp.path.string());
        if (!strata::kernels::cpu::expert_layout_load(tmp.path.string(), g.n_layers, g.n_expert, g.n_embd, g.n_ff, err)) die("expert layout: " + err);
        cudaStream_t cs = nullptr;
        ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
        {   // the Prefill is destroyed (its release syncs the stream) before the stream is
        strata::prefill::Prefill pf;
        if (!pf.init(trunk.wt, g, ss, &trunk.experts, nullptr, nullptr, 64, cs, err)) die("Prefill::init: " + err);
        const int64_t n_kv = 64;
        core::qsa_state_zero(mtp.kv_state_rw(), g, nullptr);
        ck(cudaDeviceSynchronize(), "zero");
        err.clear();
        const bool batched = pf.draft_kv(mtp, R_dev, tokens.data() + 1, n_kv, 0, err);
        check(batched && err.empty(), "Prefill::draft_kv took the 64 cells (" + (err.empty() ? std::string(batched ? "batched" : "declined") : err) + ")");
        if (batched) {
            Compare after;
            after.q8 = &q8_mtp; after.f32 = nullptr; after.il = g.n_layers;
            int bad = 0;
            auto h = [&](const char* s, const float* d, int64_t n) { after.on_seam(s, d, n); };
            for (int k = (int) n_kv + 1; k <= n_pos; ++k) {
                after.row = k - 1;
                int32_t t1 = -1;
                if (!mtp.debug_row(R_dev + (size_t) (k - 1) * N, tokens[(size_t) k], k - 1, k - 1, core::MtpDrafter::DebugOpts{}, h, &t1, err))
                    die("debug_row after draft_kv at position " + std::to_string(k) + ": " + err);
                if (t1 != top1[(size_t) (k - 1)]) ++bad;   // the drafter's own pass's draft at that position
            }
            std::printf("  after draft_kv's 64 cells, positions %lld..%d: attn_pregate %s, attn_out %s, ffn_out %s, head norm %s, logits %s (vs Q8 oracle)\n",
                        (long long) n_kv + 1, n_pos, fmt(after.st_q8["mtp_attn_pregate"].worst_l2).c_str(), fmt(after.st_q8["mtp_attn_out"].worst_l2).c_str(),
                        fmt(after.st_q8["mtp_ffn_out"].worst_l2).c_str(), fmt(after.st_q8["mtp_shared_head_norm"].worst_l2).c_str(),
                        fmt(after.st_q8["result_output"].worst_l2).c_str());
            for (const char* s : {"mtp_attn_pregate", "mtp_attn_out", "mtp_ffn_out", "mtp_shared_head_norm", "result_output"})
                check(after.st_q8[s].worst_l2 <= gate_for(s), std::string("after draft_kv: ") + s + " within " + fmt(gate_for(s)));
            check(bad == 0, "after draft_kv: the drafts at the later positions equal the drafter's own pass's (" + std::to_string(bad) + " differ)");
        }
        }
        cudaStreamDestroy(cs);
    }
    core::set_native_embed(nullptr);
    cudaFree(window_R);
    cudaFree(R_dev);
    cudaFree(sbase);
    return g_fail ? 1 : 0;
}

// ---- the real file: the drafter's top-1 per position against llama.cpp's MTP arm and the prompt's next token
int run_real(const std::string& gguf_path, const std::string& pack, const std::string& dump, const std::string& mtp_dump,
             const std::vector<int32_t>& tokens) {
    strata::GgufFile gguf(gguf_path);
    core::ModelGeometry g;
    std::string err;
    if (!core::read_model_geometry(gguf, g, err)) die("geometry: " + err);
    if (g.arch != core::ModelArch::Qwen35Moe) die("not a qwen35moe file");
    const int64_t N = g.n_embd;
    // the trunk's output norm (F32 from the file), the head and embedding rows (shape only: native)
    core::WeightTable wt;
    int64_t n_vocab = 0;
    for (const auto& t : gguf.tensors()) {
        if (t.name != "output_norm.weight" && t.name != "output.weight" && t.name != "token_embd.weight") continue;
        core::WeightRef r;
        r.ne0 = (int64_t) t.shape[0];
        r.ne1 = t.shape.size() > 1 ? (int64_t) t.shape[1] : 1;
        r.elements = (int64_t) t.elements();
        r.bytes = strata::tensor_payload_bytes(t);
        if (t.name == "output_norm.weight") {
            if (t.type != 0) die("output_norm.weight is not F32");
            r.kind = core::WeightKind::F32;
            r.data = upload<uint8_t>(gguf.tensor_data(t), r.bytes);
        } else {
            r.resident = false;
            r.native_type = (int) t.type;
            if (t.name == "output.weight") n_vocab = r.ne1;
        }
        wt.set(t.name, r);
    }
    core::NativeHead head;
    if (!head.load({gguf_path}, N, n_vocab, err)) die("native head: " + err);
    core::NativeEmbed embed;
    if (!embed.load({gguf_path}, N, n_vocab, err)) die("native embedding: " + err);
    core::set_native_embed(&embed);
    const int n_tok = (int) tokens.size(), n_pos = n_tok - 1;
    Dumps d, dm;
    d.dir = dump; dm.dir = mtp_dump;
    const std::vector<float>& R_all = d.get("l_out-" + std::to_string(g.n_layers - 1));
    if (R_all.size() != (size_t) n_tok * N) die("l_out-" + std::to_string(g.n_layers - 1) + ": " + std::to_string(R_all.size()) + " floats for " + std::to_string(n_tok) + " tokens");
    float* R_dev = upload<float>(R_all.data(), R_all.size() * 4);
    const std::vector<float>& ll = dm.get("result_output");   // llama.cpp's MTP logits, every position (the dumper asks for all)
    if (ll.size() != (size_t) n_pos * n_vocab) die("the MTP dump's result_output holds " + std::to_string(ll.size() / (size_t) n_vocab) + " rows, expected " + std::to_string(n_pos));
    const std::vector<float>& lh = dm.get("mtp_shared_head_norm");
    const int64_t max_cells = 256, max_t = 4;
    core::SessionState ss;
    void* sbase = nullptr;
    ck(cudaMalloc(&sbase, core::session_bytes(g, max_cells, g.n_expert_used)), "session");
    if (core::session_init(g, max_cells, g.n_expert_used, sbase, ss) == 0) die("session_init failed");
    ss.k = g.n_expert_used;
    core::MtpDrafter mtp;
    if (!mtp.load(pack, g, ss, (int) max_t, err, 0)) die("mtp load: " + err);
    float* window_R = nullptr;
    ck(cudaMalloc((void**) &window_R, (size_t) max_t * N * 4), "window_R");
    if (!mtp.bind(wt, &head, window_R, err)) die("mtp bind: " + err);
    std::printf("drafter from %s: experts %d/%d blob %lld B, %.0f MiB of VRAM; %d positions\n", pack.c_str(), mtp.expert_gu_type(),
                mtp.expert_d_type(), (long long) mtp.expert_blob_bytes(), (double) mtp.vram_bytes() / 1048576.0, n_pos);
    std::vector<float> host;
    Stat s_head, s_eh, s_attn, s_ffn;
    std::vector<float> logits;
    int agree_llama = 0, agree_next = 0, llama_next = 0, n_next = 0;
    std::printf("  %4s %8s %8s %8s  %s\n", "pos", "engine", "llama", "next", "head-norm relL2");
    for (int k = 1; k <= n_pos; ++k) {
        int32_t t1 = -1;
        double hn_err = 0.0;
        auto hook = [&](const char* s, const float* dev, int64_t n) {
            const std::string sn(s);
            if (sn != "mtp_shared_head_norm" && sn != "mtp_eh_proj" && sn != "mtp_attn_out" && sn != "mtp_ffn_out" && sn != "result_output") return;
            host.resize((size_t) n);
            ck(cudaMemcpy(host.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost), "seam copy");
            if (sn == "result_output") { logits = host; return; }
            const std::string stem = sn == "mtp_shared_head_norm" ? sn : sn + "-" + std::to_string(g.n_layers);
            if (!dm.has(stem)) return;
            Stat& st = sn == "mtp_shared_head_norm" ? s_head : sn == "mtp_eh_proj" ? s_eh : sn == "mtp_attn_out" ? s_attn : s_ffn;
            const std::vector<float>& o = dm.get(stem);
            Stat one;
            one.add(host.data(), o.data() + (size_t) (k - 1) * n, n);
            st.add(host.data(), o.data() + (size_t) (k - 1) * n, n);
            if (sn == "mtp_shared_head_norm") hn_err = one.worst_l2;
        };
        if (!mtp.debug_row(R_dev + (size_t) (k - 1) * N, tokens[(size_t) k], k - 1, k - 1, core::MtpDrafter::DebugOpts{}, hook, &t1, err))
            die("debug_row at position " + std::to_string(k) + ": " + err);
        const int lt = argmax(ll.data() + (size_t) (k - 1) * n_vocab, n_vocab);
        const int nx = k + 1 < n_tok ? tokens[(size_t) k + 1] : -1;
        agree_llama += t1 == lt;
        if (nx >= 0) { ++n_next; agree_next += t1 == nx; llama_next += lt == nx; }
        std::printf("  %4d %8d %8d %8d  %s%s%s\n", k, t1, lt, nx, fmt(hn_err).c_str(), t1 == lt ? "" : "  (differs from llama.cpp)",
                    nx >= 0 && t1 == nx ? "  accepted" : "");
    }
    std::printf("top-1 agreement with llama.cpp's MTP arm: %d of %d (%.1f%%)\n", agree_llama, n_pos, 100.0 * agree_llama / n_pos);
    std::printf("top-1 equal to the prompt's next token (teacher-forced acceptance): engine %d of %d (%.1f%%), llama.cpp's MTP %d of %d (%.1f%%)\n",
                agree_next, n_next, 100.0 * agree_next / std::max(1, n_next), llama_next, n_next, 100.0 * llama_next / std::max(1, n_next));
    std::printf("seams vs llama.cpp (relative L2 per row, worst over positions; relative L1 over all): eh_proj %s / %s, attn_out %s / %s, ffn_out %s / %s, shared_head_norm %s / %s\n",
                fmt(s_eh.worst_l2).c_str(), fmt(s_eh.l1()).c_str(), fmt(s_attn.worst_l2).c_str(), fmt(s_attn.l1()).c_str(),
                fmt(s_ffn.worst_l2).c_str(), fmt(s_ffn.l1()).c_str(), fmt(s_head.worst_l2).c_str(), fmt(s_head.l1()).c_str());
    (void) lh;
    core::set_native_embed(nullptr);
    cudaFree(window_R);
    cudaFree(R_dev);
    cudaFree(sbase);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qwen36_mtp_test <fixture dir>\n       qwen36_mtp_test --real --gguf FILE --pack DIR --dump D --mtp-dump D_mtp --tokens t,t,...\n");
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // the lines reach a log as they are printed
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("qwen36_mtp_test: no CUDA device (skip)\n"); return 77; }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "device");
    std::printf("qwen36_mtp_test on %s (sm_%d%d)\n", p.name, p.major, p.minor);
    if (std::strcmp(argv[1], "--real") == 0) {
        std::string gguf, pack, dump, mtp_dump, toks;
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            if (a == "--gguf") gguf = argv[i + 1];
            else if (a == "--pack") pack = argv[i + 1];
            else if (a == "--dump") dump = argv[i + 1];
            else if (a == "--mtp-dump") mtp_dump = argv[i + 1];
            else if (a == "--tokens") toks = argv[i + 1];
            else die("unknown option " + a);
        }
        if (gguf.empty() || pack.empty() || dump.empty() || mtp_dump.empty() || toks.empty()) die("--real needs --gguf, --pack, --dump, --mtp-dump and --tokens");
        std::vector<int32_t> tokens;
        std::istringstream is(toks);
        std::string t;
        while (std::getline(is, t, ',')) tokens.push_back((int32_t) std::atoi(t.c_str()));
        return run_real(gguf, pack, dump, mtp_dump, tokens);
    }
    const int rc = run_synthetic(argv[1]);
    std::printf("%s\n", rc == 0 ? "qwen36_mtp_test: all checks passed" : rc == 77 ? "qwen36_mtp_test: skipped" : "qwen36_mtp_test: FAILURES");
    return rc;
}
