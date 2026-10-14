// tests/core/pack_load_check.cpp - a native pack loaded the way `strata generate --native SHARD1 --pack DIR` loads it,
// up to the point where the first token would run, and every dense weight the step-3 code reads accounted for.
//
//   pack_load_check --native FILE.gguf --pack DIR        (or STRATA_QWEN36_GGUF / STRATA_QWEN36_PACK; neither: exit 77)
//
// Why a separate binary (Qwen3.6 port, step 4, docs/qwen36-port/step4-pack.md): `generate.cpp` stops a qwen35moe
// start at the step-2 gate, which fires BEFORE the arena, the native projections, the head and the embedding are
// loaded, so a run through the gate proves nothing about the pack.  This does what the lines after the gate do, in
// their order, with the same classes and the same skip set, then `check_all` (the per-layer shape and form check the
// kernels rely on, which generate.cpp does not call itself), and prints what each site will read: the name, the
// engine form (WeightKind), the bytes, and for a natively served projection its ggml type.  Exit 0 when every load
// and every check passes, 1 with the first refusal on stderr, 77 without a CUDA device or a model.
//
// What it is NOT: it runs no kernel.  The first Qwen3.6 token through this pack is the gate-lifting agent's.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/model_geometry.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using strata::core::ModelGeometry;
using strata::core::WeightKind;
using strata::core::WeightRef;
using strata::core::WeightTable;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int fail(const std::string& what, const std::string& err) {
    std::fprintf(stderr, "pack_load_check: %s: %s\n", what.c_str(), err.c_str());
    return 1;
}

const char* kind_name(WeightKind k) {
    switch (k) {
        case WeightKind::Verbatim: return "verbatim";
        case WeightKind::Bf16InF32: return "bf16";
        case WeightKind::F32: return "f32";
        case WeightKind::F16InF32: return "f16";
    }
    return "?";
}

/// One weight as a site reads it: `name kind bytes [native TYPE]`, or `MISSING`.
std::string describe(const WeightTable& wt, const std::string& name) {
    const WeightRef* r = wt.find(name);
    if (!r) return name + " MISSING";
    char buf[256];
    if (r->native_data) {
        std::snprintf(buf, sizeof buf, "%s native %s [%lld x %lld]", name.c_str(),
                      strata::ggml_type_name((uint32_t) r->native_type), (long long) r->ne0, (long long) r->ne1);
    } else {
        std::snprintf(buf, sizeof buf, "%s %s %llu B (%lld elems)%s", name.c_str(), kind_name(r->kind),
                      (unsigned long long) r->bytes, (long long) r->elements, r->resident ? "" : " NOT RESIDENT");
    }
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, pack;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--native") == 0) gguf = argv[++i];
        else if (std::strcmp(argv[i], "--pack") == 0) pack = argv[++i];
    }
    if (gguf.empty() && std::getenv("STRATA_QWEN36_GGUF")) gguf = std::getenv("STRATA_QWEN36_GGUF");
    if (pack.empty() && std::getenv("STRATA_QWEN36_PACK")) pack = std::getenv("STRATA_QWEN36_PACK");
    if (gguf.empty() || pack.empty()) {
        std::fprintf(stderr, "pack_load_check: no --native FILE / --pack DIR (nor STRATA_QWEN36_GGUF / STRATA_QWEN36_PACK): "
                             "skipped\n");
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "pack_load_check: no CUDA device: skipped\n");
        return 77;
    }
    std::string err;

    // ---- 1. the geometry from the file, the fused forms refused, as generate.cpp:2071-2095
    ModelGeometry g;
    if (!strata::core::read_model_geometry(gguf, g, err)) return fail("read_model_geometry", err);
    std::printf("geometry (%s):\n%s", gguf.c_str(), strata::core::model_geometry_text(g).c_str());
    std::vector<std::string> shards;
    try {
        shards = strata::gguf_split_paths(gguf);
        const strata::GgufModel model(shards);
        if (!strata::core::check_model_tensor_forms(model, g, err)) return fail("check_model_tensor_forms", err);
    } catch (const std::exception& e) {
        return fail("opening the shards", e.what());
    }

    // ---- 2. the expert table against the geometry (generate.cpp:2102)
    if (!strata::kernels::cpu::expert_layout_load(pack, g.n_layers, g.n_expert, g.n_embd, g.n_ff, err))
        return fail("expert_layout_load", err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) return fail("expert layout", pack + " has no native_experts.txt (not a native pack)");
    std::printf("native_experts.txt: v%d, %lld layers x %lld experts, %llu B of experts, largest blob %zu B\n",
                lay.version, (long long) lay.fmt.size(), (long long) lay.n_expert, (unsigned long long) lay.total,
                lay.max_blob);
    if ((int64_t) lay.fmt.size() != g.n_layers)
        return fail("expert layout", "the table has " + std::to_string(lay.fmt.size()) + " layers, the geometry " +
                                     std::to_string(g.n_layers));

    // ---- 3. the skip set and the arena (generate.cpp:2433-2516): natively served names are not loaded in
    //         canonical form; the head and the embedding come from the GGUF
    std::set<std::string> skip;
    if (!strata::core::NativeDense::served_names(shards, false, skip, err)) return fail("served_names", err);
    skip.insert("output.weight");
    skip.insert("token_embd.weight");
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack, pool, err, &skip)) return fail("pool_bytes", err);
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool) != cudaSuccess) return fail("cudaMalloc", "arena of " + std::to_string(pool) + " B");
    WeightTable wt;
    double t0 = now_ms();
    if (!wt.load(pack, arena, pool, err, &skip)) return fail("WeightTable::load", err);
    std::printf("dense.bin: %zu index rows, %llu B arena (%.1f MiB), %zu rows skipped as natively served, "
                "read %.0f ms + upload %.0f ms\n", wt.report().tensors, (unsigned long long) pool, pool / 1048576.0,
                skip.size(), wt.report().read_ms, wt.report().upload_ms);

    // ---- 4. the native projections, bounded to the trunk (the draft block of a -MTP file is the drafter's)
    strata::core::NativeDense dense;
    t0 = now_ms();
    if (!dense.load(shards, wt, err, false, 0, g.n_layers)) return fail("NativeDense::load", err);
    std::printf("native projections: %zu matrices, %llu B (%.1f MiB) in VRAM, %.0f ms\n", dense.tensor_count(),
                (unsigned long long) dense.weight_bytes(), dense.weight_bytes() / 1048576.0, now_ms() - t0);

    // ---- 5. every layer's shapes and forms, as the kernels require them (src/core/layout.cpp)
    if (!strata::core::check_all(wt, g, err)) return fail("check_all", err);
    std::printf("check_all: %lld layers (%lld attention, %lld GDN): every name, shape and engine form the kernels "
                "require\n", (long long) g.n_layers, (long long) g.n_qsa_layers(), (long long) g.n_gdn_layers());

    // ---- 6. the head and the embedding from the GGUF (generate.cpp:3277-3283, :2424); the vocabulary is
    //         output.weight's ne1, as generate.cpp takes it (ModelGeometry::n_vocab is the Flash-Next default)
    const WeightRef* wo = wt.find("output.weight");
    const WeightRef* wn = wt.find("output_norm.weight");
    if (!wo) return fail("output.weight", "missing from the index");
    if (!wn || wn->kind != WeightKind::F32 || wn->elements != g.n_embd)
        return fail("output_norm.weight", wn ? "wrong form or size" : "missing");
    const int64_t n_vocab = wo->ne1;
    strata::core::NativeHead head;
    t0 = now_ms();
    if (!head.load(shards, g.n_embd, n_vocab, err)) return fail("NativeHead::load", err);
    std::printf("native head: output.weight %s, %llu B, n_vocab %lld, %.0f ms\n",
                strata::ggml_type_name((uint32_t) head.type()), (unsigned long long) head.weight_bytes(),
                (long long) n_vocab, now_ms() - t0);
    strata::core::NativeEmbed embed;
    t0 = now_ms();
    if (!embed.load(shards, g.n_embd, n_vocab, err)) return fail("NativeEmbed::load", err);
    std::printf("native embedding: token_embd.weight %s, %llu B mapped, %.0f ms\n",
                strata::ggml_type_name((uint32_t) embed.type()), (unsigned long long) embed.bytes(), now_ms() - t0);

    // ---- 7. the expert profile, when the pack ships one
    const std::string profile = pack + "/expert-profile.bin";
    if (std::filesystem::exists(profile)) {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        int64_t slots = 0;
        if (!strata::core::read_expert_profile(profile, g.n_layers, g.n_expert, ranked, slots, err))
            return fail("read_expert_profile", err);
        std::printf("expert-profile.bin: %zu ranked pairs of %lld x %lld, %lld slots\n", ranked.size(),
                    (long long) g.n_layers, (long long) g.n_expert, (long long) slots);
    } else {
        std::printf("expert-profile.bin: none in the pack\n");
    }

    // ---- 8. what each site reads, per layer family: the names from layout.cpp's tables and the mixers' own
    //         lookups (layer.cpp gdn_layer / qsa_layer / moe_route / moe_shared, verify.cpp, prefill.cpp)
    const char* gdn[] = {"attn_norm.weight", "post_attention_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                         "ssm_out.weight", "ssm_conv1d.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_a",
                         "ssm_dt.bias", "ssm_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                         "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    const char* attn[] = {"attn_norm.weight", "post_attention_norm.weight", "attn_q.weight", "attn_k.weight",
                          "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                          "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
                          "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    int missing = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const strata::core::LayerView v(wt, l);
        const bool qsa = strata::core::is_qsa_layer(g, l);
        const bool show = l == 0 || l == g.qsa_interval - 1;   // one GDN and one attention layer in full
        if (show) std::printf("layer %lld (%s):\n", (long long) l, qsa ? "attention" : "GDN");
        const char* const* names = qsa ? attn : gdn;
        const size_t n = qsa ? sizeof attn / sizeof *attn : sizeof gdn / sizeof *gdn;
        for (size_t i = 0; i < n; ++i) {
            const std::string d = describe(wt, v.name(names[i]));
            if (d.find(" MISSING") != std::string::npos || d.find("NOT RESIDENT") != std::string::npos) {
                ++missing;
                std::printf("  %s\n", d.c_str());
            } else if (show) {
                std::printf("  %s\n", d.c_str());
            }
        }
        // the draft block's names must not have leaked into the trunk's table
        if (wt.find("blk." + std::to_string(g.n_layers) + ".attn_norm.weight") != nullptr) {
            std::printf("  blk.%lld (a draft block) is in the index: the pack is not trunk-only\n", (long long) g.n_layers);
            ++missing;
        }
    }
    std::printf("output_norm.weight: %s\n", describe(wt, "output_norm.weight").c_str());
    std::printf("output.weight: index row %lld x %lld, served by the native head\n", (long long) wo->ne0, (long long) wo->ne1);
    if (missing) return fail("site reads", std::to_string(missing) + " weight(s) missing or not resident (above)");
    std::printf("pack_load_check: %s against %s: OK\n", pack.c_str(), gguf.c_str());
    cudaFree(arena);
    return 0;
}
