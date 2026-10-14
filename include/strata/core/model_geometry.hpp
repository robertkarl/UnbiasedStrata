// include/strata/core/model_geometry.hpp - `ModelGeometry` from the model file's own metadata.
//
// Until Phase 2 step 2 of docs/PLAN-qwen36-35b-a3b.md the geometry was a struct of Flash-Next defaults and only
// `expert_count`, `expert_used_count` and the rope keys were read, under the `qwen4exp.` prefix and only with
// `--native` (docs/qwen36-port/geometry-inventory.md rows 7 to 10).  This header reads every field from the
// `<arch>.*` keys llama.cpp's own loader reads (`src/llama-model.cpp:1231-1424`, `src/models/qwen35moe.cpp:4-34`
// in the pinned submodule; the key set is docs/QWEN36-35B-A3B.md section 3.8), for both architectures, and names
// the tensor forms this engine refuses rather than mis-reads.
//
// Header-only and CUDA-free on purpose: the unit test (tests/core/model_geometry_test.cpp) links nothing but the
// GGUF reader, and the engine's start-up reads the geometry BEFORE any weight moves.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

namespace detail {

/// One integer key: `<arch>.<suffix>`.  `required`: absent is an error; otherwise the default stays.
inline bool geometry_int(const strata::GgufFile& meta, const std::string& prefix, const char* suffix, int64_t& out,
                         bool required, std::string& err) {
    const strata::MetaValue* v = meta.get(prefix + suffix);
    if (v == nullptr) {
        if (required) err = "missing " + prefix + suffix;
        return !required;
    }
    if (!v->is_num()) {
        err = prefix + suffix + " is not a number";
        return false;
    }
    out = (int64_t) v->num();
    return true;
}

inline bool geometry_num(const strata::GgufFile& meta, const std::string& prefix, const char* suffix, double& out,
                         bool required, std::string& err) {
    const strata::MetaValue* v = meta.get(prefix + suffix);
    if (v == nullptr) {
        if (required) err = "missing " + prefix + suffix;
        return !required;
    }
    if (!v->is_num()) {
        err = prefix + suffix + " is not a number";
        return false;
    }
    out = v->num();
    return true;
}

}  // namespace detail

/// Fills `g` from the metadata shard.  Every field of `ModelGeometry` comes from the file where the file has a
/// key for it; the two models' keys differ only in the prefix.  What the file cannot say is decided by the
/// architecture string: `hc` (4 with hyper-connections, 1 without), the indexer, the PLE.  `n_layers` is
/// `block_count` minus the draft blocks (`nextn_predict_layers`, 0 when absent).  A missing key the kernels
/// depend on, a value that is not a number, or a derived quantity the file states differently (`ssm.inner_size`
/// against `state_size * time_step_rank`) is an error that names the key.
inline bool read_model_geometry(const strata::GgufFile& meta, ModelGeometry& g, std::string& err) {
    const strata::MetaValue* arch = meta.get("general.architecture");
    if (arch == nullptr) { err = "missing general.architecture"; return false; }
    if (arch->s == "qwen4exp") {
        g = ModelGeometry{};   // Flash-Next's defaults, every one of them overridable below
        g.arch = ModelArch::Qwen4Exp;
    } else if (arch->s == "qwen35moe") {
        g = ModelGeometry{};
        g.arch = ModelArch::Qwen35Moe;
        g.hc = 1;              // plain residual: attn_norm / post_attention_norm / output_norm
        g.hc_lr = 0;
        g.has_ple = false;     // no per-layer token embedding, no n-gram table
        g.idx_q_heads = 0;     // no sparse indexer: the attention is dense
        g.idx_key_dim = 0;
    } else {
        err = "architecture is '" + arch->s + "', this engine reads 'qwen4exp' (Qwen3.8-Flash-Next) and 'qwen35moe' "
              "(Qwen3.6-35B-A3B)";
        return false;
    }
    const std::string p = arch->s + ".";
    const bool q36 = g.arch == ModelArch::Qwen35Moe;
    // the keys both loaders require (the architecture guard already demands the first six)
    int64_t block_count = 0;
    if (!detail::geometry_int(meta, p, "block_count", block_count, true, err)) return false;
    if (!detail::geometry_int(meta, p, "nextn_predict_layers", g.nextn_layers, false, err)) return false;
    if (!detail::geometry_int(meta, p, "embedding_length", g.n_embd, true, err)) return false;
    if (!detail::geometry_int(meta, p, "attention.head_count", g.n_head, true, err)) return false;
    if (!detail::geometry_int(meta, p, "attention.head_count_kv", g.n_head_kv, true, err)) return false;
    if (!detail::geometry_int(meta, p, "expert_count", g.n_expert, true, err)) return false;
    if (!detail::geometry_int(meta, p, "expert_used_count", g.n_expert_used, true, err)) return false;
    // the keys Flash-Next's files may leave to the defaults (its loader has them hard-coded in places) and Qwen3.6's
    // must carry: the GDN shape, the expert widths, the layer pattern, the attention widths
    if (!detail::geometry_int(meta, p, "attention.key_length", g.head_dim, q36, err)) return false;
    if (!detail::geometry_int(meta, p, "full_attention_interval", g.qsa_interval, q36, err)) return false;
    if (!detail::geometry_int(meta, p, "ssm.conv_kernel", g.ssm_d_conv, q36, err)) return false;
    if (!detail::geometry_int(meta, p, "ssm.state_size", g.ssm_state_size, q36, err)) return false;
    if (!detail::geometry_int(meta, p, "ssm.group_count", g.ssm_k_heads, q36, err)) return false;
    if (!detail::geometry_int(meta, p, "ssm.time_step_rank", g.ssm_v_heads, q36, err)) return false;
    if (!detail::geometry_int(meta, p, "expert_feed_forward_length", g.n_ff, q36, err)) return false;
    g.n_ff_shexp = g.n_ff;
    if (!detail::geometry_int(meta, p, "expert_shared_feed_forward_length", g.n_ff_shexp, false, err)) return false;
    if (!detail::geometry_int(meta, p, "rope.dimension_count", g.n_rot, false, err)) return false;
    if (!detail::geometry_num(meta, p, "rope.freq_base", g.rope_base, false, err)) return false;
    if (!detail::geometry_num(meta, p, "attention.layer_norm_rms_epsilon", g.rms_eps, false, err)) return false;
    if (const strata::MetaValue* v = meta.get(p + "rope.dimension_sections")) {
        if (v->type != strata::MetaType::ARRAY || v->items.size() != 4) {
            err = p + "rope.dimension_sections is not an array of 4";
            return false;
        }
        for (size_t i = 0; i < 4; ++i) g.rope_sections[i] = (int64_t) v->items[i].num();
    }
    // derived, then checked against what the file says of the same quantity
    g.n_layers = block_count - g.nextn_layers;
    g.derive();
    int64_t inner = g.ssm_value_dim;
    if (!detail::geometry_int(meta, p, "ssm.inner_size", inner, false, err)) return false;
    auto bad = [&](const std::string& what) { err = p.substr(0, p.size() - 1) + ": " + what; return false; };
    if (g.n_layers <= 0 || g.nextn_layers < 0)
        return bad("block_count " + std::to_string(block_count) + " minus nextn_predict_layers " +
                   std::to_string(g.nextn_layers) + " leaves no trunk layer");
    if (inner != g.ssm_value_dim)
        return bad("ssm.inner_size is " + std::to_string(inner) + ", but ssm.state_size * ssm.time_step_rank is " +
                   std::to_string(g.ssm_value_dim));
    if (g.qsa_interval <= 0 || g.n_layers % g.qsa_interval != 0)
        return bad("block_count " + std::to_string(g.n_layers) + " is not a multiple of full_attention_interval " +
                   std::to_string(g.qsa_interval));
    if (g.ssm_k_heads <= 0 || g.ssm_v_heads % g.ssm_k_heads != 0)
        return bad("ssm.time_step_rank " + std::to_string(g.ssm_v_heads) + " is not a multiple of ssm.group_count " +
                   std::to_string(g.ssm_k_heads) + " (the recurrence pairs v-heads to k-heads by h % h_k)");
    if (g.n_head_kv <= 0 || g.n_head % g.n_head_kv != 0)
        return bad("attention.head_count " + std::to_string(g.n_head) + " is not a multiple of head_count_kv " +
                   std::to_string(g.n_head_kv));
    if (g.n_expert_used <= 0 || g.n_expert_used > g.n_expert)
        return bad("expert_used_count " + std::to_string(g.n_expert_used) + " over expert_count " +
                   std::to_string(g.n_expert));
    if (g.n_rot <= 0 || g.n_rot > g.head_dim || g.n_rot % 2 != 0)
        return bad("rope.dimension_count " + std::to_string(g.n_rot) + " does not fit attention.key_length " +
                   std::to_string(g.head_dim));
    if (g.rope_sections[0] + g.rope_sections[1] + g.rope_sections[2] + g.rope_sections[3] != g.n_rot / 2)
        return bad("rope.dimension_sections do not sum to rope.dimension_count / 2 = " + std::to_string(g.n_rot / 2));
    // the vocabulary: `output.weight`'s rows are what the engine samples over (generate.cpp takes n_vocab from that
    // tensor), `token_embd.weight`'s rows are what the native embedding loads with g.n_vocab, and the tokenizer's
    // token list is the third statement of the same number.  Whichever of the three the metadata shard carries must
    // agree; the default (248,320, the shipped files') stands only when none is present (a metadata-only fixture).
    {
        int64_t seen[3] = {-1, -1, -1};
        const char* names[3] = {"output.weight", "token_embd.weight", "tokenizer.ggml.tokens"};
        if (const strata::TensorInfo* t = meta.find(names[0])) seen[0] = t->shape.size() >= 2 ? (int64_t) t->shape[1] : 0;
        if (const strata::TensorInfo* t = meta.find(names[1])) seen[1] = t->shape.size() >= 2 ? (int64_t) t->shape[1] : 0;
        if (const strata::MetaValue* v = meta.get(names[2]))
            seen[2] = v->type == strata::MetaType::ARRAY ? (int64_t) v->count : 0;   // `count` is the full length; `items` is capped
        int64_t vocab = -1;
        const char* from = nullptr;
        for (int i = 0; i < 3; ++i) {
            if (seen[i] < 0) continue;
            if (seen[i] <= 0) return bad(std::string(names[i]) + " gives no vocabulary size");
            if (vocab < 0) { vocab = seen[i]; from = names[i]; continue; }
            if (seen[i] != vocab)
                return bad(std::string(from) + " says " + std::to_string(vocab) + " tokens but " + names[i] + " says " +
                           std::to_string(seen[i]));
        }
        if (vocab > 0) g.n_vocab = vocab;
    }
    return true;
}

/// Reads the geometry from `any_shard`'s model (every shard found beside it; the metadata shard decides).
inline bool read_model_geometry(const std::string& any_shard, ModelGeometry& g, std::string& err) {
    try {
        const strata::GgufModel model = strata::GgufModel::open(any_shard);
        return read_model_geometry(model.meta(), g, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

/// The tensor FORMS this engine refuses by name.  llama.cpp accepts two fused layouts that no shipped Qwen3.6 file
/// uses (`gguf-metadata.md`, "What was unexpected"): experts fused as `ffn_gate_up_exps` (gate rows then up rows)
/// and an attention layer's projections fused as `attn_qkv` (Q | K | V rows).  The engine reads three separate
/// expert tensors (`expert_source.cpp`, check_experts_gguf) and separate `attn_q` / `attn_k` / `attn_v`; `attn_qkv`
/// is legitimate ONLY in a GDN layer, where it is the conv input.  A file in a fused form is refused here, naming
/// the tensor, instead of failing later on a shape or being split silently.  Returns true when nothing is fused.
inline bool check_model_tensor_forms(const strata::GgufModel& model, const ModelGeometry& g, std::string& err) {
    for (int64_t l = 0; l < g.n_layers + g.nextn_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        if (model.find(blk + "ffn_gate_up_exps.weight") != nullptr) {
            err = blk + "ffn_gate_up_exps.weight: the experts are fused (llama.cpp's gate-then-up form); this engine "
                  "reads separate ffn_gate_exps / ffn_up_exps / ffn_down_exps tensors.  No shipped Qwen3.6 file is "
                  "fused; re-convert the model without the fused experts";
            return false;
        }
        const bool attention = l >= g.n_layers || is_qsa_layer(g, l);
        if (attention && model.find(blk + "attn_qkv.weight") != nullptr && model.find(blk + "attn_q.weight") == nullptr) {
            err = blk + "attn_qkv.weight: the attention projections are fused (llama.cpp's Q|K|V form); this engine "
                  "reads separate attn_q / attn_k / attn_v tensors in an attention layer (attn_qkv is the conv "
                  "input of a GDN layer only).  No shipped Qwen3.6 file is fused; re-convert the model without it";
            return false;
        }
    }
    return true;
}

/// One line per field, for the start-up log and the test's --real mode.
inline std::string model_geometry_text(const ModelGeometry& g) {
    std::string s;
    auto add = [&](const char* k, int64_t v) { s += std::string("  ") + k + " " + std::to_string(v) + "\n"; };
    s += std::string("  arch ") + model_arch_name(g.arch) + "\n";
    add("n_embd", g.n_embd);
    add("n_vocab", g.n_vocab);
    add("n_layers", g.n_layers);
    add("nextn_layers", g.nextn_layers);
    add("qsa_interval", g.qsa_interval);
    add("n_qsa_layers", g.n_qsa_layers());
    add("n_gdn_layers", g.n_gdn_layers());
    add("ssm_state_size", g.ssm_state_size);
    add("ssm_k_heads", g.ssm_k_heads);
    add("ssm_v_heads", g.ssm_v_heads);
    add("ssm_d_conv", g.ssm_d_conv);
    add("ssm_conv_channels", g.ssm_conv_channels);
    add("ssm_value_dim", g.ssm_value_dim);
    add("n_head", g.n_head);
    add("n_head_kv", g.n_head_kv);
    add("head_dim", g.head_dim);
    add("n_rot", g.n_rot);
    s += "  rope_sections " + std::to_string(g.rope_sections[0]) + "," + std::to_string(g.rope_sections[1]) + "," +
         std::to_string(g.rope_sections[2]) + "," + std::to_string(g.rope_sections[3]) + "\n";
    s += "  rope_base " + std::to_string(g.rope_base) + "\n";
    s += "  rms_eps " + std::to_string(g.rms_eps) + "\n";
    add("idx_q_heads", g.idx_q_heads);
    add("idx_key_dim", g.idx_key_dim);
    add("hc", g.hc);
    add("hc_lr", g.hc_lr);
    add("n_expert", g.n_expert);
    add("n_expert_used", g.n_expert_used);
    add("n_ff", g.n_ff);
    add("n_ff_shexp", g.n_ff_shexp);
    s += std::string("  has_ple ") + (g.has_ple ? "yes" : "no") + "\n";
    s += std::string("  has_indexer ") + (g.has_indexer() ? "yes" : "no") + "\n";
    return s;
}

/// Phase 2 step 2's gate (docs/PLAN-qwen36-35b-a3b.md section 5): what this engine can and cannot yet do with a
/// `qwen35moe` file, said once and precisely, so a start fails here and not inside a kernel.  Empty when the
/// architecture needs no gate.
inline std::string qwen35moe_not_yet(const ModelGeometry& g) {
    if (g.arch != ModelArch::Qwen35Moe) return {};
    return "this is a Qwen3.6-35B-A3B file (qwen35moe, " + std::to_string(g.n_layers) + " layers, " +
           std::to_string(g.n_embd) + " wide, " + std::to_string(g.n_expert) + " experts x top-" +
           std::to_string(g.n_expert_used) + "): Phase 2 step 2 (geometry, metadata, the VRAM plan above) is done, "
           "but the engine cannot run it yet.  Not done: step 3 (kernels and the block: the GDN gate SiLU(z) "
           "instead of sigmoid(z), dense attention at " + std::to_string(g.n_head) + " heads with no indexer, the "
           "block and the verify window without hyper-connections, prefill at this geometry), step 4 (the native "
           "pack and the MTP drafter from blk." + std::to_string(g.n_layers) + ", the expert profile at " +
           std::to_string(g.n_layers) + " x " + std::to_string(g.n_expert) + ").  See docs/PLAN-qwen36-35b-a3b.md "
           "section 5 and docs/qwen36-port/step2-notes.md";
}

}  // namespace strata::core
