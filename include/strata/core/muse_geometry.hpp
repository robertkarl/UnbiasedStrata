#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

// Muse has a different residual width and expert width. Keep these separate from
// the Qwen geometry, whose kernels assume the experts read the residual stream.
struct MuseGeometry {
    int layers, context, embedding, latent, feed_forward, experts, active;
    int heads, kv_heads, head_dim, vocab, window;
    float rms_eps, post_eps, rope_base, logit_scale, logit_cap, expert_scale;
    std::vector<bool> sliding;

    static MuseGeometry read(const GgufFile& file) {
        auto require = [&](const std::string& key) -> const MetaValue& {
            const auto* v = file.get(key);
            if (!v) throw std::runtime_error("Muse: missing " + key);
            return *v;
        };
        if (require("general.architecture").s != "muse-spark")
            throw std::runtime_error("Muse: expected general.architecture=muse-spark");
        auto number = [&](const char* suffix) {
            const std::string key = std::string("muse-spark.") + suffix;
            const auto& v = require(key);
            if (!v.is_num() || !std::isfinite(v.num()) || v.num() <= 0)
                throw std::runtime_error("Muse: expected positive finite " + key);
            return v.num();
        };
        auto integer = [&](const char* suffix) {
            const double n = number(suffix);
            if (n > std::numeric_limits<int>::max() || n != std::floor(n))
                throw std::runtime_error(std::string("Muse: invalid integer ") + suffix);
            return static_cast<int>(n);
        };
        MuseGeometry g{};
        g.layers = integer("block_count");
        g.context = integer("context_length");
        g.embedding = integer("embedding_length");
        g.latent = integer("moe_latent_size");
        g.feed_forward = integer("expert_feed_forward_length");
        g.experts = integer("expert_count");
        g.active = integer("expert_used_count");
        g.heads = integer("attention.head_count");
        g.kv_heads = integer("attention.head_count_kv");
        g.head_dim = integer("attention.key_length");
        g.window = integer("attention.sliding_window");
        g.rms_eps = number("attention.layer_norm_rms_epsilon");
        g.post_eps = number("post_norm_epsilon");
        g.rope_base = number("rope.freq_base_swa");
        g.logit_scale = number("logit_scale");
        g.logit_cap = number("final_logit_softcapping");
        g.expert_scale = number("expert_weights_scale");
        if (integer("attention.value_length") != g.head_dim || g.head_dim % 2 ||
            g.heads % g.kv_heads || g.active > g.experts)
            throw std::runtime_error("Muse: inconsistent attention or routing dimensions");
        const auto& gating = require("muse-spark.expert_gating_func");
        const auto& norm = require("muse-spark.expert_weights_norm");
        if (!gating.is_num() || gating.num() != 2 || norm.type != MetaType::BOOL || !norm.u)
            throw std::runtime_error("Muse: expected normalized sigmoid expert routing");
        const auto& pattern = require("muse-spark.attention.sliding_window_pattern");
        if (pattern.type != MetaType::ARRAY || pattern.elem != MetaType::BOOL ||
            pattern.count != static_cast<uint64_t>(g.layers) || pattern.items.size() != pattern.count)
            throw std::runtime_error("Muse: incomplete sliding_window_pattern");
        for (const auto& v : pattern.items) g.sliding.push_back(v.u != 0);
        const auto& tokens = require("tokenizer.ggml.tokens");
        if (tokens.type != MetaType::ARRAY || tokens.elem != MetaType::STRING ||
            tokens.count == 0 || tokens.count > std::numeric_limits<int>::max())
            throw std::runtime_error("Muse: invalid tokenizer.ggml.tokens");
        g.vocab = static_cast<int>(tokens.count);
        return g;
    }
};

} // namespace strata::core
