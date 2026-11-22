#include "inkling_vocab.hpp"

#include "gguf.h"
#include "nlohmann/json.hpp"
#include "strata/artifact/strata_reader.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

bool identify_inkling_pack(const std::string & path, bool & inkling, std::string & err) {
    strata::artifact::StrataFileReader reader;
    if (!reader.open(path, err)) return false;
    try {
        const auto manifest = json::parse(reader.manifest_json());
        inkling = manifest.value("arch", std::string()) == "inkling2_loop" ||
                  manifest.at("config").at("text_config").value("model_type", std::string()) == "inkling2_loop_text";
        return true;
    } catch (const std::exception & e) {
        err = path + ": " + e.what();
        return false;
    }
}

// llama.cpp tokenizes the request and renders the original template. The child maps the weights.
llama_model * load_inkling_vocab(const std::string & path, llama_model_params params, std::string & err) {
    try {
        strata::artifact::StrataFileReader reader;
        if (!reader.open(path, err)) {
            return nullptr;
        }
        const auto manifest = json::parse(reader.manifest_json());
        const auto & config = manifest.at("config").at("text_config");
        if (config.at("model_type") != "inkling2_loop_text") {
            throw std::runtime_error("expected an Inkling 2 Mini text checkpoint");
        }
        json tok;
        if (manifest.contains("tokenizer_json") && !manifest.at("tokenizer_json").get_ref<const std::string &>().empty()) {
            tok = json::parse(manifest.at("tokenizer_json").get_ref<const std::string &>());
        } else {
            const auto sidecar = std::filesystem::path(path).parent_path() / "tokenizer.json";
            std::ifstream in(sidecar);
            if (!in) throw std::runtime_error("missing Inkling tokenizer " + sidecar.string());
            in >> tok;
        }
        const auto & bpe = tok.at("model");
        if (bpe.at("type") != "BPE" || bpe.value("byte_fallback", false)) {
            throw std::runtime_error("unsupported Inkling tokenizer");
        }

        std::vector<std::string> tokens(config.at("unpadded_vocab_size").get<size_t>());
        std::vector<int32_t> types(tokens.size(), LLAMA_TOKEN_TYPE_NORMAL);
        for (const auto & [piece, id] : bpe.at("vocab").items()) {
            tokens.at(id.get<size_t>()) = piece;
        }
        for (const auto & token : tok.at("added_tokens")) {
            const size_t id = token.at("id").get<size_t>();
            const auto & piece = token.at("content").get_ref<const std::string &>();
            if (!tokens.at(id).empty() && tokens[id] != piece) {
                throw std::runtime_error("conflicting Inkling vocabulary entries");
            }
            tokens[id] = piece;
            types[id] = token.at("special").get<bool>() ? LLAMA_TOKEN_TYPE_CONTROL : LLAMA_TOKEN_TYPE_USER_DEFINED;
        }
        std::vector<const char *> pieces;
        for (const auto & piece : tokens) {
            if (piece.empty()) {
                throw std::runtime_error("incomplete Inkling vocabulary");
            }
            pieces.push_back(piece.c_str());
        }
        std::vector<std::string> merges;
        for (const auto & pair : bpe.at("merges")) {
            merges.push_back(pair.is_string() ? pair.get<std::string>() :
                pair.at(0).get<std::string>() + " " + pair.at(1).get<std::string>());
        }
        std::vector<const char *> rules;
        for (const auto & merge : merges) {
            rules.push_back(merge.c_str());
        }
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), gguf_free);
        auto * gg = metadata.get();
        // Keep the real architecture in the metadata used to select the vision projector.
        gguf_set_val_str(gg, "general.architecture", "inkling2");
        gguf_set_val_str(gg, "general.name", "Inkling 2 Mini");
        gguf_set_val_str(gg, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(gg, "tokenizer.ggml.pre", "granite-embed-multi-97m");
        gguf_set_arr_str(gg, "tokenizer.ggml.tokens", pieces.data(), pieces.size());
        gguf_set_arr_data(gg, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        gguf_set_arr_str(gg, "tokenizer.ggml.merges", rules.data(), rules.size());
        const auto eos = manifest.at("config").at("eos_token_id").get<uint32_t>();
        gguf_set_val_u32(gg, "tokenizer.ggml.eos_token_id", eos);
        gguf_set_val_u32(gg, "tokenizer.ggml.eot_token_id", eos);
        gguf_set_val_u32(gg, "tokenizer.ggml.padding_token_id", 199999);
        gguf_set_val_bool(gg, "tokenizer.ggml.add_bos_token", false);
        gguf_set_val_bool(gg, "tokenizer.ggml.add_eos_token", false);
        gguf_set_val_bool(gg, "tokenizer.ggml.clean_spaces", false);
        gguf_set_val_u32(gg, "inkling2.embedding_length", config.at("hidden_size").get<uint32_t>());
        llama_model_kv_override overrides[2] = {};
        snprintf(overrides[0].key, sizeof overrides[0].key, "general.architecture");
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        snprintf(overrides[0].val_str, sizeof overrides[0].val_str, "llama");
        params.kv_overrides = overrides;
        gguf_set_val_str(gg, "tokenizer.chat_template", manifest.at("chat_template").get_ref<const std::string &>().c_str());
        params.vocab_only = true;
        return llama_model_init_from_user(gg, nullptr, nullptr, params);
    } catch (const std::exception & e) {
        err = path + ": " + e.what();
        return nullptr;
    }
}
