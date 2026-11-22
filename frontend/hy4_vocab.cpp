#include "hy4_vocab.hpp"

#include "gguf.h"
#include "nlohmann/json.hpp"
#include "strata/artifact/strata_reader.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

// llama.cpp tokenizes the request and renders the original template. The child maps the weights.
llama_model * load_hy4_vocab(const std::string & path, llama_model_params params, std::string & err) {
    try {
        strata::artifact::StrataFileReader reader;
        if (!reader.open(path, err)) {
            return nullptr;
        }
        const auto manifest = json::parse(reader.manifest_json());
        const auto & config = manifest.at("config").at("text_config");
        if (config.at("model_type") != "hy_v4_flash") {
            throw std::runtime_error("expected a Hy4-Flash text checkpoint");
        }
        const auto tok = json::parse(manifest.at("tokenizer_json").get_ref<const std::string &>());
        const auto & bpe = tok.at("model");
        if (bpe.at("type") != "BPE" || bpe.value("byte_fallback", false) || bpe.value("ignore_merges", false)) {
            throw std::runtime_error("unsupported Hy4 tokenizer");
        }

        std::vector<std::string> tokens(config.at("vocab_size").get<size_t>());
        std::vector<int32_t> types(tokens.size(), LLAMA_TOKEN_TYPE_NORMAL);
        for (const auto & [piece, id] : bpe.at("vocab").items()) {
            tokens.at(id.get<size_t>()) = piece;
        }
        for (const auto & token : tok.at("added_tokens")) {
            const size_t id = token.at("id").get<size_t>();
            const auto & piece = token.at("content").get_ref<const std::string &>();
            if (!tokens.at(id).empty() && tokens[id] != piece) {
                throw std::runtime_error("conflicting Hy4 vocabulary entries");
            }
            tokens[id] = piece;
            types[id] = token.at("special").get<bool>() ? LLAMA_TOKEN_TYPE_CONTROL : LLAMA_TOKEN_TYPE_USER_DEFINED;
        }
        std::vector<const char *> pieces;
        for (const auto & piece : tokens) {
            if (piece.empty()) {
                throw std::runtime_error("incomplete Hy4 vocabulary");
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
        // The pinned llama.cpp has the HyV4 tokenizer. Its vocabulary-only path skips model hparams.
        gguf_set_val_str(gg, "general.architecture", "hy_v4");
        gguf_set_val_str(gg, "general.name", "Hy4-Flash");
        gguf_set_val_str(gg, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(gg, "tokenizer.ggml.pre", "hy_v4");
        gguf_set_arr_str(gg, "tokenizer.ggml.tokens", pieces.data(), pieces.size());
        gguf_set_arr_data(gg, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        gguf_set_arr_str(gg, "tokenizer.ggml.merges", rules.data(), rules.size());
        gguf_set_val_u32(gg, "tokenizer.ggml.bos_token_id", config.at("bos_token_id").get<uint32_t>());
        gguf_set_val_u32(gg, "tokenizer.ggml.eos_token_id", config.at("eos_token_id").get<uint32_t>());
        gguf_set_val_u32(gg, "tokenizer.ggml.padding_token_id", config.at("pad_token_id").get<uint32_t>());
        gguf_set_val_u32(gg, "tokenizer.ggml.eot_token_id", 120008); // also terminates the upstream runner
        gguf_set_val_bool(gg, "tokenizer.ggml.add_bos_token", false);
        gguf_set_val_str(gg, "tokenizer.chat_template", manifest.at("chat_template").get_ref<const std::string &>().c_str());
        params.vocab_only = true;
        return llama_model_init_from_user(gg, nullptr, nullptr, params);
    } catch (const std::exception & e) {
        err = path + ": " + e.what();
        return nullptr;
    }
}
