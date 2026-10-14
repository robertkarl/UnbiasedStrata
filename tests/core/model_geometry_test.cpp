// tests/core/model_geometry_test.cpp - `ModelGeometry` from a model file's own metadata (no model, no GPU).
//
//   model_geometry_test <dir>        the synthetic qwen35moe GGUFs tools/qwen36_synth_gguf.py wrote into <dir>
//                                    (CMake's qwen36_synth_fixtures test): every field of the toy geometry, the
//                                    nextn block subtracted from the layer count, the fused forms refused by name
//   model_geometry_test --real FILE  a real file: prints the geometry and, for Qwen3.6-35B-A3B, asserts the values
//                                    of docs/qwen36-port/gguf-metadata.md
//
// The negative cases: a `qwen4exp` file read with Flash-Next's `hc = 4` and PLE, an unknown architecture refused
// with its name, a file whose `ssm.inner_size` contradicts its head counts, and the two fused tensor layouts.
#include "gguf_fixture.hpp"

#include "strata/core/model_geometry.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::ModelArch;
using strata::core::ModelGeometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-model-geometry-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

bool read(const std::string& path, ModelGeometry& g, std::string& err) {
    return strata::core::read_model_geometry(path, g, err);
}

bool forms_ok(const std::string& path, const ModelGeometry& g, std::string& err) {
    const strata::GgufModel m = strata::GgufModel::open(path);
    return strata::core::check_model_tensor_forms(m, g, err);
}

// ---- the toy geometry tools/qwen36_synth_gguf.py writes by default (its docstring)
void toy(const fs::path& dir) {
    ModelGeometry g;
    std::string err;
    const std::string plain = (dir / "toy.gguf").string();
    const bool ok = read(plain, g, err);
    check(ok, "toy.gguf reads" + (ok ? std::string() : ": " + err));
    if (!ok) return;
    check(g.arch == ModelArch::Qwen35Moe, "arch qwen35moe");
    check(g.n_embd == 64, "n_embd 64 (embedding_length)");
    check(g.n_vocab == 128, "n_vocab 128 (output.weight / token_embd.weight rows, the 128 tokens)");
    check(g.n_layers == 4 && g.nextn_layers == 1, "n_layers 4 = block_count 5 - nextn_predict_layers 1");
    check(g.qsa_interval == 4 && g.n_qsa_layers() == 1 && g.n_gdn_layers() == 3,
          "full_attention_interval 4: layers 0,1,2 GDN, layer 3 attention");
    check(strata::core::is_qsa_layer(g, 3) && !strata::core::is_qsa_layer(g, 2), "is_qsa_layer: (il+1) % 4 == 0");
    check(g.ssm_state_size == 16 && g.ssm_k_heads == 4 && g.ssm_v_heads == 8 && g.ssm_d_conv == 4,
          "ssm.state_size 16, group_count 4 (k-heads), time_step_rank 8 (v-heads), conv_kernel 4");
    check(g.ssm_value_dim == 128 && g.ssm_conv_channels == 2 * 4 * 16 + 128,
          "derived: value dim 8*16 = 128 (= ssm.inner_size), conv channels 2*4*16 + 128 = 256");
    check(g.n_head == 2 && g.n_head_kv == 1 && g.head_dim == 32, "attention.head_count 2, head_count_kv 1, key_length 32");
    check(g.n_rot == 8 && g.rope_sections[0] == 2 && g.rope_sections[1] == 1 && g.rope_sections[2] == 1 &&
              g.rope_sections[3] == 0,
          "rope.dimension_count 8 (0.25 * 32), dimension_sections [2,1,1,0]");
    check(std::fabs(g.rope_base - 10000.0) < 1e-6, "rope.freq_base 1e4 (the toy's; the real model is 1e7)");
    check(std::fabs(g.rms_eps - 1e-6) < 1e-12, "attention.layer_norm_rms_epsilon 1e-6");
    check(g.n_expert == 8 && g.n_expert_used == 2, "expert_count 8, expert_used_count 2");
    check(g.n_ff == 32 && g.n_ff_shexp == 32, "expert_feed_forward_length 32, expert_shared_feed_forward_length 32");
    check(g.hc == 1 && g.hc_dim() == g.n_embd, "hc 1 (no hyper-connections): hc_dim == n_embd");
    check(!g.has_ple && !g.has_indexer() && g.idx_q_heads == 0, "no PLE, no indexer");
    check(forms_ok(plain, g, err), "no fused tensor form: " + err);

    const std::string nonextn = (dir / "toy-nonextn.gguf").string();
    ModelGeometry g2;
    check(read(nonextn, g2, err) && g2.n_layers == 4 && g2.nextn_layers == 0,
          "toy-nonextn.gguf: block_count 4, no nextn_predict_layers key -> 4 trunk layers, 0 draft blocks");
    check(forms_ok(nonextn, g2, err), "toy-nonextn.gguf: no fused tensor form");

    const std::string fgu = (dir / "toy-fused-gate-up.gguf").string();
    ModelGeometry g3;
    check(read(fgu, g3, err), "toy-fused-gate-up.gguf: the metadata reads (the refusal is by tensor name)");
    const bool fgu_refused = !forms_ok(fgu, g3, err);
    check(fgu_refused && err.find("blk.0.ffn_gate_up_exps.weight") == 0,
          "toy-fused-gate-up.gguf: refused by name: " + err.substr(0, 60));

    const std::string fqkv = (dir / "toy-fused-qkv.gguf").string();
    ModelGeometry g4;
    check(read(fqkv, g4, err), "toy-fused-qkv.gguf: the metadata reads");
    const bool fqkv_refused = !forms_ok(fqkv, g4, err);
    check(fqkv_refused && err.find("blk.3.attn_qkv.weight") == 0,
          "toy-fused-qkv.gguf: attn_qkv in the attention layer refused by name (blk.3), the GDN layers' attn_qkv "
          "accepted: " + err.substr(0, 40));
}

// ---- Flash-Next's file (the six guard keys, as gguf_fixture writes them): the defaults stay, hc 4, PLE
void qwen4exp() {
    TempDir d;
    const fs::path p = d.path / "flash.gguf";
    fixture::write(p, {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
                       fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
                       fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
                       fixture::u32("qwen4exp.attention.head_count_kv", 2)},
                   {});
    ModelGeometry g;
    std::string err;
    const bool ok = read(p.string(), g, err);
    check(ok, "a qwen4exp file reads" + (ok ? std::string() : ": " + err));
    check(g.arch == ModelArch::Qwen4Exp && g.hc == 4 && g.hc_lr == 320 && g.has_ple && g.has_indexer(),
          "qwen4exp: hc 4, hc_lr 320, PLE and the indexer");
    check(g.n_layers == 48 && g.n_embd == 2560 && g.n_expert == 512 && g.n_expert_used == 10 && g.n_head == 24,
          "qwen4exp: the six keys");
    check(g.ssm_v_heads == 48 && g.ssm_conv_channels == 10240 && g.ssm_value_dim == 6144 && g.n_ff == 640,
          "qwen4exp: the GDN and expert widths keep Flash-Next's defaults when the file has no key for them");
    check(strata::check_architecture(strata::GgufFile(p.string())).empty(), "qwen4exp passes the architecture guard");

    // the real Flash-Next key set (the Q2 file's metadata: ssm.*, key_length, the interval, the expert widths, the
    // rope keys): the reader's optional reads and its four refusals run on the working model's values and must land
    // on the defaults field by field - this is the one path that could refuse the model the engine already serves
    const fs::path full = d.path / "flash-full.gguf";
    // (block_count 48 with no draft block: the guard's qwen4exp branch wants the literal 48, as it always did)
    fixture::write(full, {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
                          fixture::u32("qwen4exp.embedding_length", 2560),
                          fixture::u32("qwen4exp.expert_count", 512), fixture::u32("qwen4exp.expert_used_count", 10),
                          fixture::u32("qwen4exp.attention.head_count", 24), fixture::u32("qwen4exp.attention.head_count_kv", 2),
                          fixture::u32("qwen4exp.attention.key_length", 256), fixture::u32("qwen4exp.full_attention_interval", 4),
                          fixture::u32("qwen4exp.ssm.conv_kernel", 4), fixture::u32("qwen4exp.ssm.state_size", 128),
                          fixture::u32("qwen4exp.ssm.group_count", 16), fixture::u32("qwen4exp.ssm.time_step_rank", 48),
                          fixture::u32("qwen4exp.ssm.inner_size", 6144), fixture::u32("qwen4exp.expert_feed_forward_length", 640),
                          fixture::u32("qwen4exp.expert_shared_feed_forward_length", 640),
                          fixture::u32("qwen4exp.rope.dimension_count", 64), fixture::u32s("qwen4exp.rope.dimension_sections", {11, 11, 10, 0}),
                          fixture::f32("qwen4exp.rope.freq_base", 10000000.0f),
                          fixture::f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f)},
                   {});
    ModelGeometry gf;
    const bool okf = read(full.string(), gf, err);
    check(okf, "a qwen4exp file with the full key set reads" + (okf ? std::string() : ": " + err));
    const ModelGeometry def;   // the struct's defaults are Flash-Next's
    check(gf.arch == def.arch && gf.n_embd == def.n_embd && gf.n_layers == def.n_layers && gf.nextn_layers == 0 &&
              gf.qsa_interval == def.qsa_interval && gf.n_vocab == def.n_vocab,
          "full key set: n_embd, 48 trunk layers, the interval, n_vocab equal the defaults");
    check(gf.ssm_state_size == def.ssm_state_size && gf.ssm_k_heads == def.ssm_k_heads && gf.ssm_v_heads == def.ssm_v_heads &&
              gf.ssm_d_conv == def.ssm_d_conv && gf.ssm_conv_channels == def.ssm_conv_channels &&
              gf.ssm_value_dim == def.ssm_value_dim,
          "full key set: the GDN fields and the derived conv channels / value dim equal the defaults");
    check(gf.n_head == def.n_head && gf.n_head_kv == def.n_head_kv && gf.head_dim == def.head_dim && gf.n_rot == def.n_rot &&
              gf.rope_sections[0] == 11 && gf.rope_sections[1] == 11 && gf.rope_sections[2] == 10 && gf.rope_sections[3] == 0 &&
              gf.rope_base == def.rope_base && std::fabs(gf.rms_eps - def.rms_eps) < 1e-12,
          "full key set: the attention and rope fields equal the defaults");
    check(gf.idx_q_heads == def.idx_q_heads && gf.idx_key_dim == def.idx_key_dim && gf.hc == def.hc && gf.hc_lr == def.hc_lr &&
              gf.n_expert == def.n_expert && gf.n_expert_used == def.n_expert_used && gf.n_ff == def.n_ff &&
              gf.n_ff_shexp == def.n_ff_shexp && gf.has_ple == def.has_ple && gf.has_indexer(),
          "full key set: the indexer, hc, the MoE widths and the PLE equal the defaults");
    const std::string guard_full = strata::check_architecture(strata::GgufFile(full.string()));
    check(guard_full.empty(), "the full-key file passes the guard" + (guard_full.empty() ? std::string() : ": " + guard_full));

    // the guard and the reader refuse an unknown architecture by its name
    const fs::path q = d.path / "other.gguf";
    fixture::write(q, {fixture::str("general.architecture", "llama"), fixture::u32("llama.block_count", 32)}, {});
    const std::string guard = strata::check_architecture(strata::GgufFile(q.string()));
    check(guard.find("'llama'") != std::string::npos, "guard: 'llama' refused by name: " + guard);
    check(!read(q.string(), g, err) && err.find("'llama'") != std::string::npos, "reader: 'llama' refused by name");

    // the vocabulary is read from the tensors; two statements of it that disagree are refused naming both
    const fs::path vq = d.path / "vocab.gguf";
    fixture::write(vq, {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
                        fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
                        fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
                        fixture::u32("qwen4exp.attention.head_count_kv", 2)},
                   {fixture::Tensor{"output.weight", {64, 100}, 0, 1}, fixture::Tensor{"token_embd.weight", {64, 128}, 0, 2}});
    ModelGeometry gv;
    check(!read(vq.string(), gv, err) && err.find("output.weight says 100 tokens but token_embd.weight says 128") != std::string::npos,
          "output.weight 100 rows against token_embd.weight 128: refused naming both: " + err.substr(0, 70));
    const fs::path vo = d.path / "vocab-ok.gguf";
    fixture::write(vo, {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
                        fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
                        fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
                        fixture::u32("qwen4exp.attention.head_count_kv", 2)},
                   {fixture::Tensor{"output.weight", {64, 100}, 0, 1}});
    check(read(vo.string(), gv, err) && gv.n_vocab == 100, "output.weight alone sets n_vocab 100");

    // a qwen35moe file with a key llama.cpp's loader requires missing: the guard names it
    const fs::path r = d.path / "short.gguf";
    fixture::write(r, {fixture::str("general.architecture", "qwen35moe"), fixture::u32("qwen35moe.block_count", 40)}, {});
    const std::string g2 = strata::check_architecture(strata::GgufFile(r.string()));
    check(g2.find("missing qwen35moe.embedding_length") == 0, "guard: a short qwen35moe file: " + g2);

    // a qwen35moe file whose ssm.inner_size contradicts state_size * time_step_rank
    const fs::path t = d.path / "inner.gguf";
    fixture::write(t, {fixture::str("general.architecture", "qwen35moe"), fixture::u32("qwen35moe.block_count", 4),
                       fixture::u32("qwen35moe.embedding_length", 64), fixture::u32("qwen35moe.expert_count", 8),
                       fixture::u32("qwen35moe.expert_used_count", 2), fixture::u32("qwen35moe.attention.head_count", 2),
                       fixture::u32("qwen35moe.attention.head_count_kv", 1), fixture::u32("qwen35moe.attention.key_length", 32),
                       fixture::u32("qwen35moe.full_attention_interval", 4), fixture::u32("qwen35moe.ssm.conv_kernel", 4),
                       fixture::u32("qwen35moe.ssm.state_size", 16), fixture::u32("qwen35moe.ssm.group_count", 4),
                       fixture::u32("qwen35moe.ssm.time_step_rank", 8), fixture::u32("qwen35moe.ssm.inner_size", 100),
                       fixture::u32("qwen35moe.expert_feed_forward_length", 32)},
                   {});
    check(!read(t.string(), g, err) && err.find("ssm.inner_size is 100") != std::string::npos,
          "reader: ssm.inner_size 100 against 16 * 8 refused: " + err);
}

int real(const std::string& path) {
    ModelGeometry g;
    std::string err;
    if (!read(path, g, err)) {
        std::printf("  refused: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s", strata::core::model_geometry_text(g).c_str());
    try {
        const std::string guard = strata::check_architecture(strata::GgufFile(path));
        std::printf("  architecture guard: %s\n", guard.empty() ? "passes" : guard.c_str());
    } catch (const std::exception& e) {
        std::printf("  architecture guard: %s\n", e.what());
    }
    try {
        if (!forms_ok(path, g, err)) std::printf("  tensor forms: %s\n", err.c_str());
        else std::printf("  tensor forms: separate experts, separate attention projections\n");
    } catch (const std::exception& e) {
        std::printf("  tensor forms: %s\n", e.what());
    }
    if (g.arch == ModelArch::Qwen35Moe && g.n_embd == 2048) {
        // Qwen3.6-35B-A3B exactly (docs/qwen36-port/gguf-metadata.md, metadata table)
        check(g.n_layers == 40 && (g.nextn_layers == 0 || g.nextn_layers == 1), "40 trunk layers (41 blocks in the MTP files)");
        check(g.n_qsa_layers() == 10 && g.n_gdn_layers() == 30, "10 attention layers (3, 7, ..., 39), 30 GDN");
        check(g.ssm_state_size == 128 && g.ssm_k_heads == 16 && g.ssm_v_heads == 32 && g.ssm_d_conv == 4,
              "GDN: state 128, 16 k-heads, 32 v-heads, conv 4");
        check(g.ssm_conv_channels == 8192 && g.ssm_value_dim == 4096, "conv channels 8192 = 2*2048 + 4096, value dim 4096");
        check(g.n_head == 16 && g.n_head_kv == 2 && g.head_dim == 256, "attention 16 heads / 2 KV / 256");
        check(g.n_rot == 64 && g.rope_sections[0] == 11 && g.rope_sections[1] == 11 && g.rope_sections[2] == 10 &&
                  g.rope_sections[3] == 0 && std::fabs(g.rope_base - 1e7) < 1.0,
              "rope 64 of 256, sections [11, 11, 10, 0], base 1e7");
        check(g.n_expert == 256 && g.n_expert_used == 8 && g.n_ff == 512 && g.n_ff_shexp == 512,
              "256 experts x top-8, ff 512, shared ff 512");
        check(g.hc == 1 && !g.has_ple && !g.has_indexer(), "plain residual, no PLE, no indexer");
        check(std::fabs(g.rms_eps - 1e-6) < 1e-9, "rms eps 1e-6");
        check(g.n_vocab == 248320, "n_vocab 248320 (output.weight's rows, the tokenizer's list)");
    }
    if (g.arch == ModelArch::Qwen35Moe) std::printf("  gate: %s\n", strata::core::qwen35moe_not_yet(g).c_str());
    return g_fail ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real(argv[2]);
    if (argc != 2) {
        std::printf("usage: model_geometry_test <dir of tools/qwen36_synth_gguf.py fixtures> | --real FILE.gguf\n");
        return 2;
    }
    if (fs::exists(fs::path(argv[1]) / "SKIP")) {   // the fixture step found no numpy / gguf-py
        std::printf("model_geometry_test: the synthetic fixtures were not written (python3 with numpy and gguf-py "
                    "is needed) - skipped\n");
        return 3;
    }
    std::printf("model_geometry_test\n");
    toy(argv[1]);
    qwen4exp();
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
