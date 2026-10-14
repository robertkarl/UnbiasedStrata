// tools/qwen36_teacher_llama.cpp - llama.cpp's own greedy continuation of a prompt, with the head's logits at every
// generated position and the routed experts at probed layers: the reference the Qwen3.6 runner is held to on the real
// files (docs/qwen36-port/step4-runner.md; the comparison is tools/qwen36_compare_logits.py).
//
// Build against a llama.cpp build of the pinned submodule (CUDA or CPU), e.g. on VM 221:
//   c++ -std=c++17 -O1 tools/qwen36_teacher_llama.cpp -o qwen36-teacher -I$LL/include -I$LL/ggml/include \
//       $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
//       $B/ggml/src/libggml-base.a -lcudart -lcublas -lcublasLt -lcuda -lgomp -lpthread -ldl -lm
//
//   qwen36-teacher MODEL.gguf OUTDIR --prompt-file FILE [--n-gen 64] [--ngl 99] [--ctx 8192] [--probe 0,3,20,39]
//   qwen36-teacher MODEL.gguf --detok FILE        the ids in FILE (commas or whitespace) as text, to stdout
//   --prompt-ids FILE   the prompt as token ids (commas or whitespace) instead of --prompt-file's text
//   --force-ids FILE    teacher forcing: row j's logits are written, then FILE's j-th token is decoded instead of the
//                       argmax (continuation.txt still holds the argmax per row); --n-gen is capped at FILE's length
//
// Writes OUTDIR/prompt_ids.txt (the prompt's token ids, parse_special on: the ChatML tags of the regina prompts are
// tokens), OUTDIR/continuation.txt (the N greedy tokens), OUTDIR/logits.f32 (N rows of n_vocab F32: row j is the
// distribution the j-th generated token was picked from, i.e. the position n_prompt - 1 + j), OUTDIR/meta.txt
// ("n_vocab N n_prompt P n_gen G"), OUTDIR/routes.txt ("<row> <layer> e0 .. e7" for the probed layers of every
// generated row; the prompt batch is not traced) and OUTDIR/text.txt (the continuation as text).  Greedy means the
// argmax of the F32 logits; the KV cache is F16 as the engine's --kv fp16 is.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {
struct Routes {
    std::set<int> probe;
    FILE* out = nullptr;
    int row = -1;   // the generated row being decoded (-1: the prompt batch, not traced)
};
bool cb_routes(struct ggml_tensor* t, bool ask, void* ud) {
    if (ask) return true;
    auto* r = (Routes*) ud;
    if (r->row < 0 || r->out == nullptr || std::strncmp(t->name, "ffn_moe_topk-", 13) != 0) return true;
    const int layer = std::atoi(t->name + 13);
    if (!r->probe.count(layer) || t->type != GGML_TYPE_I32) return true;
    const int64_t k = t->ne[0], n = t->ne[1];
    std::vector<int32_t> ids((size_t) (k * n));
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    for (int64_t i = 0; i < n; ++i) {
        std::fprintf(r->out, "%d %d", r->row, layer);
        for (int64_t j = 0; j < k; ++j) std::fprintf(r->out, " %d", ids[(size_t) (i * k + j)]);
        std::fprintf(r->out, "\n");
    }
    return true;
}
std::vector<llama_token> read_ids(const std::string& path) {
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream is(text);
    std::vector<llama_token> v;
    for (long long x; is >> x;) v.push_back((llama_token) x);
    return v;
}
std::string piece(const llama_vocab* vocab, llama_token t) {
    char buf[512];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTDIR --prompt-file FILE [--n-gen N] [--ngl N] [--ctx N] [--probe L,L,..]\n"
                             "       %s MODEL.gguf --detok FILE\n", argv[0], argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    std::string outdir, prompt_file, prompt_ids, force_ids, detok;
    int n_gen = 64, ngl = 99, n_ctx = 8192;
    std::set<int> probe = {0, 3, 20, 39};
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); } return argv[++i]; };
        if (a == "--prompt-file") prompt_file = next();
        else if (a == "--prompt-ids") prompt_ids = next();
        else if (a == "--force-ids") force_ids = next();
        else if (a == "--n-gen") n_gen = std::atoi(next());
        else if (a == "--ngl") ngl = std::atoi(next());
        else if (a == "--ctx") n_ctx = std::atoi(next());
        else if (a == "--detok") detok = next();
        else if (a == "--probe") {
            probe.clear();
            std::string s = next();
            for (char& c : s) if (c == ',') c = ' ';
            std::istringstream is(s);
            for (int l; is >> l;) probe.insert(l);
        } else if (outdir.empty()) outdir = a;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = detok.empty() ? ngl : 0;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!detok.empty()) {
        for (llama_token t : read_ids(detok)) std::fputs(piece(vocab, t).c_str(), stdout);
        std::fputc('\n', stdout);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    if (outdir.empty() || (prompt_file.empty() && prompt_ids.empty())) { std::fprintf(stderr, "OUTDIR and --prompt-file or --prompt-ids are required\n"); return 1; }
    mkdir(outdir.c_str(), 0755);
    std::vector<llama_token> toks;
    int n = 0;
    if (!prompt_ids.empty()) {
        toks = read_ids(prompt_ids);
        n = (int) toks.size();
        if (n <= 0) { std::fprintf(stderr, "no ids in %s\n", prompt_ids.c_str()); return 1; }
        prompt_file = prompt_ids;
    } else {
        std::ifstream pf(prompt_file);
        std::string prompt((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        while (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();   // the files end in one newline the prompt does not
        toks.resize(prompt.size() + 16);
        n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), false, true);
        if (n < 0) { toks.resize((size_t) -n); n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), false, true); }
        if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
        toks.resize((size_t) n);
    }
    std::vector<llama_token> forced;
    if (!force_ids.empty()) {
        forced = read_ids(force_ids);
        if (forced.empty()) { std::fprintf(stderr, "no ids in %s\n", force_ids.c_str()); return 1; }
        n_gen = std::min(n_gen, (int) forced.size());
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);

    Routes routes;
    routes.probe = probe;
    routes.out = std::fopen((outdir + "/routes.txt").c_str(), "w");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = cp.n_ubatch = 2048;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.cb_eval = cb_routes;
    cp.cb_eval_user_data = &routes;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    // the prompt, logits for its last position only
    routes.row = -1;
    for (int i = 0; i < n; i += 2048) {
        const int m = std::min(2048, n - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, m))) { std::fprintf(stderr, "prompt decode failed\n"); return 1; }
    }
    FILE* lf = std::fopen((outdir + "/logits.f32").c_str(), "wb");
    std::vector<llama_token> gen;
    std::string text;
    for (int j = 0; j < n_gen; ++j) {
        const float* lg = llama_get_logits_ith(ctx, -1);
        if (lg == nullptr) { std::fprintf(stderr, "no logits at row %d\n", j); return 1; }
        std::fwrite(lg, sizeof(float), (size_t) n_vocab, lf);
        const llama_token t = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        gen.push_back(t);
        text += piece(vocab, t);
        if (j + 1 == n_gen) break;
        routes.row = j + 1;   // the next decode's logits are row j + 1
        llama_token one = forced.empty() ? t : forced[(size_t) j];
        if (llama_decode(ctx, llama_batch_get_one(&one, 1))) { std::fprintf(stderr, "decode failed at row %d\n", j + 1); return 1; }
    }
    std::fclose(lf);
    std::fclose(routes.out);
    {
        std::ofstream o(outdir + "/prompt_ids.txt");
        for (size_t i = 0; i < toks.size(); ++i) o << (i ? "," : "") << toks[i];
        o << "\n";
        std::ofstream c(outdir + "/continuation.txt");
        for (size_t i = 0; i < gen.size(); ++i) c << (i ? "," : "") << gen[i];
        c << "\n";
        std::ofstream m(outdir + "/meta.txt");
        m << "n_vocab " << n_vocab << " n_prompt " << n << " n_gen " << gen.size() << "\n";
        std::ofstream tx(outdir + "/text.txt");
        tx << text << "\n";
    }
    std::printf("%s: %d prompt tokens, %zu greedy tokens, logits %zu x %d in %s\n", prompt_file.c_str(), n, gen.size(),
                gen.size(), n_vocab, outdir.c_str());
    std::printf("---\n%s\n---\n", text.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
