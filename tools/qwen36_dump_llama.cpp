// tools/qwen36_dump_llama.cpp - run llama.cpp's own graph on a GGUF and dump every named tensor, in full.
//
// `llama-eval-callback` prints only 3 elements per edge and a float32 sum at 4 decimals.  This does the same
// thing (a cb_eval callback on every node of the pinned submodule's graph) but writes each tensor named by
// llama.cpp's cb() - "attn_norm-0", "ffn_moe_topk-3", "result_output" - to <out>/<name>.f32 in full, in ggml
// memory order (ne0 fastest), I32 tensors converted to float.  A repeated name (llama.cpp names "Kcur" twice
// in the attention graph) gets "#2", "#3" for the later occurrences.  <out>/manifest.txt lists
// "<file-stem> <ggml type> <ne0>,<ne1>,<ne2>,<ne3>" in the order the nodes ran.  Same format as ref/qwen36.py.
//
// Build against the submodule (CPU only), outside the repo:
//   cmake -S third_party/llama.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF \
//         -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF && cmake --build $B --target llama -j8
//   c++ -std=c++17 -O1 tools/qwen36_dump_llama.cpp -o $B/qwen36-dump \
//       -Ithird_party/llama.cpp/include -Ithird_party/llama.cpp/ggml/include \
//       -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin
// Run: $B/qwen36-dump MODEL.gguf OUTDIR 104,101,108,108,111 [CHUNK [mtp]]
//   CHUNK > 0 decodes the tokens CHUNK at a time (1 = the autoregressive path) and dumps only the last decode call,
//   so the recurrent state and KV cache carried from the earlier calls are what is being checked.
//   "mtp" loads the nextn block and, after the trunk pass, runs llama.cpp's MTP graph on (tokens[1:], trunk hidden
//   states[:-1]) at positions 1.. - the pairing common/speculative.cpp builds (token k with the hidden of k-1) -
//   and dumps it to OUTDIR_mtp.  Needs CHUNK = 0.
//
// KV cache is F32 and flash attention is off, so on an F32 file the only difference to the numpy oracle is float
// summation order.  Threads: 1, or $QWEN36_THREADS.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>

struct sink {
    std::string dir;
    std::map<std::string, int> seen;
    FILE * manifest = nullptr;
    std::vector<uint8_t> buf;
    bool active = true;                  // only the last decode call is dumped when chunking
    std::vector<float> h_nextn;          // the trunk's post-output-norm hidden states, for the MTP pass
};

static bool keep(const char * n) {
    if (!n[0]) return false;
    if (strncmp(n, "node_", 5) == 0 || strncmp(n, "leaf_", 5) == 0) return false;
    if (strstr(n, " (") != nullptr) return false;       // "x (view)", "x (reshaped)", "x (copy of y)"
    if (strncmp(n, "cache_", 6) == 0) return false;
    return true;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    if (ask) return true;
    auto * s = (sink *) ud;
    if (!s->active) return true;
    if (!keep(t->name) || ggml_is_quantized(t->type) || ggml_nelements(t) == 0) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32 && t->type != GGML_TYPE_F16) return true;

    std::string stem = t->name;
    int n = ++s->seen[stem];
    if (n > 1) stem += "#" + std::to_string(n);

    const size_t nb = ggml_nbytes(t);
    const bool host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data = (const uint8_t *) t->data;
    if (!host) { s->buf.resize(nb); ggml_backend_tensor_get(t, s->buf.data(), 0, nb); data = s->buf.data(); }

    std::vector<float> out((size_t) ggml_nelements(t));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
        const uint8_t * p = data + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
        float v;
        if (t->type == GGML_TYPE_F32)      v = *(const float *) p;
        else if (t->type == GGML_TYPE_I32) v = (float) *(const int32_t *) p;
        else                               v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        out[k++] = v;
    }
    if (stem == "h_nextn") s->h_nextn = out;
    std::string path = s->dir + "/" + stem + ".f32";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    fprintf(s->manifest, "%s %s %lld,%lld,%lld,%lld\n", stem.c_str(), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

static sink * open_sink(const std::string & dir) {
    auto * s = new sink;
    s->dir = dir;
    mkdir(dir.c_str(), 0755);
    s->manifest = fopen((dir + "/manifest.txt").c_str(), "w");
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL.gguf OUTDIR tok,tok,... [CHUNK [mtp]]\n", argv[0]); return 1; }
    std::vector<llama_token> toks;
    for (char * p = strtok(argv[3], ","); p; p = strtok(nullptr, ",")) toks.push_back((llama_token) atoi(p));
    const int chunk = argc > 4 ? atoi(argv[4]) : 0;
    const bool want_mtp = argc > 5 && strcmp(argv[5], "mtp") == 0;
    if (want_mtp && chunk != 0) { fprintf(stderr, "mtp needs CHUNK = 0\n"); return 1; }

    sink * s = open_sink(argv[2]);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = want_mtp;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 256;
    const char * th = getenv("QWEN36_THREADS");
    cp.n_threads = cp.n_threads_batch = th ? atoi(th) : 1;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = s;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    const int n = (int) toks.size();
    const int step = chunk > 0 ? chunk : n;
    for (int i = 0; i < n; i += step) {
        const int m = std::min(step, n - i);
        s->active = (i + m >= n);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, m))) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    fclose(s->manifest);

    if (want_mtp) {
        const int n_embd = llama_model_n_embd(model);
        sink * s2 = open_sink(std::string(argv[2]) + "_mtp");
        llama_context_params cp2 = cp;
        cp2.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp2.cb_eval_user_data = s2;
        llama_context * mctx = llama_init_from_model(model, cp2);
        if (!mctx) { fprintf(stderr, "mtp context init failed\n"); return 1; }
        const int m = n - 1;
        std::vector<llama_token> mt(toks.begin() + 1, toks.end());
        std::vector<float> h(s->h_nextn.begin(), s->h_nextn.begin() + (size_t) m * n_embd);
        std::vector<llama_pos> pos(m);
        std::vector<int32_t> nsid(m, 1);
        std::vector<llama_seq_id> sid0(1, 0);
        std::vector<llama_seq_id *> sid(m, sid0.data());
        std::vector<int8_t> lg(m, 1);
        for (int i = 0; i < m; i++) pos[i] = i + 1;
        llama_batch b = {};
        b.n_tokens = m; b.token = mt.data(); b.embd = h.data(); b.pos = pos.data();
        b.n_seq_id = nsid.data(); b.seq_id = sid.data(); b.logits = lg.data();
        if (llama_decode(mctx, b)) { fprintf(stderr, "mtp decode failed\n"); return 1; }
        fclose(s2->manifest);
        llama_free(mctx);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
