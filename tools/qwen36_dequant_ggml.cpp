// tools/qwen36_dequant_ggml.cpp - dequantize a slab of one GGUF tensor with ggml's own to_float, to check ref/load.py (gguf-py).
//   qwen36-dequant MODEL.gguf TENSOR FIRST_ELEMENT COUNT OUT.f32     (FIRST_ELEMENT and COUNT multiples of the block size)
// Build: c++ -std=c++17 -O1 tools/qwen36_dequant_ggml.cpp -o $B/qwen36-dequant -I$LL/ggml/include -L$B/bin -lggml -lggml-base -Wl,-rpath,$B/bin
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    gguf_init_params ip = { true, nullptr };
    gguf_context * g = gguf_init_from_file(argv[1], ip);
    if (!g) return 1;
    int64_t id = gguf_find_tensor(g, argv[2]);
    if (id < 0) { fprintf(stderr, "no tensor\n"); return 1; }
    ggml_type t = gguf_get_tensor_type(g, id);
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    int64_t first = atoll(argv[3]), n = atoll(argv[4]);
    if (first % tr->blck_size || n % tr->blck_size) { fprintf(stderr, "not block aligned\n"); return 1; }
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id) + (size_t)(first / tr->blck_size) * tr->type_size;
    size_t nb = (size_t)(n / tr->blck_size) * tr->type_size;
    std::vector<char> raw(nb);
    FILE * f = fopen(argv[1], "rb"); fseeko(f, off, SEEK_SET);
    if (fread(raw.data(), 1, nb, f) != nb) return 1;
    fclose(f);
    std::vector<float> out(n);
    if (tr->to_float) tr->to_float(raw.data(), out.data(), n); else return 2;
    f = fopen(argv[5], "wb"); fwrite(out.data(), 4, n, f); fclose(f);
    printf("%s %s %s\n", argv[2], ggml_type_name(t), "ok");
}
