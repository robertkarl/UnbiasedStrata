// The expert pool's native-pack path (run_split_multi_native: gate/up rows, the intermediate's quantization, down
// rows, each split across the workers) must use the LAYER'S widths, NativeFmt::n_ff and n_embd, not the canonical
// Q2_0 pack's FF / H (640 / 2560).  Qwen3.6-35B-A3B's experts are 2048 x 512: with FF / H the down phase wrote 512
// rows past every output row and the first verify window with a CPU-computed expert miss decoded to token 0.
//
// Three Q8_0 experts at 2048 x 512, four tokens; the pool's rows against the same kernels called directly, and a
// guard after every output row that must stay untouched.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels::cpu;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    const int64_t n_embd = 2048, n_ff = 512;   // Qwen3.6-35B-A3B's expert geometry
    const int kType = (int) GGML_TYPE_Q8_0;
    NativeFmt f;
    std::string err;
    if (!native_fmt(kType, kType, n_embd, n_ff, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 1; }
    check(f.n_embd == n_embd && f.n_ff == n_ff, "the format carries the layer's widths 2048 x 512");
    check(n_embd < H && n_ff < FF, "the layer is narrower than the canonical FF / H, so the old row counts overran");
    NativeFmt wide;
    const bool wide_ok = native_fmt(kType, kType, H + 64, n_ff, wide, err);
    check(!wide_ok, "a layer wider than the pool's buffers is refused: " + err);

    const int E = 3, T = 4;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<std::vector<uint8_t>> blobs((size_t) E);
    std::vector<float> row((size_t) std::max(n_embd, n_ff));
    for (int e = 0; e < E; ++e) {
        auto& b = blobs[(size_t) e];
        b.assign(f.bytes, 0);
        for (int part = 0; part < 3; ++part) {   // gate, up: n_ff rows of n_embd; down: n_embd rows of n_ff
            const int64_t rows = part < 2 ? n_ff : n_embd, cols = part < 2 ? n_embd : n_ff;
            std::vector<float> m((size_t) (rows * cols));
            for (float& v : m) v = nd(rng);
            uint8_t* dst = b.data() + (part == 0 ? 0 : part == 1 ? f.up_off : f.down_off);
            ggml_quantize_chunk((ggml_type) kType, m.data(), dst, 0, rows, cols, nullptr);
        }
    }
    std::vector<std::vector<float>> x((size_t) T, std::vector<float>((size_t) n_embd));
    std::vector<std::vector<uint8_t>> act((size_t) T, std::vector<uint8_t>(kNativeActBytes));
    for (int t = 0; t < T; ++t) {
        for (float& v : x[(size_t) t]) v = nd(rng);
        native_quant_act(f, x[(size_t) t].data(), act[(size_t) t].data());
    }
    // token -> expert: tokens 0,1 on expert 0; token 2 on expert 1; token 3 on expert 2 (then 1 again: two experts)
    const int routes[][2] = {{0, 0}, {1, 0}, {2, 1}, {3, 2}, {3, 1}};
    const int R = 5;

    // the reference: the kernels directly, one expert at a time, over exactly the layer's rows
    std::vector<std::vector<float>> ref((size_t) R, std::vector<float>((size_t) n_embd));
    for (int e = 0; e < E; ++e) {
        const void* nact[MAXT] = {};
        float* ffp[MAXT] = {};
        const void* hq[MAXT] = {};
        float* out[MAXT] = {};
        std::vector<std::vector<float>> ff;
        std::vector<std::vector<uint8_t>> h;
        int nt = 0;
        for (int r = 0; r < R; ++r)
            if (routes[r][1] == e) {
                ff.emplace_back((size_t) n_ff);
                h.emplace_back(kNativeHBytes);
                ++nt;
            }
        int i = 0;
        for (int r = 0; r < R; ++r)
            if (routes[r][1] == e) {
                nact[i] = act[(size_t) routes[r][0]].data();
                ffp[i] = ff[(size_t) i].data();
                hq[i] = h[(size_t) i].data();
                out[i] = ref[(size_t) r].data();
                ++i;
            }
        native_gu_rows(f, blobs[(size_t) e].data(), nact, nt, ffp, 0, (int) n_ff);
        for (int k = 0; k < nt; ++k) native_quant_h(f, ffp[k], h[(size_t) k].data());
        native_down_rows(f, blobs[(size_t) e].data(), hq, nt, out, 0, (int) n_embd);
    }

    // the pool: every output row followed by a guard of H floats (the old overrun was H - n_embd rows)
    const int64_t guard = H, pitch = n_embd + guard;
    const float sentinel = 1234.5f;
    std::vector<float> outbuf((size_t) (R * pitch), sentinel);
    ExpertJobMulti jobs[E];
    for (int e = 0; e < E; ++e) jobs[e].blob = blobs[(size_t) e].data();
    for (int r = 0; r < R; ++r) {
        ExpertJobMulti& j = jobs[routes[r][1]];
        j.nact[j.nt] = act[(size_t) routes[r][0]].data();
        j.out[j.nt] = outbuf.data() + (size_t) r * pitch;
        ++j.nt;
    }
    ExpertPool pool(2, false, true);
    pool.run_split_multi_native(f, jobs, E);

    double worst = 0;
    bool guards = true;
    for (int r = 0; r < R; ++r) {
        const float* got = outbuf.data() + (size_t) r * pitch;
        for (int64_t i = 0; i < n_embd; ++i)
            worst = std::max(worst, (double) std::fabs(got[i] - ref[(size_t) r][(size_t) i]));
        for (int64_t i = n_embd; i < pitch; ++i)
            if (got[i] != sentinel) { guards = false; break; }
    }
    check(guards, "no output row's guard was written (the down phase stopped at n_embd rows)");
    check(worst <= 1e-5, "the pool's rows equal the kernels' direct rows (worst abs diff " + std::to_string(worst) + ")");
    std::printf("%s: %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
