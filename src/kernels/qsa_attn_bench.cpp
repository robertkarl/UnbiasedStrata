// src/kernels/qsa_attn_bench.cpp - what the split-K decode attention costs per call (GPU, synthetic, no model).
//
//   qsa_attn_bench [--heads N,KV] [--kv fp16|int8|q4_0] [--cells C1,C2,...] [--reps R]
//
// One query row over `cells` cells (the identity selection, width = cells) through `qsa_decode_attn_step` on random
// pools behind a shuffled page table; the time is the median of `reps` back-to-back calls bracketed by events.
// Uses only the entry points the kernel had before Qwen3.6 step 3 (explicit ids), so the same file builds against
// the old kernel and the new one: the before / after numbers of docs/qwen36-port/step3-attn.md come from it.
// Exit 77 without a device.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 256), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
uint16_t f2h(float f) {   // round-to-nearest fp16 bits, host side
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int exp = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t man = u & 0x7fffffu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t) exp << 10) | (man >> 13);
    if ((man & 0x1fffu) > 0x1000u || ((man & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
}  // namespace

int main(int argc, char** argv) {
    int64_t n_head = 24, n_kv = 2;
    std::string kv = "fp16";
    std::vector<int64_t> cells = {2051};
    int reps = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--heads") { if (std::sscanf(next(), "%lld,%lld", (long long*) &n_head, (long long*) &n_kv) != 2) return 2; }
        else if (a == "--kv") kv = next();
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--cells") {
            cells.clear();
            std::string v = next();
            for (size_t p = 0; p < v.size();) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); cells.push_back(std::atoll(v.substr(p, q - p).c_str())); p = q + 1; }
        } else { std::fprintf(stderr, "usage: qsa_attn_bench [--heads N,KV] [--kv fp16|int8|q4_0] [--cells C,...] [--reps R]\n"); return 2; }
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("SKIP: no CUDA device\n"); return 77; }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = n_head; s.n_head_kv = n_kv;
    const int64_t HD = s.head_dim, PS = s.page_size;
    const int64_t max_cells = *std::max_element(cells.begin(), cells.end());
    const int64_t pages = (max_cells + PS - 1) / PS, rows = pages * n_kv * PS;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    k::QsaAttnPools pl;
    uint64_t bytes_per_cell = 0;
    if (kv == "fp16") {
        std::vector<uint16_t> kh((size_t) rows * HD), vh((size_t) rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 0.5f);
        for (auto& x : vh) x = f2h(nd(rng));
        pl.k_pool = up(kh); pl.v_pool = up(vh);
        bytes_per_cell = (uint64_t) n_kv * HD * 2 * 2;
    } else if (kv == "int8") {
        std::uniform_int_distribution<int> code(-127, 127);
        std::vector<int8_t> kq((size_t) rows * HD), vq((size_t) rows * HD);
        std::vector<uint16_t> ks((size_t) rows * (HD / k::KV_Q8_GROUP), f2h(0.01f)), vs = ks;
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs);
        bytes_per_cell = k::kv_q8_bytes_per_cell(s);
    } else if (kv == "q4_0") {
        std::uniform_int_distribution<int> byte(0, 255);
        const uint64_t ROW = k::kv_q4_bytes_per_head((int) HD);
        std::vector<uint8_t> k4((size_t) rows * ROW), v4((size_t) rows * ROW);
        for (auto* pool : {&k4, &v4})
            for (int64_t b = 0; b < rows * (HD / k::QK4_0); ++b) {
                const uint16_t d = f2h(0.05f);
                std::memcpy(pool->data() + b * sizeof(k::block_q4_0), &d, 2);
                for (int j = 0; j < 16; ++j) (*pool)[b * sizeof(k::block_q4_0) + 2 + j] = (uint8_t) byte(rng);
            }
        pl.k_q4 = up(k4); pl.v_q4 = up(v4);
        bytes_per_cell = k::kv_q4_bytes_per_cell(s);
    } else { std::fprintf(stderr, "--kv must be fp16, int8 or q4_0\n"); return 2; }
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    pl.page_table = up(table);
    std::vector<int32_t> ids((size_t) max_cells);
    for (int64_t i = 0; i < max_cells; ++i) ids[i] = (int32_t) i;
    const int32_t* d_ids = up(ids);
    std::vector<float> q((size_t) n_head * HD);
    for (auto& x : q) x = nd(rng);
    const float* d_q = up(q);
    float* d_attn = nullptr;
    ck(cudaMalloc(&d_attn, (size_t) n_head * HD * 4), "malloc");
    int32_t* d_step = nullptr;
    ck(cudaMalloc(&d_step, 64), "malloc");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event");
    std::printf("qsa_attn_bench on %s (sm_%d%d): %lld heads / %lld KV heads, %s KV, page_size %lld, %d reps\n",
                prop.name, prop.major, prop.minor, (long long) n_head, (long long) n_kv, kv.c_str(), (long long) PS, reps);
    std::printf("  %10s %12s %12s %12s\n", "cells", "us/call", "scratch MiB", "KV GB/s");
    for (const int64_t c : cells) {
        const int64_t cap = c;
        const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
        float* d_scr = nullptr;
        ck(cudaMalloc(&d_scr, scr * 4 + 256), "malloc scratch");
        const int32_t step[4] = {(int32_t) (c - 1), (int32_t) c, (int32_t) (c / 4), (int32_t) c};
        ck(cudaMemcpy(d_step, step, 16, cudaMemcpyHostToDevice), "step");
        for (int i = 0; i < 3; ++i) k::qsa_decode_attn_step(d_q, pl, d_ids, d_step, cap, s, d_scr, d_attn, nullptr);
        ck(cudaDeviceSynchronize(), "warm");
        std::vector<float> us;
        for (int r = 0; r < reps; ++r) {
            ck(cudaEventRecord(e0), "rec");
            k::qsa_decode_attn_step(d_q, pl, d_ids, d_step, cap, s, d_scr, d_attn, nullptr);
            ck(cudaEventRecord(e1), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0.f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            us.push_back(ms * 1000.f);
        }
        std::sort(us.begin(), us.end());
        const float med = us[us.size() / 2];
        std::printf("  %10lld %12.1f %12.2f %12.1f\n", (long long) c, (double) med, (double) scr * 4 / 1048576.0,
                    (double) c * (double) bytes_per_cell / (med * 1e-6) / 1e9);
        cudaFree(d_scr);
    }
    return 0;
}
