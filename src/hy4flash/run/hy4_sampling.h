#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hy4 {
struct SampleOpts {
    float temperature = 0.f;  // 0 = greedy
    float top_p = 1.f;
    int top_k = 64;
    float min_p = 0.f;
    float rep = 1.f, freq = 0.f, pres = 0.f;
    int last_n = 64;
    uint64_t seed = 0;
    bool has_seed = false;
    bool greedy() const { return temperature <= 0.f; }
    bool penalties() const { return rep != 1.f || freq != 0.f || pres != 0.f; }
};

using SparseDist = std::vector<std::pair<int32_t, double>>;

inline void compute_dist_cand(const int32_t* ids, const float* scores, int N, const SampleOpts& s, const std::vector<int32_t>& hist, SparseDist& out) {
    out.clear();
    if (N <= 0) return;
    std::unordered_map<int32_t, int> count;
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t)s.last_n ? hist.size() - (size_t)s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
    }
    const int K = s.top_k <= 0 ? N : std::min(s.top_k, N);
    std::vector<std::pair<float, int32_t>> cand((size_t)N);
    for (int i = 0; i < N; ++i) {
        float x = scores[i];
        int32_t id = ids ? ids[i] : i;
        if (!count.empty()) {
            auto it = count.find(id);
            if (it != count.end()) {
                if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= (float)it->second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
            }
        }
        cand[i] = {x, id};
    }
    auto higher = [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    const auto best = *std::min_element(cand.begin(), cand.end(), higher);
    if (s.greedy()) {
        out.emplace_back(best.second, 1.0);
        return;
    }
    // With no rank-based cutoff the input order is sufficient. Avoid sorting the whole vocabulary.
    if (K < N || s.top_p < 1.f || s.min_p > 0.f || K <= 64)
        std::partial_sort(cand.begin(), cand.begin() + K, cand.end(), higher);
    std::vector<double> p(K);
    const double mx = best.first;
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp(((double)cand[i].first - mx) / (double)s.temperature);
        sum += p[i];
    }
    for (int i = 0; i < K; ++i) p[i] /= sum;
    int keep = K;
    if (s.min_p > 0.f) {
        const double thr = (double)s.min_p * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= (double)s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    out.reserve(keep);
    for (int i = 0; i < keep; ++i) out.push_back({cand[i].second, p[i] / z});
}

inline void compute_dist_full(const float* lg_in, int V, const SampleOpts& s, const std::vector<int32_t>& hist, SparseDist& out) {
    if (s.top_k <= 0 || s.top_k > 64) {
        compute_dist_cand(nullptr, lg_in, V, s, hist, out);
        return;
    }
    out.clear();
    std::unordered_map<int32_t, int> count;
    if (s.penalties() && s.last_n > 0) {
        size_t from = hist.size() > (size_t)s.last_n ? hist.size() - (size_t)s.last_n : 0;
        for (size_t i = from; i < hist.size(); ++i) ++count[hist[i]];
    }
    auto adj = [&](int id) -> float {
        float x = lg_in[id];
        if (!count.empty()) {
            auto it = count.find(id);
            if (it != count.end()) {
                if (s.rep != 1.f) x = x > 0 ? x / s.rep : x * s.rep;
                x -= (float)it->second * s.freq + (s.pres != 0.f ? s.pres : 0.f);
            }
        }
        return x;
    };
    const int K = std::min(std::max(1, s.top_k), 64);
    // Single-pass top-K maintenance avoids allocating & partial_sorting a 120K index vector on every verify row!
    std::pair<float, int> top[65];
    int ntop = 0;
    for (int i = 0; i < V; ++i) {
        float v = adj(i);
        if (ntop < K || v > top[ntop - 1].first) {
            int pos = ntop < K ? ntop++ : K - 1;
            while (pos > 0 && (v > top[pos - 1].first || (v == top[pos - 1].first && i < top[pos - 1].second))) {
                top[pos] = top[pos - 1];
                --pos;
            }
            top[pos] = {v, i};
        }
    }
    if (ntop == 0) return;
    if (s.greedy()) {
        out.push_back({top[0].second, 1.0});
        return;
    }
    double p[64];
    const double mx = top[0].first;
    double sum = 0.0;
    for (int i = 0; i < ntop; ++i) {
        p[i] = std::exp(((double)top[i].first - mx) / (double)s.temperature);
        sum += p[i];
    }
    for (int i = 0; i < ntop; ++i) p[i] /= sum;
    int keep = ntop;
    if (s.min_p > 0.f) {
        const double thr = (double)s.min_p * p[0];
        keep = 1;
        while (keep < ntop && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= (double)s.top_p) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    out.reserve(keep);
    for (int i = 0; i < keep; ++i) out.push_back({top[i].second, p[i] / z});
}

inline int32_t sample_from_dist(const SparseDist& dist, std::mt19937_64& rng) {
    if (dist.empty()) return 0;
    double z = 0.0;
    for (const auto& kv : dist) z += kv.second;
    if (z <= 0.0) return dist[0].first;
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng), c = 0.0;
    for (const auto& kv : dist) {
        c += kv.second;
        if (r <= c) return kv.first;
    }
    return dist.back().first;
}

inline double dist_prob_of(const SparseDist& dist, int32_t tok) {
    for (const auto& kv : dist)
        if (kv.first == tok) return kv.second;
    return 0.0;
}

// After rejecting a draft token, normalize the positive part of target minus draft.
inline bool subtract_draft(SparseDist& target, const SparseDist& draft, int vocab) {
    std::vector<double> indexed;
    if (target.size() > 64 && draft.size() > 1) {
        indexed.resize(vocab, 0.0);
        for (const auto& kv : draft) indexed[kv.first] = kv.second;
    }
    double total = 0.0;
    for (auto& kv : target) {
        const double q = indexed.empty() ? dist_prob_of(draft, kv.first) : indexed[kv.first];
        kv.second = std::max(0.0, kv.second - q);
        total += kv.second;
    }
    if (total <= 1e-15) return false;
    for (auto& kv : target) kv.second /= total;
    return true;
}
}  // namespace hy4
