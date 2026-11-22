#include "../../src/hy4flash/run/hy4_sampling.h"

#include <cstdlib>
#include <iostream>
#include <numeric>

using namespace hy4;

static void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

int main() {
    constexpr int vocab = 512;
    std::vector<float> logits(vocab, 0.f);
    std::vector<int32_t> ids(vocab);
    std::iota(ids.rbegin(), ids.rend(), 0);
    SampleOpts s;
    s.temperature = 0.9f;
    SparseDist target, draft;
    for (int k : {0, 20, 64, 100, 700}) {
        s.top_k = k;
        compute_dist_full(logits.data(), vocab, s, {}, target);
        compute_dist_cand(ids.data(), logits.data(), vocab, s, {}, draft);
        const int size = k == 0 ? vocab : std::min(k, vocab);
        check(target.size() == size && draft.size() == size, "support size");
        for (int i = 0; i < vocab; ++i) {
            const double expected = i < size ? 1.0 / size : 0.0;
            check(std::abs(dist_prob_of(target, i) - expected) < 1e-12, "target probability");
            check(std::abs(dist_prob_of(draft, i) - expected) < 1e-12, "draft probability");
        }
    }
    s.top_k = 0;
    s.top_p = 0.5f;
    compute_dist_full(logits.data(), vocab, s, {}, target);
    check(target.size() == vocab / 2, "nucleus mass");
    s.top_p = 1.f;
    s.min_p = 0.1f;
    s.freq = 5.f;
    logits[417] = 9.f;
    logits[416] = 4.f;
    compute_dist_full(logits.data(), vocab, s, {417}, target);
    check(target.size() == 2 && target[0].first == 416 && target[1].first == 417, "history and min_p");
    s.min_p = 0.f;
    std::fill(logits.begin(), logits.end(), 0.f);
    compute_dist_full(logits.data(), vocab, s, {}, target);
    check(subtract_draft(target, {{0, 0.75}, {1, 0.25}}, vocab), "residual mass");
    check(target[0].second == 0 && target[1].second == 0, "rejected tokens");
    check(std::abs(target.back().second - 1.0 / (vocab - 2)) < 1e-12, "residual normalization");
}
