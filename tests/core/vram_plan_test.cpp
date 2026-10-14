// tests/core/vram_plan_test.cpp - the VRAM planner's arithmetic against docs/QWEN36-35B-A3B.md table 4.3 (no GPU).
//
// The inputs are the study's fixed set (table 4.1: weights 2,038 / 1,597 MiB, drafter ~520, KV 1,440 MiB at
// 262,144 as q4_0 over 10 attention layers = 5,760 B per token, session ~164, CUDA context ~300, the 300 MiB small
// reserve, prompt buffers 1,904 / 1,878 MiB when prefill does not borrow) and the census of the two files
// (tests/data/native_experts/qwen36-*.txt: 19,635,634,176 and 11,108,614,144 B over 40 x 256 experts, the "1.829 /
// 1.035 MiB mean blob").  Each ladder row is checked two ways: exactly against the derivation
// floor((budget - fixed) / blob), and within the table's rounding (its values are rounded to tens) of the number
// the study printed.  Then the rule's other branches: the context lowered toward 131,072 when the request does
// not fit, the refusal below the floor naming the IQ3 file, the RAM tier at the replay's 9.5 GB, and the disk
// tier that remains on the 1080 Ti.
#include "strata/plan/vram_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using strata::plan::VramPlan;
using strata::plan::VramPlanInput;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr uint64_t MiB = 1u << 20;
constexpr int64_t kExperts = 40 * 256;
constexpr uint64_t kQ4Total = 19635634176ull, kIq3Total = 11108614144ull;   // the two files' expert bytes (census)
constexpr uint64_t kQ4Blob = kQ4Total / kExperts, kIq3Blob = kIq3Total / kExperts;
constexpr uint64_t kKvPerToken = 5760;   // q4_0 K and V, 2 KV heads x 256, 10 layers: 1,440 MiB at 262,144

VramPlanInput arm(bool q4, bool borrow, uint64_t budget_mib) {
    VramPlanInput in;
    in.budget_bytes = budget_mib * MiB;
    in.weights_bytes = (q4 ? 2038 : 1597) * MiB;
    in.drafter_bytes = 520 * MiB;
    in.kv_bytes_per_token = kKvPerToken;
    in.session_bytes = 164 * MiB;
    in.cuda_context_bytes = 300 * MiB;
    in.reserve_bytes = 300 * MiB;
    in.prompt_bytes = borrow ? 0 : (q4 ? 1904 : 1878) * MiB;
    in.context = 262144;
    in.context_floor = 131072;
    in.n_experts = kExperts;
    in.mean_blob = q4 ? kQ4Blob : kIq3Blob;
    in.ram_available_bytes = 0;
    in.smaller_file = "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf";
    return in;
}

/// The derivation the table was made by: floor((budget - fixed) / blob), capped at every expert; -1 = no fit.
int64_t derive(const VramPlanInput& in) {
    const uint64_t fixed = strata::plan::vram_plan_fixed(in, in.context);
    if (fixed > in.budget_bytes) return -1;
    const int64_t s = (int64_t) ((in.budget_bytes - fixed) / in.mean_blob);
    return s < in.n_experts ? s : in.n_experts;
}

struct Row { const char* label; uint64_t budget_mib; int64_t q4_noborrow, q4_borrow, iq3_borrow; };
constexpr int64_t ALL = kExperts, NOFIT = -1;
// table 4.3's "derived" columns (the study's rounding: to tens; "all" is every expert)
const Row rows[] = {
    {"6 GB (5,722 MiB)", 5722, NOFIT, 525, 1350},
    {"8 GB (7,629 MiB)", 7629, 525, 1570, 3200},
    {"11 GB decimal (10,490 MiB)", 10490, 2090, 3130, 5960},
    {"1080 Ti (11,264 MiB)", 11264, 2510, 3550, 6700},
    {"12 GB (11,444 MiB)", 11444, 2610, 3650, 6880},
    {"16 GB (15,258 MiB)", 15258, 4700, 5740, ALL},
    {"24 GB (24,576 MiB)", 24576, 9790, ALL, ALL},
    {"32 GB (30,517 MiB)", 30517, ALL, ALL, ALL},
};

void ladder() {
    for (const Row& r : rows) {
        struct Col { const char* name; bool q4, borrow; int64_t table; } cols[] = {
            {"Q4 no borrow", true, false, r.q4_noborrow},
            {"Q4 borrow", true, true, r.q4_borrow},
            {"IQ3 borrow", false, true, r.iq3_borrow},
        };
        for (const Col& c : cols) {
            const VramPlanInput in = arm(c.q4, c.borrow, r.budget_mib);
            const VramPlan p = strata::plan::make_vram_plan(in);
            const int64_t d = derive(in);
            char what[256];
            if (c.table == NOFIT) {
                // the fixed set does not fit at 262,144; nor at the floor (6 GB at Q4 without borrowing)
                std::snprintf(what, sizeof what, "%s, %s: does not fit at 262,144 nor at 131,072 (table: 0)", r.label, c.name);
                check(!p.fits && d == NOFIT && p.refusal.find("Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf") != std::string::npos &&
                          p.refusal.find("131072") != std::string::npos,
                      what);
                continue;
            }
            const int64_t got = p.vram_slots;
            const int64_t diff = got > c.table ? got - c.table : c.table - got;
            std::snprintf(what, sizeof what, "%s, %s: %lld slots (derivation %lld, table %s%lld)", r.label, c.name,
                          (long long) got, (long long) d, c.table == ALL ? "all = " : "~", (long long) c.table);
            // exact against the derivation; the printed table rounds to tens (one cell, the 1080 Ti at IQ3, to the
            // hundred: 6,711 printed as ~6,700), so the figure it printed is within 15 of the slot count
            check(p.fits && !p.lowered && p.context == 262144 && got == d && diff <= 15, what);
            if (c.table == ALL) {
                std::snprintf(what, sizeof what, "%s, %s: every expert resident, spare %llu MiB", r.label, c.name,
                              (unsigned long long) (p.spare_bytes >> 20));
                check(p.vram_slots == kExperts && p.mapped == 0 && p.spare_bytes == (in.budget_bytes - p.fixed_bytes) - kExperts * in.mean_blob, what);
            }
        }
    }
    // the 24 GB Q4-borrow row's spare: the table says ~1,090 MiB
    const VramPlan p24 = strata::plan::make_vram_plan(arm(true, true, 24576));
    check(p24.spare_bytes / MiB >= 1080 && p24.spare_bytes / MiB <= 1095,
          "24 GB, Q4 borrow: ~1,090 MiB spare (" + std::to_string(p24.spare_bytes / MiB) + ")");
}

// Qwen3.6 step 4 (step4-drafter.md section 7): the drafter's term is its weights, its K/V over the WHOLE context in
// the --kv format, the draft head subset with the logits, and the identity ids; the step-2 line (weights + 16 MiB)
// undercounted it.  The per-token part lowers with the context.
void drafter() {
    VramPlanInput in = arm(true, true, 24576);
    in.drafter_bytes = 506 * MiB;
    in.drafter_kv_bytes_per_token = 2 * 2 * 256 * 2;      // fp16 K and V, 2 KV heads x 256: 1,024 B per cell
    in.drafter_head_bytes = 106299ull * 2176 + 8ull * 248320 * 4;   // the Q8_0 head's subset (221 MiB) + logits
    in.drafter_ids_bytes_per_token = 4 * 4;              // --spec 4
    const uint64_t drafter = strata::plan::vram_plan_drafter(in, 262144);
    check(drafter / MiB >= 506 + 512 + 221 + 7 + 4 && drafter / MiB <= 506 + 512 + 229 + 4 + 2,
          "24 GB, Q4 borrow, --mtp at 262,144 fp16: the drafter is weights 506 + K/V 512 + head 221 + logits 7.6 + ids 4 MiB (" +
          std::to_string(drafter / MiB) + ")");
    const VramPlan p = strata::plan::make_vram_plan(in);
    const VramPlan p0 = strata::plan::make_vram_plan(arm(true, true, 24576));
    check(p.fits && p.fixed_bytes == p0.fixed_bytes - 520 * MiB + drafter && p.vram_slots == derive(in),
          "the fixed set grows by the drafter's whole share and the slots derive from it (" + std::to_string(p.vram_slots) +
          " vs " + std::to_string(p0.vram_slots) + " at the step-2 line)");
    // the lowering counts the drafter's per-token K/V and ids: a budget that holds the fixed set at 0 tokens plus
    // exactly 200,000 tokens' worth of per-token bytes lowers to 196,608 (the step), not to what the trunk's KV alone
    // would give
    VramPlanInput low = in;
    low.budget_bytes = strata::plan::vram_plan_fixed(in, 0) + strata::plan::vram_plan_per_token(in) * 200000;
    const VramPlan pl = strata::plan::make_vram_plan(low);
    check(pl.fits && pl.lowered && pl.context == 196608,
          "a budget 200,000 tokens of trunk KV + drafter K/V + ids above the fixed set lowers to 196,608 (" +
          std::to_string(pl.context) + ")");
    VramPlanInput none = arm(true, true, 24576);
    none.drafter_bytes = 0;
    check(strata::plan::vram_plan_text(none, strata::plan::make_vram_plan(none)).find("(no --mtp)") != std::string::npos,
          "without --mtp the plan says so and counts no drafter");
}

void tiers() {
    // the RAM tier at the replay's 9.5 GB usable (section 4.3: Q4 ~4,950 experts, IQ3 ~8,760), the 1080 Ti's VRAM
    // slots, and the disk tier that remains at Q4 (~1,740 with borrowing, ~2,780 without)
    VramPlanInput q4 = arm(true, true, 11264);
    q4.ram_available_bytes = 9500000000ull;   // 9.5 GB decimal, as the replay
    q4.ram_headroom_bytes = 0;
    const VramPlan p = strata::plan::make_vram_plan(q4);
    check(p.ram_copies >= 4940 && p.ram_copies <= 4960, "1080 Ti, Q4 borrow, 9.5 GB RAM: ~4,950 RAM copies (" + std::to_string(p.ram_copies) + ")");
    check(p.mapped >= 1730 && p.mapped <= 1750 && p.vram_slots + p.ram_copies + p.mapped == kExperts,
          "1080 Ti, Q4 borrow: ~1,740 experts stay on disk (" + std::to_string(p.mapped) + "), the three tiers sum to 10,240");
    VramPlanInput q4n = arm(true, false, 11264);
    q4n.ram_available_bytes = 9500000000ull;
    const VramPlan pn = strata::plan::make_vram_plan(q4n);
    check(pn.mapped >= 2770 && pn.mapped <= 2790, "1080 Ti, Q4 no borrow: ~2,780 on disk (" + std::to_string(pn.mapped) + ")");
    VramPlanInput iq3 = arm(false, true, 11264);
    iq3.ram_available_bytes = 9500000000ull;
    const VramPlan pi = strata::plan::make_vram_plan(iq3);
    check(pi.mapped == 0 && pi.ram_copies == kExperts - pi.vram_slots && pi.ram_copies < 8760,
          "1080 Ti, IQ3: no disk tier (RAM would hold ~8,760, " + std::to_string(kExperts - pi.vram_slots) + " are needed)");
    // the headroom comes off the RAM before anything is placed
    VramPlanInput h = arm(true, true, 11264);
    h.ram_available_bytes = 12ull << 30;
    h.ram_headroom_bytes = 4ull << 30;
    const VramPlan ph = strata::plan::make_vram_plan(h);
    check(ph.ram_copies == (int64_t) ((8ull << 30) / kQ4Blob), "12 GiB available minus 4 GiB headroom: the tier is sized from 8 GiB");
}

void lowering() {
    // Q4 without borrowing on a card between the floor and the request: the context is lowered, a multiple of 4,096,
    // never below 131,072, and the plan says so
    const VramPlanInput in = arm(true, false, 6400);
    const VramPlan p = strata::plan::make_vram_plan(in);
    const uint64_t without_kv = strata::plan::vram_plan_fixed(in, 0);
    const int64_t expect = (int64_t) (((6400 * MiB - without_kv) / kKvPerToken) / 4096 * 4096);
    check(p.fits && p.lowered && p.context == expect && p.context >= 131072 && p.context < 262144 && p.context % 4096 == 0,
          "Q4 no borrow at 6,400 MiB: context lowered to " + std::to_string(p.context) + " (a multiple of 4,096 at or above the floor)");
    check(strata::plan::vram_plan_fixed(in, p.context) <= in.budget_bytes &&
              strata::plan::vram_plan_fixed(in, p.context + 4096) > in.budget_bytes,
          "the lowered context is the largest step that fits");
    const std::string text = strata::plan::vram_plan_text(in, p);
    check(text.find("LOWERED") != std::string::npos && text.find("131072") != std::string::npos, "the log says LOWERED and names the floor");
    // IQ3 without borrowing on 6 GB: the floor fits (6,200 - 720 = 5,480 MiB), 262,144 does not
    const VramPlanInput i6 = arm(false, false, 5722);
    const VramPlan p6 = strata::plan::make_vram_plan(i6);
    check(p6.fits && p6.lowered && p6.context >= 131072 && p6.vram_slots > 0 && p6.vram_slots == (int64_t) ((5722 * MiB - strata::plan::vram_plan_fixed(i6, p6.context)) / kIq3Blob),
          "IQ3 no borrow on 6 GB: lowered to " + std::to_string(p6.context) + " with " + std::to_string(p6.vram_slots) + " slots");
    // the refusal: the floor itself does not fit, the numbers and the smaller file are in the message, nothing is chosen
    const VramPlanInput r = arm(true, false, 5722);
    const VramPlan pr = strata::plan::make_vram_plan(r);
    check(!pr.fits && pr.vram_slots == 0 && pr.context_needed_bytes_short > 0 &&
              pr.refusal.find("MiB short even at the floor") != std::string::npos &&
              pr.refusal.find("Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf") != std::string::npos &&
              pr.refusal.find("expert profile") != std::string::npos,
          "Q4 no borrow on 6 GB: refused with the shortfall at the floor, the IQ3 file and the borrowing hint");
    const std::string rt = strata::plan::vram_plan_text(r, pr);
    check(rt.find("REFUSED") != std::string::npos, "the log says REFUSED");
    // --vram-cap below the fixed set: the same refusal (the engine clamps the budget; the rule does the rest)
    const VramPlanInput cap = arm(true, true, 4000);
    check(!strata::plan::make_vram_plan(cap).fits, "a 4,000 MiB budget (a --vram-cap below the fixed set) refuses");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--at") {
        // vram_plan_test --at <q4|iq3> <budget MiB>: print the plan for one budget (prefill borrowing)
        const VramPlanInput in = arm(std::string(argv[2]) == "q4", true, (uint64_t) std::atoll(argv[3]));
        std::printf("%s", strata::plan::vram_plan_text(in, strata::plan::make_vram_plan(in)).c_str());
        return 0;
    }
    std::printf("vram_plan_test (docs/QWEN36-35B-A3B.md table 4.3)\n");
    ladder();
    drafter();
    tiers();
    lowering();
    std::printf(g_fail ? "vram_plan_test: %d FAILED\n" : "vram_plan_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
