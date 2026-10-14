// include/strata/plan/vram_plan.hpp - the VRAM planner's arithmetic, as a pure function (Qwen3.6 port, step 2).
//
// docs/QWEN36-35B-A3B.md section 4.2, in order: the fixed set first (the non-expert weights, the drafter, the KV
// cache for the requested context, the session, the CUDA context, the reserve, the prompt buffers unless prefill
// borrows cache slots); when it does not fit, the context is lowered toward the floor (131,072), and below the
// floor the start refuses with the numbers and the smaller file's name - never a silently shorter context; every
// remaining byte becomes expert slots; then the RAM tier from the available RAM minus a headroom; whatever is left
// stays mapped from the model file on disk.  KV streaming is not a lever for this model: dense attention reads
// every cell, so the KV stays in VRAM (3.3 item 6).
//
// Header-only and GPU-free on purpose: the engine feeds it `cudaMemGetInfo` (clamped by --vram-cap) and the file's
// census, the unit test (tests/core/vram_plan_test.cpp) feeds it the study's table 4.1 and checks table 4.3, and
// `strata-plan` can print it for any budget.  The slot count is at the MEAN blob: the engine's per-layer sizing
// (generate.cpp, `sized_slots`) prices the same bytes at each profile entry's own blob and lands within a few slots
// of it; the table's ladder is at the mean, so the check is at the mean.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::plan {

struct VramPlanInput {
    uint64_t budget_bytes = 0;         ///< VRAM the engine may use: free at start, or `cap - used` under --vram-cap
    uint64_t weights_bytes = 0;        ///< the non-expert weights on the GPU (census of the file, token_embd excluded)
    uint64_t drafter_bytes = 0;        ///< the MTP draft block's weights in VRAM (dense + experts); 0 without --mtp
    /// Qwen3.6 step 4 (docs/qwen36-port/step4-drafter.md section 7): the drafter's K/V is the WHOLE context, not a
    /// ring (dense attention: no --kv-resident), in the format --kv asks for, over its one attention layer; counted
    /// per token here so a lowered context lowers it too.  0 without --mtp.
    uint64_t drafter_kv_bytes_per_token = 0;
    /// the draft head over the draft vocabulary's subset (n_dvocab x the head's row bytes) plus the draft logits:
    /// what MtpDrafter::bind allocates; 0 without --mtp
    uint64_t drafter_head_bytes = 0;
    /// the drafter's identity selection ids, T x max_cells x 4 B (T = --spec): per token; 0 without --mtp
    uint64_t drafter_ids_bytes_per_token = 0;
    uint64_t kv_bytes_per_token = 0;   ///< the KV format's bytes per token over every attention layer
    uint64_t session_bytes = 0;        ///< rope table, GDN state, scratch
    uint64_t cuda_context_bytes = 0;   ///< the driver's own share, ~230 to 300 MiB, when `budget_bytes` is a card's
                                       ///< size (the study's table); 0 when it is cudaMemGetInfo's free figure, which
                                       ///< is read after the context exists and so is already net of it
    uint64_t reserve_bytes = 0;        ///< --vram-reserve-mib (700 default, 300 on a small card)
    uint64_t prompt_bytes = 0;         ///< the prompt path's chunk buffers; 0 when prefill borrows cache slots
    int64_t context = 262144;          ///< requested
    int64_t context_floor = 131072;    ///< never below this without refusing
    int64_t context_step = 4096;       ///< a lowered context is a multiple of this
    int64_t n_experts = 0;             ///< n_layers x n_expert
    uint64_t mean_blob = 0;            ///< bytes per expert slot (the census mean)
    uint64_t ram_available_bytes = 0;  ///< host RAM the tier may take (MemAvailable, clamped by --ram-cap)
    uint64_t ram_headroom_bytes = 0;   ///< left to the OS, the page cache and the KV stream
    std::string smaller_file;          ///< named in a refusal (the IQ3_XXS file for the Q4 arm)
};

struct VramPlan {
    bool fits = false;
    int64_t context = 0;               ///< the context the plan holds (the request, or lowered)
    bool lowered = false;              ///< the context is below the request
    uint64_t kv_bytes = 0;             ///< at `context`
    uint64_t fixed_bytes = 0;          ///< everything but the expert slots, at `context`
    uint64_t expert_bytes = 0;         ///< budget - fixed (0 when it does not fit)
    int64_t vram_slots = 0;            ///< expert slots in VRAM (at most n_experts)
    uint64_t spare_bytes = 0;          ///< left over when every expert fits
    int64_t ram_copies = 0;            ///< experts the RAM tier holds (the next by profile rank)
    int64_t mapped = 0;                ///< the rest: read from the model file on a miss
    uint64_t ram_tier_bytes = 0;       ///< ram_copies x mean_blob
    int64_t context_needed_bytes_short = 0;   ///< on a refusal: how many bytes short at the floor
    std::string refusal;               ///< non-empty when !fits: the numbers and the smaller file
};

/// Everything that grows with the context: the trunk's KV and the drafter's K/V and ids, per token.
inline uint64_t vram_plan_per_token(const VramPlanInput& in) {
    return in.kv_bytes_per_token + in.drafter_kv_bytes_per_token + in.drafter_ids_bytes_per_token;
}
/// The drafter's whole share at `context` tokens: its weights, its K/V over the context, its head and ids.
inline uint64_t vram_plan_drafter(const VramPlanInput& in, int64_t context) {
    return in.drafter_bytes + in.drafter_head_bytes +
           (in.drafter_kv_bytes_per_token + in.drafter_ids_bytes_per_token) * (uint64_t) context;
}
/// The fixed set at `context` tokens.
inline uint64_t vram_plan_fixed(const VramPlanInput& in, int64_t context) {
    return in.weights_bytes + vram_plan_drafter(in, context) + in.kv_bytes_per_token * (uint64_t) context +
           in.session_bytes + in.cuda_context_bytes + in.reserve_bytes + in.prompt_bytes;
}

inline VramPlan make_vram_plan(const VramPlanInput& in) {
    VramPlan p;
    p.context = in.context;
    p.fixed_bytes = vram_plan_fixed(in, in.context);
    if (p.fixed_bytes > in.budget_bytes) {
        // lower the context toward the floor: the largest multiple of `context_step` the budget holds
        const uint64_t without_kv = vram_plan_fixed(in, 0);
        const uint64_t per_token = vram_plan_per_token(in);
        int64_t ctx = 0;
        if (in.budget_bytes > without_kv && per_token > 0) {
            ctx = (int64_t) ((in.budget_bytes - without_kv) / per_token);
            ctx -= ctx % (in.context_step > 0 ? in.context_step : 1);
        }
        if (ctx < in.context_floor) {
            const uint64_t at_floor = vram_plan_fixed(in, in.context_floor);
            p.context_needed_bytes_short = (int64_t) (at_floor > in.budget_bytes ? at_floor - in.budget_bytes : 0);
            const auto mib = [](uint64_t b) { return std::to_string((b + (1u << 20) - 1) >> 20); };
            p.refusal = "the fixed set does not fit: at " + std::to_string(in.context) + " tokens it is " +
                        mib(p.fixed_bytes) + " MiB (weights " + mib(in.weights_bytes) + ", drafter " +
                        mib(vram_plan_drafter(in, in.context)) + " (weights " + mib(in.drafter_bytes) + ", K/V " +
                        mib(in.drafter_kv_bytes_per_token * (uint64_t) in.context) + ", head " + mib(in.drafter_head_bytes) +
                        ", ids " + mib(in.drafter_ids_bytes_per_token * (uint64_t) in.context) + "), KV " +
                        mib(in.kv_bytes_per_token * (uint64_t) in.context) +
                        ", session " + mib(in.session_bytes) + ", CUDA context " + mib(in.cuda_context_bytes) +
                        ", reserve " + mib(in.reserve_bytes) + ", prompt buffers " + mib(in.prompt_bytes) +
                        "), at the floor of " + std::to_string(in.context_floor) + " tokens " + mib(at_floor) +
                        " MiB, and this card gives " + mib(in.budget_bytes) + " MiB: " +
                        mib((uint64_t) p.context_needed_bytes_short) + " MiB short even at the floor.  " +
                        (in.smaller_file.empty() ? std::string("Use a smaller quantization of the model")
                                                 : "Use the smaller file " + in.smaller_file) +
                        (in.prompt_bytes > 0 ? ", or an expert profile (prefill then borrows cache slots instead of its "
                                               "own " + mib(in.prompt_bytes) + " MiB of prompt buffers)"
                                             : std::string()) +
                        "; a context below " + std::to_string(in.context_floor) + " is not chosen for you";
            p.fits = false;
            p.context = in.context_floor;
            p.kv_bytes = in.kv_bytes_per_token * (uint64_t) in.context_floor;
            p.fixed_bytes = at_floor;
            p.mapped = in.n_experts;
            return p;
        }
        p.context = ctx;
        p.lowered = true;
        p.fixed_bytes = vram_plan_fixed(in, ctx);
    }
    p.fits = true;
    p.kv_bytes = in.kv_bytes_per_token * (uint64_t) p.context;
    p.expert_bytes = in.budget_bytes - p.fixed_bytes;
    const int64_t slots = in.mean_blob > 0 ? (int64_t) (p.expert_bytes / in.mean_blob) : 0;
    p.vram_slots = slots < in.n_experts ? slots : in.n_experts;
    if (p.vram_slots == in.n_experts) p.spare_bytes = p.expert_bytes - (uint64_t) in.n_experts * in.mean_blob;
    const int64_t rest = in.n_experts - p.vram_slots;
    const uint64_t ram_budget = in.ram_available_bytes > in.ram_headroom_bytes ? in.ram_available_bytes - in.ram_headroom_bytes : 0;
    const int64_t ram_fit = in.mean_blob > 0 ? (int64_t) (ram_budget / in.mean_blob) : 0;
    p.ram_copies = ram_fit < rest ? ram_fit : rest;
    p.mapped = rest - p.ram_copies;
    p.ram_tier_bytes = (uint64_t) p.ram_copies * in.mean_blob;
    return p;
}

/// The start-up log's lines (and `/health`'s counts): the three tiers and what they were sized from.
inline std::string vram_plan_text(const VramPlanInput& in, const VramPlan& p, const char* indent = "  ") {
    const auto mib = [](uint64_t b) { return std::to_string(b >> 20); };
    std::string s;
    s += std::string(indent) + "VRAM budget " + mib(in.budget_bytes) + " MiB\n";
    s += std::string(indent) + "fixed set at " + std::to_string(p.context) + " tokens: " + mib(p.fixed_bytes) +
         " MiB = weights " + mib(in.weights_bytes) + " + drafter " + mib(vram_plan_drafter(in, p.context)) +
         (vram_plan_drafter(in, p.context) ? " (weights " + mib(in.drafter_bytes) + ", K/V " +
                                                  mib(in.drafter_kv_bytes_per_token * (uint64_t) p.context) + ", head " +
                                                  mib(in.drafter_head_bytes) + ", ids " +
                                                  mib(in.drafter_ids_bytes_per_token * (uint64_t) p.context) + ")"
                                            : std::string(" (no --mtp)")) +
         " + KV " + mib(p.kv_bytes) +
         " + session " + mib(in.session_bytes) + " + CUDA context " + mib(in.cuda_context_bytes) + " + reserve " +
         mib(in.reserve_bytes) + (in.prompt_bytes ? " + prompt buffers " + mib(in.prompt_bytes) : std::string(" (prefill borrows cache slots)")) + "\n";
    if (!p.fits) {
        s += std::string(indent) + "REFUSED: " + p.refusal + "\n";
        return s;
    }
    if (p.lowered)
        s += std::string(indent) + "context LOWERED from " + std::to_string(in.context) + " to " + std::to_string(p.context) +
             ": the fixed set at the request would not fit (the floor is " + std::to_string(in.context_floor) + ")\n";
    s += std::string(indent) + "expert slots in VRAM: " + std::to_string(p.vram_slots) + " of " + std::to_string(in.n_experts) +
         " (" + std::to_string(in.n_experts ? 100 * p.vram_slots / in.n_experts : 0) + "%) at " + mib(p.expert_bytes) +
         " MiB, " + std::to_string(in.mean_blob) + " B mean blob" +
         (p.vram_slots == in.n_experts ? ", every expert resident, " + mib(p.spare_bytes) + " MiB spare" : std::string()) + "\n";
    s += std::string(indent) + "expert copies in RAM: " + std::to_string(p.ram_copies) + " (" + mib(p.ram_tier_bytes) +
         " MiB of " + mib(in.ram_available_bytes) + " MiB available minus " + mib(in.ram_headroom_bytes) + " headroom)\n";
    s += std::string(indent) + "experts mapped from the model file: " + std::to_string(p.mapped) +
         (p.mapped ? " (read from disk on a miss)" : "") + "\n";
    return s;
}

}  // namespace strata::plan
