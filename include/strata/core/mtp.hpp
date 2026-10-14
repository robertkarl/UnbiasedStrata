// include/strata/core/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768);
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    uint64_t vram_bytes() const { return vram_; }
    /// The draft layer's K/V state (read-only: --serve's STRATA_STATE_HASH check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// KV streaming: refill the ring of the drafter's window from its host copy for a sequence that continues at
    /// `upto` (a conversation-cache resume). No-op unless the drafter's K/V is a ring.
    void kv_restore(int64_t upto);
    /// The VRAM bind() will allocate for a native head of `head_row_bytes` per vocabulary row: the draft logits and
    /// the draft head over rt/draft_vocab.bin's subset.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const;
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err);

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// COUPLED DRAFT SAMPLING (core/coupled_draft.hpp; STRATA_SPEC_COUPLED=1, set up by bind()): the request's
    /// sampling.  A sampled request (temperature > 0, not greedy) then drafts by SAMPLING with the target's chain and
    /// the target's Philox draw for the verifying row, and `probs` is the draft's probability under that chain; a
    /// greedy one keeps the argmax drafts and their graphs.  A no-op when the switch is off.
    void set_draft_sampling(const strata::kernels::SamplerParams& sp);
    /// Coupled mode with penalties: the history the next draft() chain starts from - what the session holds after the
    /// window's commit (`tail`) and the window's pick at its last accepted row (`next`, the next window's row 0).
    void set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next);
    void set_ple_session(SessionState* s) { ple_ss_ = s; }
    bool coupled() const { return coupled_active_; }

    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

    /// E-9: the prompt path computes this layer's prompt K/V in batches (Prefill::draft_kv): its tensors, its K/V
    /// state, the first cell a round can still read, its device, and a wait for its own stream.
    const float* tensor_f32(const char* name) const { return f32(name); }
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    /// A projection of the pack in its GGUF form (Qwen3.6's pack keeps every projection as stored: `nextn.eh_proj.weight`,
    /// `attn_k.weight`, ...): its blocks and ggml type (`*type`), null when the name is not a native projection.
    const void* tensor_native(const char* name, int* type) const;

    // ---- QWEN3.6 (qwen35moe; docs/qwen36-port/step4-drafter.md).  The drafter is the -MTP file's nextn block through
    // the plain graph: concat(rms_norm(embed(tok)) * enorm, rms_norm(h) * hnorm) (embedding first), eh_proj, one plain
    // attention + MoE block over the drafter's own dense KV, shared_head_norm, the main head.  `h` is the trunk's
    // OUTPUT-NORMED hidden state (llama.cpp's h_nextn): the drafter applies `output_norm` to the residual rows it is
    // handed, so the caller passes the same final residual it passes for Flash-Next.  Pairing: token k with the hidden
    // state of position k-1 (common/speculative.cpp), the cell convention `prefill` / `draft` already use.  The second
    // path is chosen by the geometry's architecture; the Flash-Next path is unchanged.
    bool qwen36() const;
    /// The harness's one-row forward, outside any graph: the row's seams by llama.cpp's cb() names (`mtp_tok_embd`,
    /// `mtp_enorm`, `mtp_hnorm`, `mtp_concat`, `mtp_eh_proj`, `mtp_attn_norm`, `mtp_Vcur`, `mtp_Kcur_normed` (before
    /// RoPE), `mtp_Qcur_full`, `mtp_Qcur_normed` (before RoPE), `mtp_attn_pregate`, `mtp_attn_out`, `mtp_attn_residual`,
    /// `mtp_attn_post_norm`, `ffn_moe_logits`, `ffn_moe_topk` (int32 ids behind the float pointer), `ffn_moe_weights_norm`,
    /// `mtp_ffn_shexp_gated`, `mtp_ffn_out`, `mtp_post_ffn`, `mtp_shared_head_norm`, `result_output`), each handed to
    /// `seam` after a stream sync with the device pointer and the element count; `top1` gets the draft (a vocabulary
    /// id).  `R_row` is the trunk's final residual (device), the K/V is written at `cell`, RoPE at `pos`.  The rival
    /// readings of `DebugOpts` are the harness's negative controls and nothing else sets them.
    struct DebugOpts {
        bool hidden_first = false;   ///< concat(hnorm, enorm): the wrong order
        bool skip_hnorm = false;     ///< the hidden state's second norm (hnorm) left out
    };
    using SeamFn = std::function<void(const char* seam, const float* dev, int64_t n)>;
    bool debug_row(const float* R_row, int32_t token, int64_t cell, int32_t pos, const DebugOpts& o, const SeamFn& seam,
                   int32_t* top1, std::string& err);
    /// The draft block's expert blob (Qwen3.6): the native layout's types and bytes, for a reader of `experts.bin`.
    int expert_gu_type() const { return exp_gu_type_; }
    int expert_d_type() const { return exp_d_type_; }
    int64_t expert_blob_bytes() const { return exp_blob_; }
    QsaState& kv_state_rw() { return st_; }
    int64_t first_needed() const { return (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0; }
    int device() const { return device_; }
    bool idle(std::string& err) {
        if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
        return true;
    }

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    bool record_forward_qwen36(int T, int step_row0, cudaStream_t cs, std::string& err);
    bool load_qwen36_experts(const std::string& rt_dir, const ModelGeometry& g, uint64_t& bytes, std::string& err);
    bool check_qwen36_pack(const std::string& rt_dir, const ModelGeometry& g, std::string& err);
    int exp_gu_type_ = -1, exp_d_type_ = -1;   ///< Qwen3.6: the draft experts' GGUF types (native_expert_layout)
    int64_t exp_blob_ = 0;                     ///< Qwen3.6: bytes per expert blob in experts_
    float* cat_ = nullptr;                     ///< Qwen3.6: concat(enorm, hnorm), T x 2 n_embd
    const DebugOpts* dbg_ = nullptr;           ///< set by debug_row only
    const SeamFn* seam_ = nullptr;             ///< set by debug_row only
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< E-4: without the mapped staging (inputs copied on device)
    bool capture_round(int T, bool coupled, std::string& err);
    bool capture_step(int j, bool coupled, std::string& err);
    cudaGraphExec_t step_exec_[9] = {};
    // coupled draft sampling: its own round/step graphs (the argmax ones stay as they were), the request's
    // parameters and the penalty ring (mapped staging + device copies), the split scratch, token id -> subset index
    bool setup_coupled(std::string& err);
    cudaGraphExec_t round_exec_c_[9] = {};
    cudaGraphExec_t step_exec_c_[9] = {};
    bool coupled_ok_ = false, coupled_active_ = false;
    bool coupled_rec_ = false;   ///< record_forward: the full layer ends in the coupled sampler (draft coupled_j_)
    int coupled_j_ = 0;
    strata::kernels::SamplerParams *h_cparams_ = nullptr, *m_cparams_ = nullptr, *cparams_ = nullptr;
    int32_t *h_chist_ = nullptr, *m_chist_ = nullptr, *cring_ = nullptr, *dinv_ = nullptr;
    void* cscratch_ = nullptr;
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    SessionState* ple_ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int max_drafts_ = 1 << 30;
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t prefill_dev_exec_[9] = {};
    int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaGraphExec_t round_exec_[9] = {};

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; int type = -1; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
    float* probs_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
