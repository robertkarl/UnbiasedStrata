// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"
#include "strata/kernels/gdn.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// Which architecture the model file declares.  The engine was written for ONE model (Qwen3.8-Flash-Next,
/// `qwen4exp`); Qwen3.6-35B-A3B (`qwen35moe`) is the same family one generation earlier with the hyper-connection
/// residual, the sparse-attention indexer and the PLE table removed.  The string decides what the file cannot say:
/// `hc`, the indexer and the PLE (docs/QWEN36-35B-A3B.md section 3.8).
enum class ModelArch { Qwen4Exp, Qwen35Moe };

inline const char* model_arch_name(ModelArch a) { return a == ModelArch::Qwen4Exp ? "qwen4exp" : "qwen35moe"; }

/// The model's geometry.  Every field here is a number a kernel depends on, so a change is a change to a kernel
/// contract and not a tuning knob.  The defaults are Qwen3.8-Flash-Next's (`docs/semantics.md`); a model file
/// overrides every one of them through `read_model_geometry` (strata/core/model_geometry.hpp), which reads the
/// `<arch>.*` keys of the GGUF and derives the dependent members below with `derive()`.
struct ModelGeometry {
    ModelArch arch = ModelArch::Qwen4Exp;

    int64_t n_embd = 2560;
    int64_t n_layers = 48;         ///< trunk layers: `block_count` minus the draft (`nextn_predict_layers`) blocks
    int64_t nextn_layers = 0;      ///< MTP draft blocks at the end of the file (0 when the key is absent)
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... (`full_attention_interval`)
    int64_t n_vocab = 248320;      ///< rows of `token_embd.weight` (both models share the tokenizer)

    // GDN (36 of 48 layers; 30 of 40 for Qwen3.6)
    int64_t ssm_state_size = 128;        ///< `ssm.state_size`: the head dim of k and v
    int64_t ssm_k_heads = 16;            ///< `ssm.group_count`
    int64_t ssm_v_heads = 48;            ///< `ssm.time_step_rank` (32 for Qwen3.6)
    int64_t ssm_d_conv = 4;              ///< `ssm.conv_kernel`
    int64_t ssm_conv_channels = 10240;   ///< DERIVED: 2*S*k_heads + S*v_heads (`derive()`); 8192 for Qwen3.6
    int64_t ssm_value_dim = 6144;        ///< DERIVED: S*v_heads (`ssm.inner_size` in the file); 4096 for Qwen3.6

    // full attention (QSA for Flash-Next: with an indexer; dense for Qwen3.6: no indexer)
    int64_t n_head = 24;                 ///< `attention.head_count` (16 for Qwen3.6)
    int64_t n_head_kv = 2;               ///< `attention.head_count_kv`
    int64_t head_dim = 256;              ///< `attention.key_length`
    int64_t n_rot = 64;                  ///< `rope.dimension_count`: the partial rotary width (0.25 * head_dim)
    int64_t rope_sections[4] = {11, 11, 10, 0};   ///< `rope.dimension_sections` (IMRoPE), sum == n_rot / 2
    double rope_base = 1e7;              ///< `rope.freq_base`
    double rms_eps = 1e-6;               ///< `attention.layer_norm_rms_epsilon`
    int64_t idx_q_heads = 4;             ///< the QSA indexer's query heads (Flash-Next only; meaningless at hc 1)
    int64_t idx_key_dim = 128;           ///< the QSA indexer's key width (Flash-Next only)

    // gated residual, on every layer.  `hc = 1` means NO hyper-connections: a plain residual with `attn_norm`,
    // `post_attention_norm` and `output_norm` (Qwen3.6).  No key says this; the architecture string does.
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;              ///< `expert_count` (256 for Qwen3.6)
    int64_t n_expert_used = 10;          ///< `expert_used_count`, the router's k (8 for Qwen3.6)
    int64_t n_ff = 640;                  ///< `expert_feed_forward_length` (512 for Qwen3.6)
    int64_t n_ff_shexp = 640;            ///< `expert_shared_feed_forward_length` (512 for Qwen3.6)

    /// The per-layer token embedding (PLE) and its n-gram table: Flash-Next has one, Qwen3.6 has none.  Decides
    /// whether `--ple-gguf` is required and whether the session carves the PLE history (strata/core/session.hpp).
    bool has_ple = true;

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
    /// Whether the full-attention layers carry the sparse indexer (QSA).  Qwen3.6's attention is dense.
    bool has_indexer() const { return arch == ModelArch::Qwen4Exp; }
    /// The GDN output gate: `rms_norm(o) * ssm_norm * act(z)` with act = sigmoid for Flash-Next and SiLU for Qwen3.6
    /// (geometry-inventory.md row 39; `kernels::GdnGate`).  From the architecture, like `hc` and the indexer: no key
    /// says it.  Every GDN output kernel takes this value from the layer; none reads a global.
    kernels::GdnGate gdn_gate() const {
        return arch == ModelArch::Qwen35Moe ? kernels::GdnGate::Silu : kernels::GdnGate::Sigmoid;
    }

    /// The members that follow from others: conv channels `2*S*k_heads + S*v_heads` and the value dim
    /// `S*v_heads`.  They are stored (the kernels read them as plain members) and derived here, so a file that
    /// changes `ssm_v_heads` cannot leave them at the defaults (geometry-inventory.md row 2).
    void derive() {
        ssm_value_dim = ssm_state_size * ssm_v_heads;
        ssm_conv_channels = 2 * ssm_state_size * ssm_k_heads + ssm_value_dim;
    }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
