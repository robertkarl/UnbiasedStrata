#!/usr/bin/env python3
"""ref/qwen36.py - the numpy oracle for Qwen3.6-35B-A3B (llama.cpp arch `qwen35moe`).

A teacher-forced forward pass over a token sequence, written from the pinned llama.cpp graph and nothing
else: third_party/llama.cpp/src/models/qwen35moe.cpp (the layer loop and both mixers), delta-net-base.cpp
(the Gated DeltaNet), src/llama-graph.cpp (build_moe_ffn, build_norm, build_qkv) and ggml-cpu/ops.cpp (the op
semantics).  Weights come through `ref/load.py`, so a real quantized GGUF dequantizes lazily (one expert at a
time) and the synthetic F32 fixture from `tools/qwen36_synth_gguf.py` loads the same way.

    python3 ref/qwen36.py MODEL.gguf --tokens 104,101,108,108,111 --dump OUTDIR
    python3 ref/qwen36.py MODEL.gguf --text hello --dump OUTDIR       # synthetic vocab: token id = ASCII code
    python3 ref/qwen36.py MODEL.gguf --tokens ... --dump OUTDIR --mtp  # also run the nextn block (needs one)

Every tensor llama.cpp names with cb() is written to OUTDIR/<name>.f32 (float32, C order of the numpy array,
which is ggml memory order: ne0 fastest), named exactly as llama.cpp names it ("attn_norm-0", "ffn_moe_topk-3",
"result_output"; a name set twice gets "#2", "#3" - llama.cpp names "Kcur" and "Vcur" twice).  I32 tensors
(the router's argsort and top-k) are written as float32 holding the integer.  OUTDIR/manifest.txt lists
"<stem> <f32|i32> <ne0>,<ne1>,<ne2>,<ne3>".  `tools/qwen36_dump_llama.cpp` writes the same format from the real
llama.cpp graph; `tools/qwen36_compare_dumps.py` compares them and docs/qwen36-port/oracle-crosscheck.md records the result.

Shapes.  Numpy shapes are tokens-major: a (T, n_embd) activation is ggml ne [n_embd, T].  Weights are
(out, in) so y = x @ W.T.  Per-head tensors are (T, heads, dim).

The recurrence implemented for Gated DeltaNet (one layer, one v-head h, key head hk = h % n_k_heads, state
S in R^{S_v x S_k} with S[j, i] = v index j, k index i, the layout ggml calls "state is stored transposed"):

    qkv   = Wqkv x ; z = Wgate x ; beta = sigmoid(Wbeta x) ; a = Walpha x
    g     = softplus(a + dt_bias) * ssm_a              (ssm_a < 0, so g < 0)          qwen35moe.cpp:386-397
    c     = silu(causal_conv1d(qkv))  per channel, kernel [K], zero or carried (K-1)-state     qwen35moe.cpp:402-419
    q,k,v = split(c) -> [Hk,S], [Hk,S], [Hv,S];  q,k /= sqrt(sum(x^2) + eps)   (build_gdn_l2_norm, models.h:14)
    per token t:
        S    *= exp(g[t, h])                                                  ops.cpp:11012  (decay first)
        d     = (v[t, h] - S @ k[t, hk]) * beta[t, h]                         ops.cpp:11016-11020  (delta rule)
        S    += d (x) k[t, hk]                                                ops.cpp:11023-11025
        o[t,h]= (S @ q[t, hk]) / sqrt(S_v)                                    ops.cpp:11028-11032
    out   = (rms_norm(o) * ssm_norm) * SiLU(z)    <- SiLU, not sigmoid        qwen35moe.cpp:272-275, 478
    y     = Wout out

Scope: one sequence, causal, fresh or carried state, no sliding window, no YaRN (ext_factor 0, attn_factor 1:
the context's defaults for this architecture), F32 throughout (the fixture is F32; a quantized file is
dequantized to F32 first, so this is the "what the weights say" oracle, not a bit-exact quantized path).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))
from _paths import add_gguf_py                                    # noqa: E402

add_gguf_py()
from ref.load import Gguf                                         # noqa: E402

F32 = np.float32
ARCH = "qwen35moe"


# ------------------------------------------------------------------------------------------------ hparams
@dataclass
class HParams:
    """What llama.cpp's loader derives from the GGUF (src/llama-model.cpp:1231-1424, qwen35moe.cpp:4-34)."""
    n_embd: int
    n_layer_all: int            # block_count: trunk + nextn blocks
    n_nextn: int                # nextn_predict_layers (0 when absent)
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    sections: tuple             # rope.dimension_sections, 4 entries
    rope_base: float
    eps: float
    n_expert: int
    n_expert_used: int
    n_ff_exp: int
    n_ff_shexp: int
    ssm_conv: int
    ssm_inner: int
    ssm_state: int
    ssm_k_heads: int            # ssm.group_count
    ssm_v_heads: int            # ssm.time_step_rank
    interval: int
    recurrent: tuple | None     # attention.recurrent_layers, when the file carries it
    # llama.cpp behaviours for this architecture that the GGUF cannot change:
    expert_weights_norm: bool = True      # qwen35moe.cpp:506 passes norm_w = true
    expert_weights_scale: float = 0.0     # hparams default; the qwen35moe loader never reads the key (0 -> no scaling)

    @property
    def n_layer(self) -> int:
        return self.n_layer_all - self.n_nextn

    @property
    def head_v_dim(self) -> int:
        return self.ssm_inner // self.ssm_v_heads

    @property
    def key_dim(self) -> int:
        return self.ssm_state * self.ssm_k_heads

    @property
    def value_dim(self) -> int:
        return self.head_v_dim * self.ssm_v_heads

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    def is_recr(self, il: int) -> bool:
        """qwen35moe.cpp:20-26: recurrent unless (il+1) % interval == 0; the nextn block is never recurrent."""
        if il >= self.n_layer:
            return False
        if self.recurrent is not None:
            return bool(self.recurrent[il])
        return (il + 1) % self.interval != 0

    def attn_layers(self) -> list[int]:
        return [il for il in range(self.n_layer) if not self.is_recr(il)]


def read_hparams(md: dict) -> HParams:
    assert md.get("general.architecture") == ARCH, f"architecture is {md.get('general.architecture')!r}, not {ARCH!r}"
    k = lambda s, d=None: md.get(f"{ARCH}.{s}", d)
    need = lambda s: md[f"{ARCH}.{s}"]
    n_head, n_embd = need("attention.head_count"), need("embedding_length")
    head_dim = k("attention.key_length", n_embd // n_head)          # llama-model.cpp:1371-1372
    n_head_kv = k("attention.head_count_kv", n_head)
    first = lambda v: v[0] if isinstance(v, (list, tuple)) else v   # get_key_or_arr accepts a scalar or per-layer array
    sec = list(k("rope.dimension_sections", [0, 0, 0, 0]))
    sec = tuple((sec + [0] * 4)[:4])
    rec = k("attention.recurrent_layers")
    return HParams(
        n_embd=n_embd, n_layer_all=need("block_count"), n_nextn=k("nextn_predict_layers", 0),
        n_head=first(n_head), n_head_kv=first(n_head_kv), head_dim=head_dim,
        n_rot=k("rope.dimension_count", head_dim), sections=sec,
        rope_base=k("rope.freq_base", 10000.0), eps=need("attention.layer_norm_rms_epsilon"),
        n_expert=need("expert_count"), n_expert_used=first(need("expert_used_count")),
        n_ff_exp=first(need("expert_feed_forward_length")),
        n_ff_shexp=k("expert_shared_feed_forward_length", 0),
        ssm_conv=need("ssm.conv_kernel"), ssm_inner=need("ssm.inner_size"), ssm_state=need("ssm.state_size"),
        ssm_k_heads=need("ssm.group_count"), ssm_v_heads=need("ssm.time_step_rank"),
        interval=k("full_attention_interval", 4), recurrent=tuple(rec) if rec is not None else None,
    )


# ------------------------------------------------------------------------------------------------ op semantics
def rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    """ggml_rms_norm over the last axis: x / sqrt(mean(x^2) + eps), mean accumulated in double (ops.cpp rms_norm)."""
    ms = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + eps)).astype(F32)).astype(F32)


def l2_norm_gdn(x: np.ndarray, eps: float) -> np.ndarray:
    """build_gdn_l2_norm (models.h:14-18) = scale(rms_norm(x, eps/n), 1/sqrt(n)) = x / sqrt(sum(x^2) + eps)."""
    n = x.shape[-1]
    return (rms_norm(x, eps / n) * F32(1.0 / np.sqrt(F32(n)))).astype(F32)


def silu(x):
    return (x / (F32(1.0) + np.exp(-x))).astype(F32)             # vec.h:1046-1048


def sigmoid(x):
    return (F32(1.0) / (F32(1.0) + np.exp(-x))).astype(F32)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(F32)   # unary-ops.cpp:80-82


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(F32)


def mm(x, w):
    """y = x @ w.T with w (out, in), float32."""
    return (x.astype(F32) @ w.astype(F32).T).astype(F32)


_K_FAMILY = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS")
_Q8_0_FAMILY = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL")


def quantize_act(x: np.ndarray, weight_type: str) -> np.ndarray:
    """What ggml's quantized matmul sees of an F32 activation: rows rounded to the weight type's vec_dot_type, then
    dequantized (the integer dot product equals the float dot of these values, up to summation order).
    K-quant and IQ weights use Q8_K (blocks of 256, float scale from the signed max, quantize_row_q8_K_ref);
    Q4_0/Q5_0/Q8_0/IQ4_NL use Q8_0 (blocks of 32, fp16 scale, quantize_row_q8_0_ref).  F32/F16/BF16 weights: unchanged."""
    if weight_type in _K_FAMILY:
        bs = 256
    elif weight_type in _Q8_0_FAMILY:
        bs = 32
    else:
        return x
    shp = x.shape
    b = x.reshape(-1, bs).astype(F32)
    if bs == 256:
        i = np.abs(b).argmax(axis=1)
        mx = b[np.arange(len(b)), i]                              # signed value of largest magnitude
        with np.errstate(divide="ignore", invalid="ignore"):
            iscale = np.where(mx != 0, F32(-127.0) / mx, F32(0)).astype(F32)
        q = np.minimum(127, np.rint(iscale[:, None] * b))         # nearest_int: round half to even
        d = np.where(iscale != 0, F32(1.0) / iscale, F32(0)).astype(F32)
        return (q * d[:, None]).astype(F32).reshape(shp)
    amax = np.abs(b).max(axis=1)
    d = (amax / F32(127.0)).astype(F32)
    d16 = d.astype(np.float16).astype(F32)                        # block_q8_0 stores the scale as fp16
    with np.errstate(divide="ignore", invalid="ignore"):
        idv = np.where(d != 0, F32(1.0) / d, F32(0)).astype(F32)
    q = np.sign(b * idv[:, None]) * np.floor(np.abs(b * idv[:, None]) + F32(0.5))   # roundf: half away from zero
    return (q * d16[:, None]).astype(F32).reshape(shp)


def rope_multi(x: np.ndarray, pos4: np.ndarray, n_rot: int, sections: tuple, base: float, pairs: str = "neox") -> np.ndarray:
    """ggml_rope_multi in GGML_ROPE_TYPE_IMROPE mode (ggml-cpu/ops.cpp:5991-6059, 6143-6233), ext_factor 0, freq_scale 1.

    x: (T, heads, head_dim); pos4: (4, T) = text position broadcast to t/h/w and 0 for e (llama-graph.cpp:131-141).
    Rotates the first n_rot dims as NeoX pairs (p, p + n_rot/2); the rest pass through.
    """
    T, H, D = x.shape
    half = n_rot // 2
    ts = F32(np.power(F32(base), F32(-2.0) / F32(n_rot)))        # theta_scale = powf(freq_base, -2/n_dims)
    sect = sum(sections)
    out = x.copy()
    for t in range(T):
        th = [F32(pos4[a, t]) for a in range(4)]                  # theta_t, theta_h, theta_w, theta_e
        cs = np.empty(half, F32)
        sn = np.empty(half, F32)
        for p in range(half):
            sector = p % sect
            if sector % 3 == 1 and sector < 3 * sections[1]:
                theta = th[1]
            elif sector % 3 == 2 and sector < 3 * sections[2]:
                theta = th[2]
            elif sector % 3 == 0 and sector < 3 * sections[0]:
                theta = th[0]
            else:
                theta = th[3]
            cs[p], sn[p] = np.cos(theta), np.sin(theta)           # attn_factor 1: mscale = 1
            th = [F32(v * ts) for v in th]                        # every theta advances each pair
        if pairs == "neox":                                       # rotate_pairs(n_dims, n_dims/2): dim p with dim p + n_rot/2
            x0, x1 = x[t, :, :half], x[t, :, half:n_rot]
            out[t, :, :half] = x0 * cs - x1 * sn
            out[t, :, half:n_rot] = x0 * sn + x1 * cs
        else:                                                     # the rival: adjacent pairs (2p, 2p+1), GGML_ROPE_TYPE_NORMAL
            x0, x1 = x[t, :, 0:n_rot:2], x[t, :, 1:n_rot:2]
            out[t, :, 0:n_rot:2] = x0 * cs - x1 * sn
            out[t, :, 1:n_rot:2] = x0 * sn + x1 * cs
    return out


# ------------------------------------------------------------------------------------------------ dump sink
class Dump:
    """Writes cb()-named tensors; a no-op when constructed without a directory."""

    def __init__(self, path=None):
        self.dir = pathlib.Path(path) if path else None
        self.seen: dict[str, int] = {}
        self.lines: list[str] = []
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str, il: int, arr: np.ndarray, kind: str = "f32") -> np.ndarray:
        stem = f"{name}-{il}" if il >= 0 else name
        n = self.seen[stem] = self.seen.get(stem, 0) + 1
        if n > 1:
            stem += f"#{n}"
        if self.dir:
            a = np.ascontiguousarray(arr, dtype=F32)
            ne = list(reversed(a.shape)) + [1] * (4 - a.ndim)
            a.tofile(self.dir / f"{stem}.f32")
            self.lines.append(f"{stem} {kind} {','.join(map(str, ne))}")
        return arr

    def close(self):
        if self.dir:
            (self.dir / "manifest.txt").write_text("\n".join(self.lines) + "\n")


# ------------------------------------------------------------------------------------------------ model
@dataclass
class LayerState:
    conv: np.ndarray | None = None      # GDN: (conv_dim, K-1)
    ssm: np.ndarray | None = None       # GDN: (H_v, S_v, S_k)  [v index, k index]
    k: np.ndarray | None = None         # attention: (n_kv_tokens, n_head_kv, head_dim), post-norm, post-rope
    v: np.ndarray | None = None


@dataclass
class State:
    layers: dict = field(default_factory=dict)
    n_past: int = 0


class Model:
    def __init__(self, path):
        self.g = Gguf(path)
        self.hp = read_hparams(self.g.g.metadata)
        self.names = set(self.g._by_name)
        self._cache: dict[str, np.ndarray] = {}
        self.md = self.g.g.metadata
        # The faithful reading of every choice in the graph.  Each entry has a named rival (the reading a port could
        # plausibly get wrong); tools/test_qwen36_oracle.py flips them one at a time to prove the fixture can tell.
        self.opts = dict(
            gdn_gate="silu",          # rival "sigmoid": the Flash-Next engine's gate (qwen4exp.cpp:481)
            head_pairing="tiled",     # rival "interleaved": v-head h pairs with k-head h // (Hv/Hk) instead of h % Hk
            decay="before",           # rival "after": S += d k^T on the undecayed state, then S *= exp(g)
            conv="causal",            # rival "flipped": the kernel window applied in the opposite direction
            attn_gate=True,           # rival False: skip the sigmoid output gate of the full-attention layer
            router_norm=None,         # None: llama.cpp's norm_w = true; rival False: raw softmax weights
            rope_pairs="neox",        # rival "adjacent"
            act_quant="none",         # "q8": quantize activations as ggml does before each quantized matmul (a noise model, not a rival)
            force_inputs=None,        # directory of a llama.cpp dump: every layer starts from ITS l_out-(il-1), so each layer's error is local
            force_topk=None,          # {layer: (T, K) int array}: use these experts instead of the router's own top-k
            gqa_map="grouped",        # rival "interleaved": query head h reads kv head h % n_head_kv instead of h // (n_head/n_head_kv)
        )

    # ---- tensor access (small tensors cached; experts always lazy per expert)
    def qtype(self, name):
        return self.g.info(name).type_name

    def mm(self, x, wname):
        """y = x @ W.T for the named weight; with opts["act_quant"] == "q8" the activation first goes through the
        rounding ggml's quantized matmul applies (see quantize_act)."""
        if self.opts["act_quant"] == "q8":
            x = quantize_act(x, self.qtype(wname))
        return mm(x, self.t(wname))

    def has(self, name):
        return name in self.names

    def t(self, name):
        """A whole tensor as float32.  Small ones are cached; the big dense matrices of a real file (tens of MB each
        in float32) are dequantized again on each use so a 40-layer pass stays within a few GB."""
        if name in self._cache:
            return self._cache[name]
        a = self.g.tensor(name)
        if a.size <= 4_000_000:
            self._cache[name] = a
        return a

    def expert_ffn_weights(self, il, e):
        """(gate (ff, n_embd), up (ff, n_embd), down (n_embd, ff)) of expert e in block il."""
        p = f"blk.{il}"
        ff = self.hp.n_ff_exp
        if self.has(f"{p}.ffn_gate_up_exps.weight"):        # create_tensor_gate_up_exps: gate rows first, then up
            gu = self.g.expert(f"{p}.ffn_gate_up_exps.weight", e)
            gate, up = gu[:ff], gu[ff:]
        else:
            gate = self.g.expert(f"{p}.ffn_gate_exps.weight", e)
            up = self.g.expert(f"{p}.ffn_up_exps.weight", e)
        return gate, up, self.g.expert(f"{p}.ffn_down_exps.weight", e)

    def embed(self, ids, name="token_embd.weight"):
        return self.g.rows(name, ids)

    # ---- the pieces of the graph
    def norm(self, x, wname, il, dump):
        """build_norm(LLM_NORM_RMS) (llama-graph.cpp:1583-1616): cb("norm") on the unweighted result, then * weight."""
        n = dump("norm", il, rms_norm(x, self.hp.eps))
        return (n * self.t(wname)).astype(F32)

    def gdn(self, x, il, st: LayerState, dump):
        """build_layer_attn_linear (qwen35moe.cpp:359-492) over x (T, n_embd), already attn_norm'ed."""
        hp = self.hp
        p = f"blk.{il}"
        T = x.shape[0]
        Hk, Hv, S, K = hp.ssm_k_heads, hp.ssm_v_heads, hp.ssm_state, hp.ssm_conv
        assert hp.head_v_dim == S, "the graph assumes head_v_dim == head_k_dim == ssm.state_size"
        assert Hv % Hk == 0
        if st.conv is None:
            st.conv = np.zeros((hp.conv_dim, K - 1), F32)
            st.ssm = np.zeros((Hv, S, S), F32)
        dump("conv_states", il, st.conv.reshape(1, -1))                # ne [(K-1)*C, 1]: ggml's [K-1, C] flattened
        dump("conv_states_reshaped", il, st.conv)                      # ne [K-1, C]

        qkv = dump("linear_attn_qkv_mixed", il, self.mm(x, f"{p}.attn_qkv.weight"))        # (T, conv_dim)
        dump("qkv_mixed_transposed", il, qkv.T)
        z = dump("z", il, self.mm(x, f"{p}.attn_gate.weight"))                              # (T, value_dim)
        beta_raw = dump("beta", il, self.mm(x, f"{p}.ssm_beta.weight")[:, :, None])         # ne [1, Hv, T]
        beta = dump("beta_sigmoid", il, sigmoid(beta_raw))[:, :, 0]
        alpha = dump("alpha", il, self.mm(x, f"{p}.ssm_alpha.weight"))                      # (T, Hv)
        sp = dump("a_softplus", il, softplus(alpha + self.t(f"{p}.ssm_dt.bias")))
        gate = dump("gate", il, (sp * self.t(f"{p}.ssm_a")).astype(F32))                        # (T, Hv), < 0

        # causal conv1d over [state | qkv], per channel (ggml_ssm_conv, ggml-cpu/ops.cpp:9703-9754)
        conv_in = dump("conv_input", il, np.concatenate([st.conv, qkv.T], axis=1))             # (C, K-1+T)
        w = self.t(f"{p}.ssm_conv1d.weight")                                                   # (C, K)
        if self.opts["conv"] == "flipped":
            w = w[:, ::-1]
        raw = np.zeros((T, hp.conv_dim), F32)
        for t in range(T):
            acc = np.zeros(hp.conv_dim, F32)
            for i in range(K):
                acc = acc + conv_in[:, t + i] * w[:, i]       # sequential float32 sum over the window, as ops.cpp:9747
            raw[t] = acc
        new_conv = dump("conv_state_last", il, conv_in[:, -(K - 1):].copy())
        dump("conv_output_raw", il, raw)
        c = dump("conv_output_silu", il, silu(raw))
        kd = hp.key_dim
        q = dump("q_conv", il, c[:, :kd].reshape(T, Hk, S))
        k = dump("k_conv", il, c[:, kd:2 * kd].reshape(T, Hk, S))
        v = c[:, 2 * kd:].reshape(T, Hv, S)
        q = dump("q_conv_predelta", il, l2_norm_gdn(q, hp.eps))      # fused GDN: no repeat to Hv heads (qwen35moe.cpp:462)
        k = dump("k_conv_predelta", il, l2_norm_gdn(k, hp.eps))
        dump("v_conv_predelta", il, v)

        # the recurrence (ggml-cpu/ops.cpp:10967-11045, the fused op; delta-net-base.cpp:289-371 is the same math)
        dump("state_predelta", il, st.ssm)
        S_ = st.ssm.copy()
        out = np.empty((T, Hv, S), F32)
        rep = np.arange(Hv) % Hk if self.opts["head_pairing"] == "tiled" else np.arange(Hv) // (Hv // Hk)   # ggml_repeat tiling / iv1 % nek1
        scale = F32(1.0 / np.sqrt(F32(S)))
        for t in range(T):
            kt, qt = k[t][rep], q[t][rep]                             # (Hv, S)
            if self.opts["decay"] == "before":
                S_ = S_ * np.exp(gate[t])[:, None, None]              # decay
            sk = np.einsum("hji,hi->hj", S_, kt).astype(F32)
            d = ((v[t] - sk) * beta[t][:, None]).astype(F32)          # delta rule
            S_ = (S_ + d[:, :, None] * kt[:, None, :]).astype(F32)
            if self.opts["decay"] == "after":
                S_ = (S_ * np.exp(gate[t])[:, None, None]).astype(F32)
            out[t] = np.einsum("hji,hi->hj", S_, qt) * scale
        st.ssm = S_
        dump("new_state", il, S_)
        o = dump("attn_output", il, out)                              # (T, Hv, S)

        # gated norm (build_norm_gated, qwen35moe.cpp:267-276): rms_norm(o) * ssm_norm * SiLU(z)
        normed = (dump("norm", il, rms_norm(o, hp.eps)) * self.t(f"{p}.ssm_norm.weight")).astype(F32)
        act = silu if self.opts["gdn_gate"] == "silu" else sigmoid
        gated = (normed * act(z.reshape(T, Hv, S))).astype(F32)
        fo = dump("final_output", il, gated.reshape(T, hp.value_dim))
        st.conv = new_conv
        return dump("linear_attn_out", il, self.mm(fo, f"{p}.ssm_out.weight"))

    def attention(self, x, il, st: LayerState, pos, dump, mtp=False):
        """build_layer_attn (qwen35moe.cpp:278-357): joint Q+gate projection, Q/K RMS norm, IMRoPE on 25%, causal
        GQA softmax attention, sigmoid output gate, output projection.  x (T, n_embd); pos (T,) absolute positions.
        mtp=True emits only the names graph_mtp (qwen35moe.cpp:621-668) sets, under its "mtp_" spelling."""
        hp = self.hp
        p = f"blk.{il}"
        T, hd, nh, nkv = x.shape[0], hp.head_dim, hp.n_head, hp.n_head_kv

        def emit(trunk, mtp_name, arr):          # a name the trunk graph sets, and the spelling (or None) in graph_mtp
            if mtp and mtp_name is None:
                return arr
            return dump(mtp_name if mtp else trunk, il, arr)

        if self.has(f"{p}.attn_qkv.weight"):                          # build_qkv fused path (llama-graph.cpp:1649-1686)
            qkv = emit("wqkv", None, self.mm(x, f"{p}.attn_qkv.weight"))
            nq, nk = hd * 2 * nh, hd * nkv
            qfull, kc, vc = qkv[:, :nq], qkv[:, nq:nq + nk], qkv[:, nq + nk:nq + 2 * nk]
        else:
            qfull = self.mm(x, f"{p}.attn_q.weight")
            kc = self.mm(x, f"{p}.attn_k.weight")
            vc = self.mm(x, f"{p}.attn_v.weight")
        qfull = emit("Qcur_full", "mtp_Qcur_full", qfull)
        kc = emit("Kcur", None, kc)
        vc = emit("Vcur", None, vc)
        qh = qfull.reshape(T, nh, 2, hd)                              # per head: [q (hd) | gate (hd)]
        q = emit("Qcur_reshaped", None, qh[:, :, 0, :])
        q = emit("Qcur_normed", "mtp_Qcur_normed",
                 (dump("norm", il, rms_norm(q, hp.eps)) * self.t(f"{p}.attn_q_norm.weight")).astype(F32))
        gate = emit("gate_reshaped_pre", "mtp_gate", qh[:, :, 1, :].reshape(T, nh * hd)) if mtp else qh[:, :, 1, :].reshape(T, nh * hd)
        k = kc.reshape(T, nkv, hd)
        k = emit("Kcur_normed", "mtp_Kcur_normed",
                 (dump("norm", il, rms_norm(k, hp.eps)) * self.t(f"{p}.attn_k_norm.weight")).astype(F32))
        v = emit("Vcur", "mtp_Vcur", vc.reshape(T, nkv, hd))
        pos4 = np.stack([pos, pos, pos, np.zeros_like(pos)]).astype(np.int64)
        q = emit("Qcur", None, rope_multi(q, pos4, hp.n_rot, hp.sections, hp.rope_base, self.opts["rope_pairs"]))
        k = emit("Kcur", None, rope_multi(k, pos4, hp.n_rot, hp.sections, hp.rope_base, self.opts["rope_pairs"]))

        if st.k is None:
            st.k = np.zeros((0, nkv, hd), F32)
            st.v = np.zeros((0, nkv, hd), F32)
        n_past = st.k.shape[0]
        st.k = np.concatenate([st.k, k]); st.v = np.concatenate([st.v, v])
        n_kv = st.k.shape[0]
        grp = nh // nkv
        kvmap = np.arange(nh) // grp if self.opts["gqa_map"] == "grouped" else np.arange(nh) % nkv
        scale = F32(1.0 / np.sqrt(F32(hd)))                           # hparams.f_attention_scale == 0 -> 1/sqrt(head_dim)
        # kq[h, t, j] = k_j . q_t  (ggml: ne [n_kv, T, n_head]); causal mask j <= n_past + t
        kq = np.einsum("tnd,jnd->ntj", q, st.k[:, kvmap]).astype(F32)
        dump("kq", il, kq)
        mask = np.arange(n_kv)[None, :] > (n_past + np.arange(T))[:, None]
        sc = np.where(mask[None], F32(-np.inf), kq * scale)
        pr = dump("kq_soft_max", il, softmax(sc))
        kqv = dump("kqv", il, np.einsum("ntj,jnd->ntd", pr, st.v[:, kvmap]).astype(F32))
        pre = emit("attn_pregate", "mtp_attn_pregate", kqv.transpose(1, 0, 2).reshape(T, nh * hd))
        if not mtp:
            gate = dump("gate_reshaped", il, gate)
            gs = dump("gate_sigmoid", il, sigmoid(gate))
        else:
            gs = sigmoid(gate)
        gated = emit("attn_gated", None, (pre * gs).astype(F32) if self.opts["attn_gate"] else pre)
        return emit("attn_output", "mtp_attn_out", self.mm(gated, f"{p}.attn_output.weight"))

    def moe(self, x, il, dump, mtp=False):
        """build_layer_ffn (qwen35moe.cpp:494-548) with build_moe_ffn (llama-graph.cpp:1993-2355): softmax router,
        top-k, weights renormalised to sum 1 (norm_w = true), SwiGLU experts, plus the sigmoid-gated shared expert."""
        hp = self.hp
        p = f"blk.{il}"
        T, E, K = x.shape[0], hp.n_expert, hp.n_expert_used
        logits = dump("ffn_moe_logits", il, self.mm(x, f"{p}.ffn_gate_inp.weight"))        # (T, E)
        probs = dump("ffn_moe_probs", il, softmax(logits))                                     # SOFTMAX gating
        order = np.argsort(-probs, axis=1, kind="stable")                                     # argsort descending
        dump("ffn_moe_argsort", il, order, "i32")
        sel = order[:, :K] if self.opts["force_topk"] is None else self.opts["force_topk"][il]
        topk = dump("ffn_moe_topk", il, sel, "i32")                                   # (T, K)
        w = dump("ffn_moe_weights", il, np.take_along_axis(probs, topk, 1)[:, :, None])        # ne [1, K, T]
        w2 = w[:, :, 0]
        ssum = dump("ffn_moe_weights_sum", il, w2.sum(axis=1, keepdims=True, dtype=F32))        # ne [1, T]
        sc = dump("ffn_moe_weights_sum_clamped", il, np.maximum(ssum, F32(6.103515625e-5)))     # clamp (llama-graph.cpp:2141)
        do_norm = hp.expert_weights_norm if self.opts["router_norm"] is None else self.opts["router_norm"]
        wn = dump("ffn_moe_weights_norm", il, (w2 / sc).astype(F32)) if do_norm else w2
        if hp.expert_weights_scale not in (0.0, 1.0):
            wn = dump("ffn_moe_weights_scaled", il, (wn * F32(hp.expert_weights_scale)).astype(F32))
        q8 = self.opts["act_quant"] == "q8"
        gu_name = f"{p}.ffn_gate_up_exps.weight" if self.has(f"{p}.ffn_gate_up_exps.weight") else f"{p}.ffn_gate_exps.weight"
        gate_all = np.empty((T, K, hp.n_ff_exp), F32)
        up_all = np.empty_like(gate_all)
        for t in range(T):
            for j in range(K):
                g_, u_, _ = self.expert_ffn_weights(il, int(topk[t, j]))
                xt = quantize_act(x[t], self.qtype(gu_name)) if q8 else x[t]
                gate_all[t, j], up_all[t, j] = xt @ g_.T, xt @ u_.T
        if self.has(f"{p}.ffn_gate_up_exps.weight"):
            dump("ffn_moe_gate_up", il, np.concatenate([gate_all, up_all], axis=2))      # ne [2*ff, K, T]: gate then up
        dump("ffn_moe_gate", il, gate_all)
        dump("ffn_moe_up", il, up_all)
        act = dump("ffn_moe_swiglu", il, (silu(gate_all) * up_all).astype(F32))
        down = np.empty((T, K, hp.n_embd), F32)
        for t in range(T):
            for j in range(K):
                _, _, d_ = self.expert_ffn_weights(il, int(topk[t, j]))
                at = quantize_act(act[t, j], self.qtype(f"{p}.ffn_down_exps.weight")) if q8 else act[t, j]
                down[t, j] = at @ d_.T
        dump("ffn_moe_down", il, down)
        weighted = dump("ffn_moe_weighted", il, (down * wn[:, :, None]).astype(F32))
        moe_out = weighted[:, 0]
        for j in range(1, K):                                         # sequential adds in top-k order (llama-graph.cpp:2339-2345)
            moe_out = (moe_out + weighted[:, j]).astype(F32)
        # graph_mtp renames these tensors (cb sets the final name), so only the "mtp_" spelling is ever visible there
        nm = (lambda trunk, m: (m or trunk) if mtp else trunk)
        moe_out = dump(nm("ffn_moe_out", "mtp_ffn_moe_out"), il, moe_out)
        # shared expert, build_ffn(LLM_FFN_SILU, LLM_FFN_PAR) then a per-token sigmoid gate (qwen35moe.cpp:516-545)
        up = dump("ffn_up", il, self.mm(x, f"{p}.ffn_up_shexp.weight"))
        gt = dump("ffn_gate", il, self.mm(x, f"{p}.ffn_gate_shexp.weight"))
        sw = dump("ffn_swiglu", il, (silu(gt) * up).astype(F32))
        sh = dump(nm("ffn_shexp", "mtp_ffn_shexp"), il, self.mm(sw, f"{p}.ffn_down_shexp.weight"))
        sg = mm(x, self.t(f"{p}.ffn_gate_inp_shexp.weight")[None, :])   # F32 weight: never activation-quantized                            # (T, 1)
        if not mtp:
            dump("shared_expert_gate", il, sg)
        sgs = dump(nm("shared_expert_gate_sigmoid", "mtp_shared_expert_gate_sigmoid"), il, sigmoid(sg))
        shg = dump(nm("ffn_shexp_gated", "mtp_ffn_shexp_gated"), il, (sh * sgs).astype(F32))
        return dump(nm("ffn_out", "mtp_ffn_out"), il, (moe_out + shg).astype(F32))

    # ---- the trunk
    def forward(self, tokens, dump: Dump | None = None, state: State | None = None, all_logits=False):
        """Teacher-forced pass over `tokens` (list of ids) from `state` (fresh when None).  Returns
        (logits of the last token (n_vocab,), hidden (T, n_embd) after the output norm [= h_nextn], state)."""
        dump = dump or Dump()
        hp = self.hp
        st = state or State()
        ids = np.asarray(tokens, dtype=np.int64)
        T = len(ids)
        pos = st.n_past + np.arange(T)
        x = dump("model.input_embed", -1, self.embed(ids))
        for il in range(hp.n_layer):
            ls = st.layers.setdefault(il, LayerState())
            if self.opts["force_inputs"] and il > 0:
                x = np.fromfile(f"{self.opts['force_inputs']}/l_out-{il - 1}.f32", dtype=F32).reshape(T, -1)
            inp = x
            an = (dump("norm", il, rms_norm(x, hp.eps)) * self.t(f"blk.{il}.attn_norm.weight")).astype(F32)
            an = dump("attn_norm", il, an)
            mix = self.gdn(an, il, ls, dump) if hp.is_recr(il) else self.attention(an, il, ls, pos, dump)
            res = dump("attn_residual", il, (mix + inp).astype(F32))
            pn = (dump("norm", il, rms_norm(res, hp.eps)) * self.t(f"blk.{il}.post_attention_norm.weight")).astype(F32)
            pn = dump("attn_post_norm", il, pn)
            ffn = self.moe(pn, il, dump)
            x = dump("l_out", il, (ffn + res).astype(F32))
        st.n_past += T
        h = dump("norm", -1, rms_norm(x, hp.eps))
        h = dump("h_nextn", -1, (h * self.t("output_norm.weight")).astype(F32))
        out_name = "output.weight" if self.has("output.weight") else "token_embd.weight"   # tied head (qwen35moe.cpp:47-52)
        last = dump("result_norm", -1, h[-1:])
        logits = dump("result_output", -1, self.mm(last, out_name))
        if all_logits:
            dump("result_output_all", -1, self.mm(h, out_name))
        return logits[0], h, st

    # ---- the nextn (MTP) block, graph_mtp (qwen35moe.cpp:550-741)
    def mtp_forward(self, tokens, h, dump: Dump | None = None, state: State | None = None, pos0=None):
        """One pass of the draft head.  `tokens`: the T token ids whose embeddings are paired with `h`; `h`: (T, n_embd)
        trunk hidden states after the output norm (the trunk's h_nextn).  Which trunk position pairs with which token
        is the caller's choice (speculative decoding feeds the NEXT token with the hidden of the one before).
        Attention is causal over the MTP block's own KV at positions pos0.. (default: the block's own n_past).  Returns (logits (T, n_vocab), state)."""
        dump = dump or Dump()
        hp = self.hp
        assert hp.n_nextn == 1, "no nextn block in this file"
        il = hp.n_layer
        p = f"blk.{il}"
        st = state or State()
        ls = st.layers.setdefault(il, LayerState())
        ids = np.asarray(tokens, dtype=np.int64)
        T = len(ids)
        pos = (st.n_past if pos0 is None else pos0) + np.arange(T)
        emb_name = f"{p}.nextn.embed_tokens.weight" if self.has(f"{p}.nextn.embed_tokens.weight") else "token_embd.weight"
        tok = dump("mtp_tok_embd", il, self.embed(ids, emb_name))
        # dump order is graph execution order, and ggml_concat(e_norm, h_norm) pulls e_norm first, so "norm-4" is the
        # embedding norm and "norm-4#2" the hidden norm even though graph_mtp builds h_norm first.
        en = dump("mtp_enorm", il, (dump("norm", il, rms_norm(tok, hp.eps)) * self.t(f"{p}.nextn.enorm.weight")).astype(F32))
        hn = dump("mtp_hnorm", il, (dump("norm", il, rms_norm(h, hp.eps)) * self.t(f"{p}.nextn.hnorm.weight")).astype(F32))
        cat = dump("mtp_concat", il, np.concatenate([en, hn], axis=1))        # ggml_concat(e_norm, h_norm, dim 0)
        cur = dump("mtp_eh_proj", il, self.mm(cat, f"{p}.nextn.eh_proj.weight"))
        an = dump("mtp_attn_norm", il, (dump("norm", il, rms_norm(cur, hp.eps)) * self.t(f"{p}.attn_norm.weight")).astype(F32))
        att = self.attention(an, il, ls, pos, dump, mtp=True)
        res = dump("mtp_attn_residual", il, (att + cur).astype(F32))
        pn = dump("mtp_attn_post_norm", il, (dump("norm", il, rms_norm(res, hp.eps)) * self.t(f"{p}.post_attention_norm.weight")).astype(F32))
        ffn = self.moe(pn, il, dump, mtp=True)
        post = dump("mtp_post_ffn", il, (ffn + res).astype(F32))
        hw = f"{p}.nextn.shared_head_norm.weight" if self.has(f"{p}.nextn.shared_head_norm.weight") else "output_norm.weight"
        hh = dump("h_nextn", -1, (dump("norm", -1, rms_norm(post, hp.eps)) * self.t(hw)).astype(F32))
        dump("mtp_shared_head_norm", -1, hh)
        head = f"{p}.nextn.shared_head_head.weight" if self.has(f"{p}.nextn.shared_head_head.weight") else (
            "output.weight" if self.has("output.weight") else "token_embd.weight")
        logits = dump("result_output", -1, self.mm(hh, head))
        st.n_past += T
        return logits, st


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.6 (qwen35moe) numpy oracle: forward pass + cb()-named dumps")
    ap.add_argument("gguf")
    ap.add_argument("--tokens", help="comma-separated token ids")
    ap.add_argument("--text", help="synthetic vocabulary only: token id = ASCII code")
    ap.add_argument("--dump", help="directory for the .f32 dumps and manifest.txt")
    ap.add_argument("--split", type=int, default=0, help="feed the tokens CHUNK at a time, carrying GDN and KV state; dump only the last chunk")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="flip one reading to its rival (see Model.opts), e.g. gdn_gate=sigmoid; for negative controls")
    ap.add_argument("--act-quant", choices=["none", "q8"], default="none", help="q8: round activations to Q8_K / Q8_0 before quantized matmuls, as ggml does")
    ap.add_argument("--force-topk", metavar="DIR", help="use the expert choices in DIR/ffn_moe_topk-<layer>.f32 (e.g. a llama.cpp dump) instead of the router's own")
    ap.add_argument("--force-inputs", metavar="DIR", help="start every layer from DIR/l_out-<il-1>.f32 (a llama.cpp dump of the same tokens): measures per-layer, not accumulated, error")
    ap.add_argument("--mtp", action="store_true", help="also run the nextn block on (tokens[1:], trunk hidden[:-1])")
    ap.add_argument("--all-logits", action="store_true",
                    help="also dump result_output_all: the head's logits at every position of the (last) chunk, for a runner held to the oracle position by position")
    a = ap.parse_args()
    toks = [int(t) for t in a.tokens.split(",")] if a.tokens else [ord(c) for c in a.text]
    m = Model(a.gguf)
    for kv in a.opt:
        k, v = kv.split("=", 1)
        assert k in m.opts, f"unknown option {k!r}; known: {sorted(m.opts)}"
        m.opts[k] = {"True": True, "False": False, "None": None}.get(v, v)
    m.opts["act_quant"] = a.act_quant
    m.opts["force_inputs"] = a.force_inputs
    if a.force_topk:
        m.opts["force_topk"] = {il: np.fromfile(f"{a.force_topk}/ffn_moe_topk-{il}.f32", dtype=F32).astype(np.int64).reshape(len(toks), -1)
                                for il in range(m.hp.n_layer)}
    d = Dump(a.dump)
    if a.split:
        st = None
        for i in range(0, len(toks), a.split):
            last = i + a.split >= len(toks)
            logits, h, st = m.forward(toks[i:i + a.split], d if last else None, st, all_logits=a.all_logits)
    else:
        logits, h, _ = m.forward(toks, d, all_logits=a.all_logits)
    print(f"{len(toks)} tokens; last-token logits: argmax {int(logits.argmax())}, max {logits.max():.6f}, sum {logits.sum():.6f}")
    d.close()
    if a.mtp:
        d2 = Dump((a.dump + "_mtp") if a.dump else None)
        lg, _ = m.mtp_forward(toks[1:], h[:-1], d2, pos0=1)   # token k with the hidden of k-1, at position k
        print(f"mtp: {len(toks) - 1} positions; last argmax {int(lg[-1].argmax())}")
        d2.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
