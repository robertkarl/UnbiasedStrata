#!/usr/bin/env python3
"""tools/qwen36_synth_gguf.py - a tiny synthetic `qwen35moe` GGUF (Qwen3.6-35B-A3B at toy scale).

Real tensor names, real shape rules, F32 weights, deterministic seed, and every metadata key that
llama.cpp's loader reads for `qwen35moe` (src/models/qwen35moe.cpp:4-34 and src/llama-model.cpp:1231-1424
in the pinned submodule).  It exists so that `ref/qwen36.py` (the numpy oracle) and the submodule's own
`llama-eval-callback` can run the same forward pass on the same weights, and so the engine's parity tests
have a fixture that needs no 13 GB download.

    python3 tools/qwen36_synth_gguf.py OUT.gguf                       # default toy geometry, with the nextn layer
    python3 tools/qwen36_synth_gguf.py OUT.gguf --like-real REAL.gguf # geometry ratios, rope, eps, interval from REAL
    python3 tools/qwen36_synth_gguf.py OUT.gguf --no-nextn --fused-gate-up --fused-qkv
    python3 tools/qwen36_synth_gguf.py OUT.gguf --gdn 2,4,128 --experts 256,8 --shexp-ff 512 --quant-shexp q8_0 --bf16-exact
        # the engine's kernel fixture (step 3): state 128 (the CUDA GDN kernels are pinned to it), the real router
        # shape, the shared expert at its real width in Q8_0, and the weights the engine reads as BF16 already
        # BF16-representable, so the oracle and the kernels differ only by summation order

Default geometry (all dims small enough to read, large enough that every reshape is non-degenerate):

    hidden 64, 4 layers (interval 4: layers 0,1,2 are Gated DeltaNet, layer 3 is full attention),
    GDN 4 k-heads / 8 v-heads of dim 16, conv kernel 4, attention 2 heads / 1 KV head of dim 32,
    partial rotary 8/32 = 0.25 with IMRoPE sections [2,1,1,0], 8 experts top-2, expert ff 32, shared ff 32,
    vocab 128 (byte-level BPE with no merges, token id = ASCII code, so `-p "abc"` is tokens 97 98 99),
    plus one nextn (MTP) block at index 4, so block_count is 5 and nextn_predict_layers is 1.

Shape rules (GGUF dims are ne order, fastest first; numpy shapes below are the reverse):

    token_embd.weight / output.weight          ne [n_embd, n_vocab]
    blk.N.attn_norm / post_attention_norm      ne [n_embd]
    GDN:  attn_qkv [n_embd, 2*key_dim+value_dim]  attn_gate [n_embd, value_dim]  ssm_conv1d [d_conv, conv_dim]
          ssm_dt.bias [H_v]  ssm_a [H_v]  ssm_alpha/ssm_beta [n_embd, H_v]  ssm_norm [S]  ssm_out [value_dim, n_embd]
    attn: attn_q [n_embd, 2*n_head*hd]  attn_k/v [n_embd, n_kv*hd]  attn_output [n_head*hd, n_embd]  q_norm/k_norm [hd]
    MoE:  ffn_gate_inp [n_embd, E]  ffn_{gate,up}_exps [n_embd, ff, E]  ffn_down_exps [ff, n_embd, E]
          ffn_gate_inp_shexp [n_embd]  ffn_{gate,up}_shexp [n_embd, ff]  ffn_down_shexp [ff, n_embd]
    nextn (block n_layer): the attention-layer tensors above plus the MoE tensors plus
          nextn.eh_proj [2*n_embd, n_embd]  nextn.enorm / nextn.hnorm / nextn.shared_head_norm [n_embd]

Requires numpy and gguf-py (STRATA_GGUF_PY, or third_party/llama.cpp/gguf-py).
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _paths import add_gguf_py                      # noqa: E402

add_gguf_py()
from gguf import GGUFWriter                          # noqa: E402

ARCH = "qwen35moe"


@dataclasses.dataclass
class Cfg:
    n_embd: int = 64
    n_layer: int = 4                 # trunk layers (excludes the nextn block)
    interval: int = 4                # full attention on layers where (il + 1) % interval == 0
    n_vocab: int = 128
    n_expert: int = 8
    n_expert_used: int = 2
    n_ff_exp: int = 32
    n_ff_shexp: int = 32
    ssm_k_heads: int = 4
    ssm_v_heads: int = 8
    ssm_state: int = 16
    ssm_conv: int = 4
    n_head: int = 2
    n_head_kv: int = 1
    head_dim: int = 32
    n_rot: int = 8                   # 0.25 * head_dim, as the real model
    rope_sections: tuple = (2, 1, 1, 0)   # sum == n_rot / 2
    rope_base: float = 10000.0       # the real model is 1e7; at 8 rotated dims that rotates only the first pair visibly, so the toy uses 1e4 (--like-real restores 1e7)
    rms_eps: float = 1e-6
    n_ctx_train: int = 4096
    nextn: bool = True
    fused_gate_up: bool = False
    fused_qkv: bool = False          # attention layers: one attn_qkv instead of attn_q/attn_k/attn_v
    expert_weights_scale: float | None = None   # a stray key: llama.cpp's qwen35moe loader does not read it
    quant_shexp: str | None = None   # ggml type name for ffn_{gate,up,down}_shexp (e.g. "q8_0"); F32 when None
    bf16_exact: bool = False         # round ssm_alpha/ssm_beta/ffn_gate_inp/ffn_gate_inp_shexp to BF16-representable F32
    seed: int = 0

    @property
    def key_dim(self): return self.ssm_k_heads * self.ssm_state
    @property
    def value_dim(self): return self.ssm_v_heads * self.ssm_state
    @property
    def conv_dim(self): return 2 * self.key_dim + self.value_dim
    def is_attn(self, il): return (il + 1) % self.interval == 0

    def check(self):
        assert self.ssm_v_heads % self.ssm_k_heads == 0
        assert sum(self.rope_sections) * 2 <= self.n_rot <= self.head_dim and self.n_rot % 2 == 0
        assert self.n_head % self.n_head_kv == 0 and self.n_expert_used <= self.n_expert
        assert self.n_layer >= self.interval, "need at least one full-attention layer"


def split_sections(sec: list, slots: int) -> list:
    """Largest-remainder split of `slots` sector slots (n_rot / 2 of them) in the proportions of `sec`."""
    tot = sum(sec)
    raw = [s * slots / tot for s in sec]
    fl = [int(x) for x in raw]
    for i in sorted(range(len(sec)), key=lambda i: raw[i] - fl[i], reverse=True)[: slots - sum(fl)]:
        fl[i] += 1
    return fl


def real_config(path: str) -> tuple[Cfg, dict]:
    """Derive a toy Cfg from a real qwen35moe GGUF: keep what is a rule (rope base, eps, interval, conv
    kernel, the ratios), divide what is a size.  Returns (cfg, report-of-real-values)."""
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    m = gf.metadata
    assert m["general.architecture"] == ARCH, f"{path}: architecture {m['general.architecture']!r}"
    g = lambda k, d=None: m.get(f"{ARCH}.{k}", d)
    hd, n_rot = g("attention.key_length"), g("rope.dimension_count")
    sec = list(g("rope.dimension_sections", [11, 11, 10, 0]))
    small_hd = hd // 8
    small_rot = n_rot * small_hd // hd
    fl = split_sections(sec, small_rot // 2)
    n_head, n_kv = g("attention.head_count"), g("attention.head_count_kv")
    cfg = Cfg(
        n_embd=g("embedding_length") // 32, n_layer=4, interval=g("full_attention_interval", 4),
        n_vocab=128, n_expert=g("expert_count") // 32, n_expert_used=max(1, g("expert_used_count") // 4),
        n_ff_exp=g("expert_feed_forward_length") // 16, n_ff_shexp=g("expert_shared_feed_forward_length") // 16,
        ssm_k_heads=g("ssm.group_count") // 4, ssm_v_heads=g("ssm.time_step_rank") // 4,
        ssm_state=g("ssm.state_size") // 8, ssm_conv=g("ssm.conv_kernel"),
        n_head=n_head // 2, n_head_kv=max(1, n_kv // 2), head_dim=small_hd, n_rot=small_rot,
        rope_sections=tuple(fl), rope_base=g("rope.freq_base"), rms_eps=g("attention.layer_norm_rms_epsilon"),
        nextn=any(t.name.startswith(f"blk.{g('block_count') - 1}.nextn.") for t in gf.tensors),
        fused_gate_up=any(".ffn_gate_up_exps." in t.name for t in gf.tensors),
        fused_qkv=any(t.name == f"blk.{g('full_attention_interval', 4) - 1}.attn_qkv.weight" for t in gf.tensors),
    )
    cfg.check()
    real = {"block_count": g("block_count"), "tensors": [t.name for t in gf.tensors]}
    return cfg, real


def norm_name(name: str, n_layer_real: int | None = None) -> str:
    """blk.17.ffn_gate_exps.weight -> blk.N.ffn_gate_exps.weight, for comparing name sets."""
    import re
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


# ------------------------------------------------------------------------------------------------ weights
def build_tensors(c: Cfg) -> list[tuple[str, np.ndarray]]:
    """(gguf name, numpy array in (out, in[, experts]) order) for every tensor, in the order llama.cpp's
    loader declares them.  Numpy shape is the reverse of the GGUF ne list."""
    rng = np.random.default_rng(c.seed)
    f32 = np.float32

    def lin(out, inn, gain=1.0):
        return (rng.standard_normal((out, inn)) * gain / np.sqrt(inn)).astype(f32)

    def exps(e, out, inn, gain=1.0):
        return (rng.standard_normal((e, out, inn)) * gain / np.sqrt(inn)).astype(f32)

    def normw(n):                      # llama.cpp multiplies by the stored weight directly (no +1)
        return (1.0 + 0.2 * rng.standard_normal(n)).astype(f32)

    T: list[tuple[str, np.ndarray]] = []
    add = lambda n, a: T.append((n, np.ascontiguousarray(a, dtype=f32)))
    d = c.n_embd

    def bf16(a):
        """Round-to-nearest-even to BF16, returned as F32 (the engine's bf16_from_f32, strata/kernels/bf16_bits.hpp).
        The pack rounds these F32 tensors to BF16 exactly when the low 16 bits are already zero (upstream-tooling.md
        section 4), so a fixture written this way is one the engine reads bit-for-bit."""
        a = np.ascontiguousarray(a, dtype=f32)
        if not c.bf16_exact:
            return a
        u = a.view(np.uint32)
        u = (u + ((u >> 16) & 1) + np.uint32(0x7FFF)) & np.uint32(0xFFFF0000)
        return u.view(f32)

    add("token_embd.weight", rng.standard_normal((c.n_vocab, d)))
    add("output_norm.weight", normw(d))
    add("output.weight", lin(c.n_vocab, d, 2.0))

    def moe(p):
        add(f"{p}.ffn_gate_inp.weight", bf16(lin(c.n_expert, d, 3.0)))   # spread logits so top-k is decisive
        add(f"{p}.ffn_down_exps.weight", exps(c.n_expert, d, c.n_ff_exp))
        if c.fused_gate_up:                                              # ne [n_embd, 2*ff, E]: gate rows then up rows
            add(f"{p}.ffn_gate_up_exps.weight", exps(c.n_expert, 2 * c.n_ff_exp, d))
        else:
            add(f"{p}.ffn_gate_exps.weight", exps(c.n_expert, c.n_ff_exp, d))
            add(f"{p}.ffn_up_exps.weight", exps(c.n_expert, c.n_ff_exp, d))
        add(f"{p}.ffn_gate_inp_shexp.weight", bf16(rng.standard_normal(d) / np.sqrt(d) * 2.0))
        add(f"{p}.ffn_gate_shexp.weight", lin(c.n_ff_shexp, d))
        add(f"{p}.ffn_up_shexp.weight", lin(c.n_ff_shexp, d))
        add(f"{p}.ffn_down_shexp.weight", lin(d, c.n_ff_shexp))

    def attn(p):
        qw, kw = c.head_dim * c.n_head * 2, c.head_dim * c.n_head_kv
        if c.fused_qkv:
            add(f"{p}.attn_qkv.weight", lin(qw + 2 * kw, d))
        else:
            add(f"{p}.attn_q.weight", lin(qw, d))
            add(f"{p}.attn_k.weight", lin(kw, d))
            add(f"{p}.attn_v.weight", lin(kw, d))
        add(f"{p}.attn_output.weight", lin(d, c.head_dim * c.n_head))
        add(f"{p}.attn_q_norm.weight", normw(c.head_dim))
        add(f"{p}.attn_k_norm.weight", normw(c.head_dim))

    for il in range(c.n_layer):
        p = f"blk.{il}"
        add(f"{p}.attn_norm.weight", normw(d))
        add(f"{p}.post_attention_norm.weight", normw(d))
        if c.is_attn(il):
            attn(p)
        else:
            add(f"{p}.attn_qkv.weight", lin(c.conv_dim, d))
            add(f"{p}.attn_gate.weight", lin(c.value_dim, d))
            add(f"{p}.ssm_conv1d.weight", rng.standard_normal((c.conv_dim, c.ssm_conv)) * 0.6)
            # real layer 0: ssm_a in [-72, -0.02] (already -exp(A_log), so negative), ssm_dt.bias in [-5.3, 15.6]; the toy
            # spans both regimes (fast-forgetting and long-memory heads) over the 11-40 token sequences the tests use
            add(f"{p}.ssm_dt.bias", rng.uniform(-5.0, 1.0, c.ssm_v_heads))
            add(f"{p}.ssm_a", -np.exp(rng.uniform(np.log(0.05), np.log(20.0), c.ssm_v_heads)))
            add(f"{p}.ssm_beta.weight", bf16(lin(c.ssm_v_heads, d, 1.5)))
            add(f"{p}.ssm_alpha.weight", bf16(lin(c.ssm_v_heads, d, 1.0)))
            add(f"{p}.ssm_norm.weight", normw(c.ssm_state))
            add(f"{p}.ssm_out.weight", lin(d, c.value_dim))
        moe(p)

    if c.nextn:
        p = f"blk.{c.n_layer}"
        add(f"{p}.attn_norm.weight", normw(d))
        add(f"{p}.post_attention_norm.weight", normw(d))
        attn(p)
        moe(p)
        add(f"{p}.nextn.eh_proj.weight", lin(d, 2 * d))
        add(f"{p}.nextn.enorm.weight", normw(d))
        add(f"{p}.nextn.hnorm.weight", normw(d))
        add(f"{p}.nextn.shared_head_norm.weight", normw(d))
    return T


# ------------------------------------------------------------------------------------------------ vocab
def byte_unicode() -> list[str]:
    """GPT-2 bytes_to_unicode for bytes 0..127 (printable ASCII maps to itself)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = dict(zip(bs, (chr(c) for c in cs)))
    return [m[b] for b in range(128)]


def write(path: str, c: Cfg, comment: str = "") -> None:
    c.check()
    assert c.n_vocab == 128, "the toy vocabulary is the 128 ASCII byte tokens"
    w = GGUFWriter(path, ARCH)
    k = lambda s: f"{ARCH}.{s}"
    w.add_name(f"qwen36-synth{comment}")
    w.add_uint32(k("context_length"), c.n_ctx_train)
    w.add_uint32(k("embedding_length"), c.n_embd)
    w.add_uint32(k("block_count"), c.n_layer + (1 if c.nextn else 0))
    if c.nextn:
        w.add_uint32(k("nextn_predict_layers"), 1)
    w.add_uint32(k("attention.head_count"), c.n_head)
    w.add_uint32(k("attention.head_count_kv"), c.n_head_kv)
    w.add_uint32(k("attention.key_length"), c.head_dim)
    w.add_uint32(k("attention.value_length"), c.head_dim)
    w.add_float32(k("attention.layer_norm_rms_epsilon"), c.rms_eps)
    w.add_uint32(k("rope.dimension_count"), c.n_rot)
    w.add_array(k("rope.dimension_sections"), list(c.rope_sections) + [0] * (4 - len(c.rope_sections)))
    w.add_float32(k("rope.freq_base"), c.rope_base)
    w.add_uint32(k("expert_count"), c.n_expert)
    w.add_uint32(k("expert_used_count"), c.n_expert_used)
    w.add_uint32(k("expert_feed_forward_length"), c.n_ff_exp)
    w.add_uint32(k("expert_shared_feed_forward_length"), c.n_ff_shexp)
    w.add_uint32(k("ssm.conv_kernel"), c.ssm_conv)
    w.add_uint32(k("ssm.state_size"), c.ssm_state)
    w.add_uint32(k("ssm.group_count"), c.ssm_k_heads)
    w.add_uint32(k("ssm.time_step_rank"), c.ssm_v_heads)
    w.add_uint32(k("ssm.inner_size"), c.value_dim)
    w.add_uint32(k("full_attention_interval"), c.interval)
    if c.expert_weights_scale is not None:
        w.add_float32(k("expert_weights_scale"), c.expert_weights_scale)
    w.add_file_type(0)                                     # LLAMA_FTYPE_ALL_F32

    toks = byte_unicode()
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types([1] * c.n_vocab)                     # NORMAL
    w.add_token_merges(["Ā ā"])                  # loader requires the key; this pair never occurs in ASCII text
    w.add_bos_token_id(0)
    w.add_eos_token_id(3)
    w.add_add_bos_token(False)

    qt = None
    if c.quant_shexp:
        from gguf import GGMLQuantizationType, quants
        qt = GGMLQuantizationType[c.quant_shexp.upper()]
    for name, arr in build_tensors(c):
        if qt is not None and any(name.endswith(f".{r}_shexp.weight") for r in ("ffn_gate", "ffn_up", "ffn_down")):
            w.add_tensor(name, quants.quantize(arr, qt), raw_dtype=qt)      # gguf-py's reference quantizer (bit-identical to ggml)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--like-real", metavar="REAL.gguf", help="take rope/eps/interval/ratios and the tensor-name set from a real qwen35moe GGUF")
    ap.add_argument("--heads", metavar="N,KV", help="attention heads and KV heads (default 2,1; use 4,2 to make GQA non-degenerate)")
    ap.add_argument("--head-dim", type=int, metavar="D", help="attention head_dim (default 32); n_rot becomes D/4, the IMRoPE sections the real "
                    "[11,11,10,0] in proportion, rope base 1e7: 256 is the engine's kernel width (step 3's attention parity)")
    ap.add_argument("--no-nextn", action="store_true", help="omit the MTP block (the real UD-IQ3_XXS file has none)")
    ap.add_argument("--fused-gate-up", action="store_true", help="ffn_gate_up_exps instead of ffn_gate_exps + ffn_up_exps")
    ap.add_argument("--fused-qkv", action="store_true", help="attention layers carry attn_qkv instead of attn_q/k/v")
    ap.add_argument("--expert-weights-scale", type=float, help="write a stray qwen35moe.expert_weights_scale key (llama.cpp ignores it)")
    ap.add_argument("--gdn", metavar="K,V,S", help="GDN k-heads, v-heads, state size (default 4,8,16; the engine's kernels need S = 128)")
    ap.add_argument("--experts", metavar="E,K", help="expert count and top-k (default 8,2; the real model is 256,8)")
    ap.add_argument("--shexp-ff", type=int, metavar="N", help="shared expert width (default 32; the real model is 512)")
    ap.add_argument("--quant-shexp", metavar="TYPE", help="write ffn_{gate,up,down}_shexp in this ggml type (e.g. q8_0) instead of F32")
    ap.add_argument("--bf16-exact", action="store_true",
                    help="ssm_alpha, ssm_beta, ffn_gate_inp, ffn_gate_inp_shexp hold BF16-representable F32 values (what the engine reads them as)")
    a = ap.parse_args()
    real = None
    if a.like_real:
        c, real = real_config(a.like_real)
    else:
        c = Cfg()
    c.seed = a.seed
    if a.heads:
        c.n_head, c.n_head_kv = (int(x) for x in a.heads.split(","))
    if a.head_dim:
        c.head_dim, c.n_rot, c.rope_base = a.head_dim, a.head_dim // 4, 1e7
        c.rope_sections = tuple(split_sections([11, 11, 10, 0], c.n_rot // 2))
    if a.no_nextn:
        c.nextn = False
    c.fused_gate_up |= a.fused_gate_up
    c.fused_qkv |= a.fused_qkv
    c.expert_weights_scale = a.expert_weights_scale
    if a.gdn:
        c.ssm_k_heads, c.ssm_v_heads, c.ssm_state = (int(x) for x in a.gdn.split(","))
    if a.experts:
        c.n_expert, c.n_expert_used = (int(x) for x in a.experts.split(","))
    if a.shexp_ff:
        c.n_ff_shexp = a.shexp_ff
    c.quant_shexp = a.quant_shexp
    c.bf16_exact = a.bf16_exact
    write(a.out, c, " like-real" if real else "")
    names = {norm_name(n) for n, _ in build_tensors(c)}
    print(f"wrote {a.out}: {len(build_tensors(c))} tensors, block_count {c.n_layer + int(c.nextn)}")
    if real:
        rn = {norm_name(n) for n in real["tensors"]}
        # the real file's trunk blocks vs ours; nextn tensors only exist in blk.<block_count-1>
        extra, missing = sorted(names - rn), sorted(rn - names)
        print(f"name-set check vs {a.like_real}: {len(names)} synthetic patterns, {len(rn)} real")
        print("  only in synthetic:", extra or "none")
        print("  only in real     :", missing or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
