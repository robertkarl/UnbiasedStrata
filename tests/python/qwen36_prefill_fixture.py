#!/usr/bin/env python3
"""The Qwen3.6 prompt-path harness's fixtures: synthetic qwen35moe GGUFs at the model's per-layer geometry, and the
oracle's dumps for them (docs/qwen36-port/step3-prefill.md).

The prompt path's kernels are built for the real widths (n_embd 2048, GDN 16 key heads / 32 value heads of 128 with
8192 conv channels, attention 16 heads / 2 KV heads of 256, experts of 512), so the oracle's own 64-wide toy fixture
cannot run through them.  This writes the same synthetic model (tools/qwen36_synth_gguf.py's weights) at those widths
with few layers and few experts:

    gdn3     3 layers, all Gated DeltaNet (interval 4: no layer satisfies (il + 1) % 4 == 0), 16 experts top-8
    full4    4 layers: three DeltaNet, layer 3 full attention (the attention kernels' case)

The projections and the experts are Q8_0 (what the engine serves natively: Gemm::native dequantizes them to FP16,
the prompt experts take the MMQ or the FP16 path), the router / ssm_alpha / ssm_beta / ffn_gate_inp_shexp are F32
values rounded to BF16 (the engine's forms, exact), the norms, conv, dt bias and ssm_a F32 as in the real file.  The
oracle (ref/qwen36.py) dequantizes with gguf-py, bit-identical to ggml, so every difference the harness sees is the
engine's arithmetic (FP16 activations, int8 MMQ), not the weights.

For each case: <out>/<case>.gguf, the oracle's dumps for 40 tokens in <out>/<case>/o (the faithful reading) and
<out>/<case>/o_q8 (--act-quant q8: the activations rounded as ggml does before a quantized matmul), and the negative
controls <out>/<case>/neg_<name>: the oracle with one reading flipped to its rival (gdn_gate=sigmoid,
head_pairing=interleaved, router_norm=False; attn_gate=False on full4).  Without numpy or gguf-py it writes
<out>/SKIP with the reason and exits 0, and the test exits 77 on seeing it (a skip, not a failure).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

try:
    import numpy as np
    import _paths
    import qwen36_synth_gguf as qs          # adds gguf-py to sys.path on import (tools/_paths.py)
    from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
except ImportError as e:                    # a trimmed third_party or no numpy: a skip, not a failure
    print(f"qwen36_prefill_fixture: {e}", file=sys.stderr)
    out = None
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--out" and i < len(sys.argv) - 1:
            out = sys.argv[i + 1]
        elif a.startswith("--out="):
            out = a[len("--out="):]
    if out:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out) / "SKIP").write_text(f"{e}\n")
    sys.exit(0)

# 40 tokens: the ASCII codes of this text (the fixture's vocabulary is the 128 byte tokens)
TEXT = "the quick brown fox jumps over the lazy "
TOKENS = [ord(c) for c in TEXT]
assert len(TOKENS) == 40

Q8_SUFFIXES = (".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
               ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
               ".ffn_down_shexp.weight", ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight")
BF16_SUFFIXES = (".ffn_gate_inp.weight", ".ssm_alpha.weight", ".ssm_beta.weight", ".ffn_gate_inp_shexp.weight")


def to_bf16_exact(a: np.ndarray) -> np.ndarray:
    """The F32 values rounded to the nearest BF16 (round-to-nearest-even), returned as F32: the engine's BF16 image of
    the tensor is then exact, and so is the oracle's F32 read of it."""
    u = a.astype(np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32)


def cfg(case: str, seed: int) -> qs.Cfg:
    c = qs.Cfg(n_embd=2048, n_layer=3 if case == "gdn3" else 4, interval=4, n_vocab=128, n_expert=16, n_expert_used=8,
               n_ff_exp=512, n_ff_shexp=512, ssm_k_heads=16, ssm_v_heads=32, ssm_state=128, ssm_conv=4, n_head=16,
               n_head_kv=2, head_dim=256, n_rot=64, rope_sections=(11, 11, 10, 0), rope_base=1e7, rms_eps=1e-6,
               n_ctx_train=4096, nextn=False, seed=seed)
    # Cfg.check() demands a full-attention layer; gdn3 deliberately has none (llama.cpp's rule puts attention at
    # (il + 1) % 4 == 0, which no layer of three reaches), so its invariants are checked here instead
    assert c.ssm_v_heads % c.ssm_k_heads == 0 and c.n_head % c.n_head_kv == 0 and c.n_expert_used <= c.n_expert
    assert sum(c.rope_sections) * 2 == c.n_rot
    return c


def write(path: str, c: qs.Cfg, q8_embd: bool = False) -> tuple[int, int]:
    """qwen36_synth_gguf.write with the per-tensor forms above.  Returns (tensors, q8 tensors).  `q8_embd` writes
    token_embd.weight as Q8_0 too (a native pack's embedding must be a type iq_embed_rows serves: the runner's fixture,
    tests/python/qwen36_runner_fixture.py); the prompt-path harness keeps it F32."""
    w = GGUFWriter(path, qs.ARCH)
    k = lambda s: f"{qs.ARCH}.{s}"
    w.add_name("qwen36-synth prefill-geometry")
    w.add_uint32(k("context_length"), c.n_ctx_train)
    w.add_uint32(k("embedding_length"), c.n_embd)
    w.add_uint32(k("block_count"), c.n_layer)
    w.add_uint32(k("attention.head_count"), c.n_head)
    w.add_uint32(k("attention.head_count_kv"), c.n_head_kv)
    w.add_uint32(k("attention.key_length"), c.head_dim)
    w.add_uint32(k("attention.value_length"), c.head_dim)
    w.add_float32(k("attention.layer_norm_rms_epsilon"), c.rms_eps)
    w.add_uint32(k("rope.dimension_count"), c.n_rot)
    w.add_array(k("rope.dimension_sections"), list(c.rope_sections))
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
    w.add_file_type(7)                                     # LLAMA_FTYPE_MOSTLY_Q8_0
    toks = qs.byte_unicode()
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types([1] * c.n_vocab)
    w.add_token_merges(["Ā ā"])
    w.add_bos_token_id(0)
    w.add_eos_token_id(3)
    w.add_add_bos_token(False)
    n, nq = 0, 0
    for name, arr in qs.build_tensors(c):
        n += 1
        if name.endswith(Q8_SUFFIXES) or name == "output.weight" or (q8_embd and name == "token_embd.weight"):
            w.add_tensor(name, quants.quantize(arr, Q.Q8_0), raw_dtype=Q.Q8_0)   # the byte shape; gguf-py derives the elements
            nq += 1
        elif name.endswith(BF16_SUFFIXES):
            # the router's logits at half the toy's gain: with 16 experts at the toy's spread the top-8 carry 99 % of the
            # softmax mass and the renormalisation rival (router_norm=False) moves ffn_out by 1 %, invisible to a 5 %
            # gate; at this gain the top-8 carry less and the rival is seen (the harness measures both)
            if name.endswith(".ffn_gate_inp.weight"):
                arr = (arr * 0.5).astype(np.float32)
            w.add_tensor(name, to_bf16_exact(arr))
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n, nq


def oracle(gguf: str, dump: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-I", str(ROOT / "ref" / "qwen36.py"), gguf, "--tokens", ",".join(map(str, TOKENS)),
           "--dump", dump] + extra
    env = dict(os.environ)
    env.setdefault("STRATA_GGUF_PY", _paths.gguf_py())
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


NEGATIVES = {"gate": ["--opt", "gdn_gate=sigmoid"], "pair": ["--opt", "head_pairing=interleaved"],
             "norm": ["--opt", "router_norm=False"], "attngate": ["--opt", "attn_gate=False"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", default="gdn3,full4")
    ap.add_argument("--no-oracle", action="store_true", help="write the GGUFs only")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in a.cases.split(","):
        c = cfg(case, a.seed)
        gguf = str(out / f"{case}.gguf")
        n, nq = write(gguf, c)
        print(f"wrote {gguf}: {n} tensors ({nq} Q8_0), {c.n_layer} layers, {c.n_expert} experts top-{c.n_expert_used}")
        if a.no_oracle:
            continue
        d = out / case
        oracle(gguf, str(d / "o"), [])
        oracle(gguf, str(d / "o_q8"), ["--act-quant", "q8"])
        for name, opt in NEGATIVES.items():
            if name == "attngate" and case == "gdn3":
                continue
            oracle(gguf, str(d / f"neg_{name}"), opt)
        print(f"  oracle dumps in {d}: o, o_q8, " + ", ".join(f"neg_{k}" for k in NEGATIVES if not (k == "attngate" and case == "gdn3")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
