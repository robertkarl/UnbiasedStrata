#!/usr/bin/env python3
"""tools/qwen36_router_bf16_flips.py - how often rounding Qwen3.6's F32 router weight to BF16 flips a top-8 choice.

The shipped files store `blk.N.ffn_gate_inp.weight` in F32 and llama.cpp multiplies in F32; the engine's router
kernel reads a BF16 weight (`project_bf16` / `bf16_gemv_fp32_mmvf`, F32 activation x BF16 weight) and the pack
rounds the F32 tensor to BF16 under --compat-bf16 (docs/QWEN36-35B-A3B.md 3.4, docs/qwen36-port/upstream-tooling.md
section 4).  A flipped selection is a different expert, not a rounding error, so the study asks for the flip rate to
be MEASURED on the real file, per layer, and not asserted: this script is that measurement, a number for the step-4
--compat-bf16 decision.

    python3 -I ref/qwen36.py REAL.gguf --tokens T0,..,T7 --dump DIR --act-quant q8     # the router's inputs per layer
    python3 -I tools/qwen36_router_bf16_flips.py REAL.gguf DIR [--k 8]

For every layer it takes the oracle's `attn_post_norm-N` (the FFN input of the N-th layer, (T, n_embd)) and the file's
F32 router weight, forms the logits three ways - F32 x F32 (llama.cpp and the oracle; checked against the oracle's
own `ffn_moe_logits-N`), F32 x BF16(W) (the engine's native MMVF contract) and BF16(x) x BF16(W) (the legacy
`bf16_gemv` contract, both operands rounded) - and reports per layer: positions of the top-k that differ, experts
in one top-k set and not the other, tokens with any difference, and the smallest logit gap across the top-k boundary
(a flip needs the rounding error to exceed that gap), and the share of the layer's F32 router weight that is
already BF16-representable (low 16 bits zero).  BF16 rounding is round-to-nearest-even, the engine's `bf16_from_f32`.

Measured 2026-10-11 on both shipped files (UD-IQ3_XXS and UD-Q4_K_XL): every element of ffn_gate_inp,
ffn_gate_inp_shexp, ssm_alpha, ssm_beta and ssm_conv1d is BF16-exact (the checkpoint is BF16; the converter stored
them as F32), so F32 x BF16(W) is bitwise F32 x F32 and the flip rate of the engine's native contract is zero by
construction; the pack needs no --compat-bf16 for them.  attn_norm is not BF16-exact (17%) and is read as F32.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ref"))
from load import Gguf  # noqa: E402

F32 = np.float32


def bf16_round(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even to BF16, returned as F32 (strata/kernels/bf16_bits.hpp bf16_from_f32)."""
    u = np.ascontiguousarray(a, dtype=F32).view(np.uint32)
    u = (u + ((u >> 16) & 1) + np.uint32(0x7FFF)) & np.uint32(0xFFFF0000)
    return u.view(F32)


def topk_desc(logits: np.ndarray, k: int) -> np.ndarray:
    """llama.cpp's selection: stable argsort of the softmax probabilities, descending, = of the logits."""
    return np.argsort(-logits, axis=1, kind="stable")[:, :k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("dump", help="ref/qwen36.py --dump directory (needs attn_post_norm-N and ffn_moe_logits-N)")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    g = Gguf(a.gguf)
    dump = pathlib.Path(a.dump)
    n_layers = 0
    while (dump / f"ffn_moe_logits-{n_layers}.f32").exists():
        n_layers += 1
    if n_layers == 0:
        sys.exit(f"{dump}: no ffn_moe_logits-N.f32 (run ref/qwen36.py --dump first)")
    K = a.k
    print(f"{a.gguf}: {n_layers} layers, top-{K}; router weight type "
          f"{g.info('blk.0.ffn_gate_inp.weight').type_name}; dump {dump}")
    print(f"{'layer':>5} {'oracle':>8} {'W bf16':>7} | {'F32 x BF16(W)':^30} | {'BF16(x) x BF16(W)':^30} | {'min gap':>9}")
    print(f"{'':>5} {'max|d|':>8} {'-exact':>7} | {'slots':>6} {'set':>6} {'tokens':>7} {'rate':>7} | {'slots':>6} {'set':>6} {'tokens':>7} {'rate':>7} | {'(logits)':>9}")
    tot = {"w": [0, 0, 0], "xw": [0, 0, 0], "slots": 0, "tokens": 0}
    worst_oracle = 0.0
    for il in range(n_layers):
        w = g.tensor(f"blk.{il}.ffn_gate_inp.weight").astype(F32)                      # (E, n_embd)
        E, n_embd = w.shape
        x = np.fromfile(dump / f"attn_post_norm-{il}.f32", dtype=F32).reshape(-1, n_embd)   # (T, n_embd)
        T = x.shape[0]
        ref = np.fromfile(dump / f"ffn_moe_logits-{il}.f32", dtype=F32).reshape(T, E)
        l32 = (x @ w.T).astype(F32)
        worst_oracle = max(worst_oracle, float(np.abs(l32 - ref).max()))
        wb = bf16_round(w)
        exact = float((wb == w).mean())                   # the share of the weight the rounding leaves untouched
        l_w = (x @ wb.T).astype(F32)                      # the native MMVF contract: F32 activation, BF16 weight
        l_xw = (bf16_round(x) @ wb.T).astype(F32)         # the legacy bf16_gemv contract: both rounded
        t32 = topk_desc(ref, K)                           # the oracle's own selection (identical to llama.cpp's)
        gap = float((np.sort(ref, axis=1)[:, ::-1][:, K - 1] - np.sort(ref, axis=1)[:, ::-1][:, K]).min())
        row = [f"{il:>5} {np.abs(l32 - ref).max():>8.1e} {exact:>7.4f} |"]
        for key, lg in (("w", l_w), ("xw", l_xw)):
            tk = topk_desc(lg, K)
            slots = int((tk != t32).sum())
            sets = sum(len(set(t32[t]) - set(tk[t])) for t in range(T))
            toks = int(sum(bool(set(t32[t]) != set(tk[t])) for t in range(T)))
            tot[key][0] += slots; tot[key][1] += sets; tot[key][2] += toks
            row.append(f" {slots:>6} {sets:>6} {toks:>7} {sets / (T * K):>7.3f} |")
        tot["slots"] += T * K; tot["tokens"] += T
        row.append(f" {gap:>9.2e}")
        print("".join(row))
    n = tot["slots"]
    print(f"\n{'total':>5} {worst_oracle:>8.1e} {'':>7} |", end="")
    for key in ("w", "xw"):
        s, st, tk = tot[key]
        print(f" {s:>6} {st:>6} {tk:>7} {st / n:>7.4f} |", end="")
    print(f"\n  slots = positions of the top-{K} that differ (order included); set = experts chosen by F32 and not by the "
          f"rounded weight (order ignored); tokens = tokens whose top-{K} SET differs; rate = set / ({tot['tokens']} tokens x {K}).")
    print("  'oracle max|d|' is this script's F32 logits against the oracle's ffn_moe_logits-N (a check of the inputs, not a result);")
    print("  'W bf16-exact' is the share of the layer's F32 router weight whose low 16 bits are zero (1.0000: BF16(W) == W, F32 x BF16(W) is F32 x F32).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
