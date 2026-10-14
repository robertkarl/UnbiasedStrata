#!/usr/bin/env python3
"""tools/qwen36_spread.py - per-seam-class spread between oracle dumps and a llama.cpp dump of the SAME real quantized file.

    python3 tools/qwen36_spread.py LLAMA_DIR LABEL=ORACLE_DIR [LABEL=ORACLE_DIR ...]

For every seam of a class and every layer: rel_rms = ||oracle - llama|| / ||llama||.  Prints, per class and label,
the median and the maximum over layers.  Use it to set per-seam tolerances for the CUDA parity tests (docs/qwen36-port/oracle-crosscheck.md).
"""
import re, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qwen36_compare_dumps as C

CLASSES = [("norm (attn_norm)", r"attn_norm-\d+$"), ("qkv_mixed (GDN in-proj)", r"linear_attn_qkv_mixed-\d+$"),
           ("gdn_out (linear_attn_out)", r"linear_attn_out-\d+$"), ("attn_out (attention o-proj)", r"attn_output-(3|7|11|15|19|23|27|31|35|39)$"),
           ("moe_out (ffn_moe_out)", r"ffn_moe_out-\d+$"), ("ffn_out (routed+shared)", r"ffn_out-\d+$"),
           ("l_out (layer output)", r"l_out-\d+$"), ("logits (result_output)", r"result_output$")]

def main():
    l = C.load(sys.argv[1])
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    print("| class | " + " | ".join(f"{lab} median / max" for lab, _ in runs) + " |")
    print("| --- | " + " | ".join("---" for _ in runs) + " |")
    loaded = {lab: C.load(d) for lab, d in runs}
    for cname, pat in CLASSES:
        cells = []
        for lab, _ in runs:
            o = loaded[lab]
            v = [float(np.linalg.norm(o[s][2] - l[s][2]) / np.linalg.norm(l[s][2])) for s in o if re.match(pat, s) and s in l]
            cells.append(f"{np.median(v):.1e} / {max(v):.1e}" if v else "-")
        print(f"| {cname} | " + " | ".join(cells) + " |")
    for lab, _ in runs:
        o = loaded[lab]
        n = sum(int(np.sum(o[f"ffn_moe_topk-{i}"][2] != l[f"ffn_moe_topk-{i}"][2])) for i in range(40) if f"ffn_moe_topk-{i}" in o)
        tot = sum(o[f"ffn_moe_topk-{i}"][2].size for i in range(40) if f"ffn_moe_topk-{i}" in o)
        print(f"<!-- {lab}: {n} of {tot} top-k slots differ from llama.cpp (position-wise) -->")
main()
