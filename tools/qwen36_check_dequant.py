#!/usr/bin/env python3
"""tools/qwen36_check_dequant.py - ref/load.py (gguf-py) dequantization against ggml's own to_float, on a real GGUF.

    python3 tools/qwen36_check_dequant.py MODEL.gguf /path/to/qwen36-dequant   (built from tools/qwen36_dequant_ggml.cpp)

Checks token_embd rows, one routed expert of each distinct quant type in the first and last blocks, and a dense
projection of each type.  Prints the max |difference| per tensor; exit 1 if any exceeds 1e-6 * max|x|.
"""
import subprocess, sys, tempfile, pathlib
import numpy as np
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from _paths import add_gguf_py
add_gguf_py()
from ref.load import Gguf


def main():
    path, tool = sys.argv[1], sys.argv[2]
    g = Gguf(path)
    cases, seen = [], set()
    for t in g.g.tensors:
        key = (t.type_name, len(t.shape))
        if t.name == "token_embd.weight":
            cases.append((t.name, "rows", 12345))
        elif key not in seen and len(t.shape) >= 2 and t.type_name not in ("F32", "F16", "BF16"):
            seen.add(key); cases.append((t.name, "e" if len(t.shape) == 3 else "all", 5))
    for t in g.g.tensors:          # every quant type that occurs as a routed-expert tensor, wherever it first appears
        if len(t.shape) == 3 and (t.type_name, 9) not in seen:
            seen.add((t.type_name, 9)); cases.append((t.name, "e", 7))
    bad = 0
    for name, mode, idx in cases:
        t = g.info(name)
        if mode == "rows":
            ref = g.rows(name, [idx, idx + 1, 247000])
            slabs = [(r * t.shape[0], t.shape[0]) for r in (idx, idx + 1, 247000)]
            ref = ref.reshape(-1)
        elif mode == "e":
            per = t.shape[0] * t.shape[1]
            ref = g.expert(name, idx).reshape(-1); slabs = [(idx * per, per)]
        else:
            ref = g.tensor(name).reshape(-1)[: 256 * 4096]; slabs = [(0, ref.size)]
        outs = []
        for first, n in slabs:
            with tempfile.NamedTemporaryFile() as f:
                subprocess.run([tool, path, name, str(first), str(n), f.name], check=True, capture_output=True)
                outs.append(np.fromfile(f.name, dtype=np.float32))
        got = np.concatenate(outs)
        d, m = float(np.abs(got - ref).max()), float(np.abs(got).max())
        ok = d <= 1e-6 * max(m, 1e-9)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:34s} {t.type_name:8s} {mode:4s} max|d| {d:.2e}  max|x| {m:.3g}")
    return 1 if bad else 0

sys.exit(main())
