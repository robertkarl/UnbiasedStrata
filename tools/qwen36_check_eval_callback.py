#!/usr/bin/env python3
"""tools/qwen36_check_eval_callback.py - check the printed output of the submodule's `llama-eval-callback`
against the numpy oracle's dumps.

    llama-eval-callback -m MODEL.gguf -p "hello" -ctk f32 -ctv f32 -fa off -c 64 -t 1 -ngl 0 > eval.txt 2>&1
    python3 ref/qwen36.py MODEL.gguf --text hello --dump ORA
    python3 tools/qwen36_check_eval_callback.py ORA eval.txt [--tol 1e-4]

`llama-eval-callback` prints, per graph node, the first and last 3 entries along every dimension (`%12.4f`, so each
printed value carries up to 5e-5 of rounding) and a float32 `sum` of the whole tensor.  This reconstructs which
indices were printed from the shape (common/debug.cpp:75-117), looks them up in the oracle's array of the same
cb() name, and requires |printed - oracle| <= tol on every printed value and the sum to agree to tol of the
tensor's L1 mass.  Repeated names get "#2", "#3" in execution order, as in the dumps.  Nodes llama.cpp leaves
unnamed ("node_30", "x (view)") are ignored; "kq" and "kq_soft_max" are skipped because llama.cpp pads their
KV axis to its cache size.  Exit status 1 on any mismatch or if fewer than --min-seams names were checked.
"""
import argparse
import math
import re
import sys
import pathlib

import numpy as np

HEAD = re.compile(r"^common_debug_cb_eval:\s+(.+?) = \((\w+)\)\s+\w+\(.*\) = \{(\d+), (\d+), (\d+), (\d+)\}\s*$")
NUM = re.compile(r"(?<![\w.])-?(?:\d+\.\d+|nan|inf)(?![\w.])")
N_EDGE = 3
TRUNCATED = None


def printed_indices(ne):
    def dim(n):
        return list(range(n)) if n <= 2 * N_EDGE else list(range(N_EDGE)) + list(range(n - N_EDGE, n))
    for i3 in range(ne[3]):
        for i2 in dim(ne[2]):
            for i1 in dim(ne[1]):
                for i0 in dim(ne[0]):
                    yield (i3, i2, i1, i0)


def parse(path):
    """Yield (name, type, ne, values, sum) for each printed node, in execution order."""
    lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        name, typ = m.group(1), m.group(2)
        ne = tuple(int(m.group(k)) for k in range(3, 7))
        i += 1
        body = []
        total = None
        while i < len(lines) and not lines[i].startswith("common_debug_cb_eval:"):
            ln = lines[i]
            if ln.strip().startswith("sum ="):
                total = float(ln.split("=")[1])
                i += 1
                break
            if ln.strip().startswith("["):
                body.extend(NUM.findall(ln))
            i += 1
        if total is None:                 # llama.cpp's async log drops its queue at exit: the last block can be cut off
            global TRUNCATED
            TRUNCATED = name
            continue
        yield name, typ, ne, [float(v) for v in body], total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("eval_output")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--min-seams", type=int, default=100)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--require-tail", action="store_true", help="fail unless the printout reaches result_output")
    a = ap.parse_args()
    od = pathlib.Path(a.oracle)
    man = {}
    for ln in (od / "manifest.txt").read_text().splitlines():
        s, k, ne = ln.split()
        man[s] = (k, tuple(int(x) for x in ne.split(",")))
    seen, checked, bad, skipped = {}, 0, 0, []
    worst_val = worst_sum = 0.0
    for name, typ, ne, vals, total in parse(a.eval_output):
        base = name
        if base not in man and base + "#2" not in man and base != "":
            # not a cb() name the oracle knows (views, unnamed nodes, cache plumbing)
            seen[base] = seen.get(base, 0)
            continue
        n = seen[base] = seen.get(base, 0) + 1
        stem = base if n == 1 else f"{base}#{n}"
        if stem not in man:
            continue
        kind, one = man[stem]
        if base.split("-")[0] in ("kq", "kq_soft_max") or one != ne:
            skipped.append(f"{stem} (oracle ne {one}, llama ne {ne})")
            continue
        ref = np.fromfile(od / f"{stem}.f32", dtype=np.float32).reshape(tuple(reversed(ne)))
        idx = list(printed_indices(ne))
        if len(idx) != len(vals):
            print(f"FAIL  {stem}: printed {len(vals)} values, expected {len(idx)} for ne {ne}")
            bad += 1
            continue
        got = np.array(vals, dtype=np.float64)
        want = np.array([ref[i] for i in idx], dtype=np.float64)
        dv = float(np.max(np.abs(got - want))) if len(vals) else 0.0
        osum = float(np.cumsum(ref.ravel().astype(np.float32), dtype=np.float32)[-1])   # ggml's float32 running sum, in memory order
        mass = float(np.abs(ref).sum())
        ds = abs(osum - total) / max(1.0, mass)
        worst_val, worst_sum = max(worst_val, dv), max(worst_sum, ds)
        ok = dv <= a.tol and ds <= a.tol
        checked += 1
        if not ok:
            bad += 1
        if not ok or not a.quiet:
            print(f"{'ok  ' if ok else 'FAIL'}  {stem:34s} printed {len(vals):3d} values max|d| {dv:.2e}   sum oracle {osum: .6f} llama {total: .6f}  (|d|/L1 {ds:.1e})")
    if TRUNCATED:
        print(f"note: the last block ({TRUNCATED}) was cut off by llama-eval-callback's log shutdown; rerun, or use tools/qwen36_dump_llama.cpp for the tail")
    complete = "result_output" in seen
    print(f"\nchecked {checked} named nodes ({'through result_output' if complete else 'tail missing'}), {bad} failed; worst printed-value diff {worst_val:.2e}, worst sum diff {worst_sum:.2e} (of L1 mass)")
    if skipped:
        print("skipped:", "; ".join(skipped[:6]), "..." if len(skipped) > 6 else "")
    return 1 if bad or checked < a.min_seams or (a.require_tail and not complete) else 0


if __name__ == "__main__":
    sys.exit(main())
