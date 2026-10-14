#!/usr/bin/env python3
"""tools/qwen36_compare_dumps.py - compare two directories of cb()-named .f32 dumps seam by seam.

    python3 tools/qwen36_compare_dumps.py ORACLE_DIR LLAMA_DIR [--tol 1e-4] [--quiet]

Both directories hold <stem>.f32 + manifest.txt ("<stem> <type> <ne0>,<ne1>,<ne2>,<ne3>"), written by
ref/qwen36.py and tools/qwen36_dump_llama.cpp.  For every stem in the ORACLE manifest that also appears in
the llama manifest: shapes must match (llama.cpp pads the KV axis of "kq"/"kq_soft_max" to its cache size; the
padded part is cropped, and for "kq_soft_max" required to be exactly zero), then max |a-b| is reported with the
largest magnitude seen, and the seam fails when max |a-b| > tol * max(1, max|b|).  Integer seams (router
indices) must be identical.  Stems present on only one side are listed.  Exit status 1 on any failure.
"""
import argparse
import pathlib
import sys

import numpy as np


def load(d):
    d = pathlib.Path(d)
    out = {}
    for line in (d / "manifest.txt").read_text().splitlines():
        stem, kind, ne = line.split()
        ne = tuple(int(x) for x in ne.split(","))
        a = np.fromfile(d / f"{stem}.f32", dtype=np.float32)
        out[stem] = (kind, ne, a.reshape(tuple(reversed(ne))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("llama")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    a = ap.parse_args()
    o, l = load(a.oracle), load(a.llama)
    bad, rows = 0, []
    for stem, (kind, ne, x) in o.items():
        if stem not in l:
            continue
        lk, lne, y = l[stem]
        if lne != ne:
            crop = stem.split("-")[0] in ("kq", "kq_soft_max") and lne[1:] == ne[1:] and lne[0] >= ne[0]
            if not crop:
                rows.append((stem, "SHAPE", f"oracle ne {ne} vs llama ne {lne}"))
                bad += 1
                continue
            pad = y[..., ne[0]:]
            y = y[..., :ne[0]]
            if stem.startswith("kq_soft_max") and np.any(pad != 0):
                rows.append((stem, "PAD", "padded KV columns are not zero"))
                bad += 1
                continue
        if kind == "i32":
            ok = np.array_equal(x, y)
            rows.append((stem, "ok" if ok else "FAIL", "identical" if ok else f"{int((x != y).sum())} indices differ"))
            bad += 0 if ok else 1
            continue
        finite = np.isfinite(y)
        err = np.abs(np.where(finite, x - y, 0.0)).max() if x.size else 0.0
        mag = np.abs(np.where(finite, y, 0.0)).max() if x.size else 0.0
        if not np.array_equal(np.isfinite(x), finite) or not np.array_equal(np.isneginf(x), np.isneginf(y)):
            rows.append((stem, "FAIL", "non-finite pattern differs"))
            bad += 1
            continue
        ok = err <= a.tol * max(1.0, mag)
        rows.append((stem, "ok" if ok else "FAIL", f"max|d| {err:.3e}  max|x| {mag:.4g}"))
        bad += 0 if ok else 1
    for r in rows:
        if not a.quiet or r[1] != "ok":
            print(f"{r[1]:5s} {r[0]:36s} {r[2]}")
    only_o = sorted(set(o) - set(l))
    # "conv_state_update-N" is a view of the cache that llama.cpp evaluates BEFORE the copy that updates it, so it
    # holds the old state; the new state is "conv_state_last-N", which is compared.
    only_l = sorted(x for x in set(l) - set(o) if not x.startswith("conv_state_update-"))
    print(f"\ncompared {len(rows)} seams, {bad} failed (tol {a.tol:g} x max(1, max|x|))")
    print(f"oracle only ({len(only_o)}): {only_o}")
    print(f"llama only  ({len(only_l)}): {only_l}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
