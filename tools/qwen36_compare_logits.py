#!/usr/bin/env python3
"""tools/qwen36_compare_logits.py - the engine's per-position logits against a reference's, teacher-forced.

    python3 -I tools/qwen36_compare_logits.py ENGINE.f32 REFERENCE.f32 --n-vocab N [--engine-header] [--rows R]
                                              [--routes-engine F --routes-ref F] [--json OUT]

ENGINE.f32 is `strata --dump-logits` (two int32, n_vocab and rows, then the rows; --engine-header is the default)
or raw rows; REFERENCE.f32 is raw rows of n_vocab F32 (tools/qwen36_teacher_llama.cpp's logits.f32, or the oracle's
result_output_all).  Row j of both is the distribution at the same position.  Reports, per position and summarised:

  top-1 agreement      the argmax of the two rows equal
  KL(ref || engine)    in nats, from the softmax of each row (mean, max, the position of the max)
  drift                Maya's quantized-mode gate: KL must not rise monotonically with the position - the slope of
                       KL against the position (least squares) and the Spearman rank correlation are printed; the
                       gate (--max-spearman) is one-sided - only a RISING KL is drift (Spearman > the bound), a
                       chaotic spread is flat and a falling KL is not drift
  routes               with --routes-*: "<row> <layer> e0 .. e7" lines from both sides, the share of probed
                       (row, layer) pairs whose top-k SETS are equal

Exit 1 when a gate given on the command line fails (--min-top1, --max-mean-kl, --max-max-kl, --max-spearman).
"""
import argparse
import json
import sys

import numpy as np


def load_engine(path, n_vocab, header):
    a = np.fromfile(path, dtype=np.float32)
    if header:
        h = np.frombuffer(a[:2].tobytes(), dtype=np.int32)
        if int(h[0]) != n_vocab:
            raise SystemExit(f"{path}: header says n_vocab {int(h[0])}, expected {n_vocab}")
        a = a[2:]
        rows = int(h[1])
        if a.size != rows * n_vocab:
            raise SystemExit(f"{path}: header promises {rows} rows, the file holds {a.size / n_vocab:.2f}")
    return a.reshape(-1, n_vocab)


def softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def read_routes(path):
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        row, layer = int(parts[0]), int(parts[1])
        out[(row, layer)] = frozenset(int(p) for p in parts[2:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("reference")
    ap.add_argument("--n-vocab", type=int, required=True)
    ap.add_argument("--engine-header", action="store_true", default=True)
    ap.add_argument("--no-engine-header", dest="engine_header", action="store_false")
    ap.add_argument("--rows", type=int, default=0, help="compare this many rows (0 = all common rows)")
    ap.add_argument("--routes-engine")
    ap.add_argument("--routes-ref")
    ap.add_argument("--json")
    ap.add_argument("--min-top1", type=float)
    ap.add_argument("--max-mean-kl", type=float)
    ap.add_argument("--max-max-kl", type=float)
    ap.add_argument("--max-spearman", type=float)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    e = load_engine(a.engine, a.n_vocab, a.engine_header)
    r = np.fromfile(a.reference, dtype=np.float32).reshape(-1, a.n_vocab)
    n = min(e.shape[0], r.shape[0])
    if a.rows:
        n = min(n, a.rows)
    if n == 0:
        raise SystemExit("no rows to compare")
    e, r = e[:n], r[:n]
    pe, pr = softmax(e), softmax(r)
    kl = np.sum(pr * (np.log(pr + 1e-30) - np.log(pe + 1e-30)), axis=1)
    top1 = (e.argmax(axis=1) == r.argmax(axis=1))
    top5 = np.array([len(set(np.argsort(-e[i])[:5]) & set(np.argsort(-r[i])[:5])) for i in range(n)])
    pos = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(pos, kl, 1)[0]) if n > 1 else 0.0
    rho = spearman(pos, kl) if n > 1 else 0.0
    res = {"rows": int(n), "top1": int(top1.sum()), "top1_rate": float(top1.mean()), "kl_mean": float(kl.mean()),
           "kl_max": float(kl.max()), "kl_argmax": int(kl.argmax()), "kl_median": float(np.median(kl)),
           "kl_first_half": float(kl[: n // 2].mean()), "kl_second_half": float(kl[n // 2:].mean()),
           "kl_slope_per_position": slope, "kl_spearman": rho, "top5_overlap_mean": float(top5.mean()),
           "rows_top1_differ": [int(i) for i in np.nonzero(~top1)[0]]}
    if a.routes_engine and a.routes_ref:
        re_, rr = read_routes(a.routes_engine), read_routes(a.routes_ref)
        common = sorted(set(re_) & set(rr))
        equal = sum(1 for k in common if re_[k] == rr[k])
        by_layer = {}
        for k in common:
            by_layer.setdefault(k[1], [0, 0])
            by_layer[k[1]][1] += 1
            by_layer[k[1]][0] += re_[k] == rr[k]
        res["routes_pairs"] = len(common)
        res["routes_equal"] = equal
        res["routes_by_layer"] = {str(l): f"{v[0]}/{v[1]}" for l, v in sorted(by_layer.items())}
        slot_eq = []
        for k in common:
            slot_eq.append(len(re_[k] & rr[k]) / max(1, len(rr[k])))
        res["routes_slot_overlap_mean"] = float(np.mean(slot_eq)) if slot_eq else 0.0
    if not a.quiet:
        print(f"{n} positions: top-1 agreement {res['top1']}/{n} ({res['top1_rate']:.4f}), top-5 overlap "
              f"{res['top5_overlap_mean']:.2f}/5")
        print(f"KL(ref||engine): mean {res['kl_mean']:.5f}  median {res['kl_median']:.5f}  max {res['kl_max']:.5f} "
              f"at position {res['kl_argmax']}; first half {res['kl_first_half']:.5f}, second half {res['kl_second_half']:.5f}")
        print(f"drift: KL slope {slope:+.2e} per position, Spearman(position, KL) {rho:+.3f}")
        if res["rows_top1_differ"]:
            print("top-1 differs at positions: " + ", ".join(map(str, res["rows_top1_differ"][:40])) +
                  (" ..." if len(res["rows_top1_differ"]) > 40 else ""))
        if "routes_pairs" in res:
            print(f"routes: {res['routes_equal']}/{res['routes_pairs']} probed (row, layer) top-k sets equal "
                  f"(per layer {res['routes_by_layer']}); mean slot overlap {res['routes_slot_overlap_mean']:.3f}")
        print("per position (pos top1 kl):")
        print(" ".join(f"{i}:{'=' if top1[i] else 'x'}{kl[i]:.3f}" for i in range(n)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    bad = []
    if a.min_top1 is not None and res["top1_rate"] < a.min_top1:
        bad.append(f"top-1 {res['top1_rate']:.4f} < {a.min_top1}")
    if a.max_mean_kl is not None and res["kl_mean"] > a.max_mean_kl:
        bad.append(f"mean KL {res['kl_mean']:.5f} > {a.max_mean_kl}")
    if a.max_max_kl is not None and res["kl_max"] > a.max_max_kl:
        bad.append(f"max KL {res['kl_max']:.5f} > {a.max_max_kl}")
    if a.max_spearman is not None and rho > a.max_spearman:
        bad.append(f"Spearman(position, KL) {rho:.3f} > {a.max_spearman}: KL rises with position (a leaking state); "
                   f"a negative correlation is not drift")
    if bad:
        print("GATE FAILED: " + "; ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
