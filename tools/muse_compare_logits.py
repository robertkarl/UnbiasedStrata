"""Compare teacher-forced native and reference dumps from muse-check."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("--vocab", type=int, default=202048)
    p.add_argument("--max-kl", type=float, default=.01)
    p.add_argument("--min-top1", type=float, default=.95)
    args = p.parse_args()
    if args.vocab <= 0:
        p.error("--vocab must be positive")
    token_files = [Path(str(x) + ".tokens").read_text() for x in (args.reference, args.native)]
    if token_files[0] != token_files[1]:
        p.error("token sequences differ; comparison must be teacher-forced")
    arrays = [np.fromfile(x, dtype="<f4") for x in (args.reference, args.native)]
    if arrays[0].size == 0 or arrays[0].size != arrays[1].size or arrays[0].size % args.vocab:
        p.error("empty or incompatible logit shapes")
    if arrays[0].size // args.vocab != len(token_files[0].splitlines()):
        p.error("token count does not match logit rows")
    if not all(np.isfinite(a).all() for a in arrays):
        p.error("non-finite logits")
    ref, native = [a.reshape(-1, args.vocab).astype(np.float64) for a in arrays]

    def log_probs(a):
        a = a - a.max(axis=1, keepdims=True)
        return a - np.log(np.exp(a).sum(axis=1, keepdims=True))

    lp, lq = log_probs(ref), log_probs(native)
    kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    agreement = float((ref.argmax(axis=1) == native.argmax(axis=1)).mean())
    result = {"positions": len(ref), "top1_agreement": agreement,
              "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
              "max_abs_error": float(np.abs(ref-native).max()),
              "rms_error": float(np.sqrt(np.mean((ref-native)**2)))}
    result["passed"] = agreement >= args.min_top1 and result["max_kl"] <= args.max_kl
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
