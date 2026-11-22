"""Teacher-force Inkling continuations and compare full-vocabulary CUDA logits.

python3 parity.py --maya /path/to/maya-ik2 --strata /path/to/strata-ik2 --model model.gguf
The optional image case takes separately encoded Maya and Strata SVE1 files and a
Maya CLI with IK2_EVAL_PICTURES wired to its existing GENI load_pictures function.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def write_ids(path, ids):
    path.write_text(",".join(map(str, ids)))


def logits(exe, model, prompt, targets, work, name, picture):
    # IK2_LOGITS_OUT starts after the first forced token. Feed the last prompt
    # token first so row zero predicts the first assistant token.
    write_ids(work / "prompt", prompt[:-1])
    write_ids(work / "force", [prompt[-1], *targets])
    out = work / (name + ".f32")
    cmd = [exe, "--model", model, "--ctx", "4096", "--chunk", "2048",
           "--tokens", "@" + str(work / "prompt"), "--gen", str(len(targets) + 1)]
    env = {**os.environ, "IK2_FORCE": str(work / "force"), "IK2_LOGITS_OUT": str(out),
           "IK2_KV": "fp8", "IK2_GATE_BF16": "1", "IK2_GATE_TILE": "64",
           "IK2_Q8_UNROLL": "1", "IK2_BATCH_PREFILL": "1"}
    if picture:
        if name == "strata":
            cmd.extend(["--pictures", picture])
        else:
            env["IK2_EVAL_PICTURES"] = picture
    with (work / (name + ".log")).open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = np.memmap(out, dtype="<f4", mode="r")
    return rows.reshape(len(targets), 200058)


def score(reference, candidate, targets):
    kls, nlls = [], [[], []]
    agree = 0
    correct = [0, 0]
    for ref, test, target in zip(reference, candidate, targets):
        pair = []
        for i, row in enumerate((ref, test)):
            lp = row.astype(np.float64)
            if not np.isfinite(lp).all():
                raise ValueError("Nonfinite logits")
            lp -= lp.max()
            lp -= np.log(np.exp(lp).sum())
            nlls[i].append(-lp[target])
            correct[i] += int(lp.argmax() == target)
            pair.append(lp)
        p, q = pair
        kls.append(float((np.exp(p) * (p - q)).sum()))
        agree += int(p.argmax() == q.argmax())
    return dict(tokens=len(targets), identical=bool(np.array_equal(reference.view('u4'), candidate.view('u4'))),
                mean_kl=float(np.mean(kls)), max_kl=max(kls), top1_agree=agree,
                oracle_correct=correct[0], candidate_correct=correct[1],
                oracle_ppl=float(np.exp(np.mean(nlls[0]))), candidate_ppl=float(np.exp(np.mean(nlls[1]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("maya", "strata", "model"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--maya-image-engine")
    parser.add_argument("--maya-picture")
    parser.add_argument("--strata-picture")
    args = parser.parse_args()
    image_args = (args.maya_image_engine, args.maya_picture, args.strata_picture)
    if any(image_args) and not all(image_args):
        parser.error("the image case requires all three image options")
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    for name, case in cases.items():
        image = name == "picture"
        if image and not all(image_args):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            try:
                ref = logits(args.maya_image_engine if image else args.maya, args.model,
                             case["prompt"], case["targets"], work, "maya", args.maya_picture if image else None)
                test = logits(args.strata, args.model, case["prompt"], case["targets"], work,
                              "strata", args.strata_picture if image else None)
                print(json.dumps({"case": name, **score(ref, test, case["targets"])}), flush=True)
            except subprocess.CalledProcessError:
                import sys
                for log in sorted(work.glob("*.log")):
                    print(log.read_text()[-4000:], file=sys.stderr)
                raise


if __name__ == "__main__":
    main()
