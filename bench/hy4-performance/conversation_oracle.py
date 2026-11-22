#!/usr/bin/env python3
"""Prepare matched conversation IDs or compare teacher-forced assistant distributions."""

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "correctness"))
from hy4_oracle import measure_logits
from score import logsoftmax


def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def save_ids(path, ids):
    path.write_text(" ".join(map(str, ids)) + "\n")


def prepare(args, cases):
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for name, case in cases.items():
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=False)
        prompt = post(args.url + "/apply-template", {
            "messages": case["messages"], "reasoning_effort": case["reasoning_effort"]
        })["prompt"]
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        front_ids = post(args.url + "/tokenize", {"content": prompt, "parse_special": True})["tokens"]
        if ids != front_ids:
            raise ValueError(f"{name}: checkpoint and server tokenizers disagree")
        target = tokenizer.encode(case["assistant_continuation"], add_special_tokens=False).ids
        (directory / "rendered.txt").write_text(prompt)
        save_ids(directory / "prompt.ids", ids)
        save_ids(directory / "target.ids", target)


def sampled_distribution(row):
    # Conditional target distribution for temperature 0.9, top-k 64, top-p 1.
    # Native top-k breaks equal-logit ties by ascending vocabulary ID.
    top = np.argsort(-row, kind="stable")[:64]
    probabilities = np.zeros(len(row))
    probabilities[top] = np.exp(logsoftmax(row[top] / 0.9))
    return probabilities


def score(reference, candidate, targets):
    rows = []
    for a, b, token in zip(reference, candidate, targets, strict=True):
        a, b = a.astype(np.float64), b.astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite logits")
        la, lb = logsoftmax(a), logsoftmax(b)
        pa, pb = np.exp(la), np.exp(lb)
        sampled_tv = np.abs(sampled_distribution(a) - sampled_distribution(b)).sum() / 2
        rows.append([pa @ (la-lb), np.abs(pa-pb).sum()/2, sampled_tv,
                     -la[token], -lb[token], a.argmax()==b.argmax(), a.argmax()==token, b.argmax()==token])
    rows = np.asarray(rows)
    kl, tv, sampled_tv, loss_a, loss_b, agree, acc_a, acc_b = rows.mean(axis=0)
    return dict(positions=len(targets), mean_kl_nats=kl, max_kl_nats=rows[:, 0].max(),
                mean_tv=tv, max_tv=rows[:, 1].max(), mean_sampled_tv=sampled_tv,
                max_sampled_tv=rows[:, 2].max(),
                oracle_ppl=math.exp(loss_a), candidate_ppl=math.exp(loss_b),
                top1_agreement=agree, oracle_accuracy=acc_a, candidate_accuracy=acc_b,
                passed=bool(kl<=0.001 and abs(math.expm1(loss_b-loss_a))<=0.01 and agree>=0.99
                            and sampled_tv<=0.01))


def compare(args, cases):
    report = {}
    for name in cases:
        directory = args.out / name
        prompt = list(map(int, (directory/"prompt.ids").read_text().split()))
        targets = list(map(int, (directory/"target.ids").read_text().split()))
        ids = prompt + targets[:-1]
        # The GPU runner exports its final chunk. Use one chunk for the whole input
        # so every row from the last prompt token through the assistant answer is available.
        flags = ["--model", args.model, "--engram", args.engram, "--ctx", "4096",
                 "--chunk", str(len(ids)), "--gpus", "0,1", "--draft", "off"]
        arrays = [measure_logits(engine, flags, ids, directory / arm)
                  for engine, arm in [(args.oracle, "stock"), (args.candidate, "fork")]]
        if any(a.shape != (len(ids), 120832) for a in arrays):
            raise ValueError(f"{name}: incomplete full-vocabulary dump")
        result = score(*(a[len(prompt)-1:] for a in arrays), targets)
        result.update(prompt_tokens=len(prompt), vocabulary=120832,
                      prompt_sha256=hashlib.sha256((directory/"prompt.ids").read_bytes()).hexdigest(),
                      byte_identical=bool(np.array_equal(arrays[0].view(np.uint32), arrays[1].view(np.uint32))))
        report[name] = result
        (args.out/"distributions.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(result), flush=True)
    return all(row["passed"] for row in report.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("conversations.json"))
    parser.add_argument("--out", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--url", required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    cmp = sub.add_parser("compare")
    for key in ("oracle", "candidate", "model", "engram"):
        cmp.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if args.command == "prepare":
        prepare(args, cases)
        return 0
    return 0 if compare(args, cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
