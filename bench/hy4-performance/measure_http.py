#!/usr/bin/env python3
"""Measure /completion using the prompt.ids saved by measure.py."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def trial(url, ids, limit, temperature, top_k):
    payload = {
        "prompt": ids, "n_predict": limit, "temperature": temperature,
        "top_k": top_k, "top_p": 1, "min_p": 0, "repeat_penalty": 1,
        "stream": True, "return_tokens": True, "cache_prompt": False,
    }
    request = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    output_count = 0
    timings = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: {"):
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            tokens = event.get("tokens", [])
            if tokens:
                last = time.monotonic()
                if first is None:
                    first = last
                output_count += len(tokens)
            if event.get("stop"):
                timings = event["timings"]
    if timings is None or first is None or output_count < 2 or last <= first:
        raise RuntimeError("incomplete streamed completion")
    if timings["prompt_n"] != len(ids) or timings["predicted_n"] != output_count:
        raise RuntimeError(f"unexpected token accounting: {timings}, streamed {output_count}")
    return dict(input=len(ids), output=output_count, requested=limit, ttft_s=first-start,
                client_pp_tps=len(ids)/(first-start), client_tg_tps=(output_count-1)/(last-first),
                timings=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18864/completion")
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--generate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=128)
    args = parser.parse_args()
    ids = [int(token) for token in args.prompt_ids.read_text().split()]
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(url=args.url, prompt_sha256=hashlib.sha256(args.prompt_ids.read_bytes()).hexdigest(),
                  temperature=args.temperature, top_k=args.top_k, top_p=1, min_p=0, trials=[])
    if args.warmup:
        result["warmup"] = trial(args.url, ids, args.warmup, args.temperature, args.top_k)
    for _ in range(args.trials):
        result["trials"].append(trial(args.url, ids, args.generate, args.temperature, args.top_k))
        (args.out/"trials.json").write_text(json.dumps(result, indent=2) + "\n")
    result["median"] = {key: statistics.median(t[key] for t in result["trials"])
                        for key in ("client_pp_tps", "client_tg_tps")}
    (args.out/"result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median"]), flush=True)


if __name__ == "__main__":
    main()
