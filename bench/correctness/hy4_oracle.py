#!/usr/bin/env python3
"""Compare two native Hy4 runners on the same weights. Requires numpy and tokenizers."""
import argparse
import json
import math
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from score import logsoftmax


def full_vocab_scores(reference, candidate, targets):
    """One finite log-softmax per full vocabulary. Row t predicts token t+1."""
    if reference.shape != candidate.shape or reference.shape != (len(targets) + 1, 120832):
        raise ValueError(f"unexpected logit shapes {reference.shape} and {candidate.shape}")
    scores = []
    for a, b, target in zip(reference, candidate, targets):
        a, b = a.astype(np.float64), b.astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite logits")
        a, b = logsoftmax(a), logsoftmax(b)
        scores.append([np.sum(np.exp(a) * (a - b)), -a[target], -b[target],
                       a.argmax() == b.argmax(), a.argmax() == target, b.argmax() == target])
    return np.array(scores)


def measure_logits(exe, args, ids, directory):
    directory.mkdir()
    tokens = directory / "ids.txt"
    tokens.write_text(" ".join(map(str, ids)))
    with (directory / "engine.log").open("w") as log:
        subprocess.run([exe, *args, "--tokens", tokens, "--all-logits", "--save-logits",
                        "--out", directory], stdout=log, stderr=log, check=True)
    return np.load(directory / "logits.npy", mmap_mode="r")


def serve_trials(exe, args, ids, log_path, count=3):
    """Engine protocol timings after one full warm-up. Require every requested token."""
    rates, client_rates, sequences = [], [], []
    n = 32
    with log_path.open("w") as log:
        p = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        try:
            for line in p.stdout:
                if line.startswith("READY "):
                    break
                if line.startswith(("ERR", "FATAL")):
                    raise RuntimeError(line)
            else:
                raise RuntimeError(f"engine exited before READY (see {log.name})")
            for trial in range(count + 1):
                start = time.monotonic()
                p.stdin.write(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
                p.stdin.flush()
                generated = []
                for line in p.stdout:
                    fields = line.split()
                    if line.startswith("T "):
                        arrived = time.monotonic()
                        if not generated:
                            first = arrived
                        generated.append(int(fields[1]))
                    elif line.startswith("DONE "):
                        if (len(generated) != n or int(fields[1]) != n or int(fields[2]) != len(ids)
                                or fields[5] != "length" or (len(fields) > 8 and int(fields[8]) != 0)):
                            raise RuntimeError(f"unexpected generation: {line}")
                        if trial:
                            rates.append((len(ids) * 1000 / float(fields[3]), n * 1000 / float(fields[4])))
                            client_rates.append((len(ids) / (first - start), (n - 1) / (arrived - first)))
                            sequences.append(generated)
                        break
                    elif line.startswith(("ERR", "FATAL")):
                        raise RuntimeError(line)
                else:
                    raise RuntimeError(f"engine exited during generation (see {log.name})")
            p.stdin.write("QUIT\n")
            p.stdin.close()
            if p.wait(timeout=60) != 0:
                raise RuntimeError(f"engine failed at shutdown (see {log.name})")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
    return rates, client_rates, sequences


def http_trials(opt, args, ids, tokenizer, log_path):
    base = f"http://127.0.0.1:{opt.port}"

    def request(path, payload):
        return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(payload).encode(),
                                      {"Content-Type": "application/json"}), timeout=600)

    with log_path.open("w") as log:
        cmd = [opt.frontend, "--engine", opt.candidate, "-m", opt.model, "--port", str(opt.port),
               "--alias", "hy4-oracle-test", "--jinja", "--", *args]
        p = subprocess.Popen(cmd, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 300
            while p.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(base + "/health", timeout=3) as response:
                        if json.load(response).get("model") == "hy4-oracle-test":
                            break
                except OSError:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f"front end did not become ready (see {log.name})")
            text = "简体中文123456, café\nカタカナ <think:opensource>"
            with request("/tokenize", {"content": text, "parse_special": True}) as r:
                if json.load(r)["tokens"] != tokenizer.encode(text, add_special_tokens=False).ids:
                    raise RuntimeError("HTTP tokenizer differs from the checkpoint")
            with request("/v1/chat/completions", {
                "messages": [{"role": "user", "content": "What is 2 plus 2? Give only the number."}],
                "reasoning_effort": "none", "temperature": 0, "max_tokens": 24,
            }) as r:
                chat = json.load(r)["choices"][0]["message"]
                if chat.get("content", "").strip().rstrip(".") != "4":
                    raise RuntimeError(f"expected 4 from the arithmetic chat check: {chat}")
            rates, sequences = [], []
            for i in range(4):
                generated = []
                payload = dict(prompt=ids, n_predict=32, temperature=0, stream=True, return_tokens=True)
                start = time.monotonic()
                with request("/completion", payload) as stream:
                    for line in stream:
                        if not line.startswith(b"data: {"):
                            continue
                        delta = json.loads(line[6:])
                        if delta.get("tokens"):
                            arrived = time.monotonic()
                            if not generated:
                                first = arrived
                            generated.extend(delta["tokens"])
                if len(generated) != 32:
                    raise RuntimeError(f"expected 32 HTTP tokens, got {len(generated)}")
                if i:
                    # Client-observed time to first token and throughput between first and last.
                    rates.append((len(ids) / (first - start), (len(generated) - 1) / (arrived - first)))
                    sequences.append(generated)
            return rates, sequences, chat
        finally:
            p.terminate()
            try:
                status = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                raise RuntimeError(f"front end timed out during shutdown (see {log.name})")
            if status != 0:
                raise RuntimeError(f"front end exited with {status} (see {log.name})")


def measure_speed(opt, engine_args, ids, tokenizer, draft):
    serve_args = ["--model", opt.model, *engine_args, "--draft", draft]
    result = {}
    result["engine_args"] = list(map(str, serve_args))
    sequences = []
    for arm, exe in [("oracle", opt.oracle), ("candidate", opt.candidate)]:
        rates, client_rates, tokens = serve_trials(exe, serve_args, ids, opt.out / f"{arm}-{draft}.log")
        sequences.extend(tokens)
        result[arm + "_client_pp"], result[arm + "_client_tg"] = map(statistics.median, zip(*client_rates))
        result[arm + "_trials_pp_tg"] = rates
        result[arm + "_pp"], result[arm + "_tg"] = map(statistics.median, zip(*rates))
    if opt.frontend:
        rates, tokens, chat = http_trials(opt, [*engine_args, "--draft", draft], ids, tokenizer, opt.out / f"http-{draft}.log")
        result["http_trials_pp_tg"] = rates
        result["http_pp"], result["http_tg"] = map(statistics.median, zip(*rates))
        result["chat_message"] = chat
        sequences.extend(tokens)
    result["greedy_match"] = all(x == sequences[0] for x in sequences)
    result["greedy_ids"] = sequences[0]
    speed_ratios = {k: result["candidate_" + k] / result["oracle_" + k] for k in ("pp", "tg")}
    if opt.frontend:
        for k in ("pp", "tg"):
            speed_ratios["http_" + k] = result["http_" + k] / result["oracle_client_" + k]
    result["speed_ratios"] = speed_ratios
    result["pass"] = result["greedy_match"] and all(0.8 <= r <= 1.2 for r in speed_ratios.values())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("oracle", "candidate", "model", "engram"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--frontend", type=Path, help="also check real HTTP chat, token IDs and streaming speed")
    ap.add_argument("--port", type=int, default=18549)
    ap.add_argument("--draft", choices=["off", "dfly2", "both"], default="off", help="drafter mode for throughput trials")
    ap.add_argument("--out", type=Path, required=True, help="new directory for raw results")
    ap.add_argument("--long", action="store_true", help="also score positions past the 2048-entry index")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="common engine options after --")
    opt = ap.parse_args()
    opt.out.mkdir(parents=True, exist_ok=False)
    engine_args = ["--engram", opt.engram, "--ctx", "4096", "--chunk", "128", "--session-reuse", "0",
                   *[s for s in opt.extra if s != "--"]]
    args = ["--model", opt.model, *engine_args]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([opt.oracle, "--export-tokenizer", opt.model, tmp], check=True)
        tokenizer = Tokenizer.from_file(str(Path(tmp) / "tokenizer.hf.json"))
    corpus = json.loads((Path(__file__).resolve().parents[1] / "profile_corpus.json").read_text())
    text = "\n".join(row["p"] for row in corpus)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    cases = {"first": ids[:128], "last": ids[-128:]}
    if opt.long:
        cases["long"] = (ids * math.ceil(2176 / len(ids)))[:2176]
    metrics = []
    for label, prompt in cases.items():
        rows = [measure_logits(exe, [*args, "--draft", "off"], prompt, opt.out / f"{label}-{arm}")
                for arm, exe in [("oracle", opt.oracle), ("candidate", opt.candidate)]]
        # Both runners export only the FINAL chunk, even with --all-logits.
        scores = full_vocab_scores(*rows, prompt[-127:])
        metrics.extend(scores)
        print(f"{label}: {len(scores)} positions, mean KL {scores[:, 0].mean():.8g}", flush=True)
    kl, la, lb, agree, acc_a, acc_b = np.mean(metrics, axis=0)
    result = dict(positions=len(metrics), mean_kl=float(kl), top1_agreement=float(agree),
                  oracle_ppl=math.exp(la), candidate_ppl=math.exp(lb),
                  oracle_next_token_accuracy=float(acc_a), candidate_next_token_accuracy=float(acc_b))
    result["quality_pass"] = bool(kl <= 0.001 and abs(math.expm1(lb - la)) <= 0.01 and agree >= 0.99)
    (opt.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    modes = ["off", "dfly2"] if opt.draft == "both" else [opt.draft]
    for draft in modes:
        result[draft] = measure_speed(opt, engine_args, cases["first"], tokenizer, draft)
        print(f"{draft}: {result[draft]['speed_ratios']}", flush=True)
        (opt.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    result["pass"] = result["quality_pass"] and all(result[mode]["pass"] for mode in modes)
    print(json.dumps(result, indent=2))
    (opt.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
