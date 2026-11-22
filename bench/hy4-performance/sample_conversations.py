#!/usr/bin/env python3
"""Pair seeded, one-token draws on all prepared conversations in one native server."""

import argparse
import json
import subprocess
from pathlib import Path


def draw(child, prompt, prompt_length, seed):
    child.stdin.write(f"GEN 1 temperature=0.9 top_p=1 top_k=64 seed={seed} {prompt}\n")
    child.stdin.flush()
    tokens = []
    for line in child.stdout:
        fields = line.split()
        if line.startswith("T "):
            tokens.append(int(fields[1]))
        elif line.startswith("DONE "):
            if (len(tokens) != 1 or int(fields[1]) != 1 or int(fields[2]) != prompt_length
                    or fields[5] not in ("stop", "length")
                    or int(fields[8]) != 0):
                raise RuntimeError(f"unexpected one-token completion: {line}")
            return tokens[0]
        elif line.startswith(("ERR", "FATAL")):
            raise RuntimeError(line)
    raise RuntimeError("native server exited before DONE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("engine", "model", "engram", "conversations", "out"):
        parser.add_argument("--" + opt, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.conversations.glob("*/prompt.ids"))
    if not paths:
        parser.error("no prepared conversations")
    argv = [args.engine, "--serve", "--model", args.model, "--engram", args.engram,
            "--gpus", "0,1", "--ctx", "4096", "--draft", "dfly2", "--session-reuse", "0"]
    with (args.out / "engine.log").open("w") as log:
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=log, text=True, bufsize=1)
        try:
            if not any(line.startswith("READY ") for line in child.stdout):
                raise RuntimeError("native server exited before READY")
            for path in paths:
                ids = path.read_text().split()
                prompt = ",".join(ids)
                rows = [{"seed": s, "token": draw(child, prompt, len(ids), s)} for s in range(1, 97)]
                (args.out / (path.parent.name + ".json")).write_text(json.dumps(rows) + "\n")
            child.stdin.write("QUIT\n")
            child.stdin.close()
            if child.wait(timeout=30):
                raise RuntimeError("native server exited abnormally")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
