"""Score SQ pack next-token distributions on Metal from two strata-inkling executables.

For prompts longer than 64 positions, the Strata 0.1.65 oracle dump-ref driver
must submit at most 64 output rows per backend call. See README.md.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from parity import score


def dump(engine, model, ids, path):
    subprocess.run([engine, "--native", model, "--dump-ref", path, "--prompt-ids", ",".join(map(str, ids)),
                    "--max-context", "4096", "--chunk", "128", "--parallel", "1", "--draft", "off"], check=True)
    with path.open("rb") as f:
        n, vocab = struct.unpack("<ii", f.read(8))
        if (n, vocab) != (len(ids), 200058) or np.fromfile(f, dtype="<i4", count=n).tolist() != ids:
            raise ValueError("Unexpected reference header or input token IDs")
        return np.memmap(path, dtype="<f4", mode="r", offset=f.tell(), shape=(n, vocab))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("oracle", "candidate", "model"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    for name, case in json.loads(Path(__file__).with_name("cases.json").read_text()).items():
        if name == "picture":
            continue
        targets, prompt = case["targets"], case["prompt"]
        ids = prompt + targets[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            rows = [dump(engine, args.model, ids, Path(tmp) / f"{i}.bin")[len(prompt)-1:]
                    for i, engine in enumerate((args.oracle, args.candidate))]
            print(json.dumps({"case": name, **score(*rows, targets)}), flush=True)


if __name__ == "__main__":
    main()
