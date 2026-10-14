#!/usr/bin/env python3
"""The streaming runner (the `strata` binary) on the synthetic native pack, held to the oracle at every position
(docs/qwen36-port/step4-runner.md; the fixture is tests/python/qwen36_runner_fixture.py, the window-level test
tests/cuda/qwen36_verify_test.cpp).

    python3 -I tests/python/qwen36_runner_check.py --strata build/strata --fixture DIR [--keep]

Three runs of the 40-token sequence ("the quick brown fox jumps over the lazy "), each teacher-forced to the oracle's
own tokens (--spec-oracle + --teacher-force: every window row committed, the next window from the sequence's token)
with --dump-logits writing one row of head logits per position:

    c7    the oracle's base_c7 shape: positions 0..34 through the prompt path in chunks of 7, then the decode windows
          from position 35 (rows 35..39 compared)
    c40   one prompt chunk of 35, then the windows (the same rows; a different prompt chunking must give the same)
    ar    position 0 through a one-token prompt chunk, then every other position through the decode windows
          (the oracle's base_ar shape; rows 1..39 compared)

Each row is held to the Q8 oracle's `result_output_all` (--act-quant q8: the policy's reference for a path that
rounds activations to Q8, oracle-crosscheck.md "Tolerance policy") at the window test's accumulated class (5e-2
relative L2), with the argmax required to agree at every compared position; the F32 oracle's numbers are reported.  Then the
mutation-sensitivity gate: 4 KB of `blk.1.ffn_gate_inp.weight` in a copy of the pack's dense.bin are flipped and the
c7 run repeated; its logits must move by more than 1e-2 relative L1 from the intact run's (a pack whose bytes do not
reach the logits is not being read).  Exit 0 / 1 / 77 (no fixture, no CUDA device: the binary says so).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

TEXT = "the quick brown fox jumps over the lazy "
TOKENS = [ord(c) for c in TEXT]
N_VOCAB = 128
TOL_LOGITS = 5e-2     # the accumulated class against the Q8 oracle (tests/cuda/qwen36_verify_test.cpp's kAccum)
MUTATION_MIN = 1e-2


def run_engine(strata, fixture, pack, tokens, prefill_chunk, prefill_until, oracle, out_logits, max_new, log):
    cmd = [strata, "--native", str(fixture / "full4.gguf"), "--pack", str(pack), "--tokens", ",".join(map(str, tokens)),
           "--max-new", str(max_new), "--greedy", "--spec", "4", "--prefill", str(prefill_chunk),
           "--prefill-until", str(prefill_until), "--max-context", "128", "--kv", "fp16",
           "--expert-profile", str(pack / "expert-profile.bin"), "--expert-cache", "64",
           "--spec-oracle", str(oracle), "--teacher-force", "--dump-logits", str(out_logits), "--suffix-draft", "0"]
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def load_rows(path):
    a = np.fromfile(path, dtype=np.float32)
    h = np.frombuffer(a[:2].tobytes(), dtype=np.int32)
    if int(h[0]) != N_VOCAB:
        raise SystemExit(f"{path}: header n_vocab {int(h[0])}")
    rows = a[2:].reshape(-1, N_VOCAB)
    if rows.shape[0] != int(h[1]):
        raise SystemExit(f"{path}: header promises {int(h[1])} rows, the file holds {rows.shape[0]}")
    return rows


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def rel_l1(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strata", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    a = ap.parse_args()
    fx = pathlib.Path(a.fixture)
    if (fx / "SKIP").exists() or not (fx / "pack" / "index.txt").exists() or not (fx / "o40" / "manifest.txt").exists():
        print(f"qwen36_runner_check: no fixture in {fx}: skipped", file=sys.stderr)
        return 77
    work = pathlib.Path(tempfile.mkdtemp(prefix="qwen36-runner-check-"))
    ref = np.fromfile(fx / "o40" / "result_output_all.f32", dtype=np.float32).reshape(-1, N_VOCAB)
    refq = np.fromfile(fx / "o40_q8" / "result_output_all.f32", dtype=np.float32).reshape(-1, N_VOCAB)
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(f"  {what:<100} {'ok' if ok else 'FAIL'}")
        fails += not ok

    # the three chunkings: (name, prompt chunk, prefill-until, the first compared position)
    cases = [("c7", 7, 35, 35), ("c40", 40, 35, 35), ("ar", 7, 1, 1)]
    logits = {}
    for name, chunk, until, first in cases:
        # the oracle for the windows: the sequence's tokens after `first`, then one more row's worth (the last row's
        # draft is never emitted; any id will do and 0 is a token of this vocabulary)
        oracle = TOKENS[first + 1:] + [0]
        oracle_file = work / f"oracle-{name}.txt"
        oracle_file.write_text(",".join(map(str, oracle)) + "\n")
        out = work / f"logits-{name}.f32"
        log = work / f"run-{name}.log"
        max_new = len(TOKENS) - first   # rows for positions first .. 39
        rc = run_engine(a.strata, fx, fx / "pack", TOKENS, chunk, until, oracle_file, out, max_new, log)
        if rc == 77 or (rc != 0 and "no CUDA device" in log.read_text()):
            print(f"qwen36_runner_check: {log.read_text().strip().splitlines()[-1]}: skipped", file=sys.stderr)
            return 77
        if rc != 0:
            print(log.read_text()[-3000:])
            check(False, f"{name}: the engine exited {rc} (log {log})")
            continue
        rows = load_rows(out)
        if rows.shape[0] != max_new:
            check(False, f"{name}: {rows.shape[0]} rows dumped, expected {max_new}")
            continue
        logits[name] = rows
        worst, worstq, top1, top1q, miss, ambiguous = 0.0, 0.0, 0, 0, [], []
        for i in range(max_new):
            p = first + i
            worst = max(worst, rel_l2(rows[i], ref[p]))
            worstq = max(worstq, rel_l2(rows[i], refq[p]))
            # a position where the F32 and Q8 oracles themselves disagree is a near tie: it cannot discriminate the
            # engine (which rounds activations as the Q8 oracle does), so it is reported, not gated
            if int(ref[p].argmax()) != int(refq[p].argmax()):
                ambiguous.append(p)
                continue
            same = int(rows[i].argmax()) == int(refq[p].argmax())
            top1q += same
            top1 += int(rows[i].argmax()) == int(ref[p].argmax())
            if not same:
                miss.append(p)
        n_gated = max_new - len(ambiguous)
        check(worstq <= TOL_LOGITS, f"{name}: logits per position {first}..39: worst rel L2 F32 {worst:.2e} | Q8 {worstq:.2e} (Q8 <= {TOL_LOGITS:.0e})")
        check(top1q == n_gated, f"{name}: argmax equals the Q8 oracle's at {top1q} of {n_gated} unambiguous positions"
                                + (f" (differs at {miss})" if miss else "")
                                + (f"; {len(ambiguous)} near-tie positions where the oracles disagree were not gated: {ambiguous}" if ambiguous else ""))
    if "c7" in logits and "c40" in logits:
        d = rel_l2(logits["c7"], logits["c40"])
        check(d <= TOL_LOGITS, f"c7 vs c40: the two prompt chunkings' rows 35..39 agree: rel L2 {d:.2e} (<= {TOL_LOGITS:.0e})")
    if "c7" in logits and "ar" in logits:
        d = rel_l2(logits["c7"], logits["ar"][34:])
        check(d <= TOL_LOGITS, f"c7 vs ar: the prompt path's rows 35..39 against the decode path's: rel L2 {d:.2e}")

    # the mutation-sensitivity gate: 4 KB of blk.1.ffn_gate_inp.weight (BF16 in dense.bin) flipped in a copy of the pack
    if "c7" in logits:
        mpack = work / "pack-mutated"
        shutil.copytree(fx / "pack", mpack)
        src_off = None
        for line in (mpack / "index.txt").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "blk.1.ffn_gate_inp.weight":
                src_off, src_bytes = int(parts[3]), int(parts[4])
                break
        if src_off is None:
            check(False, "mutation: blk.1.ffn_gate_inp.weight is not in index.txt")
        else:
            with open(mpack / "dense.bin", "r+b") as f:
                f.seek(src_off)
                blk = bytearray(f.read(4096))
                for i in range(len(blk)):
                    blk[i] ^= 0x55   # every byte flipped: the row's BF16 values are no longer the router's
                f.seek(src_off)
                f.write(blk)
            oracle_file = work / "oracle-c7.txt"
            out = work / "logits-mutated.f32"
            log = work / "run-mutated.log"
            rc = run_engine(a.strata, fx, mpack, TOKENS, 7, 35, oracle_file, out, 5, log)
            if rc != 0:
                print(log.read_text()[-2000:])
                check(False, f"mutation: the engine exited {rc} on the mutated pack (log {log})")
            else:
                moved = rel_l1(load_rows(out), logits["c7"])
                check(moved > MUTATION_MIN, f"mutation: 4 KB of blk.1.ffn_gate_inp.weight flipped moves the logits by rel L1 {moved:.2e} (> {MUTATION_MIN:.0e})")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"work directory kept: {work}")
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {fails} failure(s)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
