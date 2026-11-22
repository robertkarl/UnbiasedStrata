#!/usr/bin/env python3
"""Run the published Hy4 native line protocol with fresh prompts and save each trial."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def workload(tokenizer, size):
    # The budget includes the chat framing. Only inventory records are cut to length.
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    head = encode("<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:high<｜hy_end:opensource｜><｜hy_start:opensource｜>user<｜hy_middle:opensource｜>Here are warehouse audit records.\n")
    tail = encode("\nWrite a detailed engineering design for an inventory service using these records. Cover the data model, units, migration, idempotent receiving, stock reservations, reversals, reconciliation, monitoring, backup and recovery. For each part provide worked examples and Python or SQL code. Include at least 2500 words so a new engineer can implement and operate it.<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>")
    inventory = encode("".join(f"Audit {i}: warehouse {i % 7}, received {100 + i} parts, shipped {i % 13}, verified.\n" for i in range(size // 8 + 1)))
    n = size - len(head) - len(tail)
    if not 0 < n <= len(inventory):
        raise ValueError("prompt length outside supported range")
    return head + inventory[:n] + tail


def trial(child, ids, limit, temperature, protocol, token_stream, seed=None):
    start = time.monotonic()
    seed_opt = "" if seed is None else f" seed={seed}"
    child.stdin.write(f"GEN {limit} temperature={temperature} top_p=1 top_k=64{seed_opt} " + ",".join(map(str, ids)) + "\n")
    child.stdin.flush()
    generated = []
    for line in child.stdout:
        fields = line.split()
        if line.startswith("T "):
            if not generated:
                first = time.monotonic()
            generated.append(int(fields[1]))
        else:
            protocol.write(line)
            protocol.flush()
        if line.startswith("DONE "):
            n, prompt = int(fields[1]), int(fields[2])
            pp, tg = float(fields[3]), float(fields[4])
            cached = int(fields[8]) if len(fields) >= 9 else 0
            if (fields[5] not in ("length", "stop") or not 0 < n <= limit or
                    n != len(generated) or prompt != len(ids) or cached or
                    not all(math.isfinite(ms) and ms > 0 for ms in (pp, tg))):
                raise RuntimeError(f"unexpected native completion: {line}")
            token_stream.write(json.dumps(generated) + "\n")
            return dict(requested=limit, generated=n, prompt=prompt, cached=cached,
                        prefill_ms=pp, decode_ms=tg, pp_tps=1000*prompt/pp,
                        tg_tps=1000*n/tg, ttft_s=first-start, elapsed_s=time.monotonic()-start,
                        finish=fields[5], seed=seed, output_sha256=hashlib.sha256(
                            ",".join(map(str, generated)).encode()).hexdigest())
        if line.startswith(("ERR", "FATAL")):
            raise RuntimeError(line.rstrip())
    raise RuntimeError(f"engine exited before DONE (status {child.poll()})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("engine", "model", "engram", "tokenizer", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, help="reuse an exact saved token sequence instead of generating the inventory fixture")
    parser.add_argument("--gpus", required=True, help="ordered physical NVIDIA UUIDs, comma separated")
    parser.add_argument("--draft", choices=("off", "dfly2"), default="dfly2")
    parser.add_argument("--chunk", type=int, default=0, help="0 selects the engine default")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--prompt", type=int, default=2048)
    parser.add_argument("--generate", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, help="first measured seed; increment for each trial")
    opt = parser.parse_args()
    if opt.seed is not None and not 0 <= opt.seed <= 2**32 - 1 - opt.trials:
        parser.error("measured seeds must fit in uint32 and exclude the default seed sentinel")
    if opt.prompt + opt.generate > opt.context:
        parser.error("input and output exceed the configured context")
    opt.out.mkdir(parents=True, exist_ok=False)
    if opt.prompt_ids:
        ids = [int(token) for token in opt.prompt_ids.read_text().split()]
        if len(ids) != opt.prompt:
            parser.error("--prompt and the number of saved token IDs differ")
    else:
        ids = workload(Tokenizer.from_file(str(opt.tokenizer)), opt.prompt)
    (opt.out/"prompt.ids").write_text(" ".join(map(str, ids))+"\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=opt.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    env["LD_LIBRARY_PATH"] = str(opt.engine.resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    argv = [str(opt.engine), "--serve", "--model", str(opt.model), "--engram", str(opt.engram),
            "--ctx", str(opt.context), "--chunk", str(opt.chunk), "--draft", opt.draft, "--session-reuse", "0",
            "--gpus", ",".join(str(i) for i in range(len(opt.gpus.split(","))))]
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(), argv=argv,
                    devices=opt.gpus.split(","), model=opt.model.name, temperature=opt.temperature,
                    top_p=1, top_k=64, prompt=len(ids), requested=opt.generate, trials=opt.trials,
                    warmup_tokens=opt.warmup, prompt_sha256=hashlib.sha256((opt.out/"prompt.ids").read_bytes()).hexdigest(),
                    binary_sha256=subprocess.check_output(["sha256sum",opt.engine],text=True).split()[0],
                    environment={k:v for k,v in env.items() if k.startswith(("HY4_", "STRATA_HY4_", "CUDA_"))})
    write_json(opt.out/"manifest.json",manifest)
    rows=[]
    with (opt.out/"native.log").open("w") as log, (opt.out/"protocol.log").open("w") as wire, (opt.out/"generated.jsonl").open("w") as tokens:
        child=subprocess.Popen(argv,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            for line in child.stdout:
                wire.write(line)
                if line.startswith("READY "): break
                if line.startswith(("ERR","FATAL")): raise RuntimeError(line)
            else: raise RuntimeError("engine failed to load; see native.log")
            processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,process_name,pid","--format=csv,noheader"],text=True)
            (opt.out/"gpu-processes.txt").write_text(processes)
            if {l.split(",")[0] for l in processes.splitlines() if l.rsplit(",",1)[-1].strip()==str(child.pid)} != set(manifest["devices"]):
                raise RuntimeError("CUDA device attribution differs from requested UUIDs")
            if opt.warmup:
                write_json(opt.out/"warmup.json",trial(child,ids,opt.warmup,opt.temperature,wire,tokens, opt.seed))
            for i in range(opt.trials):
                rows.append(trial(child,ids,opt.generate,opt.temperature,wire,tokens,
                                  None if opt.seed is None else opt.seed + i))
                write_json(opt.out/"trials.json",rows)
            child.stdin.write("QUIT\n")
            child.stdin.close()
            try:
                exit_code=child.wait(timeout=15)
                shutdown={"returncode":exit_code}
            except subprocess.TimeoutExpired:
                child.terminate()
                shutdown={"timed_out_after_done":True,"returncode":child.wait(timeout=30)}
            write_json(opt.out/"shutdown.json",shutdown)
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=90)
    result=dict(pp_tps=statistics.median(r["pp_tps"] for r in rows),
                tg_tps=statistics.median(r["tg_tps"] for r in rows),
                generated=[r["generated"] for r in rows], prompt=len(ids))
    write_json(opt.out/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
