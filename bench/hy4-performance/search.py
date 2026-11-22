#!/usr/bin/env python3
"""Screen SQ-64 layouts sequentially and repeat promising layouts with long generations."""
import argparse
import json
import os
import select
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PP_REFERENCE, TG_REFERENCE = 1068.0, 41.8


def qualifies(row):
    return row["pp_tps"] >= 0.8 * PP_REFERENCE and row["tg_tps"] >= 0.8 * TG_REFERENCE


def record(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--primary",required=True,help="4090 UUID")
    ap.add_argument("--peer",required=True,help="A6000 UUID")
    ap.add_argument("--after-pid",type=int,help="wait for a previously launched Linux benchmark")
    args=ap.parse_args()
    study=args.study
    args.out.mkdir(parents=True,exist_ok=True)
    if args.after_pid:
        try:
            fd=os.pidfd_open(args.after_pid)
        except ProcessLookupError:
            pass
        else:
            poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
    existing={}
    ledger=args.out/"attempts.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row=json.loads(line);existing[row["id"]]=row

    def measure(stage,layout,gpus,chunk,temperature):
        name=f"{stage}-{layout}-chunk{chunk}-t{temperature}"
        if name in existing:return existing[name]
        output=args.out/name
        confirm=stage=="confirm"
        command=[sys.executable,str(Path(__file__).with_name("measure.py")),
                 "--engine",str(study/"published-0.1.64/strata-hy4"),"--model",str(study/"checkpoints/Hy4-Flash-SQ-64.strata"),
                 "--engram",str(study/"checkpoints/Hy4-Flash-Engram-Q8.strata"),"--tokenizer",str(study/"published-tokenizer-check/tokenizer.hf.json"),
                 "--context","4096","--prompt","2048","--chunk",str(chunk),"--temperature",str(temperature),"--gpus",gpus,
                 "--generate","2048" if confirm else "256","--trials","3" if confirm else "1","--warmup","128","--out",str(output)]
        row=dict(id=name,stage=stage,layout=layout,chunk=chunk,temperature=temperature,utc=datetime.now(timezone.utc).isoformat(),command=command)
        if not (output/"result.json").exists():
            with (args.out/(name+".console.log")).open("a") as console:
                try:
                    subprocess.run(command,stdout=console,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError as exc:
                    row.update(status="error",exit_code=exc.returncode)
                    record(ledger,row)
                    return row
        row.update(json.loads((output/"result.json").read_text()))
        markers=("[load] multi-GPU experts", "[load] experts tiered", "[hybrid] prefill streaming", "[runner] kv cache")
        row["placement"]=[l for l in (output/"native.log").read_text().splitlines() if l.startswith(markers)]
        row["status"]="measured"
        record(ledger,row)
        return row

    normal=args.primary+","+args.peer
    reverse=args.peer+","+args.primary
    configs=[("4090-a6000",normal,0),("a6000-4090",reverse,0),
             ("4090-a6000",normal,1024),("4090-a6000",normal,2048),
             ("a6000-4090",reverse,1024),("a6000-4090",reverse,2048),
             ("a6000",args.peer,0),("4090",args.primary,0)]
    for temperature in (0.9,0.0):
        for layout,gpus,chunk in configs:
            screen=measure("screen",layout,gpus,chunk,temperature)
            if screen["status"]=="measured" and qualifies(screen):
                long=measure("confirm",layout,gpus,chunk,temperature)
                if long["status"]=="measured" and qualifies(long) and len(long["generated"])==3 and min(long["generated"])>=1600:
                    (args.out/"qualified.json").write_text(json.dumps(long,indent=2)+"\n")
                    return 0
    return 2


if __name__=="__main__":
    raise SystemExit(main())
