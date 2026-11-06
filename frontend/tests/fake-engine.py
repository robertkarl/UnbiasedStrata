#!/usr/bin/env python3
"""A stand-in for strata / strata-poly that speaks the line protocol (docs/BACKENDS.md), for tests without a GPU.

  GEN <max_new> [key=value ...] <id,id,...>   -> T <id> per token, then DONE
  STOP                                         -> the current request ends with finish=cancel
  QUIT                                         -> exit 0

Environment variables choose the answers, so one script covers the cases:
  FAKE_TOKENS     comma-separated ids to emit (default: the prompt ids echoed back, capped by max_new)
  FAKE_N_TOKENS   emit this many tokens of id FAKE_TOKEN_ID instead (burst tests)
  FAKE_TOKEN_ID   the id used by FAKE_N_TOKENS (default 100)
  FAKE_DELAY_MS   sleep this long between tokens (default 0)
  FAKE_CONTEXT    the context in the READY line (default 4096)
  FAKE_ERR        if set, answer every GEN with "ERR <text>" instead
  FAKE_DIE_AFTER  if set, exit 3 after emitting this many tokens of the first request
  FAKE_NO_STOP    if set, READY without "stop": STOP is ignored and the request runs to its end
  FAKE_PROMPT_N   if set, lie about the prompt size in DONE (mismatch test)
"""
import os
import queue
import sys
import threading
import time

out = sys.stdout


def say(s):
    out.write(s + "\n")
    out.flush()


env = os.environ.get
say("INFO engine=fake batch_slots=0")
say("READY %s%s" % (env("FAKE_CONTEXT", "4096"), "" if env("FAKE_NO_STOP") else " stop"))

first = True
pending = []  # lines read while checking for STOP that are not STOP/QUIT

# stdin is read on a thread of its own (select() does not take a pipe on Windows); None marks its end
lines = queue.Queue()


def _reader():
    for line in sys.stdin:
        lines.put(line.rstrip("\r\n"))
    lines.put(None)


threading.Thread(target=_reader, daemon=True).start()


def readline():
    if pending:
        return pending.pop(0)
    line = lines.get()
    if line is None:
        os._exit(0)
    return line


def stop_requested():
    """Consume what is readable now; True if a STOP was among it."""
    stop = False
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            return stop
        if line is None:
            os._exit(0)
        if line == "STOP":
            stop = True
        elif line == "QUIT":
            os._exit(0)
        else:
            pending.append(line)


while True:
    line = readline()
    if line == "QUIT":
        sys.exit(0)
    if line == "STOP" or line == "":
        continue
    parts = line.split()
    if parts[0] not in ("GEN", "GENI") or len(parts) < 3:
        say("ERR expected: GEN <max_new> <id,id,...>")
        continue
    max_new = int(parts[1])
    prompt = [int(x) for x in parts[-1].split(",") if x]
    prompt_n = len(prompt)
    if env("FAKE_ERR"):
        say("ERR " + env("FAKE_ERR"))
        continue
    say("RESUME 0")
    say("PP %d %d 1 1000" % (prompt_n, prompt_n))
    if env("FAKE_N_TOKENS"):
        tokens = [int(env("FAKE_TOKEN_ID", "100"))] * int(env("FAKE_N_TOKENS"))
    elif env("FAKE_TOKENS"):
        tokens = [int(x) for x in env("FAKE_TOKENS").split(",") if x]
    else:
        tokens = prompt
    delay = float(env("FAKE_DELAY_MS", "0")) / 1000.0
    die_after = int(env("FAKE_DIE_AFTER", "0") or 0)
    produced = 0
    finish = "stop"
    for t in tokens:
        if produced >= max_new:
            finish = "length"
            break
        if stop_requested() and not env("FAKE_NO_STOP"):
            finish = "cancel"
            break
        say("T %d" % t)
        produced += 1
        if die_after and first and produced >= die_after:
            sys.exit(3)
        if delay:
            time.sleep(delay)
    first = False
    # DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused>
    say("DONE %d %s 12.5 %.1f %s 0 0 0" % (produced, env("FAKE_PROMPT_N", str(prompt_n)), produced * 10.0, finish))
