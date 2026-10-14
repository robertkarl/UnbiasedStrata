#!/usr/bin/env python3
"""A running server must accept a request that asks for more tokens than the context has left.

The engine refuses a request unless prompt + max_new + 8 <= context, so the front end has to keep those 8 slots out
of max_new (REGINA-112).  Without that, a request with no max_tokens, or with max_tokens at or above
context - prompt - 7, is answered 500 "prompt (N tokens) + max_new (M) exceeds the context".

usage: serve_max_tokens_check.py [HOST:]PORT      (any supported model; a few tokens are generated per case)
"""
import http.client, json, sys

target = sys.argv[1] if len(sys.argv) > 1 else "8080"
host, _, port = target.rpartition(":")
host = host or "127.0.0.1"

def call(method, path, body=None):
    c = http.client.HTTPConnection(host, int(port), timeout=600)
    c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse(); raw = r.read(); c.close()
    return r.status, raw

status, raw = call("GET", "/v1/models")
assert status == 200, (status, raw)
ctx = json.loads(raw)["data"][0]["context_length"]
base = {"messages": [{"role": "user", "content": "Reply with the single word: ok"}], "temperature": 0,
        "reasoning_effort": "none"}
cases = {"no max_tokens": {}, "max_tokens = context": {"max_tokens": ctx}, "max_tokens = context - 20": {"max_tokens": ctx - 20},
         "max_completion_tokens = 2 x context": {"max_completion_tokens": 2 * ctx}}
failed = 0
for name, extra in cases.items():
    status, raw = call("POST", "/v1/chat/completions", dict(base, **extra))
    d = json.loads(raw)
    ok = status == 200 and d["choices"][0]["finish_reason"] == "stop" and d["usage"]["completion_tokens"] >= 1
    failed += not ok
    print("PASS" if ok else "FAIL", name, status, d.get("error", {}).get("message", "") if status != 200 else d["usage"])
# streamed: the refusal arrives as an error event after a 200
c = http.client.HTTPConnection(host, int(port), timeout=600)
c.request("POST", "/v1/chat/completions", json.dumps(dict(base, stream=True)), {"Content-Type": "application/json"})
text = c.getresponse().read().decode(errors="replace"); c.close()
ok = '"error"' not in text and '"finish_reason":"stop"' in text.replace(" ", "") and text.rstrip().endswith("[DONE]")
failed += not ok
print("PASS" if ok else "FAIL", "no max_tokens, streamed", "" if ok else text[-300:])
print(f"context {ctx}: {len(cases) + 1 - failed} of {len(cases) + 1} pass")
sys.exit(1 if failed else 0)
