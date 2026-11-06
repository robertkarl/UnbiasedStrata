#!/bin/bash
# PLAN.md step 13 (c), the API-only acceptance of a served model: sampled requests as the model card has them, a fixed
# seed, decode and prompt-reading speed against the step-6/9 figures for the same box.
#   acceptance-real-engine.sh <url> <model-alias> <outdir> [expected-decode-tok/s] [expected-pp-tok/s] [band=0.15]
# Exit 0 when every check passes (speed checks only when an expectation is given). Raw responses are kept in <outdir>.
set -u
URL=$1; MODEL=$2; OUT=$3; EXP_DEC=${4:-}; EXP_PP=${5:-}; BAND=${6:-0.15}; mkdir -p "$OUT"
post() { curl -s -m "${2:-900}" "$URL/v1/chat/completions" -H 'content-type: application/json' -d "$1"; }
SAMP='"temperature":0.6,"top_p":0.95,"top_k":20,"seed":7'
fail=0; note() { echo "$@"; }

note "== /v1/models"; curl -s -m 10 "$URL/v1/models" > "$OUT/models.json" || { echo "no server at $URL"; exit 2; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print([m["id"] for m in d["data"]])' "$OUT/models.json"

note "== 1. sampled chat with thinking: answer present, reasoning separated, finish=stop"
post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer with the number only.\"}],\"max_tokens\":1024,$SAMP}" > "$OUT/think.json"
python3 - "$OUT/think.json" <<'PY' || fail=1
import json, sys; d = json.load(open(sys.argv[1])); m = d["choices"][0]["message"]
ok = "391" in (m.get("content") or "") and d["choices"][0]["finish_reason"] == "stop" and "</think>" not in (m.get("content") or "")
print(json.dumps({"content": (m.get("content") or "")[:80], "reasoning_chars": len(m.get("reasoning_content") or ""), "finish": d["choices"][0]["finish_reason"], "ok": ok})); sys.exit(0 if ok else 1)
PY

note "== 2. same request, same seed, twice: identical ids (sampling is seeded)"
post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the sea.\"}],\"max_tokens\":120,$SAMP,\"chat_template_kwargs\":{\"enable_thinking\":false,\"thinking\":false}}" > "$OUT/seed-a.json"
post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the sea.\"}],\"max_tokens\":120,$SAMP,\"chat_template_kwargs\":{\"enable_thinking\":false,\"thinking\":false}}" > "$OUT/seed-b.json"
python3 - "$OUT/seed-a.json" "$OUT/seed-b.json" <<'PY' || note "   (seeded sampling not reproducible: recorded, not failed; drafting changes the path)"
import json, sys; a, b = (json.load(open(p))["choices"][0]["message"].get("content") for p in sys.argv[1:3])
print(json.dumps({"a": a[:100], "same": a == b})); sys.exit(0 if a == b else 1)
PY

note "== 3. tool call, pi-shaped: a tool_calls entry with a JSON-parsable argument object"
post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Read the file /etc/hostname using the read tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}],\"max_tokens\":1024,$SAMP}" > "$OUT/toolcall.json"
python3 - "$OUT/toolcall.json" <<'PY' || fail=1
import json, sys; d = json.load(open(sys.argv[1])); tc = d["choices"][0]["message"].get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "read" and "path" in json.loads(tc[0]["function"]["arguments"])
print(json.dumps({"tool_calls": tc[:1], "ok": ok})); sys.exit(0 if ok else 1)
PY

note "== 4. decode speed: 3 x 256 tokens of prose, thinking off, median tokens/s from the server's timings"
python3 - "$URL" "$MODEL" "$OUT" "$EXP_DEC" "$BAND" <<'PY' || fail=1
import json, sys, urllib.request, statistics
url, model, out, exp, band = sys.argv[1:]; rates = []
for i in range(3):
    body = {"model": model, "messages": [{"role": "user", "content": f"Write a 300-word story about a lighthouse keeper, take {i+1}."}], "max_tokens": 256,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7 + i, "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    r = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900)); json.dump(d, open(f"{out}/decode-{i}.json", "w"))
    rates.append(d["timings"]["predicted_per_second"])
med = statistics.median(rates); res = {"decode_tok_s": [round(x, 1) for x in rates], "median": round(med, 1)}
if exp:
    lo = float(exp) * (1 - float(band)); res.update(expected=float(exp), floor=round(lo, 1), ok=med >= lo)
print(json.dumps(res)); sys.exit(0 if res.get("ok", True) else 1)
PY

note "== 5. prompt reading: one ~6k-token prompt, tokens/s from the server's timings"
python3 - "$URL" "$MODEL" "$OUT" "$EXP_PP" "$BAND" <<'PY' || fail=1
import json, sys, urllib.request
url, model, out, exp, band = sys.argv[1:]
text = " ".join(f"Line {i}: the quick brown fox number {i} jumps over the lazy dog, as recorded in the ledger on day {i % 365}." for i in range(260))
body = {"model": model, "messages": [{"role": "user", "content": text + "\n\nHow many lines are above? Answer with the number only."}], "max_tokens": 32,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7, "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
r = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=900)); json.dump(d, open(f"{out}/pp.json", "w")); t = d["timings"]
res = {"prompt_tokens": t["prompt_n"], "pp_tok_s": round(t["prompt_per_second"], 1), "answer": (d["choices"][0]["message"].get("content") or "")[:40]}
if exp:
    lo = float(exp) * (1 - float(band)); res.update(expected=float(exp), floor=round(lo, 1), ok=t["prompt_per_second"] >= lo)
print(json.dumps(res)); sys.exit(0 if res.get("ok", True) else 1)
PY

note "== 6. thinking budget: reasoning_budget_tokens caps the thinking, which closes with the wrap-up; the answer follows"
post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer with the number only.\"}],\"max_tokens\":1024,\"reasoning_budget_tokens\":16,$SAMP}" > "$OUT/budget.json"
python3 - "$OUT/budget.json" <<'PY' || fail=1
import json, sys; d = json.load(open(sys.argv[1])); m = d["choices"][0]["message"]
content, reasoning = m.get("content") or "", m.get("reasoning_content") or ""
ok = ("391" in content and d["choices"][0]["finish_reason"] == "stop" and "</think>" not in content
      and "time to give my answer" in reasoning)
print(json.dumps({"content": content[:80], "reasoning_tail": reasoning[-60:], "completion_tokens": d["usage"]["completion_tokens"], "ok": ok}))
sys.exit(0 if ok else 1)
PY

note "== /health"; curl -s -m 10 "$URL/health"; echo
[ $fail = 0 ] && echo "ACCEPTANCE_OK $MODEL" || echo "ACCEPTANCE_FAIL $MODEL"; exit $fail
