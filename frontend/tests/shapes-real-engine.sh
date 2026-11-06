#!/bin/bash
# Step 7 request shapes (PLAN.md section 9): the shapes a coding agent sends, against a live server. Raw responses in $OUT.
#   us2-shapes.sh http://127.0.0.1:18091 <outdir> <model-alias>
set -u
URL=$1; OUT=$2; MODEL=$3; mkdir -p "$OUT"
j() { python3 -c 'import json,sys; d=json.load(sys.stdin); r=eval(sys.argv[1]); print(json.dumps(r) if isinstance(r,(dict,list)) else r)' "$1" 2>/dev/null; }
post() { curl -s -m "${3:-600}" "$URL$1" -H 'content-type: application/json' -d "$2"; }
SAMP='"temperature":0.6,"top_p":0.95,"top_k":20'
TOOLS='[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}}}]'

echo "== 1. no max_tokens at all"
post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],$SAMP}" > "$OUT/1-no-max-tokens.json"
j '{"content": d["choices"][0]["message"]["content"], "finish": d["choices"][0]["finish_reason"], "completion_tokens": d["usage"]["completion_tokens"]}' < "$OUT/1-no-max-tokens.json" || head -c 300 "$OUT/1-no-max-tokens.json"

echo "== 2. max_tokens 32768 (more than the context leaves): clamped, not refused"
post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":32768,$SAMP}" > "$OUT/2-max-tokens-32768.json"
j '{"content": d["choices"][0]["message"]["content"], "finish": d["choices"][0]["finish_reason"], "http_error": d.get("error")}' < "$OUT/2-max-tokens-32768.json" || head -c 300 "$OUT/2-max-tokens-32768.json"

echo "== 3. reasoning_content on a prior assistant message, then a tool result (pi's history shape)"
post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Read /etc/hostname with the read tool, then tell me the hostname.\"},{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"The user wants the hostname; I should call read.\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"/etc/hostname\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"gpu-vm\\n\"}],\"tools\":$TOOLS,$SAMP}" > "$OUT/3-reasoning-history.json"
j '{"content": d["choices"][0]["message"]["content"], "tool_calls": d["choices"][0]["message"].get("tool_calls"), "finish": d["choices"][0]["finish_reason"]}' < "$OUT/3-reasoning-history.json" || head -c 400 "$OUT/3-reasoning-history.json"

echo "== 4. parallel tool calls asked for"
post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Using the tools, read /etc/hostname and /etc/os-release. Issue both tool calls now, in one turn, before saying anything.\"}],\"tools\":$TOOLS,\"max_tokens\":1024,$SAMP,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > "$OUT/4-parallel-tools.json"
j '{"n_tool_calls": len(d["choices"][0]["message"].get("tool_calls") or []), "calls": [(t["function"]["name"], t["function"]["arguments"]) for t in d["choices"][0]["message"].get("tool_calls") or []], "finish": d["choices"][0]["finish_reason"]}' < "$OUT/4-parallel-tools.json" || head -c 400 "$OUT/4-parallel-tools.json"

echo "== 5. optional arguments first: a tool result whose next call the model writes with limit/offset (model-dependent; we record what it wrote)"
post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Call read on /var/log/syslog with limit 20 and offset 100. Put the limit and offset arguments BEFORE the path argument in your call.\"}],\"tools\":$TOOLS,\"max_tokens\":512,$SAMP,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > "$OUT/5-optional-args-first.json"
j '{"calls": [(t["function"]["name"], t["function"]["arguments"]) for t in d["choices"][0]["message"].get("tool_calls") or []], "content": d["choices"][0]["message"].get("content"), "finish": d["choices"][0]["finish_reason"]}' < "$OUT/5-optional-args-first.json" || head -c 400 "$OUT/5-optional-args-first.json"

echo "== 6. client disconnects mid-stream (2 s into a long answer), then a clean request"
curl -s -N --max-time 2 "$URL/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Write a 2000-word essay on the history of the sea.\"}],\"max_tokens\":4000,$SAMP}" > "$OUT/6-disconnected.sse" 2>/dev/null
echo "chunks received before disconnect: $(grep -c '^data: {' "$OUT/6-disconnected.sse")"; sleep 1
t0=$(date +%s.%N); post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":32,$SAMP,\"chat_template_kwargs\":{\"enable_thinking\":false}}" > "$OUT/6-after-disconnect.json"; t1=$(date +%s.%N)
echo "next request: $(j 'd["choices"][0]["message"]["content"]' < "$OUT/6-after-disconnect.json") in $(python3 -c "print(round($t1-$t0,2))") s"

echo "== 7. a long prompt (~24k tokens) streamed: time to first byte, SSE pings while the prompt is read"
# the prompt is made inside Python, not passed as an argument: Windows caps a command line at 32K characters
python3 - "$URL" "$MODEL" "$OUT/7-long-prompt.sse" <<'PY'
import sys,json,time,urllib.request,random
url,model,out=sys.argv[1:]
random.seed(1); w="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
big=" ".join(random.choice(w) for _ in range(16000))
body=json.dumps({"model":model,"stream":True,"messages":[{"role":"user","content":"Here is a list of Greek letters: "+big+"\n\nHow many times does the word 'omicron' appear? Give a rough estimate in one sentence."}],"max_tokens":64,"temperature":0.6,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"enable_thinking":False}}).encode()
req=urllib.request.Request(url+"/v1/chat/completions",data=body,headers={"content-type":"application/json"})
t0=time.time(); first=None; pings=0; chunks=0; raw=b""
with urllib.request.urlopen(req,timeout=900) as r:
    while True:
        line=r.readline()
        if not line: break
        if first is None: first=time.time()-t0
        raw+=line
        if line.startswith(b":"): pings+=1
        if line.startswith(b"data: {"): chunks+=1
open(out,"wb").write(raw)
print(json.dumps({"first_byte_s":round(first or -1,2),"total_s":round(time.time()-t0,2),"ping_lines":pings,"data_chunks":chunks,"prompt_tokens":(json.loads([l for l in raw.decode().splitlines() if l.startswith("data: {")][-1][6:]).get("usage") or {}).get("prompt_tokens")}))
PY
echo "== /health after"; curl -s "$URL/health"; echo
