"""Check a running Muse server's text, reasoning, streaming and tool round trip.

Usage: python3 tools/muse_http_smoke.py http://127.0.0.1:8080 results.json
Uses a simulated weather result; no external tool is executed.
"""
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    url = sys.argv[1].rstrip("/") + "/v1/chat/completions"
    output = Path(sys.argv[2])
    results = {}

    def save():
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    def request(name, messages, **extra):
        body = {"messages": messages, "temperature": 0, "reasoning_effort": "low",
                "max_tokens": 256, **extra}
        start = time.monotonic()
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as response:
            if body.get("stream"):
                chunks = []
                done = False
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    chunks.append(json.loads(data))
                result = {"chunks": chunks, "done": done}
            else:
                result = json.load(response)
        results[name] = {"seconds": time.monotonic() - start, "response": result}
        save()
        print(f"{name}: {results[name]['seconds']:.2f}s", flush=True)
        return result

    question = [{"role": "user", "content": "What is 2 + 2? Reply with just the answer."}]
    try:
        request("invalid_effort", question, reasoning_effort="none")
        raise AssertionError("unsupported reasoning effort was accepted")
    except urllib.error.HTTPError as error:
        results["invalid_effort"] = {"status": error.code, "body": error.read().decode()}
        save()
        assert error.code == 400

    normal = request("text_and_reasoning", question)
    choice = normal["choices"][0]
    assert choice["finish_reason"] == "stop", "text response did not finish"
    assert re.search(r"\b4\b", choice["message"].get("content", "")), "incorrect arithmetic answer"
    assert choice["message"].get("reasoning_content"), "reasoning was not returned separately"

    streamed = request("streaming", question, stream=True)
    assert streamed["done"], "missing SSE completion"
    deltas = [chunk["choices"][0]["delta"] for chunk in streamed["chunks"] if chunk.get("choices")]
    assert "".join(d.get("content", "") for d in deltas) == choice["message"]["content"]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == choice["message"]["reasoning_content"]

    tools = [{"type": "function", "function": {"name": "weather", "description": "Get current weather for a city.",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    messages = [{"role": "user", "content": "Call the weather tool for Paris and tell me the result. Do not guess the weather."}]
    called = request("tool_call", messages, tools=tools)
    message = called["choices"][0]["message"]
    calls = message.get("tool_calls", [])
    assert len(calls) == 1, "expected one weather call"
    call = calls[0]
    assert call["function"]["name"] == "weather"
    assert json.loads(call["function"]["arguments"])["city"].lower() == "paris"
    assert called["choices"][0]["finish_reason"] == "tool_calls"
    messages += [message, {"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 20 degrees Celsius."}]
    returned = request("tool_result", messages, tools=tools)
    answer = returned["choices"][0]["message"].get("content", "").lower()
    assert "sunny" in answer and "20" in answer, "tool result was not used"
    assert returned["choices"][0]["finish_reason"] == "stop"
    results["passed"] = True
    save()
    print("PASS: text, separate reasoning, streaming, tool call, and tool-result round trip")


if __name__ == "__main__":
    main()
