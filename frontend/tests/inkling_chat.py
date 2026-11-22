"""Exercise the Inkling 2 Mini GGUF tokenizer and chat API with scripted output.

python3 frontend/tests/inkling_chat.py build-frontend/bin/unbiased-strata Inkling-2-Mini-UD-Q4_K_XL.gguf [mmproj-BF16.gguf]
"""

import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def main(binary, model, mmproj=None):
    fake = Path(__file__).resolve().with_name("fake-engine.py")
    # From the checkpoint's tokenizer.json: a thought, an answer, then a weather call.
    tokens = [200001, 200008, 78168, 200010, 200001, 200004, 32269, 200010,
              200001, 30847, 200049, 90, 1, 930, 1, 25, 1, 30847, 1, 11, 1, 70747,
              1, 25, 90, 1, 2958, 1, 25, 1, 3967, 1, 92, 92, 200010]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    def post(path, body):
        req = Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urlopen(req, timeout=60)

    command = [binary, "--engine", str(fake), "-m", model, "--port", str(port)]
    if mmproj:
        command.extend(["--mmproj", mmproj, "--threads", "4"])
    proc = subprocess.Popen(command,
                            env={**os.environ, "FAKE_TOKENS": ",".join(map(str, tokens))})
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                with urlopen(url + "/health", timeout=1):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError("server did not become ready")
        with post("/tokenize", {"content": "हिन्दी में पढ़ें"}) as r:
            assert json.load(r)["tokens"] == [117266, 150208, 165789, 68187]
        messages = [{"role": "user", "content": "Hi"}]
        with post("/apply-template", {"messages": messages, "reasoning_effort": "none"}) as r:
            prompt = json.load(r)["prompt"]
            assert "Thinking effort level: 0<|end_message|>" in prompt
            assert prompt.endswith("<|message_model|><|content_text|>")
        tool = {"type": "function", "function": {"name": "weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}}}}
        body = {"messages": messages, "tools": [tool], "max_tokens": 64}
        with post("/apply-template", body) as r:
            assert '"parameters":{"properties":{"city":{"type":"string"}},"type":"object"}' in json.load(r)["prompt"]
        with post("/v1/chat/completions", body) as r:
            choice = json.load(r)["choices"][0]
        reply = choice["message"]
        assert choice["finish_reason"] == "tool_calls" and reply["content"] == "Hello"
        assert reply["reasoning_content"] == "thinking"
        call, = reply["tool_calls"]
        assert call["function"]["name"] == "weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        with post("/v1/chat/completions", {**body, "stream": True}) as r:
            events = [json.loads(line[6:]) for line in r if line.startswith(b"data: {")]
        deltas = [c["delta"] for event in events for c in event["choices"]]
        assert "".join(d.get("content") or "" for d in deltas) == "Hello"
        assert "".join(d.get("reasoning_content") or "" for d in deltas) == "thinking"
        args = "".join(c["function"].get("arguments", "") for d in deltas for c in d.get("tool_calls", []))
        assert json.loads(args) == {"city": "Paris"}
        messages += [reply, {"role": "tool", "tool_call_id": call["id"], "content": "sunny"}]
        with post("/apply-template", body) as r:
            prompt = json.load(r)["prompt"]
            assert '<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Paris"}}' in prompt
            assert "<|message_tool|>weather<|content_text|>sunny" in prompt
        if mmproj:
            png = Path(__file__).resolve().parents[2] / "bench/inkling/shapes.png"
            image = {"type": "image_url", "image_url": {"url":
                     "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()}}
            parts = [{"type": "text", "text": "Before"}, image, {"type": "text", "text": "Between"}, image]
            picture = {"messages": [{"role": "user", "content": parts}], "reasoning_effort": "none"}
            with post("/apply-template", picture) as r:
                prompt = json.load(r)["prompt"]
            blocks = prompt.split("<|message_user|>")[1:]
            assert len(blocks) == 4 and blocks[0] == "<|content_text|>Before<|end_message|>"
            assert blocks[2] == "<|content_text|>Between<|end_message|>"
            assert blocks[1].startswith("<|content_image|>") and blocks[1].endswith("<|end_message|>")
            assert blocks[3] == blocks[1] + "<|message_model|><|content_text|>"
            marker = blocks[1][len("<|content_image|>"):-len("<|end_message|>")]
            # A 165 x 93 image occupies a 5 x 3 grid of merged patches.
            with post("/tokenize", {"content": prompt.replace(marker, "<|unused_200054|>" * 15)}) as r:
                expected = json.load(r)["tokens"]
            with post("/v1/chat/completions", {**picture, "max_tokens": 1}) as r:
                assert json.load(r)["usage"]["prompt_tokens"] == len(expected)
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    print("Inkling 2 Mini chat passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
