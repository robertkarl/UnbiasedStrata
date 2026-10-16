[Codebase docs](../README.md) › [Front end](README.md) › HTTP API

# HTTP API, options and request handling

## Purpose

The user-visible surface of `unbiased-strata`: command-line options, the start-up sequence, the three HTTP
endpoints, how a chat request becomes an engine request (`max_tokens`, sampling), the JSON and SSE response
shapes, and the error codes. All of it is in `frontend/main.cpp`; the engine call goes through
`frontend/engine.hpp` ([engine-process.md](engine-process.md)), the template and parser through llama.cpp
([chat-and-parsing.md](chat-and-parsing.md)).

## Command-line options

`Options` (`frontend/main.cpp:53-66`), parsed by `parse_args` (`main.cpp:611-646`); `usage()` at `main.cpp:68-86`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model PATH` | required | first GGUF shard; for Flash-Next `...-00001-of-00002.gguf` with shard 2 beside it; Qwen3.6-35B-A3B is one file |
| `--pack DIR` | required | prepared pack; `index.txt` and `expert-profile.bin` must exist |
| `--mtp DIR` | required | prepared draft head; `experts.bin` must exist (`draft_vocab.bin` is optional, `main.cpp:666-667`) |
| `--mmproj PATH` | off | enable image input with this vision encoder (CPU) |
| `--vision-tokens N` | 1024 | maximum tokens per image, 16..4096 |
| `--ctx N` | 32768 | context length requested from the engine; at least 1024 |
| `--threads N` | 0 | CPU expert workers (`--pool-workers`); 0 = cgroup-quota default or every physical core |
| `--host ADDR` | `127.0.0.1` | listen address |
| `--port N` | 8080 | listen port |
| `--vram-cap MIB` | 0 (off) | forwarded: engine uses at most this much VRAM |
| `--ram-cap GIB` | 0 (off) | forwarded: bounds the engine's RAM expert tier and pinned arena |
| `-h`, `--help` | | usage, exit 2 |
| `-- ARGS...` | | engine options, appended verbatim to the engine's command line (see [engine-process.md](engine-process.md#the-engine-command-line-the-front-end-builds)) |

Exit code 2 for a missing value, an unknown option, a missing required option, a missing file
(`missing file: ...`, `main.cpp:668-669`), `--ctx` below 1024, a bad `--mmproj`/`--vision-tokens`, or a refused
engine option after `--`. Exit code 1 for engine start failure, tokenizer/template or vision load failure, the
engine failing before `READY`, or a failed `bind`.

The model id reported everywhere is `model_name_of(--model)` (`main.cpp:45-51`): the file's base name without
`.gguf` and without a `-00001-of-...` suffix, e.g. `Qwen3.6-35B-A3B-UD-IQ3_XXS`.

## Start-up sequence (`main`, `frontend/main.cpp:650-829`)

1. `--engine` as `argv[1]`: run the engine instead (`main.cpp:651`).
2. Parse and validate options; check files exist.
3. Build the engine command line and `Engine::start` it (fork/exec) before any thread exists.
4. Load the GGUF with `vocab_only = true` (`main.cpp:698-708`), so only the tokenizer and metadata are read;
   `common_chat_templates_init(model, "")` takes the template embedded in the GGUF. llama.cpp's log is filtered to
   warnings and errors (`quiet_log`, `main.cpp:578-580`).
5. With `--mmproj`, `Vision::init` ([vision.md](vision.md)).
6. `Engine::wait_ready`; adopt the engine's context if it differs.
7. Register handlers; payload limit 64 MiB, read and write timeouts 600 s (`main.cpp:789-791`); `SIGINT`/`SIGTERM`
   stop the server.
8. `bind_to_port`, print `ready on http://HOST:PORT (context N tokens)`, start the supervisor thread
   ([engine-process.md](engine-process.md#supervisor-child-watchdog-and-exit-behaviour)), `listen_after_bind`.

## Endpoints

### `GET /health` (`main.cpp:733-769`)

- Engine down: HTTP 503, `{"status":"error"}`.
- Engine up and idle: the handler `try_lock`s `Engine::mu`, sends `STATS`, and returns 200
  `{"status":"ok","tiers":{...}}`. `tiers` is the engine's JSON verbatim (`src/program/generate.cpp:6138-6158`):
  `experts`, `vram_slots`, `vram_held`, `ram_copies`, `mapped`, `mean_blob`, `tokens`, `vram_hits`, `lookups`,
  `ram_reads`, `file_reads`, `file_mb`, `warm_hints`, `major_faults`, `max_context`.
- Engine busy with a request (lock not available): the last good reading with `"tiers_stale":true`, or
  `"busy":true` if there has never been one.
- `STATS` failed and nothing cached: `"tiers_error":"<message>"`.

The last reading is kept in a function-static string under its own mutex, because httplib calls the handler from
several threads.

### `GET /v1/models` (`main.cpp:770-785`)

503 if the engine is down. Otherwise one model:
`{"object":"list","data":[{"id":<model name>,"object":"model","owned_by":"local","context_length":<ctx>,"input_modalities":["text"(,"image")]}]}`.
`context_length` is the engine's allocated context; `image` appears only when `--mmproj` loaded an encoder.

### `POST /v1/chat/completions` (`Server::handle_chat`, `main.cpp:466-568`)

Request fields read by `Server::prepare` (`main.cpp:208-367`):

| Field | Use |
| --- | --- |
| `messages` | required; OpenAI format, parsed by `common_chat_msgs_parse_oaicompat`; `image_url` content parts are encoded first ([vision.md](vision.md)) |
| `tools`, `tool_choice` (string form only), `parallel_tool_calls` | tool calling; `parallel_tool_calls` defaults to the template's `supports_parallel_tool_calls` capability |
| `reasoning_effort` | `"none"` disables thinking; any other string is passed to the template as `reasoning_effort` |
| `chat_template_kwargs` | `enable_thinking` (bool), `reasoning_effort` (string), `preserve_thinking` (bool); other keys ignored |
| `max_completion_tokens`, `max_tokens` | output cap (below) |
| `stream` | SSE when true |
| `temperature`, `top_p`, `top_k`, `seed`, `min_p` | sampling (below) |
| `presence_penalty`, `frequency_penalty`, `repeat_penalty` | penalties (below) |

Every other field (`model`, `stop`, `n`, `response_format`, `logprobs`, `stream_options`, ...) is not read. The
request's `model` is ignored: one process serves one model.

Template and parsing details: [chat-and-parsing.md](chat-and-parsing.md).

#### `max_tokens` and the 8-slot reserve (`main.cpp:301-318`)

The engine refuses a request unless `prompt + max_new + 8 <= context` (`src/program/generate.cpp:6770-6773`,
`ERR prompt (N tokens) + max_new (M) exceeds the context (C)`). The front end therefore:

- returns 400 "the prompt is N tokens, which does not fit the context of C" when `N + 1 + 8 > ctx`
  (`kEngineReserve = 8`);
- defaults `max_new` to `ctx - N - 8`, i.e. "as much as fits";
- if `max_completion_tokens` (checked first) or `max_tokens` is a number >= 1, uses `min(that, ctx - N - 8)`. The
  first of the two keys that is present and numeric decides; a value below 1 leaves the default, and the other key
  is not consulted.

So a client asking for more than the context has left (pi sends `max_tokens` 32768 on a 64K context) gets a
clamped request instead of a refusal. This was REGINA-112 (`docs/qwen36-port/step5-serve.md`), checked by
`tests/python/serve_max_tokens_check.py`.

#### Sampling

Built into the `GEN` header as `key=value` pairs (`main.cpp:321-350`):

- `temperature` absent or <= 0: greedy; no sampling keys are sent (the engine's default is greedy).
- `temperature` > 0: `temperature=T top_p=P top_k=K seed=S`, with `top_p` default 0.95, `top_k` default 20 and
  clamped to 1..64 ("the engine's sampler takes 1..64"), `seed` from the request or `std::random_device`; plus
  `min_p=X` if given.
- `presence_penalty` -> `penalty_present`, `frequency_penalty` -> `penalty_freq`, `repeat_penalty` ->
  `penalty_repeat`, each when numeric, and `penalty_last_n=64` once if any penalty is present. Penalties are sent
  also with greedy decoding.

Header format: `GEN <max_new> [keys] ` or `GENI <max_new> [keys] <embeddings file> `, followed by the ids.

#### Non-streaming response (`main.cpp:481-500`)

`Server::run` is called with an emit callback that does nothing; the text is parsed once at the end. On success
(200):

```
{"id":"chatcmpl-<16 hex>","object":"chat.completion","created":<s>,"model":<name>,
 "choices":[{"index":0,"message":<msg.to_json_oaicompat()>,"finish_reason":...}],
 "usage":{"prompt_tokens","completion_tokens","total_tokens"},
 "timings":{"prompt_n","prompt_ms","prompt_per_second","predicted_n","predicted_ms","predicted_per_second",
            "draft_n","draft_n_accepted"}}
```

`message` carries `content`, `reasoning_content` and `tool_calls` as llama.cpp's `to_json_oaicompat` writes them.
`prompt_tokens` / `prompt_n` is the full prompt length from `DONE` (including tokens the engine reused from a
checkpoint); `prompt_ms` is the engine's prompt time; `draft_n` / `draft_n_accepted` are the MTP draft tokens
offered and accepted. A client that disconnects during a non-streaming request is not noticed; the request runs to
the end.

#### Streaming response (`main.cpp:502-567`)

Headers `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`, chunked. Events,
each `data: <json>\n\n`:

1. `chat.completion.chunk` with `delta: {"role":"assistant","content":""}`, `finish_reason: null`.
2. One chunk per parser diff, `delta` holding any of `reasoning_content`, `content`, and `tool_calls: [{index, id,
   type:"function", function:{name, arguments}}]` (`id` and `type` only when the diff carries an id). Empty diffs
   are skipped. Reasoning and content arrive as they are generated; each tool call arrives once, complete.
3. Final chunk: empty `delta`, `finish_reason`, plus `usage` and `timings` as above.
4. `data: [DONE]`.

If the engine reports an error after the 200 has gone out, step 3 is replaced by
`data: {"error":{"message":...,"type":"server_error"}}`, still followed by `[DONE]`. If the client has gone
(a failed `sink.write`), nothing more is written: the generation is stopped with `STOP` at the next token.

Keepalive: when 15 s pass without an emit, the stream writes the SSE comment `: keepalive\n\n`
(`main.cpp:390-396`, `424-430`, `546-550`), which keeps clients with an idle timeout (pi: 5 minutes) attached during
a long, silent tool-call argument and detects a vanished client. It is checked only when a token arrives, so it
does not fire during a long prompt read.

`finish_reason` (`main.cpp:461-464`): `tool_calls` if the final message has tool calls; else `stop` if the engine's
finish was `stop` (an end-of-generation token); else `length` (this includes the engine's `cancel`).

## Error codes

| Status | `error.type` | When |
| --- | --- | --- |
| 400 | `invalid_request_error` | body not valid JSON; no `messages`; empty prompt; prompt does not fit the context; any exception in `prepare` (bad template input, image errors: not a base64 data URL, undecodable, too many tokens, more than 32 images, `--mmproj` not given, encoder failure, template dropped an image marker) |
| 500 | `server_error` | non-streaming request whose generation failed: engine `ERR` (e.g. the engine's own context check), engine exit, prompt-size mismatch, response formatter exception |
| 503 | `server_error` | engine not running (`/v1/chat/completions`, `/v1/models`); `/health` returns 503 with `{"status":"error"}` |
| 200 + error event | `server_error` | streaming request whose generation failed after the headers were sent |

Error bodies are `{"error":{"message":...,"type":...}}` (`json_error`, `main.cpp:128-135`). Unknown routes get
cpp-httplib's default 404.

## Logging and provenance capture

Every request logs `request <id> start: prompt=<n> header=<GEN ...>` and `request <id> end: ok=... prompt=...
generated=... finish=... error=...` to stderr (`main.cpp:401-402`, `456-457`), plus the raw output when a
`<tool_call>` was seen but not parsed or when the formatter failed ([chat-and-parsing.md](chat-and-parsing.md)).

`SUGMEOFF_CAPTURE_FIRST_PROMPT=<path>` (`main.cpp:351-365`): opt-in; the first prepared request is written to
`<path>` as JSON with `request`, `rendered_prompt`, `token_ids` and `engine_header`. The file is opened with `"wx"`
(exclusive create), so only the first request of a process is captured and an existing file is never overwritten.

## Concurrency

httplib serves requests on a thread pool, but every generation holds `Engine::mu` for its whole duration
(`Server::run`, `main.cpp:400`), and image encoding holds it too (`prepare`, `main.cpp:215`). Requests are
therefore served one at a time; others block (subject to httplib's 600 s timeouts). There is no queue limit and
no batch mode (`--batch` after `--` is refused).

## Model-specific behaviour

None in the HTTP layer beyond the model name and `input_modalities`; the template and vocabulary come from the
GGUF. Context limits come from the engine's plan for the model and card.

## Tests

- `tests/python/serve_max_tokens_check.py` (`make test-serve PORT=[HOST:]PORT`): the `max_tokens` clamp, streamed
  and not.
- `frontend/chat_parser_test.cpp` (`make test-chat-parser MODEL=...`): `prepare` and the streaming parse.
- Details in [tests.md](tests.md).

## Gotchas

- The repository README says content and tool calls are sent once the whole message is parsed; since the
  REGINA-110 fix (`docs/qwen36-port/step5-serve.md`) only tool calls are; content streams.
- `/health` builds its JSON by string concatenation; a `tiers_error` message containing a quote would not be
  escaped (`main.cpp:755`).
- No keepalive or cancellation check during the prompt read (only `PP` lines arrive then).
- `stop` sequences are not supported; generation ends at an end-of-generation token or `max_new`.

## See also

- [engine-process.md](engine-process.md), [chat-and-parsing.md](chat-and-parsing.md), [vision.md](vision.md), [tests.md](tests.md)
- [README.md (repository)](../../../README.md), "Run".
- [docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md) - streaming measurements, REGINA-110 and REGINA-112.
