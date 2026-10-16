[Codebase docs](../README.md) › [Front end](README.md) › Tests

# Front-end tests

## Purpose

The tests that cover `frontend/`: three model-free protocol and supervision tests, one test that needs the real
GGUF (CPU only), and one check against a running server. None of them is registered with CTest; they are run from
`Makefile` targets.

| Test | Target | Needs |
| --- | --- | --- |
| `frontend/child_watch_test.cpp` | `make test-frontend` | a C++20 compiler |
| `frontend/engine_test.cpp` | `make test-frontend` | a C++20 compiler |
| `frontend/engine_backpressure_test.cpp` | `make test-frontend` | a C++20 compiler, pthreads |
| `frontend/chat_parser_test.cpp` | `make test-chat-parser MODEL=<shard 1>` | a prior `make build` (it links the llama.cpp static libraries from `build/`), the GGUF |
| `tests/python/serve_max_tokens_check.py` | `make test-serve PORT=[HOST:]PORT` | a running `unbiased-strata`, Python 3 |

`make test-frontend` (`Makefile:38-47`) compiles each test with `$(CXX) -std=c++20 -Wall -Wextra -Werror` into
`build/frontend-tests/` and runs it; each prints a `PASS: ...` line and uses `assert`, so they must not be built
with `NDEBUG`.

## `child_watch_test.cpp` (31 lines)

Forks two children that sleep 30 ms and then `_exit(17)` or `raise(SIGABRT)` (core dumps disabled with
`setrlimit(RLIMIT_CORE)`). Asserts `child_running` first reports running, then within 1 s reaps the child, sets the
pid to -1 and describes it as "engine exited with code 17" or "engine killed by signal 6". The abort case stands in
for the engine's stall watchdog. Covers `frontend/child_watch.hpp`.

## `engine_test.cpp` (62 lines)

The test binary re-executes itself with `--fake-engine` through `Engine::start(..., argv[0])` (the `executable`
parameter replaces `/proc/self/exe`). The fake engine prints `READY`, answers `STATS` and `STATS BAD`, ignores
`STOP`, and for a `GEN` whose text contains `99` prints `T 99`, `T 100`, `DONE 2 1 ...`; containing `88` it prints
a `DONE` whose prompt count is 2 for a 1-id prompt; otherwise `T 11`, `T 12`, `DONE`. Asserted:

1. An `on_token` that throws ("synthetic partial tool JSON diff error") gives `ok = false` with that message, and
   the next request still reads its own tokens (`99, 100`): the old request was drained.
2. A cancelled request (`on_token` returns false) ends `ok`, and the next request is clean.
3. `command("STATS", "STATS ", ...)` returns the JSON line; `STATS BAD` returns the `ERR` text and leaves the engine
   alive; a following request is unaffected.
4. A prompt-size mismatch in `DONE` fails the request with "prompt size mismatch" and marks the engine dead; the next
   `generate` fails immediately with nothing produced.

Covers `Engine::start`, `wait_ready`, `generate`, `command`, `quit` (`frontend/engine.hpp`).

## `engine_backpressure_test.cpp` (71 lines)

A fake backend modelled on the real engine: unbuffered stdout, a separate stdin reader thread that turns `STOP`
into a flag (as `generate.cpp`'s reader does), and up to 100,000 `T` lines per request with the flag checked per
token. That is far more than a Linux pipe holds, so the backend blocks on writes while the front end is not reading.
Asserted: a throwing formatter, then a full 100,000-token request of only `99`s; a cancellation, then another full
request of only `99`s. That is, a large response cannot fill an abandoned pipe or leak into the next request.

## `chat_parser_test.cpp` (132 lines)

Includes `main.cpp` with `main` renamed (`#define main unused_serving_main`) and provides a stub
`strata_engine_main` returning 99, so it can construct a `Server` without an engine or CUDA. It loads the GGUF given
as `argv[1]` vocab-only, sets `opt.ctx = 196608`, and checks:

1. The template advertises `supports_parallel_tool_calls`.
2. A request with a `bash` tool: two Qwen3-Coder tool calls in the output parse to two calls with the right JSON
   arguments when `parallel_tool_calls` is omitted or true, and to one call when it is false; the prompt ids are the
   same in all three cases.
3. Argument order (the any-order patch): a `read` tool with `path` required and `offset`/`limit` optional parses to
   one call with the right `path` for orders `limit offset path`, `offset path limit`, `path limit offset`, `path`.
4. Streaming (REGINA-110): seven cases (reasoning then answer; answer with thinking off; tools declared but not
   called, thinking on and off; text then a tool call; a tool call alone; two tool calls). Each output is fed to
   `StreamParse::step` one complete UTF-8 prefix at a time, then once as final. The concatenated reasoning deltas
   must equal the whole parse's reasoning; the concatenated content must equal the whole parse's content, except for
   trailing white space the final parse trims before a call; content and reasoning must arrive in many deltas
   (at least one per two bytes); each tool call must arrive in exactly one delta, only from the final parse, with a
   non-empty id. The sample answer includes `<tool` and `<function=` as plain text, a JSON code block, `<b>` tags
   and a multi-byte character.

Setting `STREAM_TEST_VERBOSE` prints the whole and streamed content per case. The `Makefile:50-62` target compiles
it against `build/llama.cpp/...` and `build/ggml/...` static libraries with the submodule's include paths, so the
submodule must be checked out and `make build` must have run.

## `tests/python/serve_max_tokens_check.py` (44 lines)

Against a running server (`[HOST:]PORT`, default `127.0.0.1:8080`): reads `context_length` from `/v1/models`,
then sends `"Reply with the single word: ok"` with `temperature 0` and `reasoning_effort "none"` four ways: no
`max_tokens`, `max_tokens = context`, `max_tokens = context - 20`, `max_completion_tokens = 2 x context`. Each must
return 200, `finish_reason: stop` and at least one completion token. A fifth, streamed request without
`max_tokens` must contain no `"error"`, a `"finish_reason":"stop"` and end with `[DONE]` (the comment notes that a
refusal would arrive as an error event after a 200). Exits 1 if any case fails. This guards the 8-slot reserve
(REGINA-112; [http-api.md](http-api.md#max_tokens-and-the-8-slot-reserve)).

## Not covered by tests

- The vision path (`vision.hpp`), the HTTP handlers' JSON shapes, `/health` and the supervisor thread have no
  automated test in the repository; they were checked by hand in `docs/qwen36-port/step5-serve.md`.

## See also

- [engine-process.md](engine-process.md), [chat-and-parsing.md](chat-and-parsing.md), [http-api.md](http-api.md), [vision.md](vision.md)
- [docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md) - where these tests were run and what they caught.
