[Codebase docs](../README.md) › Front end

# Front end (`frontend/`)

The front end is the program users actually run: the `unbiased-strata` binary. It is a small C++ HTTP server
(cpp-httplib) that speaks an OpenAI-style chat API, renders the GGUF's own Jinja chat template and tokenizes with
the GGUF's own vocabulary through llama.cpp's `common` library, and then hands token ids to the unmodified Strata
engine. The engine is not a library call: the same binary re-executes itself with `--engine` as a child process,
which runs the engine's own `main` from `src/program/generate.cpp` in `--serve` mode, and the two talk over a
pair of pipes in the engine's line protocol (`GEN ...` in, `T <id>` / `DONE ...` out). Tokens coming back are
turned into text and fed to llama.cpp's chat parser, which splits them into reasoning, answer and tool calls for
the HTTP response. Optional image input (`--mmproj`) is encoded on the CPU with llama.cpp's `mtmd` and passed to
the engine as an embeddings file. Callers: HTTP clients (chat UIs, coding agents such as pi). Callees: the engine
child ([../program/README.md](../program/README.md)) and llama.cpp's `llama`, `llama-common`, `mtmd` and
`cpp-httplib` libraries from the pinned `third_party/llama.cpp` submodule.

## Contents

| Page | What it covers |
| --- | --- |
| [http-api.md](http-api.md) | CLI options, start-up sequence, the three endpoints, request fields, response shapes, SSE streaming, error codes, `max_tokens` and the 8-slot reserve, sampling keys |
| [engine-process.md](engine-process.md) | One binary in two roles (`--engine`, the `main` rename), fork/exec, the arguments the engine is started with, the pipe protocol both sides speak, backpressure and cancellation, the child watchdog and exit behaviour, the engine's own stall watchdog |
| [chat-and-parsing.md](chat-and-parsing.md) | Chat template rendering, tokenization, detokenization, reasoning / tool-call parsing with `common_chat_parse`, the `StreamParse` publishing rules, `chat_template_kwargs`, the Qwen3-Coder argument-order patch |
| [vision.md](vision.md) | `--mmproj` image input: data-URL decoding, CPU `mtmd` encoding, the image cache, prompt splicing of `<|vision_start|>` / `<|image_pad|>` runs, the `SVE1` embeddings file and the engine's `GENI` request |
| [tests.md](tests.md) | `make test-frontend` (three model-free tests), `make test-chat-parser MODEL=...`, `make test-serve PORT=...` |

## Source map

| File | Lines | Role |
| --- | --- | --- |
| [frontend/main.cpp](../../../frontend/main.cpp) | 829 | The front end: options, start-up, request preparation (template, tokens, sampling header), streaming parse, HTTP handlers, supervisor thread; dispatches to the engine when `argv[1] == "--engine"` |
| [frontend/engine.hpp](../../../frontend/engine.hpp) | 189 | `strata_frontend::Engine`: fork/exec of the child, the pipe protocol client (`wait_ready`, `generate`, `command`, `quit`, `reap_exit`), `GenResult` |
| [frontend/engine_entry.cpp](../../../frontend/engine_entry.cpp) | 6 | Compiles `src/program/generate.cpp` into the binary with `#define main strata_engine_main` |
| [frontend/child_watch.hpp](../../../frontend/child_watch.hpp) | 22 | `child_running()`: non-blocking `waitpid` that reaps and describes a dead engine |
| [frontend/vision.hpp](../../../frontend/vision.hpp) | 153 | `Vision` (mtmd encoder + cache), `ImageFile` (temporary embeddings file), `image_data()` (base64 data-URL decoder), `write_images()` |
| [frontend/engine_test.cpp](../../../frontend/engine_test.cpp) | 62 | Protocol test against a fake engine: formatter exceptions, cancellation, `STATS`, prompt-size mismatch |
| [frontend/engine_backpressure_test.cpp](../../../frontend/engine_backpressure_test.cpp) | 71 | 100,000-token responses that overflow a pipe; STOP and drain must not leak into the next request |
| [frontend/child_watch_test.cpp](../../../frontend/child_watch_test.cpp) | 31 | `child_running()` on a normal exit and on `SIGABRT` |
| [frontend/chat_parser_test.cpp](../../../frontend/chat_parser_test.cpp) | 132 | `Server::prepare` and `StreamParse` against a real GGUF (tool calls, argument order, streaming deltas) |
| [tests/python/serve_max_tokens_check.py](../../../tests/python/serve_max_tokens_check.py) | 44 | Live-server check that requests asking for more tokens than the context has left still succeed (REGINA-112) |

Related build files (not front-end sources, but they define it): `CMakeLists.txt:1428-1444` (the `unbiased-strata`
target, only when `STRATA_ENABLE_CUDA` and `STRATA_NATIVE_EXPERTS` are on), `CMakeLists.txt:1150-1166` (applies
`patches/llama.cpp-qwen3-coder-any-argument-order.patch` at configure time), `Makefile:25-68` (`build`,
`test-frontend`, `test-chat-parser`, `test-serve`).

## Key concepts and data flow

```
 HTTP client
    |  POST /v1/chat/completions (JSON)            GET /v1/models, GET /health
    v
 +-------------------------- unbiased-strata (front end process) ---------------------------+
 | httplib thread  -> Server::handle_chat                                                    |
 |                     |-- Server::prepare: images -> Vision::encode (mtmd, CPU, cache)      |
 |                     |                    messages -> common_chat_templates_apply (Jinja)   |
 |                     |                    prompt   -> common_tokenize -> ids               |
 |                     |                    header   "GEN <max_new> key=val ..." / "GENI ..." |
 |                     |-- Server::run (holds Engine::mu: one request at a time)             |
 |                           Engine::generate(header, ids, on_token)                         |
 |                             on_token: token -> piece -> UTF-8 buffer -> StreamParse       |
 |                                       -> common_chat_parse -> diffs -> SSE chunk          |
 | supervisor thread: every 100 ms, waitpid(WNOHANG) / alive() -> _Exit(1) if engine gone    |
 +---------------------------|----------------------------^------------------------------------+
        stdin pipe:          | GEN/GENI/STOP/STATS/QUIT  | T <id>, DONE ..., ERR ..., PP/RESUME/REUSED/INFO
                             v                            |   (stdout pipe; stderr is shared with the parent)
 +----------------- unbiased-strata --engine --serve ... (child process) ---------------------+
 | strata_engine_main = generate.cpp's main; serve loop; stdin reader thread; stall watchdog   |
 +-------------------------------------------------------------------------------------------+
```

- **One process, one model, one conversation at a time.** `Engine::mu` (`frontend/engine.hpp:159`) serializes every
  generation and every image encode; concurrent HTTP requests wait on it. The request's `model` field is ignored.
- **The engine is unmodified.** All model knowledge on the front-end side comes from the GGUF (vocabulary,
  template, `embedding_length`); the engine sees only token ids, sampling keys and an optional embeddings file.
- **Context comes from the engine.** The engine's `READY <n>` line may report a smaller context than `--ctx` (its
  VRAM plan lowered it); the front end then bounds requests and reports `context_length` by that figure
  (`frontend/main.cpp:718-728`).
- **Failure model: crash and restart.** Any engine exit or protocol break makes the front end `_Exit(1)` so a
  supervisor such as systemd (`Restart=on-failure`) restarts the whole service (`frontend/main.cpp:801-820`).
- **Streaming rules.** Reasoning and answer text stream as they grow; tool calls are sent once, whole, from the
  parse of the complete message ([chat-and-parsing.md](chat-and-parsing.md)).

## See also

- [../program/README.md](../program/README.md) - the engine program (`src/program/generate.cpp`) in full, including `--serve`.
- [README.md (repository)](../../../README.md) - "Run": user-facing options, endpoints and image notes.
- [docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md) - serving checks through this front end (REGINA-110 streaming fix, `preserve_thinking`, REGINA-112).
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) - upstream Strata's notes; its server is upstream's Python one, which this front end replaces with the same process model.
