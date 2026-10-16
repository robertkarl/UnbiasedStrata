[Codebase docs](../README.md) › [Front end](README.md) › Engine process and protocol

# Engine process, pipe protocol and watchdogs

## Purpose

How the front end starts the Strata engine as a child process, the line protocol the two speak over pipes, how
cancellation and backpressure keep that protocol in step, and what happens when the engine dies or stalls.

Files covered: `frontend/engine_entry.cpp`, `frontend/engine.hpp`, `frontend/child_watch.hpp`, the start-up and
supervisor parts of `frontend/main.cpp`, and the engine's side of the protocol in `src/program/generate.cpp`
(documented in full in [../program/README.md](../program/README.md)).

## One binary, two roles

- `frontend/engine_entry.cpp:4-5` is the whole file: `#define main strata_engine_main` followed by
  `#include "../src/program/generate.cpp"`. The engine's source is compiled unchanged into the front-end binary;
  only its entry point is renamed by the preprocessor. (The rename is not done by CMake: `CMakeLists.txt:1440`
  simply lists `frontend/main.cpp frontend/engine_entry.cpp` as the sources of the `unbiased-strata` executable and
  links the engine libraries `strata_engine strata_prefill strata_spec` plus `llama-common llama mtmd cpp-httplib`.)
- `frontend/main.cpp:38` declares `int strata_engine_main(int, char**)`; `frontend/main.cpp:651` is the switch:
  if `argv[1] == "--engine"`, the process returns `strata_engine_main(argc - 1, argv + 1)`. The engine therefore
  sees `"--engine"` as its `argv[0]` and parses options from index 1 (`src/program/generate.cpp:1308`).
- The engine can be run by hand the same way, e.g. `./build/unbiased-strata --engine --native ... --serve`
  (used in `docs/qwen36-port/step2-notes.md` and step5-serve.md to drive the engine on its own stdin protocol).

## Starting the child

`Engine::start(args, err, executable = "/proc/self/exe")` (`frontend/engine.hpp:26-47`):

1. Two pipes: `to_child` (front end writes, engine's stdin) and `from_child` (engine's stdout, front end reads).
2. `fork()`. The child `dup2`s the pipe ends onto `STDIN_FILENO` / `STDOUT_FILENO`, closes all four descriptors and
   `execv`s `executable` (by default `/proc/self/exe`, i.e. the same binary) with `args`. On `execv` failure it prints
   `could not start the engine: ...` and `_exit(127)`. stderr is not redirected: the engine's log goes to the same
   stderr as the front end's.
3. The parent keeps `in_fd_` (write end) and wraps the read end in a `FILE*` (`fdopen`) for `getline`.

Ordering in `main` (`frontend/main.cpp:675-695`): the engine is started *before* the tokenizer, templates, vision
encoder or HTTP server are created, because "loading takes minutes, and it must be forked before any thread
exists". `SIGPIPE` is ignored first (`main.cpp:693`) so a dead engine shows up as a failed `write`, not a signal.

### The engine command line the front end builds

`frontend/main.cpp:676-692`:

| Argument | Source |
| --- | --- |
| `unbiased-strata --engine --serve` | fixed (`argv[0]`, role switch, engine serve mode) |
| `--pack <pack>` `--native <model>` | `--pack`, `--model` |
| `--expert-profile <pack>/expert-profile.bin` | fixed path inside the pack |
| `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5` | fixed defaults |
| `--mtp <dir>` `--max-context <ctx>` | `--mtp`, `--ctx` |
| `--ple-gguf <shard2> --ple-io ram` | only when the model path contains `-00001-of-00002` (Flash-Next: shard 2 holds the PLE table); Qwen3.6-35B-A3B is one file and gets neither |
| `--vram-cap <MiB>` / `--ram-cap <GiB>` | front-end options of the same names, when > 0 |
| `--vision` | when `--mmproj` is given |
| `--kv int8` | when `--ctx` > 8192 |
| `--pool-workers <N>` | `--threads`, or the cgroup-quota default (below), when > 0 |
| everything after `--` | appended last, unchanged |

Because user arguments come last and the engine's parser assigns options in order (e.g. `--kv` at
`src/program/generate.cpp:1371`), a later flag overrides the front end's default; the README's
`-- --kv q4_0` relies on this. `parse_args` refuses any argument after `--` that *starts with* `--gpus`,
`--layer-split`, `--split-device`, `--peer-device`, `--peer-slots`, `--remote-expert`, `--vision` or `--batch`
(exit code 2; `frontend/main.cpp:629-640`): one GPU only, no batch slots, and `--vision` is driven by `--mmproj`.

`default_threads()` (`frontend/main.cpp:586-609`): reads the cgroup v2 `cpu.max` (or v1 `cpu.cfs_quota_us` /
`cpu.cfs_period_us`); with no quota, or a quota at least the online CPU count, returns 0 (engine default: every
physical core). Otherwise `round(quota / period)` CPUs, halved when `/sys/devices/system/cpu/smt/active` is 1.

### Waiting for READY

`Engine::wait_ready(err, &context)` (`frontend/engine.hpp:52-66`) reads lines until one starts with `READY` (parses
the number after it as the allocated context) or `ERR` (returned as the error). Other lines (the engine's `INFO`
line) are ignored. EOF gives "the engine exited before it was ready (its messages are above)". The engine prints
`READY <max_context> stop` at `src/program/generate.cpp:6091`; `stop` advertises that it honours `STOP`. If the
number differs from `--ctx`, the front end logs it and adopts it (`frontend/main.cpp:724-728`).

## The wire protocol

Line-oriented ASCII over the two pipes; each message is one line ending in `\n` (`\r` is stripped on both sides).

### Front end to engine (stdin)

| Line | Meaning | Sent by |
| --- | --- | --- |
| `GEN <max_new> [key=value ...] <id>,<id>,...` | a text request: generate up to `max_new` tokens after the prompt ids | `Engine::generate` (`engine.hpp:74-81`) with the header from `Server::prepare` (`main.cpp:349-350`) |
| `GENI <max_new> [key=value ...] <file> <id>,...` | the same with images: `<file>` is the embeddings file (see [vision.md](vision.md)) | same |
| `STOP` | stop the running request at the next check (client gone or formatter failed) | `engine.hpp:97` |
| `STATS` | one-line tier census and counters | `Engine::command` from `/health` (`main.cpp:747`) |
| `QUIT` | leave the serve loop | `Engine::quit` (`engine.hpp:143`) |

Sampling keys the front end may put in the header: `temperature`, `top_p`, `top_k`, `seed`, `min_p`,
`penalty_present`, `penalty_freq`, `penalty_repeat`, `penalty_last_n` (mapping in [http-api.md](http-api.md#sampling)).
The engine also accepts `cvec`, `pcie_frac`, `spec_min_p`, the `VRAM` command and the batch commands
`BGEN`/`BGENI`/`BSTOP`/`BYIELD` (`src/program/generate.cpp:6570-6606` for the commands, `6625-6663` for the keys); the front end never sends them.

### Engine to front end (stdout)

| Line | Meaning | Front-end handling |
| --- | --- | --- |
| `INFO context=... kv=... ...` | start-up summary (`generate.cpp:6044`) | ignored |
| `READY <max_context> stop` | ready for requests | `wait_ready` |
| `RESUME <n>` | `n` prompt tokens are reused from a checkpoint (`generate.cpp:7045`) | ignored |
| `PP <reached> <total> <ms> <tok/s>` | prompt-reading progress (`generate.cpp:5736`, `7108`) | ignored |
| `REUSED <n>` | prompt read; decoding starts (`generate.cpp:7425`) | ignored |
| `T <id>` | one generated token (`generate.cpp:7583`) | `on_token(id)` |
| `DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> ...` | request finished (`generate.cpp:7775`); `finish` is `stop` (EOS), `length` or `cancel` | first seven fields parsed into `GenResult` (`engine.hpp:100-114`); the trailing tier fields are not read |
| `ERR <message>` | the request (or command) failed | `GenResult::error`; request ends |
| `STATS {json}` | reply to `STATS` (`generate.cpp:6138-6158`) | `/health` |

### Request lifecycle in `Engine::generate` (`frontend/engine.hpp:70-124`)

1. Serialize `header + comma-separated ids + "\n"` and write it fully (`send` loops over partial writes; any
   failed write sets `alive_ = false`).
2. Read lines until `DONE`, `ERR` or EOF. Each `T <id>` calls `on_token`. If `on_token` returns false or throws,
   the front end writes `STOP` once and keeps reading, discarding later `T` lines, until the request's own `DONE`
   or `ERR`. This drain is what keeps the next request from reading the previous one's tokens.
3. On `DONE`, if the engine's prompt count differs from the number of ids sent, the response is refused
   ("engine response prompt size mismatch") and `alive_` is set false: the protocol is considered out of step, and
   the supervisor then exits the front end. A formatter exception turns an otherwise good `DONE` into
   `ok = false`, "response formatter failed: ...".
4. EOF: `alive_ = false`, "the engine exited during the request".

`Engine::command(line, prefix, reply, err)` (`engine.hpp:128-139`) is the same for one-line replies between
requests; an `ERR` reply is an error but leaves the engine alive.

### Backpressure and cancellation

- The front end reads the engine's stdout synchronously on the request's HTTP thread. If the client is slow, the
  SSE write blocks, the pipe fills, and the engine blocks on `printf`/`fflush` of its next `T` line: generation
  is paced by the client, nothing is buffered without bound.
- The engine reads stdin on its own thread with `read(2)` (`src/program/generate.cpp:5938-5983`), so a `STOP` is
  seen while a request runs: it only sets `stop_req`, checked between prompt chunks and verify windows
  (`generate.cpp:7619`, `finish = "cancel"`); every other line is queued for the serve loop. A `STOP` that arrives
  between requests is cleared when the next request starts (`generate.cpp:6617`). The comment at
  `generate.cpp:5946-5948` explains the `read(2)`: glibc's `exit()` would otherwise hang on stdin's lock.
- After sending `STOP` the front end keeps draining, so an engine blocked on a full pipe is unblocked, reaches the
  check, and ends with `DONE ... cancel`. `frontend/engine_backpressure_test.cpp` checks this with 100,000-token
  responses (far more than a 64 KiB pipe).
- Cancellation is only noticed when a token arrives (the `on_token` callback is where the SSE write and the
  keepalive happen); while the engine reads a long prompt it emits only `PP` lines, which `generate` ignores, so
  a client that leaves during the prompt read is detected at the first generated token.

## Supervisor, child watchdog and exit behaviour

- `child_running(pid, why)` (`frontend/child_watch.hpp:11-21`): `waitpid(pid, WNOHANG)`. Still running (or
  `EINTR`) returns true. Otherwise it fills `why` ("engine killed by signal N", "engine exited with code N",
  "waitpid: ...") and sets `pid = -1`. It reaps, because `kill(pid, 0)` would report a zombie as alive.
- `Engine::reap_exit(why)` (`engine.hpp:152-156`) wraps it and marks the engine dead.
- The supervisor thread (`frontend/main.cpp:804-820`) runs once the HTTP port is bound: every 100 ms, if
  `reap_exit` reports an exit, or `alive()` is false (EOF, failed write, prompt-size mismatch), it prints
  `...; exiting frontend for service restart` and calls `std::_Exit(1)`. The comment explains the design: the
  engine's own watchdog intentionally aborts a stuck child, and the front end exits too so that systemd's
  `Restart=on-failure` reloads the complete service; it does not fork a replacement from a multithreaded process.
- Before the supervisor exists (during start-up), an engine that dies makes `wait_ready` fail and `main` returns 1.
- `/health` and `/v1/models` answer 503 and `/v1/chat/completions` 503 "the engine is not running" when `alive()` is
  false; in practice the window is at most one 100 ms supervisor tick.
- Clean shutdown: `SIGINT`/`SIGTERM` call `httplib::Server::stop()` (`main.cpp:574-576`, `792-793`); `listen`
  returns, the supervisor is told to stop and joined, then `Engine::quit()` sends `QUIT`, polls `waitpid(WNOHANG)`
  up to 100 x 100 ms, then sends `SIGKILL` and a blocking `waitpid` (`engine.hpp:141-150`), and `main` returns 0.

### The engine's stall watchdog

In serve mode the engine runs its own watchdog thread (`src/program/generate.cpp:6063-6090`): if a request is in
progress (`progress().busy`, set by `BusyScope` at `generate.cpp:6612-6616`) and the heartbeat counter has not
moved for `STRATA_WATCHDOG_S` seconds (default 60; 0 disables it), it prints "no progress for N s during a request
(stage) - stopping the engine so the server starts it again (issue #29)", a stall report (`stall_report`,
`generate.cpp:1007`), releases GPU spin waits and calls `std::abort()`. The front end's supervisor then sees
"engine killed by signal 6" and exits 1. Some engine `ERR` paths also `return 1` from the serve loop (fatal CUDA or
pool errors, e.g. `generate.cpp:7612-7613`), which ends the child the same way.

There is no "engine silence" timeout in this front end. Upstream's Python server has one (issue #481,
`docs/upstream/DETAILS.md` Troubleshooting); here a silent but alive engine with a stuck request relies on the
engine watchdog.

## Configuration

| Knob | Where | Effect |
| --- | --- | --- |
| `--threads N` | front end | engine `--pool-workers N`; default from the cgroup quota |
| `--vram-cap MiB`, `--ram-cap GiB` | front end | forwarded to the engine |
| `--ctx N` | front end | engine `--max-context N`; `--kv int8` above 8192 |
| `-- <engine flags>` | front end | appended verbatim (subject to the refused prefixes) |
| `STRATA_WATCHDOG_S` | engine env (inherited) | stall watchdog seconds, 0 = off |
| `STRATA_TRACE=1` | engine env (inherited) | one stderr line per request step (`generate.cpp:5994-5999`) |

The child inherits the front end's environment unchanged (`execv`).

## Model-specific behaviour

Only the shard-2 rule: a model path containing `-00001-of-00002` (Flash-Next) also requires `-00002-of-00002`
beside it and passes it as `--ple-gguf ... --ple-io ram`; a single-file Qwen3.6-35B-A3B gets neither, and the
engine decides the architecture from the file (`frontend/main.cpp:659-664`). End-of-generation tokens are the
engine's `--eos-ids` (default 248044 `<|endoftext|>`, 248046 `<|im_end|>`, `generate.cpp:510-512`).

## Tests

- `frontend/engine_test.cpp`, `frontend/engine_backpressure_test.cpp`, `frontend/child_watch_test.cpp`, run by
  `make test-frontend` (no model, no CUDA). See [tests.md](tests.md).

## Gotchas

- The prompt-size check on `DONE` is fatal by design: a mismatch means the two sides may be out of step.
- `Engine::quit()` sends `SIGKILL` and a blocking `waitpid` to the stored pid even if the polling loop already
  reaped the child (`engine.hpp:145-147`); the blocking `waitpid` then fails harmlessly with `ECHILD`.
- Only the first seven `DONE` fields are read; the engine appends more (reused tokens, tier counters) "so an older
  server reads the rest" (`generate.cpp:7768-7772`).
- `GenResult::finish` of `cancel` is reported to HTTP clients as `length` (see [http-api.md](http-api.md)).

## See also

- [http-api.md](http-api.md), [chat-and-parsing.md](chat-and-parsing.md), [vision.md](vision.md), [tests.md](tests.md)
- [../program/README.md](../program/README.md) - `generate.cpp`, the serve loop, conversation checkpoints, `STATS`.
- [docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md) - the engine driven on its stdin protocol.
