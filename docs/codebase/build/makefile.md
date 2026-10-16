[Codebase docs](../README.md) › [Build](README.md) › Makefile

# Makefile

## Purpose

[`Makefile`](../../../Makefile) (68 lines) is the supported way to build and check this fork. It wraps one
CMake configure and one `cmake --build --target unbiased-strata`, and adds three check targets that do not
go through CMake/CTest. Header comment: "Needs CMake 3.24+, a C++20 compiler and the CUDA toolkit. Nothing is
downloaded." (`Makefile:1`).

## Variables

| Variable | Default | Line | Meaning |
| --- | --- | --- | --- |
| `BUILD_DIR` | `build` | `Makefile:2` | CMake binary directory; also where `test-*` put their binaries (`$(BUILD_DIR)/frontend-tests/`) |
| `CUDA_ARCHS` | `61;70;75;86;89;120` | `Makefile:9` | Passed as `-DCMAKE_CUDA_ARCHITECTURES`. One fat binary for Pascal (61), Volta (70), Turing (75), Ampere (86), Ada (89), Blackwell (120) |
| `NVCC_ARCHS` | output of `nvcc --list-gpu-arch` with `compute_` stripped | `Makefile:10` | The archs the installed `nvcc` can compile; empty when `nvcc` is not on `PATH` |
| `CUDA_ARCHS_DROPPED` | computed | `Makefile:12` | Entries of `CUDA_ARCHS` whose number (before any `-real`/`-virtual` suffix) is not in `NVCC_ARCHS` |
| `SM60` / `SM60_FLAG` | computed | `Makefile:18-19` | `-DSTRATA_EXPERIMENTAL_SM60=ON` when any numeric arch in `CUDA_ARCHS` is below 75 |
| `JOBS` | `nproc`, capped at 16 (4 if `nproc` fails) | `Makefile:21` | Parallel build jobs; capped because in a container `nproc` reports the host's CPUs |
| `CMAKE_EXTRA` | empty | `Makefile:31` | Extra arguments appended to the configure line (e.g. `-DCMAKE_CUDA_HOST_COMPILER=g++-13`) |
| `CXX` | make's default (`g++`) | `Makefile:42,54` | Compiler for the `test-frontend` / `test-chat-parser` binaries |
| `MODEL` | none (required) | `Makefile:52` | First GGUF shard for `test-chat-parser` |
| `PORT` | none (required) | `Makefile:67` | `[HOST:]PORT` of a running server for `test-serve` |

### Arch filtering (`Makefile:10-17`)

When `nvcc` is on `PATH`, archs it cannot compile are removed from `CUDA_ARCHS` with the message
`make: nvcc cannot compile CUDA arch(s) <list>; building without them`. Per the comments and README: CUDA 12.4
has no sm_120; CUDA 13 has no sm_61 / sm_70. `SM60` is computed after the filtering, so a CUDA 13 toolkit that
drops 61 and 70 also turns off `STRATA_EXPERIMENTAL_SM60`.

## Targets

| Target | Lines | What it does |
| --- | --- | --- |
| `build` | `Makefile:25-33` | 1) Fails with "third_party/llama.cpp is empty: run 'git submodule update --init' first" if `third_party/llama.cpp/ggml/CMakeLists.txt` is missing. 2) Applies `patches/llama.cpp-qwen3-coder-any-argument-order.patch` with `git -C third_party/llama.cpp apply`, unless `git apply --check --reverse` shows it is already applied. 3) Configures with `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" $(SM60_FLAG) $(CMAKE_EXTRA)`. 4) `cmake --build $(BUILD_DIR) -j $(JOBS) --target unbiased-strata`. 5) Prints `built: $(BUILD_DIR)/unbiased-strata` |
| `clean` | `Makefile:35-36` | `rm -rf $(BUILD_DIR)` |
| `test-frontend` | `Makefile:39-47` | Compiles and runs three standalone front-end tests with `$(CXX) -std=c++20 -Wall -Wextra -Werror`: `frontend/child_watch_test.cpp`, `frontend/engine_test.cpp`, `frontend/engine_backpressure_test.cpp` (the last with `-pthread`). They include only `child_watch.hpp` / `engine.hpp`; no CUDA, no model, no CMake build needed |
| `test-chat-parser` | `Makefile:50-62` | Needs `MODEL=<first GGUF shard>` (exit 2 otherwise) and a prior `make build` (it links the static libraries that build produced). Compiles `frontend/chat_parser_test.cpp` (which `#include`s `main.cpp` with `main` renamed, and stubs `strata_engine_main`) at `-O0` against the llama.cpp headers and these archives: `llama.cpp/tools/mtmd/libmtmd.a`, `llama.cpp/vendor/hash/libvendor-hash.a`, `llama.cpp/common/libllama-common.a`, `llama.cpp/src/libllama.a`, `llama.cpp/vendor/cpp-httplib/libcpp-httplib.a`, `ggml/src/libggml.a`, `libggml-cpu.a`, `libggml-base.a`, `llama.cpp/common/libllama-common-base.a`, plus `-lpthread -ldl -lm`. Runs it on `$(MODEL)`: vocab-only model load, chat template and tool-call parsing, no engine and no CUDA context |
| `test-serve` | `Makefile:65-68` | Needs `PORT=[HOST:]PORT` of a running server. Runs `python3 tests/python/serve_max_tokens_check.py $(PORT)`: a request may ask for more tokens than the context has left (REGINA-112) |

The archive paths in `test-chat-parser` follow from where `CMakeLists.txt` places the subprojects:
`add_subdirectory(<ggml> ${CMAKE_BINARY_DIR}/ggml)` (`CMakeLists.txt:1167`) and
`add_subdirectory(<llama.cpp> ${CMAKE_BINARY_DIR}/llama.cpp)` (`CMakeLists.txt:1438`). The exact archive names
inside llama.cpp (e.g. `libvendor-hash.a`, `libllama-common-base.a`) come from the pinned llama.cpp's own CMake
files, which are not checked out here, so they were not verified against source.

## Usage

```
git submodule update --init          # once; the build fetches nothing
make build                           # -> build/unbiased-strata
make build CUDA_ARCHS=61 JOBS=8      # one-arch build (about 3 minutes per README)
make build CMAKE_EXTRA=-DCMAKE_CUDA_HOST_COMPILER=g++-13
make test-frontend
make test-chat-parser MODEL=/models/<shard 1>.gguf
make test-serve PORT=127.0.0.1:8080
```

## Gotchas

- **Arch filtering does not apply to a `CUDA_ARCHS` given on the command line.** `Makefile:15` reassigns
  `CUDA_ARCHS` with an ordinary assignment, and GNU make ignores ordinary assignments to a variable set on
  the command line. Checked with a stub `nvcc` that lists only 61/75/86: `make build CUDA_ARCHS='86;120'`
  prints "nvcc cannot compile CUDA arch(s) 120; building without them" but still passes `86;120` to CMake.
  Only the default list (or one set in the environment) is filtered.
- Without `nvcc` on `PATH` no filtering happens and CMake's own compiler check is the first failure.
- The patch is applied by `make build` with `git apply`, and again at CMake configure time with `patch`
  (see [third-party-and-patches.md](third-party-and-patches.md)); both are idempotent.
- `make build` configures with `STRATA_BUILD_TESTS=OFF` but CMake still registers some CTest entries (see
  [test-targets.md](test-targets.md#ctest-in-a-make-build-tree)); `ctest` in a `make build` tree is not a
  meaningful check.
- `test-frontend` and `test-chat-parser` are plain compiler invocations, not CMake targets: they do not see
  CMake's flags or definitions.
- The Makefile has no target for the CTest suite; the CUDA test configure line used during the Qwen3.6 port is
  in [docs/qwen36-port/step2-notes.md](../../qwen36-port/step2-notes.md) section 4.

## See also

- [cmake-options.md](cmake-options.md) for what each `-D` above does
- [targets.md](targets.md) for what `--target unbiased-strata` pulls in
- [docs/qwen36-port/pascal-build.md](../../qwen36-port/pascal-build.md) section 5 (the Makefile's Pascal switch on a GTX 1080 Ti)
- [Front end](../frontend/README.md) for the code the `test-frontend` / `test-chat-parser` targets exercise
