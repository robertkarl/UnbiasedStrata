[Codebase docs](../README.md) › [Guides](README.md) › Testing

# Testing: one index of every test in the repository

## Purpose

Tests in this tree are spread over six places and three drivers (Make, CTest, Python `unittest`). This page lists all
of them, says what each needs (CPU only, a CUDA device, Python with numpy, a real model file), how it is registered,
and how to build and run it. Kernel parity programs are only counted here; their details are in
[../kernels/parity/README.md](../kernels/parity/README.md). Build options are in [../build/README.md](../build/README.md).

| Kind | Where | Driver | Count |
| --- | --- | --- | --- |
| Front-end tests | `frontend/*_test.cpp`, `tests/python/serve_max_tokens_check.py` | `make test-frontend`, `make test-chat-parser`, `make test-serve` | 4 C++ + 1 Python |
| Engine unit tests beside their code | `src/**/*_test.cpp`, `src/kernels/cpu/{pool_stress,expert_parity}.cpp`, `src/platform/memory_test.cpp` | CTest | 21 files |
| Engine tests in `tests/` | `tests/core` (11 + 1 header), `tests/cuda` (7), `tests/platform` (1) | CTest | 19 programs |
| Kernel parity programs | `src/kernels/*_parity.cpp` (37), `src/prefill/*_parity.cu` (2), `src/kernels/cpu/router_dot_parity.cpp` | CTest | 40 programs |
| Python unit tests | `tools/test_*.py` | `python3 -m unittest` | 5 files, 77 test methods |
| Python fixtures and checks | `tests/python/*.py`, `cmake/qwen36_*.cmake`, `tools/iq_fixture.py`, `tools/qwen36_*` | CTest fixtures, or by hand | see section 5 |

---

## 1. How to build and run

### Front end (no CUDA, no model): `make test-frontend`

`Makefile:39-47` compiles three header-only tests with `$(CXX) -std=c++20 -Wall -Wextra -Werror` into
`$(BUILD_DIR)/frontend-tests/` and runs them: `child_watch_test.cpp` (31 lines), `engine_test.cpp` (62),
`engine_backpressure_test.cpp` (71, `-pthread`). Needs only a C++20 compiler; it ran green on this checkout with
`BUILD_DIR` pointed outside the tree. Details: [../frontend/tests.md](../frontend/tests.md).

### Chat template and tool-call parsing: `make test-chat-parser MODEL=<first GGUF shard>`

`Makefile:50-62` links `frontend/chat_parser_test.cpp` (132 lines; it `#include`s `main.cpp` with `main` renamed)
against the llama.cpp static libraries a previous `make build` left in `$(BUILD_DIR)`, then runs it on the real
GGUF's template. CPU only, no engine. Needs the `third_party/llama.cpp` submodule and a completed build.

### Against a running server: `make test-serve PORT=[HOST:]PORT`

`Makefile:65-68` runs `tests/python/serve_max_tokens_check.py` (44 lines, stdlib only): a request may ask for more
tokens than the context has left (REGINA-112: the engine refuses `prompt + max_new + 8 > context`, so the front end
must keep those 8 slots out of `max_new`).

### Engine tests: CMake and CTest

`make build` configures with `-DSTRATA_BUILD_TESTS=OFF` and builds only the `unbiased-strata` target (`Makefile:25-33`).
`STRATA_BUILD_TESTS` defaults ON only when both `tests/CMakeLists.txt` and `bench/micro/` exist
(`CMakeLists.txt:87-92`); neither is in this tree, so the default is OFF and the option must be given. The
configure line the port notes used on both test cards (`docs/qwen36-port/step3-integration.md`, `step2-notes.md`
section 4):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
      -DSTRATA_PORTABLE=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89            # Pascal: =61 -DSTRATA_EXPERIMENTAL_SM60=ON; gcc-13 as host compiler
cmake --build build -j 8                       # every target, not only unbiased-strata
(cd build && ctest --output-on-failure)        # or ctest -R 'qwen36|vram_plan_test|pool_native_widths_test'
```

Rules that apply to every CTest entry:

- **Skips.** Exit 77 is a skip for GPU tests (`SKIP_RETURN_CODE 77`: no device, no VMM, below sm_80 for the fused
  prompt MoE, a missing real-file env var); exit 3 is a skip for fixture generators and tests that need Python
  with numpy and gguf-py (`SKIP_RETURN_CODE 3`).
- **Python for fixtures.** CMake probes `${Python3_EXECUTABLE}`, `<repo>/.venv/bin/python`, `/usr/bin/python3`,
  `python3` for one that imports numpy (`CMakeLists.txt:239-249`, again at `:681-703` for `iq_parity`, which also tries `.venv/Scripts/python.exe`;
  `-DIQ_FIXTURE_PYTHON=` overrides the latter). Without one, the Qwen3.6 fixture tests and `iq_parity` are not
  registered at all (a configure message says so). gguf-py comes from `STRATA_GGUF_PY` or
  `third_party/llama.cpp/gguf-py` (`tools/_paths.py:16`); the fixture tests set `STRATA_GGUF_PY` to the submodule.
- **CTest fixtures.** Generators run as tests with `FIXTURES_SETUP`; consumers name them in `FIXTURES_REQUIRED`, so
  `ctest -R qwen36_block_test` also runs `qwen36_synth_fixtures` and `qwen36_block_fixtures` first. Generated files
  go to the build directory (`build/qwen36_synth`, `qwen36_block`, `qwen36_oracle`, `qwen36_attn`, `qwen36_prefill`,
  `qwen36_mtp`, `qwen36_runner`, `iq_fixture`), never the source tree.
- **Registration is not uniform.** Most kernel parity tests and a few unit tests sit in the
  `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` block (`CMakeLists.txt:281-1038`) with no `STRATA_BUILD_TESTS`
  guard: they are registered (and their executables are part of the `all` target) whenever a GPU backend is on.
  In a `make build` tree those entries exist but their programs were never built, so `ctest` there fails with
  missing executables; use a tree configured and built as above.
- **Known host failures.** The notes record the same three failures on every card: `ple_parity` (its required inputs
  `bench/micro/ple_in.bin` / `ple_out.bin`, `STRATA_PLE_FIXTURE_DIR`, and a Flash-Next shard are not in this tree),
  `platform_memory_test` (the container's `mlock` limit), `expert_multi_test` (needs AVX-512). Recorded totals:
  89 of 92 on the RTX 4090 and 87 of 92 (2 skipped) on the GTX 1080 Ti at step 3; 95 of 98 at step 4's end;
  94 of 98 with `pack_load_check` skipped at d00d383 (`docs/qwen36-port/step5-serve.md`).

### Conversation-cache tests: `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`

A separate opt-in (`CMakeLists.txt:93-104, 540-557`): `conversation_cache_test`, `conversation_memory_test` (CPU),
and, with a GPU backend, `conversation_snapshot_test`, `conversation_validation_test` and, on Linux with GCC/Clang,
`conversation_transfer_test` (the validation test relinked with `--wrap=cudaMemcpy` etc. to count transfers;
`CUDA_VISIBLE_DEVICES=-1`). `conv_cache_test`, `coupled_draft_test` and `bf16_bits_test` build with either option.

### Python unit tests

```
python3 -m unittest tools.test_draft_vocab tools.test_make_profile     # no model, no gguf-py
STRATA_GGUF_PY=third_party/llama.cpp/gguf-py python3 -m unittest tools.test_iq_pack tools.test_mtp_fetch
python3 -I tools/test_qwen36_oracle.py                                # numpy + gguf-py
```

`tools.test_draft_vocab` and `tools.test_make_profile` (13 tests) pass on a checkout without the submodule; the
other three exit early with "llama.cpp's gguf-py was not found" until it is checked out or `STRATA_GGUF_PY` is set.

---

## 2. Engine unit tests beside the code (`src/`)

| File | Lines | CTest name | Registered when | What it checks |
| --- | --- | --- | --- | --- |
| [src/artifact/gguf_reader_test.cpp](../../../src/artifact/gguf_reader_test.cpp) | 93 | `gguf_reader_test` | TESTS | the GGUF reader refuses a duplicate tensor name, naming it (synthetic GGUF) |
| [src/core/conversation_cache_test.cpp](../../../src/core/conversation_cache_test.cpp) | 305 | `conversation_cache_test` | CONV | conversation cache bookkeeping (header-only) |
| [src/core/conversation_memory_test.cpp](../../../src/core/conversation_memory_test.cpp) | 66 | `conversation_memory_test` | CONV | conversation memory accounting |
| [src/core/conversation_snapshot_test.cpp](../../../src/core/conversation_snapshot_test.cpp) | 324 | `conversation_snapshot_test` | GPU + CONV | snapshot save/restore incl. q4 KV (skip 77) |
| [src/core/conversation_validation_test.cpp](../../../src/core/conversation_validation_test.cpp) | 239 | `conversation_validation_test`, `conversation_transfer_test` | GPU + CONV | invalid restores refused before any CUDA call or write |
| [src/core/coupled_draft_test.cpp](../../../src/core/coupled_draft_test.cpp) | 115 | `coupled_draft_test` | TESTS or CONV | coupled draft sampling's Philox counters and penalty ring (CPU) |
| [src/core/pinned_shared_test.cpp](../../../src/core/pinned_shared_test.cpp) | 61 | `pinned_shared_test` | GPU, Linux | `PinnedArena`'s `MAP_SHARED` backing |
| [src/kernels/bf16_bits_test.cpp](../../../src/kernels/bf16_bits_test.cpp) | 80 | `bf16_bits_test` | TESTS or CONV | f32 -> bf16 rounding vs ggml over all 2^32 inputs, NaN included |
| [src/kernels/dequant_bf16_test.cpp](../../../src/kernels/dequant_bf16_test.cpp) | 108 | none (by hand) | GPU + TESTS | device dequantizers vs CPU ones on a real Flash-Next shard |
| [src/kernels/cpu/expert_multi_test.cpp](../../../src/kernels/cpu/expert_multi_test.cpp) | 87 | `expert_multi_test` | TESTS | multi-token CPU expert kernel bitwise vs per token (AVX-512) |
| [src/kernels/cpu/expert_parity.cpp](../../../src/kernels/cpu/expert_parity.cpp) | 184 | `expert_parity` | TESTS | CPU expert path vs scalar oracle; reads `pack/full/experts.bin` (a required input) |
| [src/kernels/cpu/pool_test.cpp](../../../src/kernels/cpu/pool_test.cpp) | 224 | `pool_test` | TESTS | expert pool = serial answer, bit for bit |
| [src/kernels/cpu/pool_stress.cpp](../../../src/kernels/cpu/pool_stress.cpp) | 99 | `pool_stress` | TESTS | tiny batches with sleeping workers (issue #29); a hang fails it |
| [src/kernels/cpu/pool_affinity_linux_test.cpp](../../../src/kernels/cpu/pool_affinity_linux_test.cpp) | 290 | `pool_affinity_test` | TESTS, Linux | worker CPU affinity modes |
| [src/kernels/cpu/pool_affinity_test.cpp](../../../src/kernels/cpu/pool_affinity_test.cpp) | 148 | `pool_affinity_test` | TESTS, Windows | the Windows variant (not built here) |
| [src/ngram/ple_reader_test.cpp](../../../src/ngram/ple_reader_test.cpp) | 298 | `ple_reader_selftest` | GPU | direct SSD row reader on a synthetic table; `--gguf` mode on the real one |
| [src/platform/memory_test.cpp](../../../src/platform/memory_test.cpp) | 72 | `platform_memory_test` | GPU | `lock_resident` on 256 MiB (fails under a low `ulimit -l`) |
| [src/program/conv_cache_test.cpp](../../../src/program/conv_cache_test.cpp) | 67 | `conv_cache_test` | TESTS or CONV | serve-loop checkpoint retention (root kept, leaves LRU) |
| [src/spec/controller_test.cpp](../../../src/spec/controller_test.cpp) | 74 | `controller_test` | TESTS | speculation controller decisions |
| [src/spec/draft_policy_test.cpp](../../../src/spec/draft_policy_test.cpp) | 63 | `draft_policy_test` | TESTS | MTP vs lookup window choice |
| [src/spec/suffix_drafter_test.cpp](../../../src/spec/suffix_drafter_test.cpp) | 141 | `suffix_drafter_test` | TESTS | suffix-lookup drafter; `--simulate` replays text offline |

"TESTS" = `STRATA_BUILD_TESTS`, "CONV" = `STRATA_BUILD_CONVERSATION_TESTS`, "GPU" = inside the CUDA/HIP block.
`strata-device --selftest` is also registered (`cuda_device_selftest`, GPU block, `CMakeLists.txt:1005-1009`): it
allocates real VRAM and fails, by design, on a machine without a device.

## 3. Engine tests in `tests/`

| File | Lines | CTest name(s) | Registered when | Needs | What it checks |
| --- | --- | --- | --- | --- | --- |
| [tests/core/gguf_fixture.hpp](../../../tests/core/gguf_fixture.hpp) | 113 | (header) | | | synthetic GGUF v3 writer for the loader tests |
| [tests/core/gguf_split_test.cpp](../../../tests/core/gguf_split_test.cpp) | 220 | `gguf_split_test` | TESTS | CPU | split-shard discovery and joint view (missing shard, mismatched split keys, duplicates, truncation) |
| [tests/core/model_geometry_test.cpp](../../../tests/core/model_geometry_test.cpp) | 273 | `qwen36_synth_fixtures`, `model_geometry_test` | TESTS + python | CPU | `ModelGeometry` from metadata on `tools/qwen36_synth_gguf.py` files; `--real FILE` by hand |
| [tests/core/vram_plan_test.cpp](../../../tests/core/vram_plan_test.cpp) | 223 | `vram_plan_test` | TESTS | CPU | the VRAM planner vs the study's table 4.3; `--at q4|iq3 MIB` prints a plan |
| [tests/core/file_expert_source_test.cpp](../../../tests/core/file_expert_source_test.cpp) | 360 | `file_expert_source_test` | GPU (file exists) | CPU | the file expert tier (`FileExpertSource`) |
| [tests/core/expert_profile_save_test.cpp](../../../tests/core/expert_profile_save_test.cpp) | 92 | `expert_profile_save_test` | GPU (file exists) | CPU | `--expert-profile-save` ranking and file format ([data-files.md](data-files.md)) |
| [tests/core/expert_layout_test.cpp](../../../tests/core/expert_layout_test.cpp) | 412 | `expert_layout_test` | GPU + TESTS + NATIVE | CPU | `native_experts.txt` v2-v4 on `tests/data/native_experts`, arena from synthetic shards |
| [tests/core/pool_native_widths_test.cpp](../../../tests/core/pool_native_widths_test.cpp) | 125 | `pool_native_widths_test` | GPU + TESTS + NATIVE | CPU | pool uses the layer's 2048 x 512, not 2560 x 640 (step-5 fault 3) |
| [tests/core/native_dense_ple_key_test.cpp](../../../tests/core/native_dense_ple_key_test.cpp) | 169 | `native_dense_ple_key_test` | GPU + TESTS + NATIVE | CUDA (77) | BF16 PLE key stays in the arena (#326) |
| [tests/core/expert_cache_per_layer_test.cpp](../../../tests/core/expert_cache_per_layer_test.cpp) | 44 | `expert_cache_per_layer_test` | GPU + TESTS + NATIVE | CUDA (77) | per-layer slot ranges on a sized cache (#369) |
| [tests/core/expert_cache_segmented_test.cpp](../../../tests/core/expert_cache_segmented_test.cpp) | 99 | `expert_cache_segmented_test` | GPU + TESTS + NATIVE | CUDA VMM (77) | `--vram-elastic` shrink/grow keeps slots (#533) |
| [tests/core/pack_load_check.cpp](../../../tests/core/pack_load_check.cpp) | 226 | `pack_load_check` | GPU + TESTS + NATIVE | `STRATA_QWEN36_GGUF` + `STRATA_QWEN36_PACK` (else 77) | a real Qwen3.6 pack loads like `--native --pack` up to the first token |
| [tests/cuda/qwen36_block_test.cpp](../../../tests/cuda/qwen36_block_test.cpp) | 531 | `qwen36_block_fixtures`, `qwen36_block_test` | GPU + TESTS + CUDA + python | CUDA | the plain (hc = 1) block vs the oracle's dumps, seam by seam |
| [tests/cuda/qwen36_verify_test.cpp](../../../tests/cuda/qwen36_verify_test.cpp) | 511 | `qwen36_runner_fixtures`, `qwen36_verify_test` | GPU + TESTS + CUDA + python | CUDA | verify window on a synthetic native pack vs oracle and vs the single-token path |
| [tests/cuda/prefill_qwen36_test.cpp](../../../tests/cuda/prefill_qwen36_test.cpp) | 483 | `prefill_qwen36_fixtures`, `prefill_qwen36_test` (gdn3), `prefill_qwen36_attn_test` (full4) | GPU + CUDA (not gfx906) + python | CUDA | the prompt path at Qwen3.6's geometry vs the oracle |
| [tests/cuda/qwen36_mtp_test.cpp](../../../tests/cuda/qwen36_mtp_test.cpp) | 622 | `qwen36_mtp_fixtures`, `qwen36_mtp_test` | GPU + CUDA (not gfx906) + python | CUDA | the Qwen3.6 drafter vs the oracle's `--mtp` dumps; `--real` by hand |
| [tests/cuda/prefill_fused_moe_test.cpp](../../../tests/cuda/prefill_fused_moe_test.cpp) | 420 | `prefill_fused_moe_test` | NATIVE + CUDA + TESTS | sm_80 (77 below) | fused int8 Q2_0 prompt experts vs double reference and MMQ (#136) |
| [tests/cuda/prefill_fused_iq_test.cpp](../../../tests/cuda/prefill_fused_iq_test.cpp) | 642 | `prefill_fused_iq_test` | NATIVE + CUDA + TESTS | sm_80 (77 below) | the same for the native i-quant formats |
| [tests/cuda/prefill_mmq_kquant_test.cpp](../../../tests/cuda/prefill_mmq_kquant_test.cpp) | 209 | `prefill_mmq_kquant_test` | NATIVE + CUDA + TESTS + `STRATA_MMQ_KQUANTS` | CUDA | MMQ for Q4_K/Q5_K/Q5_1/Q8_0 (and Q5_K/Q6_K at 2048 x 512) |
| [tests/platform/direct_file_async.cpp](../../../tests/platform/direct_file_async.cpp) | 60 | `direct_file_async_test` | TESTS, Unix non-Apple | CPU | `DirectFile` asynchronous reads |

`qwen36_runner_check` (`tests/python/qwen36_runner_check.py`) runs the `strata` binary on the `qwen36_runner`
fixture under the same guard as `qwen36_verify_test`.

## 4. Kernel parity programs

40 programs compare a kernel with a reference (a host implementation, an older kernel bit for bit, ggml's
dequantizers, or the numpy oracle's dumps). 37 are `src/kernels/*_parity.cpp`, two are `src/prefill/gdn_rec_parity.cu`
and `gemm_bf16_parity.cu`, one is `src/kernels/cpu/router_dot_parity.cpp`. Most run as `<name> --selftest` under CTest;
`native_expert_parity` alone accounts for 15 entries (seven Flash-Next-geometry format pairs, five Qwen3.6 pairs
at `--geom 2048x512`, `q5_1_min`, `bf16_embd`, and an expected refusal marked `WILL_FAIL`); `sampler_parity` runs
three times with different `STRATA_*SAMPLER*` environments. Not registered (by-hand programs):
`qsa_prompt_attn_parity` (also buildable alone with `-DSTRATA_PARITY_PROMPT_ATTN=ON`), `ple_q5_parity`,
`q5_projection_parity`, `ple_fp8_parity`. Benchmarks that assert nothing are separate targets with no CTest entry:
`fused_gr_bench`, `native_expert_bench`, `qsa_attn_bench`; `qsa_select_bench` is a HIP-only CTest entry.
See [../kernels/parity/README.md](../kernels/parity/README.md) for each program.

## 5. Python: unit tests, fixture generators, oracle checks

### Unit tests (`tools/test_*.py`)

| File | Lines | Tests | Needs | Covers |
| --- | --- | --- | --- | --- |
| [tools/test_draft_vocab.py](../../../tools/test_draft_vocab.py) | 47 | 4 | stdlib | `draft_vocab.py --corpus` coverage (#597) |
| [tools/test_make_profile.py](../../../tools/test_make_profile.py) | 145 | 9 | stdlib | `make_profile.py` ranking, trace readers, `--n-layer` |
| [tools/test_iq_pack.py](../../../tools/test_iq_pack.py) | 840 | 32 | numpy, gguf-py | `iq_pack.py`: Flash-Next and Qwen3.6 forms, `--layout-only`, v3/v4 tables, refusals, profile copy |
| [tools/test_mtp_fetch.py](../../../tools/test_mtp_fetch.py) | 201 | 8 | gguf-py path | `mtp_fetch.py` range reads and sha256 checks (mocked HTTP) |
| [tools/test_qwen36_oracle.py](../../../tools/test_qwen36_oracle.py) | 419 | 24 | numpy, gguf-py | `ref/qwen36.py` and `qwen36_synth_gguf.py` invariants; every "rival" reading must differ by > 5% |

### Fixture generators run by CTest

| File | Lines | CTest entry | Writes |
| --- | --- | --- | --- |
| [cmake/qwen36_synth_fixtures.cmake](../../../cmake/qwen36_synth_fixtures.cmake) | 27 | `qwen36_synth_fixtures` | toy `qwen35moe` GGUFs via `tools/qwen36_synth_gguf.py` |
| [cmake/qwen36_block_fixtures.cmake](../../../cmake/qwen36_block_fixtures.cmake) | 28 | `qwen36_block_fixtures` | oracle dumps (`ref/qwen36.py`) for 8 tokens |
| [cmake/qwen36_oracle_dumps.cmake](../../../cmake/qwen36_oracle_dumps.cmake) | 35 | `qwen36_oracle_dumps` | `k.gguf` and F32/Q8 dumps for `qwen36_seam_parity` |
| [cmake/qwen36_attn_fixtures.cmake](../../../cmake/qwen36_attn_fixtures.cmake) | 39 | `qwen36_attn_fixtures` | g8 / g12 / g2 attention fixtures for `qsa_dense_parity` |
| [tests/python/qwen36_prefill_fixture.py](../../../tests/python/qwen36_prefill_fixture.py) | 186 | `prefill_qwen36_fixtures` | `gdn3` / `full4` GGUFs at real widths and oracle dumps |
| [tests/python/qwen36_mtp_fixture.py](../../../tests/python/qwen36_mtp_fixture.py) | 158 | `qwen36_mtp_fixtures` | `full4` trunk + nextn block, dumps, and the draft pack |
| [tests/python/qwen36_runner_fixture.py](../../../tests/python/qwen36_runner_fixture.py) | 109 | `qwen36_runner_fixtures` | a synthetic native pack (through `tools/iq_pack.py`) and o8/o11/o40 dumps |
| [tests/python/qwen36_runner_check.py](../../../tests/python/qwen36_runner_check.py) | 186 | `qwen36_runner_check` | (a check) teacher-forced `strata` runs vs the oracle, with a 4 KB mutation gate |
| [tools/iq_fixture.py](../../../tools/iq_fixture.py) | 137 | `iq_parity_fixtures` | deterministic i-quant blocks + references for `iq_parity` |

### Oracle and cross-check tools (by hand, documented in `docs/qwen36-port/oracle-crosscheck.md`)

| File | Lines | Role |
| --- | --- | --- |
| [ref/qwen36.py](../../../ref/qwen36.py) | 683 | the numpy oracle for `qwen35moe`: teacher-forced forward pass, cb()-named dumps, `--mtp`, `--act-quant q8` |
| [ref/load.py](../../../ref/load.py) | 217 | GGUF -> FP32 with lazy per-expert dequantization (Q2_0 by its own decoder) |
| [tools/qwen36_synth_gguf.py](../../../tools/qwen36_synth_gguf.py) | 373 | the toy `qwen35moe` GGUF every fixture starts from |
| [tools/qwen36_crosscheck.sh](../../../tools/qwen36_crosscheck.sh) | 82 | builds the submodule's llama.cpp (CPU) and compares it with the oracle on the synthetic cases |
| [tools/qwen36_dump_llama.cpp](../../../tools/qwen36_dump_llama.cpp) | 171 | dumps every cb()-named llama.cpp tensor in full |
| [tools/qwen36_compare_dumps.py](../../../tools/qwen36_compare_dumps.py) | 85 | seam-by-seam comparison of two dump directories |
| [tools/qwen36_check_eval_callback.py](../../../tools/qwen36_check_eval_callback.py) | 130 | `llama-eval-callback` printed stats vs the oracle |
| [tools/qwen36_check_dequant.py](../../../tools/qwen36_check_dequant.py) | 55 | `ref/load.py` dequantization vs ggml's `to_float` (uses `qwen36_dequant_ggml.cpp`, 30 lines) |
| [tools/qwen36_spread.py](../../../tools/qwen36_spread.py) | 37 | per-seam spread oracle vs llama.cpp on a real file, for tolerances |
| [tools/qwen36_router_bf16_flips.py](../../../tools/qwen36_router_bf16_flips.py) | 111 | how often BF16 rounding of the F32 router flips a top-8 choice |
| [tools/qwen36_teacher_llama.cpp](../../../tools/qwen36_teacher_llama.cpp) | 199 | llama.cpp greedy continuation with per-position logits: the real-file reference |
| [tools/qwen36_compare_logits.py](../../../tools/qwen36_compare_logits.py) | 155 | engine `--dump-logits` vs a reference: top-1 agreement and KL per position |

## 6. Registered in CMake but absent from this tree

`CMakeLists.txt` and `cmake/hip_backend.cmake` still name upstream sources that were not imported: twelve
`tests/hip/*.cpp` (HIP backend only, each behind an `EXISTS` check or a HIP option this fork does not build) and four
`bench/micro/*` programs (`native_mmvq_multi.cpp`, `graph_registry_test.cu`, `pinned_capture.cu`,
`hit_cpu_order_parity.cu`, each behind `EXISTS`). The llama.cpp-oracle parity loop (`CMakeLists.txt:423-443`,
`native_*_parity` from `bench/micro/`) runs only with `STRATA_ORACLE_SOURCE_DIR` and `STRATA_ORACLE_BUILD_DIR` set and
would fail for the same reason. `add_subdirectory(tests)` is skipped because `tests/CMakeLists.txt` does not exist
(`CMakeLists.txt:1408-1414`). `docs/qwen36-port/upstream-tooling.md` sections 2 and 7 explain what was restored.

## Gotchas

- `ctest` in a `make build` directory is not meaningful (see "Registration is not uniform").
- `expert_parity` and `ple_parity` require real Flash-Next artifacts at fixed paths (`pack/full/experts.bin`,
  `bench/micro/ple_*.bin`) and fail, by design, when they are missing; neither exists in this checkout.
- Without Python + numpy at configure time the Qwen3.6 harnesses silently drop out of the CTest list (only a
  configure-time status line says so); check `ctest -N` for `qwen36_*` before trusting a green run.
- `make test-chat-parser` links the static libraries of a previous build; run `make build` first with the same `BUILD_DIR`.
- `cmake/qwen36_synth_fixtures.cmake` sets `CMP0007 NEW`: without it cmake < 4 dropped an empty list element and the
  fixture failed (`docs/qwen36-port/step2-notes.md`, step3-block.md).

## See also

- [../build/README.md](../build/README.md) - every CMake option and the Makefile.
- [../kernels/parity/README.md](../kernels/parity/README.md) - each parity program.
- [../frontend/tests.md](../frontend/tests.md) - the three front-end targets in detail.
- [data-files.md](data-files.md) - the `tests/data/native_experts` fixtures.
- [project-notes.md](project-notes.md) - where each step's test results are recorded.
- [docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md) - the oracle and tolerance policy.
- [docs/qwen36-port/pascal-build.md](../../qwen36-port/pascal-build.md) section 3 - ctest on the GTX 1080 Ti.
