[Codebase docs](../README.md) › [Build](README.md) › Test targets

# Test, parity and bench targets

## Purpose

Every test-like executable and every `add_test` registration in [`CMakeLists.txt`](../../../CMakeLists.txt),
grouped by the condition that creates it, with the files this fork does not have. None of these are built by
`make build` (which builds `--target unbiased-strata` only), but many are part of the default `all` target in
any GPU build because they are not gated by `STRATA_BUILD_TESTS`.

Exit-code conventions used by the registrations: `SKIP_RETURN_CODE 77` (no suitable GPU / missing real-model
input) and `SKIP_RETURN_CODE 3` (missing numpy or gguf-py for a fixture). Most parity programs take
`--selftest` (synthetic data, no model) and many accept `--bench`.

## Files referenced by CMake but absent from this fork

Checked by listing every `src/`, `tests/`, `bench/`, `tools/` path in `CMakeLists.txt` and `cmake/*.cmake`
against the tree. All `src/` paths exist; these do not:

| Missing path | Referenced at | Guarded by `EXISTS`? | Consequence when its branch is taken |
| --- | --- | --- | --- |
| `tests/CMakeLists.txt` | `:89`, `:1411` | yes | `STRATA_BUILD_TESTS` defaults OFF; the model-level test directory is skipped |
| `bench/micro/` (directory) | `:89`, `:937` | default-only | `STRATA_BUILD_TESTS` defaults OFF; `ple_parity`'s default fixture dir does not exist |
| `bench/micro/<name>_parity.cpp` (16 files) | `:424-425` | no | `-DSTRATA_ORACLE_SOURCE_DIR=... -DSTRATA_ORACLE_BUILD_DIR=...` fails at generate |
| `bench/micro/native_mmvq_multi.cpp` | `:978` | yes | target silently absent |
| `bench/micro/graph_registry_test.cu` | `:1014` | yes | target silently absent |
| `bench/micro/pinned_capture.cu` | `:1026` | yes | target silently absent |
| `bench/micro/hit_cpu_order_parity.cu` | `:1358` | yes | target silently absent |
| `tests/hip/prefill_gemm.cpp`, `q2_zero.cpp`, `expert_cache_staging.cpp`, `router_fast.cpp` | `:583`, `:587`, `:591`, `:629` | no | **`STRATA_ENABLE_HIP=ON` cannot generate** |
| `tests/hip/router_fast.cpp` | `:864` | no | **`STRATA_HIP_GFX906=ON` cannot generate** |
| `tests/hip/prefill_mmq_parity.cpp` | `:1297` | no | `STRATA_PREFILL_MMQ=ON` cannot generate |
| `tests/hip/ple_iq4.cpp` | `:1306` | no | `STRATA_ENABLE_HIP` + native experts cannot generate |
| `tests/hip/intrinsics.cpp`, `handoff.cpp`, `mapped_alias.cpp`, `native_qsa_score.cpp`, `prefill_native_batch.cpp`, `prefill_hipblaslt_gemm.cpp` | `:595-646`, `hip_backend.cmake:104` | yes | targets silently absent |
| `tools/hip/tune_hipblaslt.cpp` | `:572` | no | HIP with hipBLASLt found cannot generate |
| `sycl/` | `:20` | no | `STRATA_ENABLE_SYCL=ON` fails |
| `tools/vision/*.cu` | `:1419` (glob) | glob | nothing (empty glob) |
| `pack/full/experts.bin` | `:311` | yes | `expert_arena_load` test not registered (a status message says so) |
| `tools/gpu_session.py`, `tools/ple_fp8_pack.py` | comments `:1375`, `:1352` | n/a | comments only |

"Cannot generate" is from CMake's rule that `add_executable` with a source file that does not exist is an
error ("Cannot find source file"); the AMD configurations were not run here (no ROCm in this environment).

## CPU-only tests

| Executable | Sources | Links | Gate | CTest name(s) |
| --- | --- | --- | --- | --- |
| `conversation_cache_test` | `src/core/conversation_cache_test.cpp` | include only | `STRATA_BUILD_CONVERSATION_TESTS` (`:96-104`) | same |
| `conversation_memory_test` | `src/core/conversation_memory_test.cpp`, `src/core/conversation_memory.cpp` | include only | same | same |
| `gguf_reader_test` | `src/artifact/gguf_reader_test.cpp` | `strata_artifact` | `STRATA_BUILD_TESTS` (`:225-229`) | same |
| `gguf_split_test` | `tests/core/gguf_split_test.cpp` | `strata_artifact` | `STRATA_BUILD_TESTS` (`:231-233`) | same |
| `model_geometry_test` | `tests/core/model_geometry_test.cpp` | `strata_artifact` | `STRATA_BUILD_TESTS`; test only if a python3 with numpy is found (`:237-260`) | `qwen36_synth_fixtures` (setup), `model_geometry_test` (skip 3) |
| `vram_plan_test` | `tests/core/vram_plan_test.cpp` | `strata_plan` | `STRATA_BUILD_TESTS` (`:272-277`) | same |
| `suffix_drafter_test`, `controller_test`, `draft_policy_test` | `src/spec/*_test.cpp` | `strata_spec` | `STRATA_BUILD_TESTS` (`:1054-1064`) | same |
| `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test` | `src/program/conv_cache_test.cpp`, `src/core/coupled_draft_test.cpp`, `src/kernels/bf16_bits_test.cpp` | include only | `STRATA_BUILD_TESTS OR STRATA_BUILD_CONVERSATION_TESTS` (`:1065-1078`) | same |
| `router_dot_parity` | `src/kernels/cpu/router_dot_parity.cpp` | `strata_kernels_cpu` | executable: native experts, **ungated** (`:1195`); tests: `STRATA_BUILD_TESTS` | `router_dot_parity`, `router_dot_parity_avx` (`STRATA_FORCE_ISA=avx`) |
| `pool_affinity_test` | `src/kernels/cpu/pool_affinity_test.cpp` (Windows) or `pool_affinity_linux_test.cpp` (Linux) | `strata_kernels_cpu` | `STRATA_BUILD_TESTS` (`:1364-1372`) | same |
| `expert_multi_test` | `src/kernels/cpu/expert_multi_test.cpp` | `strata_kernels_cpu` | `STRATA_BUILD_TESTS` | same |
| `expert_parity` | `src/kernels/cpu/expert_parity.cpp` | `strata_kernels_cpu` | `STRATA_BUILD_TESTS`; working dir = source tree | same |
| `pool_test`, `pool_stress` | `src/kernels/cpu/pool_test.cpp`, `pool_stress.cpp` | `strata_kernels_cpu` | `STRATA_BUILD_TESTS` | `pool_test`, `pool_stress` (arg 20) |
| `direct_file_async_test` | `tests/platform/direct_file_async.cpp`, `src/platform/direct_file.cpp` | Threads | `STRATA_BUILD_TESTS`, Unix not Apple (`:1400-1405`) | same |

**Verified in this environment** (g++ 13.3, CMake 3.28, no GPU, empty submodule):
`cmake -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON` registers
19 tests; all targets build. `ctest`: 16 pass; `qwen36_synth_fixtures` fails ("llama.cpp's gguf-py was not
found", because the submodule is not checked out, and the script's skip detection does not match that message,
see [test-fixtures-cmake.md](test-fixtures-cmake.md#gotchas)); `model_geometry_test` is then Not Run;
`expert_multi_test` fails because this host CPU lacks AVX512-VBMI (the canonical Q2_0 CPU expert kernel needs
AVX512-VNNI + VBMI; the same failure is recorded on two VMs in
[pascal-build.md](../../qwen36-port/pascal-build.md)).

## GPU parity programs not gated by `STRATA_BUILD_TESTS`

These are declared in every CUDA (and HIP) configuration, so `cmake --build build` builds them and `ctest`
lists them; `make build` does not build them. Sources are `src/kernels/<name>.cpp` unless noted; each links
`strata_kernels` (or `strata_prefill`) and the GPU runtime.

| Executable | Line | Extra condition | CTest name(s) / arguments |
| --- | --- | --- | --- |
| `file_expert_source_test` (`tests/core/`) | `:456-462` | source file exists (it does) | same, timeout 60 |
| `expert_profile_save_test` (`tests/core/`) | `:464-470` | source file exists (it does) | same, timeout 60 |
| `dequant_s2_parity` | `:658-664` | | `--selftest` |
| `s2_gemv_parity` | `:666-669` | | `--selftest` |
| `iq_parity` | `:671-716` | test needs python3 + numpy (`_iq_python`) | `iq_parity_fixtures` (setup, `tools/iq_fixture.py`), `iq_parity` |
| `mmvq_multi_parity` | `:717-728` | CUDA | `mmvq_multi_parity`; on gfx906 also `mmvq_multi_parity_cuda_layout` (`STRATA_MMVQ_WAVE=0`) |
| `iq_multi_parity` | `:731-734` | | same |
| `gdn_rec_parity` (`src/prefill/*.cu`) | `:739-741` | CUDA, not gfx906 | same |
| `gemm_bf16_parity` (`src/prefill/*.cu`) | `:744-746` | CUDA, not gfx906 | same |
| `prefill_qwen36_test` (`tests/cuda/`) | `:751-763` | CUDA, not gfx906; tests need `_iq_python` | `prefill_qwen36_fixtures`, `prefill_qwen36_test` (gdn3), `prefill_qwen36_attn_test` (full4) |
| `qwen36_mtp_test` (`tests/cuda/`) | `:768-778` | CUDA, not gfx906; tests need `_iq_python` | `qwen36_mtp_fixtures`, `qwen36_mtp_test` |
| `native_grouped_parity` | `:782-785` | | same |
| `shared_expert_parity` | `:786-789` | | `--selftest` |
| `fused_gr_bench` | `:791-793` | | none (bench) |
| `gr_parity`, `gdn_parity` | `:794-802` | | `--selftest` |
| `qwen36_seam_parity` | `:806-816` | test gated on `_q36_python` (see gotcha) | `qwen36_oracle_dumps` (setup), `qwen36_seam_parity` |
| `s2_gemv_q8_parity` | `:818-821` | | `--selftest` |
| `sampler_parity` | `:823-832` | | `sampler_parity`, `sampler_parity_one_block`, `sampler_parity_old` (env `STRATA_OLD_SAMPLER`, `STRATA_SAMPLER_ONE_BLOCK`) |
| `decode_cluster_parity` | `:835-840` | CUDA | `--selftest` (skips below sm_90 per comment) |
| `rope_parity`, `quantize_act_parity` | `:842-850` | | `--selftest` |
| `route_window_parity` | `:853-856` | | same |
| `router_top10_parity` | `:858-861` | | `--selftest` |
| `s_gemv_parity`, `elementwise_parity`, `bf16_gemv_parity`, `s_gemv_q8k_parity`, `qsa_parity` | `:869-892` | | `--selftest` |
| `qsa_dense_parity` | `:898-911` | | with `_q36_python`: `qwen36_attn_fixtures` + `qsa_dense_parity --fixtures`; else plain (skip 77) |
| `qsa_attn_bench` | `:914-916` | | none (bench) |
| `qsa_topk_parity` | `:921-924` | | `--selftest` |
| `ple_parity` | `:934-941` | | `--selftest --in $STRATA_PLE_FIXTURE_DIR/ple_in.bin --out .../ple_out.bin`; default dir `bench/micro` is absent |
| `kv_q8_parity`, `kv_stream_parity`, `kv_hybrid_parity`, `kv_q4_parity`, `cvec_parity` | `:944-968` | | same |
| `s2_expert_grouped_parity` | `:972-975` | | `--selftest` |
| `platform_memory_test` (`src/platform/memory_test.cpp`) | `:986-988` | | same (needs `ulimit -l`, per pascal-build.md) |
| `pinned_shared_test` (`src/core/`) | `:990-994` | Linux | same |
| `ple_reader_test` (`src/ngram/`) | `:997-1000` | | `ple_reader_selftest` |
| (`strata-device`) | `:1005-1009` | | `cuda_device_selftest` or `hip_device_selftest` |
| `native_expert_parity`, `native_expert_bench` | `:1309-1315` | native experts | tests only with `STRATA_BUILD_TESTS AND CUDA` (below) |

That is 44 GPU executables (counting `native_expert_parity` / `native_expert_bench`) plus `router_dot_parity`
outside the GPU section.

## GPU tests gated by `STRATA_BUILD_TESTS`

| Executable | Line | Further condition | CTest name(s) |
| --- | --- | --- | --- |
| `qsa_prompt_attn_parity` | `:372-376` | CUDA; also built with `-DSTRATA_PARITY_PROMPT_ATTN=ON` | none |
| `qsa_topk_active_parity` | `:379-385` | CUDA | same (skip 77) |
| `qsa_select_bench` | `:389-394` | HIP; or `STRATA_PARITY_PROMPT_ATTN` | `qsa_select_bench 32768 256 3 262144` |
| `expert_layout_test`, `pool_native_widths_test`, `native_dense_ple_key_test`, `expert_cache_per_layer_test`, `expert_cache_segmented_test`, `pack_load_check` (`tests/core/`) | `:471-504` | native experts | same; `pack_load_check` needs `STRATA_QWEN36_GGUF` / `STRATA_QWEN36_PACK` (skip 77) |
| `qwen36_block_test` (`tests/cuda/`) | `:507-520` | CUDA + `_q36_python` | `qwen36_block_fixtures`, `qwen36_block_test` |
| `qwen36_verify_test` (`tests/cuda/`) | `:526-540` | CUDA + `_q36_python` | `qwen36_runner_fixtures` (`tests/python/qwen36_runner_fixture.py`), `qwen36_verify_test`, `qwen36_runner_check` (runs the `strata` binary) |
| `native_mmvq_multi` | `:978-983` | `bench/micro/native_mmvq_multi.cpp` exists (**absent**) | |
| `graph_registry_test` | `:1014-1019` | `bench/micro/graph_registry_test.cu` exists (**absent**) | |
| `pinned_capture` | `:1026-1031` | CUDA; `bench/micro/pinned_capture.cu` exists (**absent**) | |
| `prefill_fused_moe_test`, `prefill_fused_iq_test` (`tests/cuda/`) | `:1245-1256` | CUDA + native experts | same (skip 77 below sm_80) |
| `prefill_mmq_kquant_test` (`tests/cuda/`) | `:1257-1263` | + `STRATA_MMQ_KQUANTS` | same |
| (tests of `native_expert_parity`) | `:1316-1343` | CUDA | 7 `native_expert_parity_<pair>` at 2560x640, 5 `native_expert_parity_qwen36_<pair>` at `--geom 2048x512`, `_q5_1_min`, `_bf16_embd`, and `_refuses_q6_K_down_at_640` (`WILL_FAIL`) |
| `ple_q5_parity`, `q5_projection_parity`, `ple_fp8_parity` | `:1346-1355` | CUDA | none (real-weight checks run by hand) |
| `hit_cpu_order_parity` | `:1358-1361` | CUDA; `bench/micro/...` exists (**absent**) | |
| `dequant_bf16_test` | `:1376-1379` | GPU | none (needs the 38 GB model) |

## Conversation tests on the GPU side (`STRATA_BUILD_CONVERSATION_TESTS`, `:541-559`)

`conversation_snapshot_test` (skip 77), `conversation_validation_test` (`CUDA_VISIBLE_DEVICES=-1`), and on
Linux with GNU/Clang and not HIP, `conversation_transfer_test`: the validation test source compiled with
`CONVERSATION_TEST_TRANSFERS` and linked with `-Wl,--wrap=` for `cudaMemcpy`, `cudaDeviceSynchronize` and
`cudaGetLastError`.

## AMD-only targets

| Executable | Line | Source exists? |
| --- | --- | --- |
| `hip_prefill_gemm`, `hip_q2_zero`, `hip_expert_cache_staging`, `hip_router_fast` | `:583-632` | no (unguarded) |
| `hip_intrinsics`, `hip_handoff`, `hip_mapped_alias`, `hip_native_qsa_score`, `hip_prefill_native_batch`, `hip_prefill_hipblaslt_gemm` | `:595-646` | no (guarded, skipped) |
| `hip_prompt_attn_wmma` (`src/kernels/qsa_prompt_attn_parity.cpp`) | `:634-638` | yes |
| `hip_prefill_mmq_parity` | `:1297-1302` | no (unguarded) |
| `hip_ple_iq4` | `:1306-1307` | no (unguarded) |
| `hip_router_fast` (gfx906 variant) | `:864-866` | no (unguarded) |

All HIP tests get a 60 s timeout (`:647-649`), 120 s for the two prefill ones.

## Oracle parity programs (`:420-443`)

With CUDA and both `STRATA_ORACLE_SOURCE_DIR` and `STRATA_ORACLE_BUILD_DIR` set: `native_mmvq`, `native_bf16`,
`native_shared`, `native_gr_norm`, `native_moe`, `native_gdn`, `native_gr_postops`, `native_router`,
`native_gdn_preprocess`, `native_gdn_pipeline`, `native_qsa`, `native_rope`, `native_ple_postops`,
`native_flash_attn`, `native_qsa_indexer`, `native_qsa_score` -> `<name>_parity` from `bench/micro/`, each
linked against a separate shared CUDA ggml build. All 16 sources are absent in this fork.

## CTest in a `make build` tree

`enable_testing()` is called whenever `tests/core/file_expert_source_test.cpp` exists (`:456-457`), which it
does, so a `make build` tree (CUDA, `STRATA_BUILD_TESTS=OFF`) still has a CTest file listing every ungated
registration above. Only `unbiased-strata` was built, so `ctest` there reports those tests as Not Run. Use a
separate configure with `-DSTRATA_BUILD_TESTS=ON` and a full `cmake --build` (the line in
[step2-notes.md](../../qwen36-port/step2-notes.md) section 4) to run them.

## Gotchas

- **`_q36_python` is undefined unless `STRATA_BUILD_TESTS` is ON** (it is set at `:240`, inside the tests
  block). At `:809` and `:901` the test is `if(NOT _q36_python STREQUAL "")`; with the variable undefined CMake
  compares the literal string `_q36_python`, so the condition is true (checked with a CMake script). In a GPU
  build without tests, `qwen36_oracle_dumps` and `qwen36_attn_fixtures` are therefore registered with an
  empty `-DPYTHON=`, and their dependent tests (`qwen36_seam_parity`, `qsa_dense_parity --fixtures`) depend on
  a fixture step that cannot work. The checks at `:507` and `:526` are safe because they also require
  `STRATA_BUILD_TESTS`.
- Several tests need real artifacts or host settings rather than synthetic data: `ple_parity` (PLE fixtures;
  pascal-build.md records it failing on a hard-coded Flash-Next shard path), `platform_memory_test`
  (`ulimit -l`), `expert_multi_test` (AVX-512 VNNI + VBMI), `expert_arena_load` (`pack/full/experts.bin`).
- [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) predates the restored
  `tests/core`, `tests/cuda`, `tests/platform` and `tests/python` files; at the time of writing all
  `tests/` sources the root CMake file names exist except `tests/CMakeLists.txt` and `tests/hip/*`.

## See also

- [test-fixtures-cmake.md](test-fixtures-cmake.md) (the fixture scripts and Python probes)
- [targets.md](targets.md), [cmake-options.md](cmake-options.md)
- [docs/qwen36-port/pascal-build.md](../../qwen36-port/pascal-build.md) section 3 (a full ctest run on sm_61 and its failures)
- docs/qwen36-port step pages that introduced tests: [step3-attn.md](../../qwen36-port/step3-attn.md),
  [step3-block.md](../../qwen36-port/step3-block.md), [step3-prefill.md](../../qwen36-port/step3-prefill.md),
  [step3-experts.md](../../qwen36-port/step3-experts.md), [step4-pack.md](../../qwen36-port/step4-pack.md),
  [step4-runner.md](../../qwen36-port/step4-runner.md), [step4-drafter.md](../../qwen36-port/step4-drafter.md)
