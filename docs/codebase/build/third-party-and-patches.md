[Codebase docs](../README.md) › [Build](README.md) › Third party and patches

# Third-party code, the llama.cpp patch, and repository metadata

## Purpose

What the repository takes from outside, how the build wires it in, and the one local modification to it.

Files covered: [.gitmodules](../../../.gitmodules), `third_party/llama.cpp` (submodule),
[third_party/ggml/ggml-common.h](../../../third_party/ggml/ggml-common.h),
[third_party/ggml/VERSION.txt](../../../third_party/ggml/VERSION.txt),
[third_party/ggml/LICENSE](../../../third_party/ggml/LICENSE),
[patches/llama.cpp-qwen3-coder-any-argument-order.patch](../../../patches/llama.cpp-qwen3-coder-any-argument-order.patch),
[.gitattributes](../../../.gitattributes), [.gitignore](../../../.gitignore).

## The llama.cpp submodule

| Item | Value |
| --- | --- |
| Declared in | `.gitmodules`: path `third_party/llama.cpp`, url `https://github.com/ggml-org/llama.cpp.git`, `shallow = true` |
| Pinned commit (gitlink) | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`git ls-files -s third_party/llama.cpp`); the same commit upstream Strata pins (UPSTREAM.md) |
| State in a fresh clone | empty directory; `git submodule update --init` fills it (about 170 MB checked out, per README) |
| Fetched by the build? | no. `make build` and CMake both stop with a message if it is empty (`Makefile:26-27`, `CMakeLists.txt:1147-1149`) |
| Override | `-DSTRATA_GGML_DIR=<another llama.cpp checkout>` (the patch is then not applied) |

What the build takes from it:

| Use | Where |
| --- | --- |
| ggml (static, CPU backend only: `GGML_CUDA OFF`) as `ggml`, `ggml-base`, `ggml-cpu`; binary dir `build/ggml` | `add_subdirectory(${STRATA_GGML_DIR}/ggml ... EXCLUDE_FROM_ALL)`, `CMakeLists.txt:1167` |
| ggml-cuda source files compiled into Strata's own `strata_mmq`: `quantize.cu` and `template-instances/mmq-instance-<type>.cu`, plus the headers in `ggml/include`, `ggml/src`, `ggml/src/ggml-cuda` | `CMakeLists.txt:1210-1231` (CUDA), `:1267-1287` (HIP) |
| llama.cpp itself (`llama`, `llama-common`, `mtmd`, `cpp-httplib`); binary dir `build/llama.cpp` | `add_subdirectory(${STRATA_GGML_DIR} ... EXCLUDE_FROM_ALL)`, `CMakeLists.txt:1438` |
| `vendor/` headers for the front end | `target_include_directories(unbiased-strata ... ${STRATA_GGML_DIR}/vendor)`, `:1441` |
| llama.cpp headers and static archives for `make test-chat-parser` | `Makefile:54-61` |
| `gguf-py` for the test fixtures | `STRATA_GGUF_PY` in `CMakeLists.txt:533`, `:758`, `:775`; default lookup in `tools/_paths.py` |

## Vendored ggml header: `third_party/ggml/`

| File | Content |
| --- | --- |
| `ggml-common.h` (1911 lines) | ggml's block layouts and lookup tables: `QK_K = 256`; structs `block_q1_0`, `block_q2_0`, `block_q4_0/1`, `block_mxfp4`, `block_nvfp4`, `block_q5_0/1`, `block_q8_0/1`, `block_tq1_0/2_0`, `block_q2_K` ... `block_q8_K`, `block_iq2_xxs/xs/s`, `block_iq3_xxs/s`, `block_iq1_s/m`, `block_iq4_nl/xs`; tables `kmask_iq2xs`, `ksigns_iq2xs`, `ksigns64`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `kvalues_iq4nl`, `kvalues_fp4`, `iq1s_grid`, `iq1s_grid_gpu`. The including file picks the language with `GGML_COMMON_DECL_{C,CPP,METAL,CUDA,HIP,SYCL}` (types, `:1-72`) and `GGML_COMMON_IMPL_*` (tables, `:467-505`) |
| `VERSION.txt` | `3cf03257f219afbe7334045ff7c6a06ac68c627d Sun Sep 20 16:08:11 2026 +0800`: the same commit as the submodule |
| `LICENSE` | MIT, "Copyright (c) 2023-2026 The ggml authors" |

Wiring: `third_party/ggml` is a PRIVATE include directory of `strata_kernels` (`CMakeLists.txt:396`, comment
"plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats") and of
`strata_kernels_cpu` (`:1173`). The files that include it:

| File | Mode |
| --- | --- |
| `src/kernels/cuda/iq_kernels.cu` | `GGML_COMMON_DECL_CUDA` + `IMPL_CUDA` |
| `src/kernels/cpu/iq_avx512.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp` | `GGML_COMMON_DECL_CPP` + `IMPL_CPP` |
| `src/prefill/moe_fused_iq.cu` | `DECL_CUDA` + `IMPL_CUDA`, but it is a `strata_mmq` source, whose include path has the submodule's `ggml/src` and not `third_party/ggml`, so it gets the submodule's copy |

Whether the vendored copy is byte-identical to the submodule's `ggml/src/ggml-common.h` at the pinned commit
was not checked (the submodule is not checked out here).

## The llama.cpp patch

`patches/llama.cpp-qwen3-coder-any-argument-order.patch` (26 lines) changes one hunk of
`common/parsers/qwen3-coder.cpp` in `common_chat_params_init_qwen3_coder`:

- before: `p.permute("tool-<name>-args", required_args)` followed by `zero_or_more(choice(optional_args))`,
  i.e. required arguments first (in any order), optional ones after;
- after: `required_args` and `optional_args` concatenated into `all_args`, and
  `zero_or_more(choice(all_args))` (or `eps()` when there are none), i.e. any order, interleaved.

Reason (patch comment and UPSTREAM.md): Qwen3.8 Flash-Next often writes the optional arguments first (`limit`,
`offset`, `path`); because the tool-call group is optional and the grammar has no end anchor, such a call was
dropped silently and the client got `finish_reason=stop` with an empty answer. Qwen3.6's template routes to
the same parser. Consequence: the grammar no longer enforces that required arguments are present; per the
patch comment that is left to the tool to report.

### How it is applied

| By | Command | Idempotence check | Line |
| --- | --- | --- | --- |
| `make build` | `git -C third_party/llama.cpp apply ../../patches/...` | `git apply --check --reverse` succeeds -> already applied, skip | `Makefile:28-29` |
| CMake configure | `patch -p1 -N -s -i <patch>` in `${STRATA_GGML_DIR}` | `patch -p1 -N -R --dry-run -s -i` returns 0 -> skip; a failed apply is `FATAL_ERROR` | `CMakeLists.txt:1150-1166` |

The CMake step runs only when `STRATA_GGML_DIR` is the submodule and the patch file exists (`:1155`). It was
added so that trees configured with CMake directly (VMs, agent worktrees, a copied submodule without `.git`)
also get it (comment `:1150-1153`, REGINA-104). Both steps modify the submodule's working tree, so the
submodule shows as modified in `git status` after a build.

### Test

`make test-chat-parser MODEL=<shard 1>` builds `frontend/chat_parser_test.cpp`, which parses a tool call
whose arguments come in the orders `limit offset path`, `offset path limit`, `path limit offset` and `path`
(`frontend/chat_parser_test.cpp:47-67`). See [makefile.md](makefile.md).

## `.gitattributes` and `.gitignore`

| File | Rule | Meaning |
| --- | --- | --- |
| `.gitattributes` | `*.sh text eol=lf` | Shell scripts keep LF endings on any checkout platform (e.g. `tools/qwen36_crosscheck.sh`) |
| `.gitignore` | `/build*/` | Every top-level build directory (`build`, `build-make`, `build-mac`, ...) |
| `.gitignore` | `*.log` | Log files anywhere |

Nothing ignores Python bytecode: running the fixture tools from a build leaves an untracked
`tools/__pycache__/`.

## Gotchas

- The CMake patch step needs the `patch` program on `PATH` whenever the submodule is used: if it is missing,
  both `execute_process` calls fail and configure stops with "could not apply ..." even when `make build`
  already applied the patch with `git apply` (inferred from the code at `:1156-1163`; not run).
- `CMakeLists.txt:1101` still says ggml is fetched from "the pinned upstream commit (third_party/ggml/VERSION.txt)"
  when `STRATA_GGML_DIR` is empty; this fork uses the submodule and fetches nothing (`:1143-1146`).
- `tools/_paths.py`'s error text tells the user to "run setup.py"; `setup.py` was removed in this fork, the
  remedy is `git submodule update --init` or `STRATA_GGUF_PY`.

## See also

- [UPSTREAM.md](../../../UPSTREAM.md) (sources, pinned commits, the patch's rationale)
- [targets.md](targets.md) (where the llama.cpp libraries are linked), [cmake-options.md](cmake-options.md)
  (forced `GGML_*` / `LLAMA_*` settings)
- [Front end](../frontend/README.md) (the code that uses llama.cpp's chat templates and parsers)
- [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md)
