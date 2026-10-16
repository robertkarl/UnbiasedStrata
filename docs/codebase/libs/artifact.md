[Codebase docs](../README.md) › [Libraries](README.md) › GGUF artifact reader

# GGUF reader, split shards, scalar dequantizers (`strata_artifact`)

## Purpose

`strata_artifact` is a header-only, GPU-free CMake `INTERFACE` target (`CMakeLists.txt:210-216`). It is how the engine
reads model files without linking ggml:

- **`gguf_reader.hpp`**: memory-maps a GGUF v3 file, parses header, metadata (all value types, arrays sampled) and
  tensor directory; joins split-model shards (`GgufModel`); guards the architecture.
- **`gguf_split.hpp`**: resolves `<name>-0000K-of-0000N.gguf` into the full shard list and refuses a missing shard.
- **`dequant.hpp`**: scalar reference dequantizers (Q2_0, Q8_0, Q4_0, Q5_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS,
  F32, F16, BF16) transcribed from ggml. They are the validated references for the GPU kernels' parity tests.
- Two small CLIs, **`strata-gguf`** and **`strata-dequant`**, exercise them on a real file.

The CMake comment explains the header-only choice: the reader and dequantizers hold no state worth hiding, and a
header-only boundary cannot add duplicate-symbol or missing-declaration bugs to code already validated against the real
artifact. `gguf_reader.hpp`'s own header says it was generated from `src/artifact/gguf_reader.cpp` by
`scripts/split_artifact.py` (not in this checkout); the Python reference is
[tools/gguf_reader.py](../../../tools/gguf_reader.py).

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/artifact/gguf_reader.hpp](../../../include/strata/artifact/gguf_reader.hpp) | 627 | `ggml_type_name`, `block_geometry`, `TensorInfo`, `Cursor`, `MetaType`/`MetaValue`, `GgufFile`, `tensor_payload_bytes`, `GgufModel`, `Qwen4ExpGuard`, `check_architecture` |
| [include/strata/artifact/gguf_split.hpp](../../../include/strata/artifact/gguf_split.hpp) | 52 | `gguf_split_paths` (no `<windows.h>`, so any TU can use it) |
| [include/strata/artifact/dequant.hpp](../../../include/strata/artifact/dequant.hpp) | 292 | fp16/bf16 conversion and per-block dequantizers |
| [src/artifact/gguf_reader.cpp](../../../src/artifact/gguf_reader.cpp) | 67 | `strata-gguf` CLI |
| [src/artifact/dequant.cpp](../../../src/artifact/dequant.cpp) | 123 | `strata-dequant` CLI (compiled with `STRATA_DEQUANT_SELFTEST`) |
| [src/artifact/gguf_reader_test.cpp](../../../src/artifact/gguf_reader_test.cpp) | 93 | Duplicate-tensor-name refusal on a synthetic GGUF |
| [tests/core/gguf_split_test.cpp](../../../tests/core/gguf_split_test.cpp) | 220 | Split-model tests on synthetic shards |

## GGUF reader (`gguf_reader.hpp`, namespace `strata`)

### Type tables

- `ggml_type_name(t)` (`include/strata/artifact/gguf_reader.hpp:53`): names for ggml type ids, including the
  engine-specific `42 = Q2_0` (PrismML ternary 2-bit), `41 = Q1_0`, `40 = NVFP4`, `39 = MXFP4`. Unknown ids give
  `"?"`.
- `block_geometry(t, elems, bytes)` (`:122`): elements and bytes per block for the types the engine reads (F32, F16,
  BF16, Q4_0/1, Q5_0/1, Q8_0/1, Q2_K..Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, I8, Q2_0).
  Q2_0 is 64 elements / 18 bytes; the comment says that was proven from the artifact's own offset brackets (the
  llama.cpp PR that introduced the type used 128). I8 (1/1) carries the FP8 PLE table written by
  `tools/ple_fp8_pack.py` (cited in comments; not in this checkout). Note that `ggml_type_name` has no entry for 29 (IQ1_M) although `block_geometry` does.

### `GgufFile` (`:340`)

| Member | What it does |
|---|---|
| `GgufFile(path)` | Opens and maps the whole file read-only (`mmap(MAP_PRIVATE)` on POSIX, `CreateFileMapping`/`MapViewOfFile` on Windows) and parses it; throws `std::runtime_error` on any problem, unmapping first. Non-copyable. |
| `parse()` (`:421`) | Magic `0x46554747` ("GGUF"), version must be 3, then `n_kv` metadata entries via `read_value`, then `n_tensors` directory entries (name, 1..4 dims, type, offset). **Duplicate tensor names are refused** naming the tensor and file (`:437-444`). `data_start` = end of directory rounded up to `general.alignment` (default 32); must not exceed the file size. |
| `tensors()`, `metadata()`, `get(key)`, `find(name)` | Accessors; `find` is a linear first-match scan. |
| `tensor_data(t)` | Pointer into the mapping: `base + data_start + t.offset`. Not bounds-checked; see `GgufModel::in_bounds`. |
| `count_type(name)`, `version()`, `data_start()`, `file_size()`, `alignment()`, `path()` | Accessors. |

`Cursor` (`:237`) bounds-checks every header read so a truncated file throws "unexpected end of file in header"
instead of crashing. `read_value` (`:279`) handles every `MetaType`; arrays deeper than 2 levels or longer than 2^24
elements are refused, and **only the first 64 items of an array are kept** (`count` holds the true length). Signed
integers are stored sign-extended in `MetaValue::u`; floats in `MetaValue::f`.

`tensor_payload_bytes(t)` (`:477`): bytes of a tensor from shape and block geometry, 0 if the type is unknown, the
first dimension is not whole blocks, or the size overflows.

### `GgufModel`: split shards (`:501`)

Constructed from the shard paths (or `GgufModel::open(any_shard)`, which calls `gguf_split_paths`). Ported from
eddoursul/Strata 8029fa9 with the split-key validation of #255. Facts the code relies on:

- Shard 1 carries the model metadata (`general.architecture` and the model keys). `meta()` returns it.
- Later shards only declare `split.count`, `split.no`, `split.tensors.count`. In Unsloth's UD-Q4_K_XL, shard 1 holds
  no tensors at all and `output.weight`, `token_embd.weight` and the PLE table live in shard 2.

`validate_split()` (`:539`) refuses: shard 1 opened alone when it says `split.count > 1`; a multi-shard model whose
first shard lacks `general.architecture`; any shard whose `split.count`/`split.no`/`split.tensors.count` disagree
with its position or with shard 1; and a total tensor count that differs from `split.tensors.count`. The constructor
also refuses a tensor name present in two shards. `find(name, &shard)` uses a map index; `in_bounds(t, s)` checks the
payload lies inside that shard's data section (used to catch truncated downloads).

### `gguf_split_paths(any)` (`include/strata/artifact/gguf_split.hpp:23`)

If `any` ends in `-NNNNN-of-MMMMM.gguf`, returns all `MMMMM` paths in order (rewriting only the five shard digits, so
directories and symlinked Hugging Face snapshot names are kept). Otherwise returns `{any}`. Throws
"missing model shard ... is the download complete?" when one does not exist. Upstream silently skipped missing
shards; this fork made it an error (header comment `:4-7`).

### Architecture guard (`:573-625`)

`check_architecture(GgufFile, Qwen4ExpGuard = {})` returns an empty string when OK, else a reason:

| `general.architecture` | Check |
|---|---|
| `qwen4exp` (Qwen3.8-Flash-Next) | Six keys must exist and, where the guard's value is non-zero, match: `block_count = 48`, `embedding_length = 2560`, `attention.head_count = 24`, `attention.head_count_kv = 2`; `expert_count` and `expert_used_count` default to 0 = presence only (pruned variants such as the GSQ-RCO Coder ship fewer than 512 experts). |
| `qwen35moe` (Qwen3.6-35B-A3B) | 14 keys must be **present** (block count, embedding length, expert counts, head counts, key length, full-attention interval, SSM conv kernel / state size / group count / time-step rank / inner size, expert FFN length). Values are read later by `strata::core::read_model_geometry`. |
| anything else | Refused, naming both accepted architectures. |

Passing the guard means the metadata is readable, not that the engine supports the run.

## Dequantizers (`dequant.hpp`)

All are `inline` in namespace `strata`, take one block and write floats. The comments name the ggml source commit
(`3cf03257`) each one was transcribed from and call out the parts that are easy to get wrong.

| Function | Line | Block | Notes |
|---|---|---|---|
| `fp16_to_fp32`, `bf16_to_fp32`, `read_u16` | `:33`, `:57`, `:64` | | fp16 handles subnormals and inf/NaN. |
| `dequantize_q2_0` | `:69` | 64 el / 18 B | `fp16 d; uint8 qs[16]`; 2-bit codes LSB-first, code {0,1,2,3} maps to {-1,0,+1,+2}, value `(code - 1) * d`. |
| `dequantize_q8_0` | `:81` | 32 / 34 | |
| `dequantize_q4_0` | `:88` | 32 / 18 | low nibble = element j, high = j+16, minus 8. |
| `dequantize_q5_0` | `:99` | 32 / 22 | plus 32-bit high-bit plane; used by `src/kernels/ngram.cpp:167` for Q5_0 PLE rows. |
| `dequantize_iq4_nl` | `:119` | 32 / 18 | non-linear codebook `kvalues_iq4nl`. |
| `dequantize_q6_K` | `:133` | 256 / 210 | `d` is last. |
| `dequantize_q4_K` | `:172` | 256 / 144 | `get_scale_min_k4` 6-bit scale/min unpacking. |
| `dequantize_q5_K` | `:196` | 256 / 176 | 5th-bit mask shifts left by 2 per 64-group. |
| `dequantize_q3_K` | `:226` | 256 / 110 | signed 6-bit scales via `kmask` repacking; `m` not reset between halves (as in ggml). |
| `dequantize_iq4_xs` | `:264` | 256 / 136 | |
| `dequantize_f32`, `_f16`, `_bf16` | `:282-290` | element types | |

`dequant.hpp` defines `STRATA_GGUF_MAIN_DISABLED` before including the reader (`:27`); that macro only matters to
`src/artifact/gguf_reader.cpp`, whose `main` it guards.

Users: the GPU kernel parity tests (`src/kernels/s_gemv_parity.cpp`, `s2_gemv_parity.cpp`,
`dequant_s2_parity.cpp`) compare against these scalar paths; `strata_kernels` links `strata_artifact` for exactly
that reason (`CMakeLists.txt:319-321`).

## CLIs

### `strata-gguf` (`src/artifact/gguf_reader.cpp:7`)

`strata-gguf <file.gguf> [--check]`. Prints version, tensor and metadata counts, `data_start`, size. With `--check`:
counts `blk.*` vs global tensors and Q2_0 / BF16 / F32 tensors; runs a geometry bracket test (each tensor's computed
byte size must fit the gap to the next tensor's offset, with less than 32 bytes of padding); runs
`check_architecture`. Exit 1 if any tensor is out of bracket or on error. It opens a single `GgufFile`, not a
`GgufModel`.

### `strata-dequant` (`src/artifact/dequant.cpp:6`)

`strata-dequant <file.gguf>`. For up to 8 Q2_0 tensors, dequantizes up to 64 blocks each and asserts the structural
invariant that every value is one of `{-d, 0, d, 2d}` for its block's scale; prints a checksum of the first one.
Then, for every Q5_0 / IQ4_NL / Q6_K / Q4_K / Q5_K / Q3_K / IQ4_XS tensor in directory order until the first IQ4_XS,
prints a checksum over 64 blocks and the first 8 values, for comparison with gguf-py. Exit 1 on an invariant
violation.

## How the engine uses it

| Caller | Use |
|---|---|
| `src/program/generate.cpp:1848-1852`, `:2107`, `:2193` | `gguf_split_paths(--native)` then `GgufModel` over the shards. |
| `src/program/generate.cpp:1136`, `:1896`, `:2403` | Single `GgufFile` opens: control-vector files, the PLE file (to detect a PLE-only `strata-ple` table), the metadata shard's rope keys. |
| `include/strata/core/model_geometry.hpp:173` | `GgufModel::open` to read Qwen3.6 geometry. |
| `src/core/native_head.cpp:29`, `:113`; `src/core/native_dense.cpp:103` | `check_architecture`. |
| `src/core/expert_source.cpp:2527-2535` | One `GgufFile` per expert file. |
| `src/kernels/ngram.cpp:182` | `GgufFile` to find the PLE table's offset (then dropped for direct I/O). |

## Configuration

| Knob | Effect |
|---|---|
| `STRATA_GGUF_MAIN_DISABLED` | Compiles `gguf_reader.cpp` without `main`. |
| `STRATA_DEQUANT_SELFTEST` | Set on `strata-dequant`; without it `dequant.cpp` has no `main`. |
| `STRATA_BUILD_TESTS` | Registers `gguf_reader_test`, `gguf_split_test`, `model_geometry_test`. |

No runtime environment variables.

## Model-specific behaviour

The reader is generic GGUF v3. Model specifics are confined to `check_architecture` (value checks for `qwen4exp`,
presence checks for `qwen35moe`) and the type table (Q2_0 is the Flash-Next routed-expert format; Qwen3.6 files use
standard llama.cpp types). Split-shard handling matters for both (Unsloth's UD-Q4_K_XL Flash-Next, multi-file Qwen3.6
downloads).

## Tests

All need no GPU and no model, and are registered when `STRATA_BUILD_TESTS` is on (`CMakeLists.txt:225-261`):

| Test | Covers |
|---|---|
| `gguf_reader_test` (`src/artifact/gguf_reader_test.cpp`) | Synthetic GGUF v3: two distinct names open and are found; the same name twice is refused and the error names tensor and file. |
| `gguf_split_test` (`tests/core/gguf_split_test.cpp`) | UD-Q4_K_XL-shaped 4-shard model (metadata-only shard 1, a layer split across shards): path resolution from any shard, lookup and `in_bounds`, missing shard, tensor in two shards, wrong split keys, count mismatch, missing architecture, shard 1 opened alone, truncated shard. |
| `model_geometry_test` | Uses `GgufModel` on synthetic Qwen3.6 files; only registered when a python3 with numpy is found. Documented with the core geometry reader. |

`strata-gguf --check` and `strata-dequant` are manual checks on real files.

## Gotchas

- `GgufFile::find` is a linear scan; `GgufModel::find` is indexed. Hot paths should use the model or cache pointers.
- Metadata arrays keep only 64 items. Code that needs a full array (for example a tokenizer vocabulary) cannot get it
  from `MetaValue::items`.
- `tensor_data` is unchecked pointer arithmetic; call `GgufModel::in_bounds` for files that may be truncated.
- The `dequant.cpp` first-line comment says it dumps a tensor as F32; it does not, it runs the checks above. The
  `dequant.hpp` comment mentions a `--check` flag; the CLI takes only the file.
- The reader's comments point at `docs/q2_0-contract.md`, `scripts/build_artifact.bat` and
  `tools/verify_q2_0_geometry.py`, none of which are in this checkout.

## See also

- [../core/README.md](../core/README.md): model geometry, native dense / head loading, expert source
- [docs/qwen36-port/gguf-metadata.md](../../qwen36-port/gguf-metadata.md): the qwen35moe keys
- [ngram-ple.md](ngram-ple.md): how the PLE table's offset from `GgufFile` feeds the direct reader
- [Libraries overview](README.md)
