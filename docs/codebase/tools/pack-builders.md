[Codebase docs](../README.md) › [Tools](README.md) › Pack builders

# Pack builders and GGUF helpers

## Purpose

The Python programs that turn a model GGUF into the directory the engine loads with `--pack`, plus the small
libraries they share: a dependency-free GGUF reader and writer, the gguf-py locator, and the tokenizer exporter.
The file formats they write are specified in [pack-format.md](pack-format.md); this page covers how the tools
decide what to write.

| File | Lines | Role |
| --- | --- | --- |
| [`tools/iq_pack.py`](../../../tools/iq_pack.py) | 763 | **The** pack builder for both supported models (native experts, served from the GGUF) |
| [`tools/gguf_reader.py`](../../../tools/gguf_reader.py) | 142 | GGUF v3 header reader, no gguf-py; knows ggml type 42 (`Q2_0`) |
| [`tools/gguf_writer.py`](../../../tools/gguf_writer.py) | 234 | Minimal GGUF v3 writer with a `Q2_0` quantizer/dequantizer |
| [`tools/_paths.py`](../../../tools/_paths.py) | 28 | Locates llama.cpp's gguf-py (`STRATA_GGUF_PY`) |
| [`tools/strata_tokenizer.py`](../../../tools/strata_tokenizer.py) | 339 | Byte-level BPE reference tokenizer; exports `<pack>/tokenizer/` |
| [`tools/strata_pack.py`](../../../tools/strata_pack.py) | 423 | Legacy canonical Q2_0 pack: `build` / `verify` / `info` (not runnable here, see Gotchas) |
| [`tools/pack_layer.py`](../../../tools/pack_layer.py) | 123 | Legacy: canonical expert arena (1,382,400-byte Flash-Next Q2_0 blobs) |
| [`tools/pack_index.py`](../../../tools/pack_index.py) | 260 | Legacy: flattens a canonical `manifest.json` into `index.txt` |

## `tools/iq_pack.py` - the native pack

### Command line

```
python3 -I tools/iq_pack.py --gguf <shard1>.gguf --out <dir>
        [--expert-profile PROFILE.bin] [--experts-bin] [--skip-experts] [--layout-only]
        [--compat-bf16] [--base <canonical pack>]
```

| Flag | Effect (`main`, `tools/iq_pack.py:613-759`) |
| --- | --- |
| `--gguf` | Shard 1. Sibling shards `-0000N-of-0000M.gguf` must all exist. The path is taken with `.absolute()`, not `.resolve()`, so a Hugging Face snapshot symlink keeps its shard name (`:634-636`). |
| `--out` | Pack directory (created). |
| `--expert-profile P` | Copy `P` in as `expert-profile.bin` after checking its `STRP` header against the trunk's layers x experts. Default for `qwen35moe`: `data/qwen36-expert-profile.bin` if it fits; none otherwise (`:655-664`). |
| `--experts-bin` | Also cut `experts.bin` (+ sidecar). Without it the engine reads experts from the GGUF. |
| `--skip-experts` | Rewrite the index and `native_experts.txt`, never touch `experts.bin`. |
| `--layout-only` | Write `native_experts.txt` alone and print a per-layer census of expert types and blob sizes (`:665-677`). |
| `--compat-bf16` | Allow lossy conversions (dequantize or round to BF16/F16/F32 with round-to-nearest-even); writes `compat-bf16.json`. Incompatible with `--base`. |
| `--base PACK` | Reuse a canonical (Q2_0) pack's `dense.bin` by hard link; floats must be byte-identical to the base model; floats that the base quantized go to `extra.bin`. Requires all tensors in shard 1. |

### Control flow

1. `Model(src)` (`:205-236`) opens every shard with `gguf_reader.GGUFFile`, memory-maps them, and builds
   `name -> (GGUFFile, TensorInfo, memmap, path)`. It refuses missing shards, truncated or unknown-type tensors,
   and duplicate names across shards. `check_split` (`:239-262`) validates llama.cpp's `split.count`, `split.no`,
   `split.tensors.count` keys exactly as the engine's `GgufModel` does.
2. `expert_layout` (`:518-574`) runs **before anything is written**: finds `blk.N.ffn_{gate,up,down}_exps.weight`,
   refuses fused `ffn_gate_up_exps` (the engine reads separate tensors), subtracts
   `<arch>.nextn_predict_layers` from the layer count, takes `n_expert` from `blk.0.ffn_gate_inp.weight`'s `ne1`
   (routers must agree in every layer), requires gate and up of the same type, and computes per-layer blob sizes
   and the shard column. Returns the `native_experts.txt` text.
3. The profile is checked (`profile_problem`, `:594-610`).
4. `index_standalone` (`:349-449`) or `index_from_base` (`:468-515`) writes `index.txt`, `dense.bin`,
   `conversions.json`.
5. The tokenizer is exported by running `strata_tokenizer.py --gguf ... --out ...` in a subprocess unless
   `tokenizer/vocab.json` and `tokenizer/chat_template.jinja` already exist (`:718-720`).
6. `expert-profile.bin` is copied in (`:721-728`).
7. `native_experts.txt` is published (the completion marker), then `experts.bin` if asked.

### How each non-expert tensor is classified (`index_standalone`)

For every tensor that is not an expert and not `per_layer_token_embd.weight` (`NOT_IN_PACK`, `:70`):

- If it belongs to a draft block (`layer_of(name) >= n_trunk`), it is skipped and counted (`:359-361`).
- `form_of(name, arch)` (`:131-132`) looks up the suffix after `blk.N.` in `FORMS[arch]`:
  `FORM` for `qwen4exp` (Flash-Next, `:83-96`) or `FORM_QWEN35MOE` (`:103-109`).
- A quantized tensor with no form is **native** (row with shape only). Flash-Next's `blk.1.ple_key.weight` in
  `Q2_0`/`Q8_0` is native too (`NATIVE_PLE_KEY`, `:127`).
- A quantized tensor that has a form needs `--compat-bf16`, else it is refused.
- For `qwen35moe`, `native_type_problem` (`:142-159`) refuses a projection, `output.weight` or `token_embd.weight`
  whose type has no native kernel: projections and the head must be in `NATIVE_PROJECTION_TYPES` (`:121-122`), the
  embedding in `NATIVE_EMBED_TYPES` (`:123-124`). The ten Qwen3.6 native projection suffixes are listed in
  `NATIVE_PROJECTIONS_QWEN35MOE` (`:116-120`). A float projection is also refused (the engine has no F32/F16 GEMV).
- All problems are collected and printed (up to 8) before anything is written; the previous pack stays intact.
- Float tensors go through `convert` (`:290-342`):

| Stored | Form wanted | Result |
| --- | --- | --- |
| same as form, or no form | - | copied, index kind from `KIND` (`BF16`=4, `F16`=5, `F32`=2) |
| F32 with every low 16 bits zero | BF16 | high halves, exact, kind 4 (default) |
| F32 with inexact values, F16, or quantized | BF16 | refused with the count of inexact values; with `--compat-bf16`, gguf-py's BF16 quantizer (round-to-nearest-even), kind 4 |
| F32 | F16 | kept as F32 bytes, index kind 3: the loader narrows it (Flash-Next PLE conv) |
| quantized/BF16 | F16 | only with `--compat-bf16`, kind 5 |
| F16 or BF16 | F32 | exact widening, kind 2 |
| quantized | F32 | only with `--compat-bf16` |

Non-finite values are refused for any conversion (`bf16_bytes`, `f16_values`, `:182-202`).

### Model-specific behaviour

- **Flash-Next (`qwen4exp`)**: `FORM` lists the hyper-connection (`hc_*`), indexer, PLE and SSM tensors. Per the
  comment above the table (`:72-82`), the GSQ-RCO files already store them in those forms; other Flash-Next-geometry
  files (for example Unsloth's UD-Q4_K_XL) need conversions, some only with `--compat-bf16`.
- **Qwen3.6 (`qwen35moe`)**: `FORM_QWEN35MOE` makes routers (`ffn_gate_inp`, `ffn_gate_inp_shexp`) and
  `ssm_alpha`/`ssm_beta` BF16 (F32-stored but BF16-exact in both shipped files, so converted exactly with no flag),
  and every norm, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d` F32. All projections, the head and the embedding are native.
  `blk.40` of a `-MTP` file is excluded from the trunk pack and left for
  [`mtp_qwen36_pack.py`](mtp-tools.md). Unused constant: `N_EXPERT = 512` (`:266`) is not referenced; the count comes
  from the router shape.

## `tools/gguf_reader.py`

`GGUFFile(path)` (`:76`) parses a GGUF **v3** header (magic `GGUF`, version, tensor count, KV count), all metadata
values (scalars, strings, arrays) into `metadata`, and the tensor directory into `TensorInfo(name, shape,
type_id, type_name, offset)` (`:52`). Tensor data is never read. `data_start` is the header end rounded up to
`general.alignment` (default 32). `GGML_TYPES` (`:20`) maps ids 0..42 to names, including `42: Q2_0`, which gguf-py's
enum lacked when it was written. `BLOCK_GEOMETRY` (`:39`) gives `(elements per block, bytes per block)` and makes
`TensorInfo.expected_bytes()` possible; it returns `None` for an unknown type or a non-multiple, which
`iq_pack.Model` treats as an error. Helpers: `by_type()`, `find(needle)`. Its docstring calls it the reference
implementation of `src/artifact/` (the C++ GGUF reader).

Used by: `iq_pack.py`, `strata_pack.py`, `pack_layer.py`, `strata_tokenizer.py`, `ref/load.py`,
`qwen36_synth_gguf.py --like-real`.

## `tools/gguf_writer.py`

- `quantize_q2_0(w)` (`:42`): blocks of 64 values, fp16 scale `d = amax`, code `clamp(round(w/d) + 1, 0, 3)`
  packed four per byte LSB-first, so symbols are `{-1, 0, +1, +2} x d`.
- `dequantize_q2_0(raw)` (`:64`): the inverse; also the `Q2_0` decoder `ref/load.py` uses.
- `GGUFWriter` (`:86`): `add(key, value, type_name)`, `add_f32`, `add_q2_0(name, arr, shape)`, `write(path)`;
  only `F32`, `F16`, `Q2_0` tensors.
- `_selfcheck()` (`:186`) is run by `python tools/gguf_writer.py`.

## `tools/_paths.py`

`gguf_py()` returns the first directory containing a `gguf/` package among: `$STRATA_GGUF_PY`,
`<repo>/third_party/llama.cpp/gguf-py`, and `../.ref/llama.cpp/gguf-py` above the repo (upstream's development
tree). It exits with a message otherwise. `add_gguf_py()` prepends it to `sys.path`. Callers: `iq_pack.py` (lazily,
only when it has to dequantize or round), `mtp_pack.py`, `mtp_rt.py`, `mtp_qwen36_pack.py`, `qwen36_synth_gguf.py`,
`ref/qwen36.py`, `qwen36_check_dequant.py`, `test_iq_pack.py`, and the `tests/python` fixtures.

## `tools/strata_tokenizer.py`

`Tokenizer` (`:70`) is a pure-Python GPT-2 byte-level BPE built from GGUF metadata (`from_gguf`, `:127`): it requires
`tokenizer.ggml.model == "gpt2"`, reads tokens, merges, token types, `pre`, and every `*_token_id`. The
pre-tokenizer is the `qwen35` regex transcribed from llama.cpp (`QWEN35_PATTERN`, `:59-67`); the comment explains
why the `\p{M}` classes matter. Merges are applied lowest rank first (`_bpe`, `:145`), switching to a heap
(`_bpe_heap`, `:171`) for words longer than 64 symbols. Token type 4 (USER_DEFINED, e.g. `<think>`) is always matched
as a literal; type 3 (CONTROL, e.g. `<|im_end|>`) only with `parse_special=True` (`:96-117`, `encode`, `:247`).
`decode` is exact for any byte sequence.

CLI: `python3 tools/strata_tokenizer.py --gguf <file> --out <pack> [--check]` writes `<pack>/tokenizer/`
(`extract`, `:278`); `--check` round-trips a fixed corpus. Encoding needs the `regex` module; the export does not.
`draft_vocab.py` uses `Tokenizer` for `--corpus` and script classification.

## Legacy canonical pack: `strata_pack.py`, `pack_layer.py`, `pack_index.py`

Upstream's Phase-1 pack, which re-encodes every tensor into "canonical planes" (codes, scales, offsets) and an
expert arena with a fixed 1,382,400-byte Flash-Next Q2_0 blob.
[docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) calls them "reference only"; both
supported models use `iq_pack.py`.

- `strata_pack.py build --gguf --out [--layers N] [--skip-hash]`, `verify --gguf --out [--limit] [--expert-every 32]`,
  `info --out`. `build` (`:206`) delegates the arena to `pack_layer.build` and encodes `dense.bin`/`embd.bin`
  per tensor (`tensor_entry`, `:107`), choosing FP16 scales only when exact (`choose_scale_dtype`, `:90`).
  `verify` (`:338`) decodes from disk using only the manifest and compares bit for bit with the source,
  including an independent inverse of the arena layout (`rebuild_role`, `:282`).
- `pack_layer.py [n_layers]` (`build`, `:68`): per layer, the 512 experts' gate/up rows interleaved
  (`g0,u0,g1,u1,...`), Q2_0 codes then fp16 scales; asserts the 1,382,400-byte blob (`:49-51`).
- `pack_index.py --pack <dir> [-o out]` (`main`, `:99`): reads `manifest.json` and writes the 19-column `index.txt`,
  deciding the loader `kind` (0-3), widening fp16 scale planes to f32 in the engine form, verifying that the
  planes are contiguous, and writing `act_kind` from `VDT` (`:86-90`). The engine's index parser
  (`src/core/weights.cpp`) and `include/strata/kernels/s_gemv.hpp` still cite it.

## Configuration

| Name | Where | Effect |
| --- | --- | --- |
| `STRATA_GGUF_PY` | `_paths.py` | gguf-py directory (else the submodule's `third_party/llama.cpp/gguf-py`) |
| `python3 -I` | UPSTREAM.md commands | isolated mode; the tools add their own directory to `sys.path` |

Dependencies: Python 3 with `numpy`; gguf-py for anything that dequantizes, rounds or writes with
`gguf.GGUFWriter`; `regex` only for tokenizing.

## Tests

- [`tools/test_iq_pack.py`](../../../tools/test_iq_pack.py): `python3 -m unittest tools.test_iq_pack` (needs gguf-py).
  Covers BF16 rounding, refusals, split shards, a layer split per role (v4), interrupted packs, stale `experts.bin`,
  `--layout-only`, and the `Qwen36PackTests` class (forms, index contents, BF16-exact rule, native type refusals,
  the nextn block, the expert profile). See [python-tests.md](python-tests.md).
- `tests/core/expert_layout_test.cpp` with `tests/data/native_experts/*.txt`, and `tests/core/pack_load_check.cpp`
  (`STRATA_QWEN36_GGUF` / `STRATA_QWEN36_PACK`), on the engine side.
- `tests/python/qwen36_runner_fixture.py` builds a native pack of a synthetic model with `iq_pack.py` for the
  runner tests.

## Gotchas

- `strata_pack.py` imports `canonical_xcheck`, which is **not in this repository**; `import strata_pack` fails
  with `ModuleNotFoundError`. `strata_pack.py` and `pack_layer.py` also default to upstream's development layout
  (`<grandparent>/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000N-of-00002.gguf`).
- `gguf_writer.py`'s self-check calls `add_q2_0(..., 64)` with an integer where the method expects the full
  shape list (`tools/gguf_writer.py:196` vs `:118`); it raises `TypeError` before writing. It would also write to
  `bench/tiny-selftest.gguf`, a directory this fork does not have.
- `_paths.py`'s error message and `iq_fixture.py`'s mention `setup.py`, which this fork removed (UPSTREAM.md).
- `strata_tokenizer.from_gguf` does not check `pre`; it always uses the `qwen35` pattern.
- An existing `experts.bin` without a matching sidecar is still read by the engine in preference to the GGUF;
  `iq_pack.py` warns or refuses but never deletes it.

## See also

- [pack-format.md](pack-format.md), [profile-and-draft-vocab.md](profile-and-draft-vocab.md),
  [mtp-tools.md](mtp-tools.md)
- [docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md),
  [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) sections 4 and 6,
  [docs/qwen36-port/gguf-metadata.md](../../qwen36-port/gguf-metadata.md)
