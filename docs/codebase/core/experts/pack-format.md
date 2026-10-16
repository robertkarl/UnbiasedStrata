[Codebase docs](../../README.md) › [Core](../README.md) › [Experts](README.md) › Pack format

# The pack files the expert code reads

## Purpose

A *pack* is the directory `--pack DIR` names. Python tools build it from the model's GGUF:
[tools/iq_pack.py](../../../../tools/iq_pack.py) for native packs (all Qwen3.6 packs and Flash-Next's IQ files), and
`tools/strata_pack.py` plus [tools/pack_index.py](../../../../tools/pack_index.py) for the canonical Q2_0 pack. This
page lists the files that matter to expert management, what reads each one, and which checks protect them. It also
covers the tests in this area's scope that exercise packs and GGUF shards. The full byte-level description of
every pack file, from the side of the tools that write them, is [Tools › Pack format](../../tools/pack-format.md).

## Files

| File | Written by | Read by | Contents |
|---|---|---|---|
| `experts.bin` | `strata_pack.py` (canonical), `iq_pack.py --experts-bin` (native, optional), or the engine itself under `STRATA_ARENA_MMAP` | `ArenaExpertSource::open`, `FileExpertSource::open` | Every blob back to back. Canonical: `n_layers * n_expert * BLOB` bytes, index `layer * n_expert + expert`, no padding. Native: per-layer blob sizes at the offsets in `native_experts.txt`. `experts.bin.src.json` records which source a native one was cut from |
| `native_experts.txt` | `iq_pack.py` (written **last**, so a pack without it is unfinished) | `expert_layout_load` (`src/kernels/cpu/expert_layout.cpp:233-349`) | The per-layer expert layout (below). Its absence means a canonical pack |
| `index.txt` | `pack_index.py` / `iq_pack.py` (`write_index`, `iq_pack.py:452-465`) | `WeightTable::load` (`src/core/weights.cpp:143` onwards) | The dense weights' flat index. Expert tensors appear as rows that carry shape only when served natively |
| `dense.bin` / `extra.bin` | the pack tools | `WeightTable::load` | The dense tensors' bytes (not expert data) |
| `conversions.json` | `iq_pack.py` (`:423-435`) | **No engine code reads it** (no reference in `src/`, `include/` or `frontend/`); `tools/test_iq_pack.py` checks it | A record of every tensor converted to the engine's float form |
| `expert-profile.bin` | `tools/make_profile.py`; copied in by `iq_pack.py --expert-profile` when its header matches the trunk's layers x experts (default `data/qwen36-expert-profile.bin`); rewritten by `--expert-profile-save` | `read_expert_profile` (see [vram-cache.md](vram-cache.md)); `frontend/main.cpp:665` requires it | `STRP` ranked `(layer, expert)` pairs |
| `manifest.json` | the canonical pack tools | `shared_arena_pack_hash` (hashed), `iq_pack.py --base` | Pack metadata |

### `native_experts.txt`

One header line, then one line per layer. Real examples are in `tests/data/native_experts/*.txt`:

```
# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert 256, total 11108614144; absolute offsets in Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf, or in the named shard beside it)
0 22 18 0 1073152 969626080 1058574816 866005472
```

| Column | Meaning |
|---|---|
| `layer` | Layer index. Lines must be contiguous and complete |
| `gu_type`, `d_type` | ggml type ids of the gate/up and down rows (for example 42 = Q2_0, 18 = IQ3_XXS) |
| `offset`, `blob_bytes` | Where the layer's blobs start in `experts.bin`, and the bytes per blob. `blob_bytes` must equal what the formats make for the model's widths |
| `gate_off`, `up_off`, `down_off` (v2+) | **Absolute** file offsets of `blk.L.ffn_{gate,up,down}_exps.weight` in the GGUF. These let the engine read experts without `experts.bin` |
| `shard` (v3+) | The GGUF file beside `--native` that holds this layer. v4: a per-role `gate,up,down` column (an empty field = the `--native` shard), written only when a shard boundary falls inside a layer (Unsloth's UD-Q4_K_XL) |

The header's `(n_expert N)` overrides the expert count (a pruned model, for example the Coder's 256). The reader
accepts versions up to `kExpertLayoutVersion = 4` and refuses newer ones (`expert_layout.hpp:83-85`). Each layer's
`NativeFmt` is built from the **model's** `n_embd` / `n_ff` (from the GGUF metadata), and its byte count is checked
against `blob_bytes`, so a pack of another geometry is refused by its numbers.

`check_experts_gguf` then proves every offset against the GGUF directory before any byte is read (see
[expert-sources.md](expert-sources.md)).

### `index.txt` (dense weights)

A `# align A pool P tensors N` header, then one space-separated row of 19 fields per tensor:

```
name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook
has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind
```

A row with fewer fields is refused (`weights.cpp:175`). `kind` values:

| Kind | Meaning |
|---|---|
| 0 | Verbatim |
| 1 | BF16 in F32 |
| 2 | F32 |
| 3 | F16 in F32 |
| 4 | Raw BF16 (native packs) |
| 5 | Raw F16 (native packs) |

The loader itself belongs to the dense-weights documentation. `shared_arena_pack_hash` hashes this file, so a shared
arena is tied to it.

### `conversions.json`

Schema 1 (`iq_pack.py:423-428`):

```json
{"schema": 1, "tool": "tools/iq_pack.py", "compat_bf16": false,
 "source_shards": [{"name": "...", "size": 123}],
 "tensors": [{"name": "...", "shard": "...", "method": "...", "exact": true, "max_abs_err": 0.0, "source_sha256": "..."}]}
```

The `tensors` records hold source and destination types too (from the conversion `rec`; the exact field set is
`iq_pack.py`'s, not determined in full here). The format rules are in the `iq_pack.py` docstring:

- F32 to BF16 is exact when the low 16 bits are zero, and is the default.
- Other conversions to BF16 round, and only happen with `--compat-bf16`.
- F32 to F16 for `ple_conv1d` uses index kind 3.

For Qwen3.6, the pack test expects every conversion to be exact (`tools/test_iq_pack.py:717`, in `Qwen36PackTests`).

## Tests

| Test | File | What it proves | Needs |
|---|---|---|---|
| `expert_layout_test` | `tests/core/expert_layout_test.cpp` | (1) the real packs' `native_experts.txt` copies (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M, Swift IQ2_XS, Qwen3.6 IQ3_XXS and Q4_K_XL) parse to the layout the pre-v4 reader produced; (2) the v4 per-role column, and refusal of v5, four names, two names and a trailing column; (3) `check_experts_gguf` and `load_experts_gguf` on synthetic shards, every byte compared, plus the refusals | `STRATA_NATIVE_EXPERTS`; argument `tests/data/native_experts` |
| `pack_load_check` | `tests/core/pack_load_check.cpp` | Loads a real native pack in `generate --native --pack` order up to the first token (arena, native projections, head, embedding, `check_all`) and reads `expert-profile.bin` (`:175-181`). Runs no kernel | `--native FILE --pack DIR` or `STRATA_QWEN36_GGUF` / `STRATA_QWEN36_PACK`; exit 77 without them or without CUDA; 900 s timeout |
| `pool_native_widths_test` | `tests/core/pool_native_widths_test.cpp` | `run_split_multi_native` uses the layer's `n_ff` / `n_embd` (Qwen3.6 2048 x 512), not the canonical 640 / 2560. Three Q8_0 experts, four tokens, guard words after each row | `STRATA_NATIVE_EXPERTS`; CPU |
| `gguf_split_test` | `tests/core/gguf_split_test.cpp` | `gguf_split_paths` / `GgufModel` on UD-Q4_K_XL-shaped synthetic shards (layer 11 split across shards 2 and 3); refusals for a missing shard, another model's shard, a duplicate tensor, a wrong `split.tensors.count`, a truncated shard, shard 1 alone | `STRATA_BUILD_TESTS`; links `strata_artifact` |
| `native_dense_ple_key_test` | `tests/core/native_dense_ple_key_test.cpp` | #326: a native pack whose PLE key row is BF16 keeps it in the dense arena, and `NativeDense` does not upload the GGUF key over it; a quantized key row stays natively served; `index_code_bits` edge cases | CUDA (exit 77 without) |
| (fixture) | `tests/core/gguf_fixture.hpp` | `fixture::Kv` / `fixture::Tensor` write GGUF v3 files (metadata strings, ints and u32 arrays; 32-byte aligned tensors; payload byte `i` = `(seed*131 + i*7 + i/4093) & 0xff`) | none |
| Python | `tools/test_iq_pack.py`, `tools/test_make_profile.py` | The pack's file set, `conversions.json` contents, `expert-profile.bin` copy rules, the profile format | python3 |

All C++ ones are registered in `CMakeLists.txt:231-233` and `:470-503`. Run them with `ctest -R <name>`.

## Model-specific behaviour

| | Flash-Next (Qwen3.8-Flash-Next) | Qwen3.6-35B-A3B |
|---|---|---|
| Pack kinds | Canonical Q2_0 (`experts.bin`, no `native_experts.txt`) or native IQ | Native only |
| Layers x experts (profile) | 48 x 512 (Coder: 256) | 40 trunk x 256. The `-MTP` draft block is left out of the pack |
| Expert widths | 2560 x 640 | 2048 x 512 |
| Shipped profile | `data/expert-profile.bin` | `data/qwen36-expert-profile.bin`, copied in as `<pack>/expert-profile.bin` |

## Gotchas

- A pack without `native_experts.txt` is treated as canonical. For a model that is not 2560 x 640 that is refused with
  "pack it with tools/iq_pack.py".
- `conversions.json` is informational only. Changing it changes nothing at run time.
- `iq_pack.py` publishes `dense.bin`, `index.txt` and `conversions.json` together after deleting `native_experts.txt`.
  It writes `native_experts.txt` last, so an interrupted pack looks unfinished.

## See also

[expert-sources.md](expert-sources.md) · [vram-cache.md](vram-cache.md) ·
[docs/qwen36-port/step4-pack.md](../../../qwen36-port/step4-pack.md) ·
[docs/qwen36-port/gguf-metadata.md](../../../qwen36-port/gguf-metadata.md) ·
[docs/qwen36-port/geometry-inventory.md](../../../qwen36-port/geometry-inventory.md)
