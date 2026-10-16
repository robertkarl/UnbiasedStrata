[Codebase docs](../README.md) › [Core](README.md) › Weights and layout

# Dense weights: loading and layout checks

## Purpose

The dense (non-expert) weights - attention and GDN projections, norms, routers, shared experts, hyper-connection
weights, the output head and (for canonical packs) the embedding - live in one VRAM arena. This page covers the
loader that copies them from a pack into that arena in **engine form**, and the checks that every tensor a kernel
reads has the name, shape and byte form the kernel assumes.

Files covered:

* [include/strata/core/weights.hpp](../../../include/strata/core/weights.hpp), [src/core/weights.cpp](../../../src/core/weights.cpp) - `WeightTable`.
* [include/strata/core/layout.hpp](../../../include/strata/core/layout.hpp), [src/core/layout.cpp](../../../src/core/layout.cpp) - `LayerView`, `check_layer`, `check_all` (the `ModelGeometry` struct in the same header is on [model-geometry.md](model-geometry.md)).

## The pack and its index

A pack directory (written by `tools/pack_index.py`, or `tools/iq_pack.py` for native packs) holds:

* `index.txt` - a header line `# align A pool P tensors N`, then one space-separated row per tensor with 19 fields:
  `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset
  codes_bytes scales_bytes offset_bytes scales_fp16 act_kind`.
* Data files by id (`file_name`, `weights.cpp:64`): 0 `dense.bin`, 1 `embd.bin`, 2 `experts.bin`, 3 `extra.bin`
  (a native pack's float tensors that the base pack quantized).

The header comment in `weights.hpp` explains why the engine parses a flat index rather than JSON: a parser bug
would be a wrong byte offset, which decodes to a plausible weight.

## Main types

### `enum class WeightKind` (`weights.hpp:41`)

What the loader must do for a tensor, decided by the pack tool and written into the index:

| Kind | Meaning | Engine form |
| --- | --- | --- |
| `Verbatim` (0) | quantized planes, or F16 already stored as 2 B | copied as is (fp16 **scale planes** are widened, see below) |
| `Bf16InF32` (1) | pack holds a bf16 value promoted to f32 | high 16 bits kept: 2 B/elem, exact |
| `F32` (2) | f32 | 4 B/elem |
| `F16InF32` (3) | an f16 value promoted to f32 | real f32-to-f16 conversion: 2 B/elem |

Index kinds 4 and 5 (native packs, `weights.cpp:182`) are BF16 / F16 already stored as 16 bits: copied raw
(`raw16`) and labelled as kinds 1 / 3.

### `struct WeightRef` (`weights.hpp:48`)

A device pointer and everything a kernel needs to read it: `ne0` (contiguous axis), `ne1`, `elements`, `kind`, the
canonical "S-form" attributes for quantized tensors (`code_bits`, `code_bias`, `group_elems`, `codebook_iq4nl`,
`has_offset`), the three plane sizes **in engine form** (`codes_bytes`, `scales_bytes` - always 4 B per scale in the
arena - and `offset_bytes`), provenance (`src_off`, `src_bytes`, `file_id`, `scales_fp16`) so tests can compare
against the pack bytes, and `act_kind` (0 = wants a Q8_0 activation, 1 = Q8_K; `wants_q8k()`), which the S-form
cannot express by itself. Native fields (`native_data`, `native_q8_1`, `native_type`) are filled by `NativeDense`
([dense-and-head.md](dense-and-head.md)). `resident == false` with `data == nullptr` marks a tensor the loader
skipped because a native form serves it.

### `class WeightTable` (`weights.hpp:107`)

| Function | Line | What it does |
| --- | --- | --- |
| `static pool_bytes(pack_dir, out, err, skip)` | `weights.cpp:89` | Reads only the header (or, with a skip set, sums the kept rows at the index alignment) so the caller can allocate the arena before loading |
| `static index_code_bits(pack_dir, name, ...)` | `weights.cpp:121` | One row's `code_bits` without loading (used to decide whether a PLE key row is unquantized, #326) |
| `load(pack_dir, arena_base, arena_bytes, err, skip)` | `weights.cpp:144` | Parses every row (refuses a row with fewer than 19 fields), compacts the arena if a skip set is given, then copies tensor by tensor |
| `find(name)` / `all()` | `weights.cpp:474` | Name lookup |
| `set(name, ref)` | `weights.hpp:135` | Test seam: insert a row the caller owns |

How `load` copies:

1. Two pinned staging buffers: `stage_in` 8 MiB (`CHUNK`) and `stage_out` 16 MiB, because widening can double a
   segment. Converted bytes go host-to-host and then cross the bus once; the comment records an earlier bug where
   conversion wrote straight to a device pointer from the host.
2. Each tensor is cut into up to three **segments** (`struct Seg`, `weights.cpp:56`): for a quantized tensor
   codes (copy), scales (copy, or `WidenF16` when `scales_fp16`), offsets (copy); otherwise one segment with
   `Bf16High`, `ToF16` or `Copy`. The segment totals are checked against the row's `src_bytes` / `dst_bytes`.
3. A row with `code_bits != 0` and `dst_bytes == 0` is a shape-only row of a native pack: loading it without
   `--native` is refused ("this pack holds the tensor only in its GGUF form").
4. `LoadReport` records tensors, arena bytes, how many were re-rounded and bytes saved, and read/upload times.

## Layout checks (`layout.cpp`)

`LayerView` (`layout.hpp:114`) resolves `blk.<layer>.<suffix>` in a table; a null result is information (a GDN
layer has no `attn_q`, an attention layer no `attn_qkv`).

`check_layer` / `check_all` (`layout.hpp:133`, `:137`; `layout.cpp:157`, `:165`) assert for every layer:

* 2-D shapes (`Want2` table, `layout.cpp:51`): the six `hc_*` gated-residual tensors (every layer, only at `hc > 1`), the
  router `ffn_gate_inp` `[n_embd, n_expert]`, the shared expert's gate/up/down; GDN-only `attn_qkv`, `attn_gate`,
  `ssm_out`, `ssm_conv1d`, `ssm_alpha`, `ssm_beta`; attention-only `attn_q` (`[n_embd, 2*n_head*head_dim]`: q and
  gate per head), `attn_k`, `attn_v`, `attn_output`, and with an indexer `indexer.q_proj` / `indexer.k_proj`.
* 1-D element counts **and engine form** (`Want1` table, `layout.cpp:103`): every 1-D tensor is `F32` except
  `ffn_gate_inp_shexp.weight` (the shared expert's scalar gate), which is `Bf16InF32`. At `hc = 1` the
  `hc_*_norm` rows are skipped and `attn_norm` / `post_attention_norm` are required instead; at `hc > 1` the
  reverse. Indexer norms are skipped without an indexer.
* `check_all` also counts the layers by `is_qsa_layer` and requires the split to equal `n_qsa_layers()` /
  `n_gdn_layers()`.

The engine-form check exists because a BF16 tensor read as f32 reads twice its length inside a multi-GiB arena
without faulting; the comment at `layout.cpp:20-28` records that this happened to `ffn_gate_inp_shexp` and made
every MoE output non-finite.

## Callers and dependencies

* `src/program/generate.cpp`: `NativeDense::served_names` builds the skip set, `WeightTable::pool_bytes`, one
  `cudaMalloc` of that size, `WeightTable::load`, `NativeDense::load`, and for `qwen35moe` `check_all`
  (around lines 2512-2607). A layer split loads a second table per stage.
* Every layer function resolves weights through `LayerView` and reads plane pointers via `plane_ptrs`
  (`layer.cpp:83`), which reads the sizes recorded here instead of re-deriving them.
* `tests/core/pack_load_check.cpp` reproduces the start-up load order on a real Qwen3.6 pack.

## Configuration

* `--pack DIR` (the pack), `--native SHARD1` (enables the skip set and native projections) - generate.cpp.
* CMake: part of `strata_core`.

## Model-specific behaviour

* Qwen3.6 packs keep norms F32 and carry no `hc_*`, `indexer.*` or `ple_*` rows; most projections are served from
  the GGUF by `NativeDense` and appear in the index as shape-only rows.
* Flash-Next canonical packs carry BF16 rows promoted to f32, which the loader re-rounds (the header's figures:
  5.305 GiB dense in the pack, 4.067 GiB in the engine).

## Tests

* `pack_load_check` (`tests/core/pack_load_check.cpp`; CTest, skip 77 without `STRATA_QWEN36_GGUF` /
  `STRATA_QWEN36_PACK` or a GPU): the real load order plus `check_all`.
* `native_dense_ple_key_test` (`tests/core/native_dense_ple_key_test.cpp`): skip sets, `index_code_bits`.
* `qwen36_block_test` writes a small pack and loads it through `WeightTable::load` (see [layer-forward.md](layer-forward.md)).

## Gotchas

* `load` checks `arena_bytes >= pool` itself; a short arena is an error, not a write past the end.
* `scales_bytes` in a `WeightRef` is the **engine** size (f32 scales), not the pack's.
* `check_all` is not called for Flash-Next starts in generate.cpp; only the Qwen3.6 path runs it.

## See also

* [model-geometry.md](model-geometry.md), [dense-and-head.md](dense-and-head.md), [layer-forward.md](layer-forward.md).
* [docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md) - the Qwen3.6 pack and its load order.
