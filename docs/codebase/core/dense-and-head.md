[Codebase docs](../README.md) › [Core](README.md) › Native dense projections, head and embedding

# Native dense projections, head and embedding

## Purpose

The canonical pack stores every dense projection in the engine's own "S-form" quantization. Since plan v0.3
("native" mode, `--native SHARD1`) the engine can instead read the large dense matrices **unchanged from the model's
GGUF** and run them through ggml-compatible native MMVQ kernels (Q8_1 activations). Three classes do that:

* `NativeDense` - the per-layer GDN, attention and shared-expert projections, attached to the canonical
  `WeightRef`s so the layer code picks them up transparently.
* `NativeHead` - the output projection `output.weight`.
* `NativeEmbed` - the token embedding `token_embd.weight`, kept in mapped pinned host memory (no VRAM).

Files covered:

* [include/strata/core/native_dense.hpp](../../../include/strata/core/native_dense.hpp), [src/core/native_dense.cpp](../../../src/core/native_dense.cpp)
* [include/strata/core/native_head.hpp](../../../include/strata/core/native_head.hpp), [src/core/native_head.cpp](../../../src/core/native_head.cpp)

## `NativeDense` (`native_dense.hpp:16`)

| Member | What it does |
| --- | --- |
| `static served_names(shards, include_ple_key, out, err)` | Lists, from GGUF headers only, the tensor names `load` would serve: eligible name, `native_mmvq_supported(type)`, 2-D. generate.cpp passes this set as the `skip` set of `WeightTable::pool_bytes` / `load`, so the canonical arena does not hold those bytes |
| `load(shards, table, err, include_ple_key, layer_lo, layer_hi)` (`native_dense.cpp:80`) | Validates every shard (architecture on the first, matching `split.*` metadata on the rest, no duplicate split numbers, tensor spans inside the payload and non-overlapping), uploads each eligible matrix with one `cudaMalloc` + `cudaMemcpy`, allocates one shared Q8_1 activation scratch sized for the widest input, and only after everything succeeded writes `native_data`, `native_type` and `native_q8_1` into the matching `WeightRef`s |
| `static set_layer_range(lb, le)` | Layer split: process-wide range for the next `load` (PLE tensors are loaded everywhere) |
| `static keep_unquantized_ple_key(pack_dir, skip, err)` | #326: when the pack's `blk.1.ple_key.weight` row is unquantized, take it out of the skip set so the PLE reads it from the arena |

Eligible names (`eligible`, `native_dense.cpp:24`): `blk.*.` tensors ending in `attn_qkv`, `attn_gate`, `ssm_out`,
`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`, and with
`include_ple_key` `blk.1.ple_key.weight` of type Q2_0 (42), IQ3_XXS (18), IQ4_XS (23) or Q8_0 (8). A canonical
row must exist for every uploaded tensor and its shape must match (`ne0`, `ne1`).

Invariant (header comment): all native projections in one table share **one** Q8_1 scratch, so they must run on
one ordered stream, and the object must outlive every graph that captured them. `gemv_quantized` in `layer.cpp:141`
takes the native branch whenever `w.native_data` is set; a caller that just quantized the same input for a previous
native projection passes `x_q8_1_ready` to skip the re-quantize (used for `attn_gate` after `attn_qkv`, `attn_v` after
`attn_k`, `attn_q` after `attn_v`).

## `NativeHead` (`native_head.hpp:15`)

* `load(shards, n_in, n_out, err)` (`native_head.cpp:19`): finds `output.weight` in whichever shard holds it, requires
  a native-MMVQ-supported type with shape `[n_in, n_out]` and `n_in % 256 == 0`, uploads it and allocates a Q8_1
  scratch.
* `run(mixed, logits, stream, err)` (`native_head.cpp:71`): Q5_K (type 13) has its own `native_q5_k_f32` kernel;
  every other type is `native_quantize_q8_1` then `native_mmvq`.
* `weights()`, `type()`, `row_bytes()` expose the blocks to the verify window's multi-column head and to the MTP
  drafter's draft-vocabulary head ([verify.md](verify.md), [mtp-draft-head.md](mtp-draft-head.md)).

The canonical head (`lm_head`, [layer-forward.md](layer-forward.md)) stays available for A/B; both share
`lm_head_mix`, so only the projection differs.

## `NativeEmbed` (`native_head.hpp:43`)

* `load(shards, n_embd, n_vocab, err)` (`native_head.cpp:108`): accepts a model GGUF or a one-tensor
  `strata-embd` file (`--embd-gguf`, written by `tools/embd_bf16_pack.py`). The table is copied into
  `cudaHostAlloc(Mapped | Portable)` memory and read over PCIe through its device alias. If pinning fails (WSL2's
  pinned budget is cited) or, on HIP, the mapped allocation has no usable device alias (#325), the table is put in
  VRAM instead and a line is printed.
* `gather_dev(tokens, n, out, stream)` (many rows, used by verify and MTP) and `gather_one(token, out, stream)`
  (used by `embed_row`).
* `set_native_embed(e)` / `native_embed()` (`native_head.hpp:65-66`) install a process-wide pointer;
  `embed_row`, `Verifier::record_window` and the drafter check it first and fall back to the canonical
  `token_embd.weight` when it is null.

## Callers

`src/program/generate.cpp`: `NativeEmbed::load` (~line 2496), `NativeDense::served_names` (~2512),
`NativeDense::load` (~2595), `NativeHead::load` (~3373), `native_head.run` on the plain decode head (~4220).
`tests/core/pack_load_check.cpp` repeats this order.

## Configuration

| Flag | Effect |
| --- | --- |
| `--native SHARD1` | Turns on every native path at once (generate.cpp ~1880): native dense from all model shards, the native head from the same file, native BF16, GDN, router, QSA, RoPE, indexer, MoE combine, and the PLE key and post-ops when the model has a PLE |
| `--native-dense-gguf PATH` (repeatable) | Shards for `NativeDense` |
| `--native-head-gguf PATH` | Source of the native head |
| `--embd-gguf PATH` | Source of the native embedding |
| `--native-ple-key` | Serve `blk.1.ple_key.weight` natively |

## Model-specific behaviour

* Qwen3.6 packs are native packs: the projections exist only as shape rows in `index.txt`, so `--native` is required.
* The PLE key path applies only to Flash-Next (`model_has_ple`).

## Tests

* `native_dense_ple_key_test` (`tests/core/native_dense_ple_key_test.cpp`, needs a GPU, skip 77): the BF16 key row
  stays in the arena and is not overwritten; a quantized key row is still served natively; `index_code_bits`.
* `pack_load_check` (see [weights-and-layout.md](weights-and-layout.md)).
* `shared_expert_parity`, `native_grouped_parity` and other `src/kernels/*_parity.cpp` cover the native kernels
  themselves (kernels area).

## Gotchas

* `NativeDense::load` refuses to run twice on one object and refuses an override on a `WeightRef` that already has
  one.
* `NativeHead::run` requires an explicit stream (null is an error).
* The native embedding's global pointer is process-wide; a second model in one process would share it.

## See also

* [weights-and-layout.md](weights-and-layout.md), [layer-forward.md](layer-forward.md).
* [docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md).
