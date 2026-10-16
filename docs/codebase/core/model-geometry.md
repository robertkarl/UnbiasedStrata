[Codebase docs](../README.md) › [Core](README.md) › Model geometry

# Model geometry

## Purpose

`ModelGeometry` is the single description of "which model is this and what shape is every tensor". Every
kernel width, buffer size, layer-type decision and per-model branch in the engine reads it. It is filled from
the GGUF file's own metadata at start-up, before any weight is moved, so a file the engine cannot run is
refused with a message naming the key instead of failing inside a kernel.

Files covered:

* [include/strata/core/layout.hpp](../../../include/strata/core/layout.hpp) - `ModelArch`, `ModelGeometry`, `is_qsa_layer` (the struct lives here because `LayerView` and `check_layer` need it).
* [include/strata/core/model_geometry.hpp](../../../include/strata/core/model_geometry.hpp) - header-only reader, fused-form refusal, the start-up text.
* [include/strata/core/qsa_shapes.hpp](../../../include/strata/core/qsa_shapes.hpp) - the single `QsaShapes` builder.
* [tests/core/model_geometry_test.cpp](../../../tests/core/model_geometry_test.cpp) - its unit test.

## Main types and functions

### `enum class ModelArch` (`include/strata/core/layout.hpp:30`)

`Qwen4Exp` (`general.architecture = "qwen4exp"`, Qwen3.8-Flash-Next) and `Qwen35Moe` (`"qwen35moe"`,
Qwen3.6-35B-A3B). `model_arch_name` (`layout.hpp:32`) maps back to the string.

### `struct ModelGeometry` (`include/strata/core/layout.hpp:38`)

The defaults are Flash-Next's; every field the file has a key for is overwritten by `read_model_geometry`.

| Group | Fields (Flash-Next default / Qwen3.6 value) | Meaning |
| --- | --- | --- |
| Trunk | `n_embd` 2560 / 2048, `n_layers` 48 / 40, `nextn_layers` 0 / 0-1, `qsa_interval` 4, `n_vocab` 248,320 | `n_layers` is `block_count - nextn_predict_layers` (the MTP blocks at the end of the file are not trunk) |
| GDN (Gated DeltaNet) | `ssm_state_size` 128, `ssm_k_heads` 16, `ssm_v_heads` 48 / 32, `ssm_d_conv` 4 | linear-attention recurrence; state per layer is `S x h_v x S` floats plus a conv history |
| GDN derived | `ssm_conv_channels` 10240 / 8192, `ssm_value_dim` 6144 / 4096 | `derive()` (`layout.hpp:97`): `value_dim = S*h_v`, `conv_channels = 2*S*h_k + value_dim` |
| Attention | `n_head` 24 / 16, `n_head_kv` 2, `head_dim` 256, `n_rot` 64, `rope_sections` {11,11,10,0}, `rope_base` 1e7, `rms_eps` 1e-6 | partial rotary (IMRoPE sections), GQA |
| QSA indexer | `idx_q_heads` 4 / 0, `idx_key_dim` 128 / 0 | Flash-Next only: the sparse-attention indexer |
| Residual | `hc` 4 / 1, `hc_lr` 320 / 0 | hyper-connection streams; `hc = 1` means the plain residual |
| MoE | `n_expert` 512 / 256, `n_expert_used` 10 / 8, `n_ff` 640 / 512, `n_ff_shexp` 640 / 512 | routed experts and the shared expert |
| PLE | `has_ple` true / false | Flash-Next's layer-1 per-layer token embedding with an n-gram table |

Derived predicates:

* `is_qsa_layer(g, l)` (`layout.hpp:105`): `l % qsa_interval == qsa_interval - 1`, i.e. layers 3, 7, 11, ...
  are full attention, the rest GDN. `n_qsa_layers()` / `n_gdn_layers()` follow (`layout.hpp:83-84`).
* `has_indexer()` (`layout.hpp:86`): true only for `Qwen4Exp`. Without it the attention is **dense**.
* `gdn_gate()` (`layout.hpp:90`): `GdnGate::Sigmoid` for Flash-Next, `GdnGate::Silu` for Qwen3.6. Every GDN output
  kernel takes this value from the call site; no kernel reads a global.
* `hc_dim()` = `hc * n_embd`, the width of the residual stack `R`.

### `read_model_geometry` (`include/strata/core/model_geometry.hpp:64`, file overload `:171`)

1. Reads `general.architecture`. `qwen4exp` keeps the Flash-Next defaults; `qwen35moe` sets `hc = 1`, `hc_lr = 0`,
   `has_ple = false`, `idx_q_heads = idx_key_dim = 0` - the things no key says. Any other architecture is refused
   by name.
2. Reads `<arch>.*` keys. Always required: `block_count`, `embedding_length`, `attention.head_count`,
   `attention.head_count_kv`, `expert_count`, `expert_used_count`. Required for Qwen3.6, optional for Flash-Next
   (whose files may leave them to defaults): `attention.key_length`, `full_attention_interval`, `ssm.conv_kernel`,
   `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank`, `expert_feed_forward_length`. Optional:
   `nextn_predict_layers`, `expert_shared_feed_forward_length` (defaults to `n_ff`), `rope.dimension_count`,
   `rope.freq_base`, `attention.layer_norm_rms_epsilon`, `rope.dimension_sections` (must be an array of 4).
3. Derives and cross-checks: `n_layers > 0`; `ssm.inner_size` (if present) equals `state_size * time_step_rank`;
   `n_layers % qsa_interval == 0`; `ssm_v_heads % ssm_k_heads == 0`; `n_head % n_head_kv == 0`;
   `0 < n_expert_used <= n_expert`; `n_rot` even and `<= head_dim`; rope sections sum to `n_rot / 2`.
4. Vocabulary: rows of `output.weight`, rows of `token_embd.weight` and the length of `tokenizer.ggml.tokens`
   (whichever are in the metadata shard) must agree; the default 248,320 stands only when none is present.

### `check_model_tensor_forms` (`model_geometry.hpp:187`)

Refuses two fused layouts llama.cpp accepts but the engine does not read: `ffn_gate_up_exps.weight` (fused expert
gate+up) and `attn_qkv.weight` in an attention layer without `attn_q.weight` (fused Q|K|V). In a GDN layer
`attn_qkv` is legitimate (it is the conv input).

### `model_geometry_text` and `qwen35moe_not_yet` (`model_geometry.hpp:208`, `:249`)

`model_geometry_text` prints one line per field for the start-up log (generate.cpp prints it for non-Flash-Next
files). `qwen35moe_not_yet` is the Phase 2 step-2 gate message; the engine no longer calls it (generate.cpp's
comment near the VRAM plan says the gate was lifted at step 4), only `model_geometry_test` prints it.

### `qsa_shapes` (`include/strata/core/qsa_shapes.hpp:19`)

The one place a `strata::kernels::QsaShapes` is built from a geometry. It starts from `qsa_real_shapes()` (which
fixes `n_rot`, `idx_block`, `page_size`, `idx_top_k`), copies the attention widths, and when `!has_indexer()` zeroes
all three indexer widths **together** (`idx_n_head`, `idx_dim`, `idx_top_k`). The header records why: five call
sites used to build it by hand, and copying only two zeros produced a shape that was neither sparse nor dense, which
the kernels' guard refused at the first KV append. Callers: `layer.cpp`, `session.cpp`, `verify.cpp`, `mtp.cpp`,
`src/prefill/prefill.cpp`.

## Control flow and callers

* `src/program/generate.cpp` reads the geometry from the first `--native` shard (`read_model_geometry`, around
  line 2091), prints it, refuses `--kv-resident` for dense attention (`qsa_kv_streaming_applies`), runs
  `check_model_tensor_forms`, and for Qwen3.6 feeds it to the VRAM plan ([vram-plan.md](vram-plan.md)).
* Without `--native` the default (Flash-Next) geometry is used.
* Everything downstream (`layout.cpp` checks, session sizing, layer kernels, verify, MTP) takes `const ModelGeometry&`.

## Model-specific behaviour

| Decision | Flash-Next | Qwen3.6 | Where it is branched |
| --- | --- | --- | --- |
| Residual | GR read/write over 4 streams | plain `rms_norm * w` and `R += out` | `g.hc == 1` in `block_layer_pre/post`, `lm_head_mix`, verify, MTP |
| Attention | sparse QSA (indexer, block top-k) | dense split-K over every cell | `g.has_indexer()` in `qsa_layer`, `layout.cpp`, `qsa_shapes` |
| GDN gate | sigmoid | SiLU | `g.gdn_gate()` |
| PLE | layer 1 | none; a wired PLE is refused | `g.has_ple`, `PleRun::ready()` |
| KV streaming | allowed | refused (dense attention reads every cell) | `qsa_kv_streaming_applies` |

## Configuration

* `--native SHARD1` selects the GGUF the geometry is read from (generate.cpp).
* No STRATA_* variable affects the geometry.

## Tests

* `model_geometry_test` (`tests/core/model_geometry_test.cpp`): with a directory argument it reads the synthetic
  `qwen35moe` GGUFs written by `tools/qwen36_synth_gguf.py` (CTest fixture `qwen36_synth_fixtures`, needs python3
  with numpy; skip code 3) and checks every field of the toy geometry, the nextn subtraction, both fused-form
  refusals by tensor name, a `qwen4exp` file keeping `hc = 4` and PLE, an unknown architecture refused by name, a
  contradicting `ssm.inner_size`, and the vocabulary cross-check. `--real FILE` prints a real file's geometry and,
  for Qwen3.6-35B-A3B (`n_embd == 2048`), asserts the values of `docs/qwen36-port/gguf-metadata.md`.
  CTest: `ctest -R model_geometry_test`.

## Gotchas

* The struct's defaults are Flash-Next's, so a code path that builds a `ModelGeometry{}` by hand silently means
  Flash-Next (`layer_verify_compatible(std::string&)` does this on purpose, `layer.cpp:489`).
* `ssm_conv_channels` and `ssm_value_dim` are stored, not computed on demand; anything that changes the GDN head
  counts must call `derive()`.
* `idx_q_heads` counts **query** heads; the cached indexer key is one head of `idx_key_dim`. Confusing the two made
  an indexer budget 4x too big upstream (`layout.hpp` header comment).

## See also

* [weights-and-layout.md](weights-and-layout.md) - the per-layer checks that use this geometry.
* [layer-forward.md](layer-forward.md) - where each branch is taken.
* [docs/qwen36-port/geometry-inventory.md](../../qwen36-port/geometry-inventory.md), [step2-notes.md](../../qwen36-port/step2-notes.md), [gguf-metadata.md](../../qwen36-port/gguf-metadata.md).
* [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) section 3.8 (the key set).
