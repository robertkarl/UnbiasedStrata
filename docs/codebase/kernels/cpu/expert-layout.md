[Codebase docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Expert layout and CPU probes

# Expert layout table and CPU feature probes

## Purpose

`expert_layout.cpp` holds two process-wide things every CPU expert path asks about:

1. **The expert layout** - where each routed expert's blob lives in `experts.bin` (or in the GGUF shards), how big it
   is, and in which formats, per layer. A canonical Q2_0 pack has one fixed blob size; a native pack has a per-layer
   table in `native_experts.txt`. "Everything that touches an expert blob - the arena, the VRAM tier, the prompt path,
   the CPU pool, the GPU window - asks this table" (`include/strata/kernels/cpu/expert_layout.hpp:1-6`).
2. **The CPU ISA probes** (`cpu_avx512_ok`, `cpu_avx2_ok`, ...), compiled **without** AVX flags so asking is safe on
   any x86-64, plus the `q2_rows_any` / `act_quant_any` dispatchers.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/expert_layout.hpp](../../../../include/strata/kernels/cpu/expert_layout.hpp) | 87 | `ExpertLayout`, probe declarations, `kExpertLayoutVersion` |
| [src/kernels/cpu/expert_layout.cpp](../../../../src/kernels/cpu/expert_layout.cpp) | 349 | Probes, dispatchers, `native_experts.txt` parser, no-ggml stubs |

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `ExpertLayout` | `expert_layout.hpp:18` | `native`, `n_layers`, `n_expert`; per layer `fmt` (`NativeFmt`), `offset`, `bytes`; per layer and role `gguf_off` and `gguf_file` (3 x n_layers); `version`, `max_blob`, `total`. Accessors `blob_bytes`, `layer_offset`, `blob_offset` |
| `expert_layout()` | `expert_layout.cpp:22` | The global (canonical Q2_0 until a native table is loaded) |
| `expert_layout_load(pack_dir, n_layers, n_expert, n_embd, n_ff, err)` | `expert_layout.cpp:233` | Parses `<pack_dir>/native_experts.txt`, or sets the canonical layout when the file is absent |
| `kExpertLayoutVersion = 4` | `expert_layout.hpp:85` | Newest table version read; a newer header is refused |
| `cpu_avx512_ok()` | `expert_layout.cpp:40` | F + BW + VL + VNNI + VBMI and OS XSAVE support for ZMM state (`xcr0 & 0xE6`). `STRATA_FORCE_AVX2=1` or `STRATA_FORCE_ISA` < 3 answers no |
| `cpu_avx512bw_ok()` | `expert_layout.cpp:73` | F + DQ + BW + VL (no VNNI/VBMI). The header says it is for `iq_avx512.cpp`, but no call site exists outside `expert_layout.cpp` |
| `cpu_avx2_ok()` | `expert_layout.cpp:105` | AVX + AVX2 + FMA + F16C with OS YMM support: the floor of every Strata expert kernel |
| `cpu_avx1_ok()`, `cpu_sse42_ok()` | `expert_layout.cpp:138`, `:164` | Floors of the experimental older-CPU builds |
| `cpu_isa_cap()` | `expert_layout.cpp:24` | `STRATA_FORCE_ISA=avx2\|avx\|sse` caps the probes (3 = no cap); an unknown value is ignored with a message |
| `isa_floor_build()` | `expert_layout.cpp:179` | `"avx"`, `"sse4.2"` or `""` from `STRATA_ISA_FLOOR_AVX` / `_NONE` |
| `q2_native_kernels(type)` | `expert_layout.hpp:65` | `type == 42 && cpu_avx2_ok()`: a native layer of Q2_0 runs on Strata's Q2_0 kernels |
| `cpu_name()` | `expert_layout.cpp:189` | CPUID brand string, for messages |
| `q2_rows_any`, `act_quant_any` | `expert_layout.cpp:212`, `:218` | AVX-512 or AVX2 variant of the Q2_0 rows / quantizer ([q2-expert.md](q2-expert.md)) |

All probes cache their answer in a function-local `static`, so environment overrides must be set before the first call.

## `native_experts.txt`

Written by `tools/iq_pack.py`. One header line and one line per layer:

```
# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert 256, ...)
0 12 13 0 1900544 1304098592 1458312992 1118435104
```

Parsing rules (`expert_layout.cpp:233-347`):

- No file: canonical Q2_0 layout, `total = n_layers * n_expert * BLOB`. Refused when the model's widths are not
  2560 x 640, because the canonical blob bakes them in ("pack it with tools/iq_pack.py").
- A file but a build without `STRATA_NATIVE_EXPERTS`: refused.
- `# strata native experts vN`: `N > 4` is refused (a newer packer). `(n_expert K)` in a header overrides the caller's
  expert count (pruned models such as the GSQ-RCO Coder ship fewer experts).
- Each layer line: `native_fmt(gu, d, n_embd, n_ff)` must succeed and its byte count must equal `blob_bytes`, so a pack
  of another geometry is refused by its numbers. `n_embd` / `n_ff` come from the model file (`ModelGeometry`), not
  the compile-time constants.
- v2 adds the three absolute GGUF offsets; v3 a shard file name; v4 allows `gate,up,down` (three names, an empty
  field meaning the `--native` shard) when a shard boundary falls inside a layer (Unsloth UD-Q4_K_XL, #255). One
  name is replicated to all three roles; two or four names, or a trailing column, are refused.
- Layers must be present and contiguous: `offset[l]` must equal the running sum of `bytes * n_expert`.

## Callers and dependencies

- `src/program/generate.cpp:2122` calls `expert_layout_load(o.pack, g.n_layers, g.n_expert, g.n_embd, g.n_ff, err)`.
- `src/core/expert_source.cpp` reads `expert_layout()` for blob addressing, `gguf_off` / `gguf_file` (filling the arena
  from GGUF shards) and the verify window's native dispatch (`expert_source.cpp:2029-2030`).
- [pool.md](pool.md) uses `q2_native_kernels`, `q2_rows_any`, `act_quant_any`.
- `src/program/generate.cpp:2382-2388` uses `cpu_avx512_ok` / `cpu_avx2_ok` to report which kernels a native pack will
  run on, and refuses canonical packs on non-AVX-512 CPUs through `cpu_require_expert_support`.
- Depends on `native_fmt` ([native-experts.md](native-experts.md)) and on CPUID / `xgetbv`.

## Configuration

| Knob | Effect |
|---|---|
| `STRATA_FORCE_AVX2=1` | `cpu_avx512_ok()` and `cpu_avx512bw_ok()` return false (tests run the AVX2 kernels on an AVX-512 CPU) |
| `STRATA_FORCE_ISA=avx2\|avx\|sse` | Caps `cpu_avx512_ok` / `cpu_avx2_ok` / `cpu_avx1_ok`; ggml-cpu is not affected (it runs what it was compiled for) |
| CMake `STRATA_ISA_FLOOR` | `avx` or `none`: defines `STRATA_ISA_FLOOR_AVX` / `STRATA_ISA_FLOOR_NONE` on `strata_kernels_cpu` |
| CMake `STRATA_NATIVE_EXPERTS` | Off: native tables are refused and the native entry points are stubs |

## Model-specific behaviour

Flash-Next canonical packs have no table (canonical layout). Every Qwen3.6-35B-A3B pack is native (2048 x 512 is
refused for the canonical form since Qwen3.6 port step 2, `expert_layout.hpp:77-80`). Examples of both models'
tables are in [tests/data/native_experts/](../../../../tests/data/native_experts/).

## Tests

- `tests/core/expert_layout_test.cpp` (ctest `expert_layout_test`, needs `STRATA_BUILD_TESTS`, a GPU backend and
  `STRATA_NATIVE_EXPERTS`): parses the eight real tables in `tests/data/native_experts/` (Q2_0, IQ2_XS, IQ3_XXS,
  IQ3_S, Coder IQ1_M, Swift IQ2_XS, Qwen3.6 IQ3_XXS and Q4_K_XL) against the pre-v4 reader's rules, the v4 shard
  column and its refusals, and fills a synthetic arena from GGUF shards.
- `router_dot_parity_avx` runs `router_dot_parity` under `STRATA_FORCE_ISA=avx`.

## Gotchas

- The probes are `static`-cached: setting `STRATA_FORCE_*` after the first call has no effect.
- `v4` is written only when some layer's roles span shards; every other pack stays v3 byte for byte
  (`expert_layout.hpp:82-84`).
- The v4 comma split copies the single name before `assign` to avoid aliasing (`expert_layout.cpp:317-320`).

## See also

- [native-experts.md](native-experts.md), [q2-expert.md](q2-expert.md), [pool.md](pool.md)
- [docs/qwen36-port/step4-pack.md](../../../qwen36-port/step4-pack.md), [docs/qwen36-port/geometry-inventory.md](../../../qwen36-port/geometry-inventory.md)
