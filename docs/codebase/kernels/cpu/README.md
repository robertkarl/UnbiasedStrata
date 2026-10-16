[Codebase docs](../../README.md) › [Kernels](../README.md) › CPU

# CPU kernels: routed experts, the worker pool, and the PLE table

The CPU half of the engine computes every routed MoE expert the GPU does not hold in VRAM, from expert bytes in host
RAM, **at the same time** as the GPU computes the resident ones. It lives in its own static library,
`strata_kernels_cpu` (built for every configuration, with or without a GPU backend), which contains the Q2_0 expert
kernel for canonical Strata packs, the ggml-cpu based native-expert path (plus hand-written multi-token AVX-512 / AVX2
kernels) for GGUF-form packs, the per-layer expert layout table, the CPU feature probes, and `ExpertPool`, the
pinned worker-thread pool that runs them. Its callers are `src/core/expert_source.cpp` (the per-layer dispatch,
driven by the session's layer loop in `src/core/session.cpp` and the verify window in `src/core/verify.cpp`) and
`src/program/generate.cpp` (construction, CPU checks, layout load). It depends only on ggml-cpu / ggml-base (native
experts), the C++ thread library and two header-only hooks in `strata/core/progress.hpp`. The n-gram / PLE table
reader (`src/kernels/ngram.cpp`) is documented here too, although it builds into the GPU library `strata_kernels`.

## Contents

| Page | Covers |
|---|---|
| [q2-expert.md](q2-expert.md) | The Q2_0 expert kernel (`expert.cpp`, `q2_avx2.cpp`): VNNI row dot, activation quantizer, multi-token and row-range variants, the oracle contract, the scalar reference |
| [native-experts.md](native-experts.md) | Native GGUF-form experts through ggml-cpu (`native_expert.cpp`) and the multi-token kernels `iq_avx512`, `iq_avx2`, `kq_avx2`, `kq_avx1` (the last also the router-prefetch dot) |
| [expert-layout.md](expert-layout.md) | `native_experts.txt` and `ExpertLayout`; the ISA probes `cpu_avx512_ok` / `cpu_avx2_ok` / ... and `STRATA_FORCE_*` |
| [pool.md](pool.md) | `ExpertPool`: topology and pinning, the epoch-tagged claim protocol, run modes, and how the CPU overlaps the GPU per layer |
| [tests.md](tests.md) | `expert_parity`, `expert_multi_test`, `pool_test`, `pool_stress`, `pool_affinity_test`, `router_dot_parity` |
| [ngram.md](ngram.md) | The PLE n-gram hash and `PleTable` (`ngram.cpp`), `--ple-io direct\|mmap\|ram` |

## Source map

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/expert.hpp](../../../../include/strata/kernels/cpu/expert.hpp) | 179 | Q2_0 expert geometry, `ActQ`, scratch, entry points |
| [src/kernels/cpu/expert.cpp](../../../../src/kernels/cpu/expert.cpp) | 683 | AVX-512 VNNI/VBMI Q2_0 expert, quantizers, oracle contract, scalar reference |
| [src/kernels/cpu/q2_avx2.cpp](../../../../src/kernels/cpu/q2_avx2.cpp) | 130 | Q2_0 GGUF-layout rows and quantizer for AVX2-only CPUs |
| [include/strata/kernels/cpu/native_expert.hpp](../../../../include/strata/kernels/cpu/native_expert.hpp) | 49 | `NativeFmt` and native row API |
| [src/kernels/cpu/native_expert.cpp](../../../../src/kernels/cpu/native_expert.cpp) | 171 | ggml-cpu traits, `native_fmt`, kernel dispatch |
| [include/strata/kernels/cpu/iq_avx512.hpp](../../../../include/strata/kernels/cpu/iq_avx512.hpp) | 18 | AVX-512 i-quant API |
| [src/kernels/cpu/iq_avx512.cpp](../../../../src/kernels/cpu/iq_avx512.cpp) | 273 | Multi-token IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ2_S rows, 512-bit |
| [include/strata/kernels/cpu/iq_avx2.hpp](../../../../include/strata/kernels/cpu/iq_avx2.hpp) | 26 | AVX2 i-quant API |
| [src/kernels/cpu/iq_avx2.cpp](../../../../src/kernels/cpu/iq_avx2.cpp) | 493 | Multi-token i-quant rows (+IQ4_XS) and IQ4_NL down rows, 256-bit |
| [include/strata/kernels/cpu/kq_avx2.hpp](../../../../include/strata/kernels/cpu/kq_avx2.hpp) | 25 | K-quant and BF16 router API |
| [src/kernels/cpu/kq_avx2.cpp](../../../../src/kernels/cpu/kq_avx2.cpp) | 265 | Q4_K / Q5_1 / Q8_0 multi-token rows bit-exact vs ggml AVX2; BF16 router dot |
| [include/strata/kernels/cpu/kq_avx1.hpp](../../../../include/strata/kernels/cpu/kq_avx1.hpp) | 18 | AVX1 router dot API |
| [src/kernels/cpu/kq_avx1.cpp](../../../../src/kernels/cpu/kq_avx1.cpp) | 71 | BF16 router dot for AVX-only CPUs |
| [include/strata/kernels/cpu/expert_layout.hpp](../../../../include/strata/kernels/cpu/expert_layout.hpp) | 87 | `ExpertLayout`, ISA probes |
| [src/kernels/cpu/expert_layout.cpp](../../../../src/kernels/cpu/expert_layout.cpp) | 349 | Probes, dispatchers, `native_experts.txt` parser, no-ggml stubs |
| [include/strata/kernels/cpu/pool.hpp](../../../../include/strata/kernels/cpu/pool.hpp) | 286 | `ExpertPool` and its job / topology / affinity types |
| [src/kernels/cpu/pool.cpp](../../../../src/kernels/cpu/pool.cpp) | 783 | Topology, pinning, worker loop, claim protocol, run modes |
| [src/kernels/cpu/pool_affinity_linux.hpp](../../../../src/kernels/cpu/pool_affinity_linux.hpp) | 69 | Linux affinity helpers (`CPU_ALLOC` masks) |
| [src/kernels/cpu/pool_affinity_win.hpp](../../../../src/kernels/cpu/pool_affinity_win.hpp) | 64 | Windows group affinity and CPU Set helpers |
| [src/kernels/cpu/expert_parity.cpp](../../../../src/kernels/cpu/expert_parity.cpp) | 184 | Test: VNNI vs scalar oracle on a real blob |
| [src/kernels/cpu/expert_multi_test.cpp](../../../../src/kernels/cpu/expert_multi_test.cpp) | 87 | Test: multi-token bitwise vs single-token; bench |
| [src/kernels/cpu/pool_test.cpp](../../../../src/kernels/cpu/pool_test.cpp) | 224 | Test: pool correctness and timing |
| [src/kernels/cpu/pool_stress.cpp](../../../../src/kernels/cpu/pool_stress.cpp) | 99 | Test: issue #29 sleep/wake stress |
| [src/kernels/cpu/pool_affinity_linux_test.cpp](../../../../src/kernels/cpu/pool_affinity_linux_test.cpp) | 290 | Test: Linux affinity round trips |
| [src/kernels/cpu/pool_affinity_test.cpp](../../../../src/kernels/cpu/pool_affinity_test.cpp) | 148 | Test: Windows CPU Set round trips |
| [src/kernels/cpu/router_dot_parity.cpp](../../../../src/kernels/cpu/router_dot_parity.cpp) | 70 | Test: router dot vs double |
| [include/strata/kernels/ngram.hpp](../../../../include/strata/kernels/ngram.hpp) | 216 | PLE hash, dequantizers, `PleTable`, I/O options |
| [src/kernels/ngram.cpp](../../../../src/kernels/ngram.cpp) | 521 | PLE hash and table reader (in `strata_kernels`) |

## Key concepts

### Two expert forms

| | Canonical Q2_0 pack | Native pack |
|---|---|---|
| Written by | `tools/strata_pack.py` | `tools/iq_pack.py` (writes `native_experts.txt`) |
| Blob | Fixed 1,382,400 B planar Strata form (2560 x 640 baked in) | Raw GGUF slices `[gate \| up \| down]`, size per layer |
| Models | Flash-Next only | Flash-Next IQ packs and every Qwen3.6-35B-A3B pack |
| CPU arithmetic | `s2_expert_*` (AVX-512 VNNI+VBMI required) | ggml-cpu `vec_dot`, or Strata multi-token kernels for i-quants / Q2_0 / IQ4_NL / opt-in K-quants (AVX2 floor) |
| Pool entry | `run`, `run_split`, `run_split_multi` | `run_split_multi_native` |

### Per-layer overlap with the GPU

```
          GPU                                    CPU (host thread + pool workers)
  pre[l]: mixer, router, top-k
     | doorbell ring (x_f, ids, weights in mapped pinned memory)
     v                                       host sees the ring
  hit experts from the VRAM cache  <-------  enqueues the GPU's share, does not wait
     |                                       expert_pool_dispatch: quantize x_f once,
     |                                       one job per missed expert -> ExpertPool
     |                                       (all cores, host included) -> y_miss
     |                               <-------  H2D copy of y_miss
  post[l]: moe_combine (router weights applied here), residual
  pre[l+1] ...
```

In a speculative verify window the same happens per layer for up to 8 tokens: each distinct missed expert is one
multi-token job whose weight rows are read once for all the tokens routed to it. Details: [pool.md](pool.md).

### ISA tiers

| Probe | Requires | Unlocks |
|---|---|---|
| `cpu_avx512_ok()` | AVX512 F, BW, VL, VNNI, VBMI + OS ZMM state | Canonical packs; `iq_avx512`; AVX-512 Q2_0 rows |
| `cpu_avx2_ok()` | AVX, AVX2, FMA, F16C + OS YMM state | Native packs' Strata kernels (`iq_avx2`, `q2_avx2`, `kq_avx2`) |
| `cpu_avx1_ok()` | AVX | AVX1 router dot; ggml-cpu in a `STRATA_ISA_FLOOR=avx` build |
| `cpu_sse42_ok()` | SSE4.2 + POPCNT | ggml-cpu in a `STRATA_ISA_FLOOR=none` build |

## Build

`CMakeLists.txt` (the `strata_kernels_cpu` and "native experts" blocks after the GPU section):

- `strata_kernels_cpu` = `expert.cpp pool.cpp expert_layout.cpp q2_avx2.cpp`, plus, with `STRATA_NATIVE_EXPERTS=ON`
  (default), `native_expert.cpp iq_avx512.cpp iq_avx2.cpp kq_avx2.cpp kq_avx1.cpp`, linking `ggml-cpu ggml-base` and
  defining `STRATA_NATIVE_EXPERTS=1` publicly.
- ISA flags are per source file, never per target: `expert.cpp` and `iq_avx512.cpp` get AVX-512 (F/BW/VL/DQ/VNNI/VBMI
  + F16C + FMA), `q2_avx2.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp` get AVX2 + FMA + F16C, `kq_avx1.cpp` gets AVX; `pool.cpp`
  and `expert_layout.cpp` (the probes) get none, so they run on any x86-64.
- ggml comes from `third_party/llama.cpp` (or `-DSTRATA_GGML_DIR`); configure fails if it is missing.
  `STRATA_PORTABLE` builds ggml-cpu for an AVX2 baseline; `STRATA_ISA_FLOOR=avx|none` for older CPUs.

## Configuration summary

CLI (`src/program/generate.cpp`): `--pool-workers`, `--pool-affinity`, `--no-host-worker`, `--no-split-rows`,
`--cpu-oracle-q8-0`, `--ple-io` and the other `--ple-*` flags. Environment: `STRATA_CPU_YMM`, `STRATA_IQ_MT_MIN`,
`STRATA_NO_IQ512`, `STRATA_NO_IQ256`, `STRATA_NO_IQ4NL`, `STRATA_KQ256`, `STRATA_IQ_PREFETCH`, `STRATA_IQ_GATHER`,
`STRATA_IQ256_GATHER`, `STRATA_FORCE_AVX2`, `STRATA_FORCE_ISA`, `STRATA_POOL_SPIN_US`, `STRATA_SSD_KEEPALIVE*`. Each
is described on the page that owns it.

## See also

- [../README.md](../README.md) (all kernels), [../parity/README.md](../parity/README.md) (GPU and cross-device tests)
- [../../core/experts/dispatch.md](../../core/experts/dispatch.md) (who computes each routed expert),
  [../../core/session.md](../../core/session.md) (the layer loop and doorbell)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md), [docs/qwen36-port/step3-experts.md](../../../qwen36-port/step3-experts.md)
