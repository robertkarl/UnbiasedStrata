[Codebase docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Tests

# CPU kernel and pool tests

## Purpose

The tests that live beside the CPU kernels in `src/kernels/cpu/`. They need no GPU. The GPU-side and cross-device
parity tests (including `native_expert_parity`, which also checks the CPU native path) are in
[../parity/README.md](../parity/README.md).

## Files

| File | Lines | Role |
|---|---|---|
| [src/kernels/cpu/expert_parity.cpp](../../../../src/kernels/cpu/expert_parity.cpp) | 184 | VNNI Q2_0 expert vs the scalar oracle on a real pack blob |
| [src/kernels/cpu/expert_multi_test.cpp](../../../../src/kernels/cpu/expert_multi_test.cpp) | 87 | Multi-token Q2_0 expert bitwise vs single-token; optional bench |
| [src/kernels/cpu/pool_test.cpp](../../../../src/kernels/cpu/pool_test.cpp) | 224 | Pool correctness (serial equality, exactly-once, repeated batches) and a layer timing |
| [src/kernels/cpu/pool_stress.cpp](../../../../src/kernels/cpu/pool_stress.cpp) | 99 | Issue #29 stress: tiny batches with sleeping workers |
| [src/kernels/cpu/pool_affinity_linux_test.cpp](../../../../src/kernels/cpu/pool_affinity_linux_test.cpp) | 290 | Linux affinity save / pin / restore |
| [src/kernels/cpu/pool_affinity_test.cpp](../../../../src/kernels/cpu/pool_affinity_test.cpp) | 148 | Windows CPU Set host pinning round trips |
| [src/kernels/cpu/router_dot_parity.cpp](../../../../src/kernels/cpu/router_dot_parity.cpp) | 70 | BF16 router dot (AVX2, AVX1) vs double |

## Test table

All are registered in the top-level `CMakeLists.txt` near the end (the `if(STRATA_BUILD_TESTS)` block after the
native-experts section). `STRATA_BUILD_TESTS` defaults to OFF in this checkout because `tests/CMakeLists.txt` and
`bench/micro/` are absent (`CMakeLists.txt:86-92`), so pass `-DSTRATA_BUILD_TESTS=ON`.

| CTest name (target) | Condition | What it checks | Reference | Inputs |
|---|---|---|---|---|
| `expert_parity` (`expert_parity --selftest`, working dir = source root) | `STRATA_BUILD_TESTS` | (1) `s2_expert_vnni` vs `s2_expert_scalar(quant_acts=true)`: rel L1 <= 3e-3 (P2.S3); (2) vs exact FP32 activations: reported; the activation contract must be observable; the blob decodes to live scales and the next expert is different data | `s2_expert_scalar`, a transcription of ggml's formula | **Required**: `pack/full/experts.bin` (a canonical Flash-Next pack); exit 2 if unreadable. Skips (exit 0, prints SKIPPED) on a CPU without AVX-512 VNNI/VBMI |
| `expert_multi_test` | `STRATA_BUILD_TESTS` | `s2_expert_vnni_multi` bitwise equal to `s2_expert_vnni_q` per token, n = 1..8; `--bench [E]` times the extra-token cost | The single-token kernel | Synthetic blobs |
| `pool_test` (`pool_test --selftest`, working dir = source root) | `STRATA_BUILD_TESTS` | Every job bitwise equal to the serial run; every job exactly once (sentinel outputs); 200 consecutive batches; batch sizes 1, w-1, w, w+1, 10; then a timing of 10 experts (GB/s) | Serial `s2_expert_vnni_q` | **Required**: `pack/full/experts.bin` (`--file`, `--layer`); skips on a CPU without the VNNI features |
| `pool_stress` (`pool_stress 20`) | `STRATA_BUILD_TESTS` | 20 s of random 0-4-job batches through `run` and `run_split`, with 21-30 ms gaps so workers sleep; every output written; a watchdog exits 3 if a batch does not return within 10 s | Sentinel check | Synthetic (zero blobs); skips without VNNI |
| `pool_affinity_test` | `STRATA_BUILD_TESTS` and Linux (`pool_affinity_linux_test.cpp`) or Windows (`pool_affinity_test.cpp`) | Linux: full-mask, sparse/nested and high-CPU-id round trips of `pin_current_thread` / `restore_thread_affinity`, worker pin, negative CPU is a no-op, failed pin keeps the mask; `--require-high-cpu` makes the high-id case mandatory. Windows: CPU Set selection round trips, inherited defaults | The OS's own affinity query | None |
| `router_dot_parity`, `router_dot_parity_avx` (`STRATA_FORCE_ISA=avx`) | `STRATA_NATIVE_EXPERTS` builds the target; ctests need `STRATA_BUILD_TESTS` | `bf16_rows_dot_multi` (AVX2) and `bf16_rows_dot_multi_avx1` (AVX) at 512 x 2560, 1..8 tokens: rel L1 < 1e-5; prints one layer's time | Double-precision dot | Synthetic BF16 rows |

Related tests outside this directory: `tests/core/pool_native_widths_test.cpp` (the pool's native path at Qwen3.6's
2048 x 512) and `tests/core/expert_layout_test.cpp` (the `native_experts.txt` reader over
`tests/data/native_experts/*.txt`); both are registered in the GPU block of `CMakeLists.txt`
(`STRATA_BUILD_TESTS AND STRATA_NATIVE_EXPERTS`, inside `STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP`).

## How to run

```
cmake -B build -DSTRATA_BUILD_TESTS=ON [-DSTRATA_ENABLE_CUDA=ON]
cmake --build build --target expert_multi_test pool_stress pool_affinity_test router_dot_parity
ctest --test-dir build -R 'expert_multi_test|pool_stress|pool_affinity_test|router_dot_parity'
```

`expert_parity` and `pool_test` additionally need a canonical pack at `pack/full/` in the source tree; neither is
shipped in this checkout. `STRATA_NATIVE_EXPERTS=ON` (the default) requires `third_party/llama.cpp` to be checked
out, since the whole configure step builds ggml from it.

## Gotchas

- A missing pack is deliberately a failure (exit 2), not a skip: "a missing file would otherwise be reported as a pass
  by a test that skipped everything" (`CMakeLists.txt`, comment above `expert_parity`).
- A CPU without AVX-512 VNNI/VBMI makes `expert_parity`, `pool_test` and `pool_stress` print SKIPPED and exit 0,
  which CTest reports as passed.
- The Linux affinity test needs `_GNU_SOURCE` and compiles to a SKIP stub on non-Linux.

## See also

- [q2-expert.md](q2-expert.md), [pool.md](pool.md), [native-experts.md](native-experts.md)
- [../parity/README.md](../parity/README.md)
