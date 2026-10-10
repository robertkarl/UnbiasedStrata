# Native upgrade validation

Validated on 2026-10-10 against upstream main
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` (0.1.42), plus the ports listed
in [UPGRADE.md](UPGRADE.md). These are source-build and fixture results, not a
full-model performance campaign or a security certification.

## Fresh integrated results

| Configuration | Result |
| --- | --- |
| Linux CPU, Ryzen 5 3600, GCC 13.3, CMake 3.28.3 | Source build completed; 42 tests passed, one AVX-512 test skipped. |
| Linux HIP, RX 5500 XT / gfx1012, HIP 5.7.1, Clang 17, CMake 3.28.3 | Native source build completed; 99 tests passed, four skipped. One external-model PLE test was excluded after documenting its missing fixture. gfx1012 remains experimental; this does not expand the fork's supported deployment matrix. |
| Linux CUDA, GTX 1050 Ti / SM61, CUDA 12.6.85, GCC 13.3, CMake 3.28.3 | Engine and original C++ frontend source builds completed. 115 distinct tests passed, four skipped; PLE and hybrid prompt-attention checks remain gaps. This is an experimental Pascal build, not the normal supported deployment. |
| CUDA SM86, portable AVX2 build, original C++ frontend | Source build completed with CUDA 12.6.85, GCC 13.3 and `STRATA_PORTABLE=ON`. No SM86 GPU runtime test was performed. |
| IQ CPU arithmetic | Default dispatch, forced ggml fallback, AVX-VNNI disabled, and gather mode each reported zero failures, including rotated groups of 1..8 tokens. |
| Frontend startup surface | The CUDA build's `unbiased-strata --help` completed, and `unbiased-strata --engine --version` reported `strata 0.1.42`. No model was loaded for these checks. |

The [per-test results](upgrade-evidence/test-results.json) retain the initial
CUDA failures as well as the successful rerun of the three loader-affected
fixtures. CUDA coverage combines those runs; it is not described as an
unqualified passing full suite. The HIP result is the final run with only the
absent-model PLE fixture excluded.

The [source SHA-256 manifest](upgrade-evidence/source-sha256.json) covers 407
native source, frontend, build, and fixture files. Both remote source trees were
hashed and compared with the local source; every listed file matched. The
llama.cpp dependency used the pinned `3cf03257f219afbe7334045ff7c6a06ac68c627d`
source. Binary hashes, when recorded, identify the tested build and do not
attest that another toolchain produces an identical executable.

## Build artifact identity and offline check

| Local build artifact | SHA-256 |
| --- | --- |
| CUDA SM61 C++ frontend | `43a50bdf896a861a303e1a4a84293ce0061b343f72f919984b43763fc41afc52` |
| CUDA SM86 portable C++ frontend | `b7872b1af5d7424b624988d892ed7b2f875b42e8081536d0e9543962587d80b7` |
| HIP gfx1012 native engine | `7fc1ffd1976170052beab1120183a2b018d40174eace9a4dd38e8f8d43888708` |

A fresh CPU configuration and an incremental SM86 build were traced with
`strace -f -e trace=network`; neither trace contained an Internet-address
connection/send operation. The incremental build had no compilation work left.
This verifies those traced commands, not every possible configuration or runtime
path. The project's dependency acquisition remains explicit, and the build has
no FetchContent/download command.

## Failures resolved and remaining limits

The fresh builds caught an obsolete device variable in a secondary-refill block
and a missing HIP invalid-value alias. Both were corrected before the final
builds. The initial HIP test run also revealed missing synthetic IQ fixture
helpers; those local, read-only/synthetic test dependencies were restored.

Three CUDA fixtures initially could not load `libcudart.so.12` from the separately
provisioned test toolchain. Setting its library directory in `LD_LIBRARY_PATH`
allowed `dma_batch_parity`, `gdn_rec_parity`, and
`secondary_refill_lifetime_fixture` to pass. The executable's normal build and
its dependencies still require explicit provisioning.

`ple_parity` requires an external Qwen model shard that was absent on both test
hosts; its failure is recorded rather than counted as a pass. On experimental
Pascal, `kv_hybrid_parity` passed its append, gather, batched-attention, and
decode-output checks, but prompt attention refused the hybrid pools. That
whole test remains failed/unvalidated. Its source was not changed to weaken
the gate. The PLE gap and this Pascal limitation must not be presented as
successful checks.

The skipped HIP fixtures require unavailable architecture/features (WMMA prompt
paths or optional exact HCD support), or AVX-512. The skipped CUDA fixtures
include the SM120 dense-T8 path, Ampere-only fused prefill paths, and AVX-512.
Those skips are not passes.

No full-model HTTP chat, saved-cache replay, cross-version token equivalence,
SM120 runtime fixture, multi-GPU accepted-token-barrier campaign, or active-layer
prefill throughput campaign was repeated for this branch. Earlier PR results
are useful background, but do not certify this integrated build or establish
a measured speedup over UnbiasedStrata 0.1.39. Windows and SYCL serving support
are not added. The C++ frontend, Makefile, dependency pin, and default model ID
were compared with the original UnbiasedStrata main and are unchanged.

## Reproduce

Provision compiler/GPU SDK dependencies and the pinned submodule explicitly.
The normal supported build remains `make build`. For a supported-architecture
CUDA source check using one installed target architecture:

```sh
cmake -S . -B build-cuda -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=ON
cmake --build build-cuda -j 6
ctest --test-dir build-cuda --output-on-failure
```

SM61 checks additionally used `-DSTRATA_EXPERIMENTAL_SM60=ON`; they do not
represent supported Ampere hardware. HIP checks used
`-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1012`. With Debian's
HIP layout, the test host also needed
`-DCMAKE_HIP_COMPILER_ROCM_LIB=/usr/lib/x86_64-linux-gnu` and an explicit
`--rocm-device-lib-path` for its installed Clang 17 device libraries. These are
test-environment flags, not hard-coded project paths.

Native IQ fixture generation uses optional Python/NumPy and the local pinned
gguf-py source. Neither CMake nor the fixture generator downloads a dependency
or model. No test helper participates in normal startup or the HTTP request path.
