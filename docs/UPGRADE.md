# Native engine upgrade from 0.1.39

This branch updates UnbiasedStrata's native engine from Strata commit
`6f32ec070f23ced9f50e704d854d775da52591ab` (0.1.39) to
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` (0.1.42), then applies the
performance and correctness ports listed below. The baseline import is separate
from the port commits so future upstream updates can review those changes independently.

The existing C++ HTTP frontend, default model ID `qwen3.8-flash-next-iq3_xxs`,
loopback bind, Makefile, and pinned llama.cpp submodule remain unchanged. This
upgrade adds no Python serving layer, updater, installer, automatic dependency
download, web app, persona/profile framework, or Codi saved-prefix files.
Upstream's normal engine caches and explicit local file operations are retained.
Dependencies and model artifacts must still be provisioned before building or starting.

## Imported main changes

The import includes native changes since 0.1.39, including CPU ISA/dispatch
work, speculative verification, multi-stage pipelining, prefill kernels,
RAM admission and shared-buffer lifetime fixes, per-stage resident/lend sizing,
and the newer single-GPU PCIe share sizing. It imports `src/`, `include/`, native
build files, and native test fixtures, rather than reintroducing upstream's
Python application, setup scripts, or SYCL frontend.

Upstream's 0.1.42 defaults are included. In particular, CUDA serving enables
route-tail skipping where upstream considers it eligible
(`STRATA_ROUTE_TAIL_SKIP=0` opts out), and refines the default single-GPU PCIe
share when no explicit `--pcie-frac` is supplied
(`STRATA_PCIE_FRAC_DEFAULT=old` retains the older rule). This is a version
upgrade, not a claim of identical output or performance to 0.1.39.

## Additional engine ports

These ports come from the independently reviewed local integration, with their
original Strata source commits recorded in Git. The additional experimental
performance paths remain opt-in. Historical PR measurements apply to those
PRs' tested configurations; they are not speedup measurements for this branch.

| Strata PR | Included change | Scope |
| --- | --- | --- |
| [#1154](https://github.com/Niko1221/Strata/pull/1154) | Per-device prefill helper shares; limit committed speculative tokens to emitted outputs, EOS and request budget | Retains main's newer arbitrary-stage pipeline instead of restoring the older scheduler. |
| [#1674](https://github.com/Niko1221/Strata/pull/1674) | One-token pipelined verifier self-commit guard | Already present in the imported main; no duplicate patch. |
| [#1336](https://github.com/Niko1221/Strata/pull/1336) | Compact read-only Q8 miss fills | CUDA, whole-model uniform Q8_0 verifier; `STRATA_Q8_MISS_CACHE_WAYS` defaults to zero. |
| [#1338](https://github.com/Niko1221/Strata/pull/1338) | Immutable secondary Q8 copies during asynchronous refill | Explicit opt-in, supported single-GPU topology only; preserves snapshot/read fences. |
| [#1339](https://github.com/Niko1221/Strata/pull/1339) | Q8 T8 QKV MMQ projection | `STRATA_Q8_DENSE_T8_MMQ=1`, SM120 only; retains main's fallback. |
| [#1340](https://github.com/Niko1221/Strata/pull/1340) | Retain RAM originals during native async exchanges | `STRATA_EXCHANGE_RETAIN_GIB` defaults to zero; bounded, eligible pinned-RAM configurations only. |
| [#1704](https://github.com/Niko1221/Strata/pull/1704) | Bounded active-layer expert cache during prefill | `STRATA_PREFILL_LAYER_CACHE=1`, explicit opt-in; retains main's CPU-share memory fallback. |
| [#1779](https://github.com/Niko1221/Strata/pull/1779) | Accepted-token adaptive cache barriers | `STRATA_ADAPT_BARRIER_TOKENS`, explicit opt-in for supported layer splits; counts committed routing and drains before adaptation. Main already includes the generic pipeline and isolated drafter scratch. |
| [#1818](https://github.com/Niko1221/Strata/pull/1818) | IQ CPU expert arithmetic independent of speculative grouping | Correctness fix and parity fixtures. |

A small integration correction uses the verifier's actual device for secondary
refills and adds the corresponding HIP invalid-value error alias. The multi-stage
asynchronous idle/diagnostic lookups address the actual stage, rather than indexing
the two-entry first/last-stage lookup as if it contained every stage.

The serving-only PRs, Responses API, Codi branching/bookmark/persistence work,
RoPE-to-YaRN saved-cache migration, web/math PRs, installer changes, and
benchmark-only PRs are not imported. Report-only PRs introduce no engine speed fix.
The optional standalone prefill measurement helper is a developer tool, not
part of startup or serving.

## Build and validation

Use the existing source-first build procedure:

```sh
git submodule update --init
make build
```

The submodule provisioning command is explicit; CMake does not fetch missing
source. A missing local ggml checkout stops configuration with an error.
The dependency stays pinned to `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

For focused native tests, after provisioning the same source:

```sh
cmake -S . -B build-tests -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build-tests -j 6
ctest --test-dir build-tests --output-on-failure
```

CUDA test builds also need `-DSTRATA_ENABLE_CUDA=ON` and a supported
`-DCMAKE_CUDA_ARCHITECTURES=...`. The initial local CPU check passed 41 tests,
with one AVX-512 test skipped. Final integrated build/test results are recorded
in [UPGRADE_VALIDATION.md](UPGRADE_VALIDATION.md).

No full-model speed campaign or cross-version token-equivalence claim is made.
The original prepared model artifacts were not regenerated, and their metadata
and a real-model HTTP chat run must be checked on the maintainer's deployment
before promoting the upgrade. The original network-volume problem is not claimed
resolved by this source import. Optional SM120, multi-GPU, and active-layer paths
require their corresponding hardware/model checks before relying on a speed benefit.
