# Relationship to upstream

## Sources

| What | Where | Version |
| --- | --- | --- |
| Engine | https://github.com/Niko1221/Strata | commit `6f32ec070f23ced9f50e704d854d775da52591ab`, engine 0.1.39 |
| ggml (CPU backend, CUDA matrix kernels) | https://github.com/ggml-org/llama.cpp, as the submodule `third_party/llama.cpp` | commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the one upstream Strata pins |

The first commit of this repository is the upstream snapshot as published. One file was not imported:
`bench/results/2026-09-30-community-rtx-5090/engine.log`, which upstream's own ignore rules exclude.
`git diff <first commit> HEAD -- src include CMakeLists.txt` shows every change to the engine and its build.

## What this fork changed

- **Build:** ggml is taken from the submodule. Upstream cloned llama.cpp during the build.
- **Removed:** the Python installer and launchers, the Python server and web app, the Python tools, the
  Intel SYCL port, Docker files, benchmark results, the test suite, translations, and all docs except
  `docs/upstream/DETAILS.md` and the paper.
- **Engine source (`src/`, `include/`):** unchanged.

Code paths for other models, AMD cards and Windows are still present in the engine source. They are not
built by `make build`, not tested and not supported here. They are left in place so the engine stays
identical to upstream and can be updated by re-importing it.

## Prepared model files

The pack and the draft head are produced once with upstream Strata's tools at the commit above
(`tools/iq_pack.py` for the pack; `tools/mtp_fetch.py`, `tools/mtp_pack.py` and `tools/mtp_rt.py` for the
draft head). The exact commands and checksums will be recorded here when the files are made.
