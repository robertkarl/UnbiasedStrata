[Codebase docs](../README.md) › Guides

# Guides (cross-cutting)

The other areas of `docs/codebase/` follow the source tree; this one covers what cuts across it. It documents the
shipped data files (`data/` and `tests/data/`) that the pack tools copy into model packs and that the engine and the
tests read; gives one index of every test in the repository, whichever directory or driver it lives in (Make, CTest,
Python), with how to build and run them; is a reading guide to the 22 existing prose documents (the README, the
upstream notes and the Qwen3.6 port's plan, study and step notes); and defines the vocabulary those documents and
the code comments use. Nothing here is executed: these pages point into `src/`, `tools/`, `tests/`, `frontend/` and
`docs/`, and are pointed at by every area page that mentions a test, a data file or an unfamiliar term.

## Contents

| Page | What it covers |
| --- | --- |
| [data-files.md](data-files.md) | `data/draft_vocab*.bin` (int32 draft-head subsets), `data/expert-profile.bin` and `data/qwen36-expert-profile.bin` (`STRP` expert-cache rankings), `tests/data/native_experts/*.txt` (expert-table fixtures): formats, readers/writers, provenance, model |
| [testing.md](testing.md) | Every test: front-end Make targets, unit tests in `src/` and `tests/`, kernel parity counts, Python unit tests, CTest fixtures and oracle tools; `STRATA_BUILD_TESTS`, `ctest`, skip codes, known host failures |
| [project-notes.md](project-notes.md) | What question each prose doc answers: README, UPSTREAM, DETAILS (upstream-only parts flagged), the plan and study, and the `docs/qwen36-port/` notes as a step-1-to-5 timeline |
| [glossary.md](glossary.md) | The domain terms (MoE, GDN, QSA, PLE, MTP, verify window, pack, profile, slot, PCIe fraction, KV formats, quant formats, oracle, seam, REGINA ids, ...) defined from this repo's usage, each with a link |

## Source map

The files this area documents directly. Test sources are inventoried file by file in [testing.md](testing.md);
the prose documents in [project-notes.md](project-notes.md).

| Path | Lines / size | Role |
| --- | --- | --- |
| [data/draft_vocab.bin](../../../data/draft_vocab.bin) | 425,196 B | draft-head subset, 106,299 int32 ids (English/code + CJK); copied into both models' draft packs |
| [data/draft_vocab_en.bin](../../../data/draft_vocab_en.bin) | 162,100 B | smaller English/code subset, 40,525 ids |
| [data/expert-profile.bin](../../../data/expert-profile.bin) | 196,632 B | Flash-Next expert-cache ranking, 48 x 512, from upstream |
| [data/qwen36-expert-profile.bin](../../../data/qwen36-expert-profile.bin) | 81,944 B | Qwen3.6 expert-cache ranking, 40 x 256, from the E0 route trace |
| [tests/data/native_experts/q2_0.txt](../../../tests/data/native_experts/q2_0.txt) | 49 | Flash-Next Q2_0 file's expert table (v2) |
| [tests/data/native_experts/iq2_xs.txt](../../../tests/data/native_experts/iq2_xs.txt) | 49 | Flash-Next IQ2_XS (v2) |
| [tests/data/native_experts/iq3_xxs.txt](../../../tests/data/native_experts/iq3_xxs.txt) | 49 | Flash-Next IQ3_XXS (v2) |
| [tests/data/native_experts/iq3_s.txt](../../../tests/data/native_experts/iq3_s.txt) | 49 | Flash-Next IQ3_S (v3) |
| [tests/data/native_experts/coder-iq1_m.txt](../../../tests/data/native_experts/coder-iq1_m.txt) | 49 | the pruned Coder, 256 experts (v3) |
| [tests/data/native_experts/swift-iq2_xs.txt](../../../tests/data/native_experts/swift-iq2_xs.txt) | 49 | Swift 1.5 IQ2_XS, layers 13-47 in shard 2 (v3 with shard column) |
| [tests/data/native_experts/qwen36-iq3_xxs.txt](../../../tests/data/native_experts/qwen36-iq3_xxs.txt) | 41 | Qwen3.6 UD-IQ3_XXS (v3) |
| [tests/data/native_experts/qwen36-q4_k_xl.txt](../../../tests/data/native_experts/qwen36-q4_k_xl.txt) | 41 | Qwen3.6 UD-Q4_K_XL (v3) |
| [Makefile](../../../Makefile) | 68 | `build`, and the test targets `test-frontend`, `test-chat-parser`, `test-serve` |
| [CMakeLists.txt](../../../CMakeLists.txt) | 1,444 | `STRATA_BUILD_TESTS`, `STRATA_BUILD_CONVERSATION_TESTS`, 115 `add_test` calls |
| [cmake/qwen36_synth_fixtures.cmake](../../../cmake/qwen36_synth_fixtures.cmake) | 27 | CTest fixture: toy `qwen35moe` GGUFs |
| [cmake/qwen36_block_fixtures.cmake](../../../cmake/qwen36_block_fixtures.cmake) | 28 | CTest fixture: oracle dumps for the block test |
| [cmake/qwen36_oracle_dumps.cmake](../../../cmake/qwen36_oracle_dumps.cmake) | 35 | CTest fixture: kernel-geometry GGUF and F32/Q8 dumps |
| [cmake/qwen36_attn_fixtures.cmake](../../../cmake/qwen36_attn_fixtures.cmake) | 39 | CTest fixture: attention fixtures g8/g12/g2 |
| [tests/python/](../../../tests/python) | 5 files, 683 lines | Qwen3.6 fixture generators, the runner check, the serve `max_tokens` check |

Readers and writers of the data files (documented in their own areas, cited in [data-files.md](data-files.md)):
`src/core/expert_cache.cpp` (696 lines, `read_expert_profile` / `write_expert_profile`),
`src/kernels/cpu/expert_layout.cpp` (349, `expert_layout_load`), `src/core/mtp.cpp` (1,372, the draft-vocab subset),
`tools/make_profile.py` (176), `tools/draft_vocab.py` (146), `tools/iq_pack.py` (763), `tools/mtp_qwen36_pack.py` (246).

## Key concepts and data flow

How the shipped data reaches a running engine, and which tests watch each step:

```
 data/qwen36-expert-profile.bin ──┐                              ┌─ tools/test_make_profile.py
 data/expert-profile.bin ─────────┤ tools/iq_pack.py             │  tests/core/expert_profile_save_test.cpp
 (made by tools/make_profile.py)  │ --expert-profile, or cp      │
                                  v                              │
 model GGUF ── tools/iq_pack.py ──> <pack>/  index.txt dense.bin native_experts.txt  expert-profile.bin
                                         │         (format fixtures: tests/data/native_experts/*.txt
                                         │          -> tests/core/expert_layout_test.cpp)
                                         v
 unbiased-strata --model GGUF --pack DIR --mtp DIR
        │  front end checks <pack>/expert-profile.bin exists, starts the engine child with
        │  --expert-profile <pack>/expert-profile.bin --expert-cache auto --spec 4 --mtp DIR
        v
 engine: read_expert_profile -> fill VRAM slots in rank order; expert_layout_load(native_experts.txt)
         MtpDrafter::bind -> <mtp dir>/draft_vocab.bin -> draft head over the subset
                                         ^
 data/draft_vocab.bin ── tools/mtp_qwen36_pack.py (Qwen3.6) or cp (Flash-Next) ──> <mtp dir>/draft_vocab.bin
 (made by tools/draft_vocab.py)
```

Testing has three layers that build on each other:

```
 Python oracle (ref/qwen36.py) ── cross-checked against llama.cpp (tools/qwen36_crosscheck.sh, oracle-crosscheck.md)
        │  writes cb()-named seam dumps of synthetic GGUFs (CTest fixtures, build/qwen36_*)
        v
 C++/CUDA tests: kernel parity (src/kernels/*_parity.cpp) -> block / prefill / drafter / verify window
                 (tests/cuda/*) -> streaming runner (tests/python/qwen36_runner_check.py)
        v
 real files by hand: teacher-forced logits vs llama.cpp (tools/qwen36_compare_logits.py, top-1 and KL),
                     make test-chat-parser MODEL=..., make test-serve PORT=...
```

Model-specific notes: the two profiles and the eight layout fixtures are each tied to one model's geometry
(Flash-Next 48 x 512, the Coder 48 x 256, Qwen3.6 40 x 256) and are refused at another; the draft vocabularies serve both
because the two models share the 248,320-token vocabulary.

## See also

- [../README.md](../README.md) - the codebase docs overview.
- [../build/README.md](../build/README.md) - build options used by the tests.
- [../kernels/parity/README.md](../kernels/parity/README.md) - the parity programs.
- [../tools/README.md](../tools/README.md) - the Python tools that produce the data files.
- [UPSTREAM.md](../../../UPSTREAM.md) - how the shipped data was turned into model packs, with checksums.
