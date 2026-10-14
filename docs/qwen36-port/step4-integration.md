# Qwen3.6 port, Phase 2 step 4: the integration branch `qwen36-step4`

Branch `qwen36-step4` (worktree `~/Code/UnbiasedStrata-step4`), from the step-3 head 355ce3c.  Unmerged, unpushed.

## What is on it, in order

| range | from | what |
| --- | --- | --- |
| 3ad2847..1db8252 (7) | `qwen36-step4-pack` | `iq_pack` FORM_QWEN35MOE, the regex-free tokenizer, `make_profile --n-layer`, the 40x256 expert profile, `pack_load_check`; notes step4-pack.md |
| 02ff47e | integrator | step-3 notes |
| ad70f3d..84b4756 (6) | `qwen36-step4-drafter` | `tools/mtp_qwen36_pack.py`, the qwen35moe drafter in `mtp.cpp`, prefill draft K/V, `[shared] generate.cpp` draft geometry, `qwen36_mtp_test`; notes step4-drafter.md |
| 6d0d817..6042a32 (6, cherry-picked clean) | `qwen36-step4-runner` | the step-2 gate lifted, `[shared] verify.cpp` indexer tail, `qwen36_verify_test` + `qwen36_runner_check.py`, the llama.cpp teacher and the logits comparison, notes, the VRAM plan's drafter term |
| 85d5d12 | `main` f07b81e | the front end's SSE keepalive and client-gone cancel (another session's patch, cherry-picked on request) |
| e73adc6 | integrator | CMake applies the chat-parser patch at configure time (REGINA-104); `make build` turns `STRATA_MMQ_KQUANTS` on |

The expected `generate.cpp` conflict between the drafter's draft-geometry hunk (:3307) and the runner's gate region
(:2184, :2557) did not occur; the three hunks sit apart.

## Builds and tests

- **VM 221, the RTX 4090**: the integrated tree at e73adc6 configures and builds with 0 errors (the common line,
  `-DSTRATA_MMQ_KQUANTS=ON`, sm_89; logs `/models/work/logs/qwen36-step4-*.log`, script `/models/work/build-step4.sh`).
  The full ctest of the integrated tree is **not** a result: 63 of 97 tests fail in under a second on
  `malloc x: out of memory`, because another session's Flash-Next service (`unbiased-strata.service`, active since
  2026-10-12 22:08 UTC) holds 23.9 of 24.5 GB.  The 34 that pass are the CPU tests (`vram_plan_test`, the pack and
  planner tests, the tokenizer).  Owed: the full run once the card has VRAM.
- **VM 211, the GTX 1080 Ti**: the runner's own tree ran steps 2-4 at 21:38 UTC 2026-10-11 (step4-runner.md section
  4): verify test and runner check pass; the real-file llama.cpp comparison did not run (fixtures not synced).  The
  integrated tree has not been built there: since 2026-10-12 22:04 UTC the card runs a Flash-Next coder service and
  the user has asked that it not be interrupted.
- **The chat parser**: `make test-chat-parser MODEL=/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf BUILD_DIR=build` passes
  on VM 221 after the fix (all four argument orders).
- **The Mac** (arm64, CUDA off): the upstream CPU expert kernel is compiled with `-mavx512*` flags unconditionally
  (CMakeLists.txt:1086), which clang rejects on arm64, so the CUDA-off Mac build has never linked past
  `strata_kernels_cpu`.  Not the port's regression; the Mac configure step was used only to verify the patch logic.

## Numbers that carry forward

- Speculation with the real drafter on the 4090 (step4-runner.md, round 8): IQ3_XXS 0.77 accepted / 3.20 tokens per
  round with the full head, 0.72 / 3.05 with the subset head; Q4_K_XL (plain trunk, MTP-file draft) 0.53 / 2.54.
- The VRAM plan's drafter share is now the whole context's K/V plus the head subset plus the ids (f2e70c8);
  `vram_plan_test` passes with the new cases.
- Correctness held on both cards: 1080 Ti argmax 40/40 vs the Q8 oracle, 4090 teacher-forced IQ3 64/64 x3 prompts.

## Owed to step 5

1. ~~Full ctest of the integrated tree on a card with free VRAM.~~  Done 2026-10-13 once the 4090's service was
   stopped: 95 of 98 at 22d72e6 (the three host-environment failures), including the serve fixes and
   `pool_native_widths_test` of step5-serve.md.
2. The 1080 Ti real-file comparison (sync `/models/work/qwen36-step4-teacher` to VM 211) and the Q4 fit at 11 GB.
3. The ladder: serve and measure on 11 GB, then `--vram-cap` 8 and 6 GB; the three instruments; engine points on
   `ladder-baseline.png`; acceptance A1-A5.
4. The step-6 fat binary, compile-checked on VM 221's CPU (`build-fat`, `/models/work/logs/qwen36-step4-fat-*.log`):
   `CUDA_ARCHS=61;70;75;86;89` with `STRATA_EXPERIMENTAL_SM60=ON` and `STRATA_MMQ_KQUANTS=ON` configures and builds
   `unbiased-strata` in 6 min 17 s at `-j 14`, 0 errors; `cuobjdump --list-elf` shows 62 ELF images per architecture
   for all five; 235 MiB.  **sm_120 cannot come from this host**: CUDA 12.4's nvcc rejects `compute_120`
   ("Unsupported gpu architecture"); Blackwell needs a 12.8+ toolkit, so the step-6 release build either uses a newer
   toolkit or ships 120 as PTX from 89.  Nothing in this binary has run on a card yet.
