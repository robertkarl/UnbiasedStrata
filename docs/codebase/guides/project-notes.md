[Codebase docs](../README.md) › [Guides](README.md) › Project notes

# Reading guide to the existing prose docs

## Purpose

Before `docs/codebase/` existed, the repository's knowledge lived in 22 hand-written Markdown files: the top-level
README and UPSTREAM notes, upstream Strata's long technical page, the Qwen3.6 port's plan and study, and 17 working
notes written while the port was done. They are dated, written for the person doing the work at the time, and cite
`file:line` against the tree as it was then. This page says what question each one answers and in what order to read
them. The code-level pages in `docs/codebase/` describe the tree as it is; these notes explain why it is that way and
hold the measured numbers.

Covered files: `README.md`, `UPSTREAM.md`, `docs/upstream/DETAILS.md` (and `docs/upstream/Strata-Paper.pdf`),
`docs/QWEN36-35B-A3B.md`, `docs/PLAN-qwen36-35b-a3b.md`, and the 17 files of `docs/qwen36-port/`.

## Where to start

| If you want to know... | Read |
| --- | --- |
| how to build, run and size the server, and what is supported | [README.md](../../../README.md) |
| what this fork changed relative to upstream Strata, and how the model packs were made | [UPSTREAM.md](../../../UPSTREAM.md) |
| how upstream Strata's engine works and what each runtime knob does | [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) ("How it works", then the speed and low-RAM sections) |
| why Qwen3.6-35B-A3B needed which engine changes | [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) sections 1-3 |
| what was done, step by step, and how it was checked | the `docs/qwen36-port/` timeline below |
| a term you do not recognise | [glossary.md](glossary.md) |

---

## 1. Top-level documents

| File | Lines | Question it answers | Notes for a reader |
| --- | --- | --- | --- |
| [README.md](../../../README.md) | 261 | What is UnbiasedStrata, how do I build and run it, which model files does it need, how big a card and how much RAM, how fast is it, what is broken? | Sections: Requirements, Build (`make build`, `CUDA_ARCHS`), Run (the two command lines, endpoint/option table), The supported models (file tables with SHA-256 for Flash-Next and Qwen3.6, Sizing tables per VRAM size), Status (machines checked; speed vs llama.cpp on the RTX 4090 and GTX 1080 Ti ladders), Known problems, Compared with upstream, Credits. |
| [UPSTREAM.md](../../../UPSTREAM.md) | 122 | Which upstream commit is this, what did the fork remove/add/patch, and exactly how were the packs and draft heads built? | Sources table (Strata `6f32ec0`, engine 0.1.39; llama.cpp `3cf0325`); the three engine source changes (`robust_read.hpp`, `expert_source.cpp`, `--ple-io ram` in `ngram.cpp`) and the one llama.cpp patch; the pack-building commands and SHA-256 of every prepared file for both models. |

Two places where the README and the code disagree, noted while writing these pages:

- The "Compared with upstream" table says the fork has 0 lines of Python. The tree now tracks about 7,800 lines of
  Python (`tools/`, `ref/`, `tests/python/`), restored or written for the Qwen3.6 port; the serving binary itself
  needs none.
- The Sizing paragraph says that when the fixed set does not fit "the KV cache is streamed to RAM above a resident
  window". For Qwen3.6 the engine refuses `--kv-resident` because its attention is dense
  (`qsa_kv_streaming_applies`, `src/core/layer.cpp:561-568`; `include/strata/plan/vram_plan.hpp:7-8`); the planner
  lowers the context toward 131,072 instead. KV streaming applies to Flash-Next's sparse (QSA) attention only.

## 2. `docs/upstream/DETAILS.md` - upstream Strata's technical page

1,127 lines, copied from upstream Strata at `6f32ec0` (UPSTREAM.md: "all docs except `docs/upstream/DETAILS.md` and
the paper" were removed). **It describes upstream Strata, not this fork.** Its links to `README.md`, `INSTALL.md`,
`MODELS.md`, `TROUBLESHOOTING.md`, `UNSLOTH_Q4.md`, `paper/...` point at upstream files that do not exist here (the
paper is `docs/upstream/Strata-Paper.pdf`). Its model is Qwen3.8-Flash-Next and its numbers are upstream's.

| Section | What it covers | Applies to this fork? |
| --- | --- | --- |
| Speed (measured) | prompt and output tokens/s per engine version and card; KV streaming (0.1.5), hybrid K8V4 KV (0.1.25), reproducible greedy output (`STRATA_IQ_MT_MIN=1`), the draft vocabulary (`--draft-vocab`, 0.1.27), low-RAM modes (0.1.26-0.1.31, `--resident-budget-gib`), `STRATA_ARENA_MMAP`, `--stats` tiers, `STRATA_DECODE_TIMING`, prompt chunking (0.1.13) | the engine mechanisms and flags do; the numbers are upstream's Flash-Next measurements |
| Other GPUs (estimated) | projected speeds on other cards | upstream only |
| Which model? (and the Coder, Swift 1.5, Unsloth UD-Q4_K_XL variants) | upstream's model menu | no: this fork supports one Flash-Next file (IQ3_XXS) and two Qwen3.6 files |
| Before you start / Windows / Linux | installer (`START-HERE.bat`, `setup.sh`), Task Scheduler, terminal chat | no: there is no installer or Windows build here; `make build` and `unbiased-strata` replace them |
| Sharing the GPU with other programs | `--vram-elastic`, `POST /v1/vram` (#533), saving the learned expert profile (#477) | engine flags exist (`--vram-elastic`, `--expert-profile-save`); the JSON config keys belong to upstream's Python server |
| Using it | upstream's HTTP API, conversation cache, multiple conversations, Responses API and Codex CLI | partly: this fork's front end is a new C++ server ([../frontend/README.md](../frontend/README.md)); compare before relying on a field |
| Tools from MCP servers / MCP server | upstream's Python server features | no |
| Images (vision) | encoder placement, `--vision-tokens`, Q8_0 mmproj | the mechanism (CPU `mtmd` encoder) matches; settings differ |
| Experimental speed projection | control vector (`--control-vector-scaled`, `--cvec-*`) | engine flags exist; upstream's setup question does not |
| Troubleshooting | upstream's error catalogue | partly |
| How it works | the tier diagram: GPU (mixers, routers, shared experts, head, MTP, KV, expert cache), RAM (all 24,576 experts pinned, CPU computes misses in place), SSD (the 28.8 GB n-gram table), speculation (MTP drafts + prompt lookup), chunked prompts with experts streamed over PCIe | yes, for Flash-Next; Qwen3.6 has no n-gram table, no QSA and only 10,240 experts |
| Credits and licenses; start the text API without the GPU; JSON response formats; API request monitor | upstream appendix | mostly upstream-only |

## 3. The Qwen3.6 port: plan and study

| File | Lines | Question it answers |
| --- | --- | --- |
| [docs/PLAN-qwen36-35b-a3b.md](../../PLAN-qwen36-35b-a3b.md) | 475 | What is the target (any single NVIDIA card from Pascal up, 6-32 GB, 256K context, no llama.cpp fallback), what are the phases, and how is success measured? Sections: 0 target and the Flash-Next vs Qwen3.6 config table; 1 the card ladder (GTX 1060 6 GB to a capped RTX A6000, with the RTX 4090 and GTX 1080 Ti VMs); 2 the memory hierarchy and the budget table; 3 Phase 0 (environment, llama.cpp baseline, the "E0" route replay that later seeded `data/qwen36-expert-profile.bin`); 4 Phase 1 (images, baseline write-up); 5 Phase 2, the engine port in seven steps; 6 acceptance A1-A5; 7 agents; 8 order of work. Written 2026-10-11 as "proposed, nothing started". |
| [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md) | 594 | The design study: what exactly is the model (section 1, from the GGUFs), what is the engine (section 2), what changes component by component (section 3: GDN gate, dense attention replacing QSA, router and shared expert, the block without hyper-connections in `layer.cpp` and `verify.cpp`, prefill literals, the native pack path, metadata and `ModelGeometry`, optional PLE, the MTP drafter, tokenizer/template/vision), the memory plan and the VRAM planner (section 4, tables 4.1-4.3), the expected envelope (5), the execution plan with effort weights for steps 2-7 (6), risks (7) and disagreements with the plan (8). The step notes cite its section numbers ("study 3.3", "table 4.3"). |

## 4. `docs/qwen36-port/` - the port as a timeline

All dated 2026-10-11 unless stated; each step was done on its own branch, often by several agents in parallel, with
results on two cards (VM 221: RTX 4090, sm_89; VM 211: GTX 1080 Ti, sm_61).

```
step 1  oracle        gguf-metadata, geometry-inventory, upstream-tooling, oracle-crosscheck   (branch qwen36-oracle)
step 2  geometry      step2-notes, pascal-build                                               (qwen36-step2)
step 3  kernels       step3-block | step3-gdn | step3-experts | step3-attn | step3-prefill     (5 branches from 8b7525e)
                      -> step3-integration                                                    (qwen36-step3)
step 4  pack/runner   step4-pack -> step4-drafter | step4-runner -> step4-integration         (qwen36-step4)
step 5  serve         step5-serve (2026-10-13 onward)                                         (qwen36-step4, then main)
```

### Before step 2: the inputs

| File | Lines | Question it answers |
| --- | --- | --- |
| [gguf-metadata.md](../../qwen36-port/gguf-metadata.md) | 102 | What metadata keys and tensors do the shipped Qwen3.6 GGUFs (UD-IQ3_XXS, and the IQ4_XS-MTP file) actually carry, in ne order, and what was unexpected? Includes the `blk.40` nextn block. |
| [geometry-inventory.md](../../qwen36-port/geometry-inventory.md) | 290 | Where does the Strata engine bake in Flash-Next's geometry? 154 numbered rows ("row N" in later notes) classed RT (runtime), CT (compile-time, must change), CT-ok, ST (structural), by subsystem A-K, plus the loader's tensor names vs llama.cpp's `qwen35moe`. Central finding: `ModelGeometry` was a default struct and only two fields were read from the file. |
| [upstream-tooling.md](../../qwen36-port/upstream-tooling.md) | 106 | Which upstream tools, references and tests were removed by the fork and must come back (`ref/`, `tools/iq_pack.py`, `make_profile.py`, `tests/`), how upstream builds parity fixtures and packs, and which tools hard-code Flash-Next (e.g. `N_LAYER = 48`). Read-only survey; nothing built. |
| [oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md) | 313 | Step 1: does the numpy oracle `ref/qwen36.py` agree with the pinned llama.cpp graph? Yes: 2,155 comparisons over 116 seam families, worst 1.1e-4; seven "rival" readings each fail at their seam; dequantization bit-identical to ggml. Also the formulas pinned by `file:line`, comparison quirks, and the tolerance policy every later parity test uses. Reproduce with `tools/qwen36_crosscheck.sh`. |

### Step 2: geometry, metadata, the VRAM planner

| File | Lines | Question it answers |
| --- | --- | --- |
| [step2-notes.md](../../qwen36-port/step2-notes.md) | 166 | What does the engine do with a Qwen3.6 file after `ModelGeometry` is read from metadata, the planner exists and `make_profile.py --n-layer` works; what goes into a Qwen3.6 pack; the exact build/test commands (Mac CUDA-off and VM 211); the planner's output against table 4.3. Ends at the "gate" (`qwen35moe_not_yet`) that step 4 lifts. |
| [pascal-build.md](../../qwen36-port/pascal-build.md) | 198 | Can the engine build and pass its tests on sm_61? Where the tree uses what Pascal lacks (tensor cores, `cp.async`, `__nanosleep`, clusters), the build, ctest on the 1080 Ti (69 pass, 4 fail, 2 skip, with causes), the pack and the gate, and the Makefile's automatic `STRATA_EXPERIMENTAL_SM60` switch. |

### Step 3: kernels and the block (five parallel areas, then integration)

| File | Lines | Question it answers |
| --- | --- | --- |
| [step3-block.md](../../qwen36-port/step3-block.md) | 202 | How does the engine run a block without hyper-connections (`hc == 1`) in `layer.cpp`, `session.cpp` and `verify.cpp`, and how is it held to the oracle (`tests/cuda/qwen36_block_test.cpp`)? Hunks for the integrator. |
| [step3-gdn.md](../../qwen36-port/step3-gdn.md) | 173 | The GDN SiLU output gate in the four decode kernels, the router at 256/top-8 and the shared expert at 512 (`gdn_parity`, `qwen36_seam_parity`, `router_top10_parity`, `shared_expert_parity`), and the measured BF16-router flip rate (study 3.4). |
| [step3-experts.md](../../qwen36-port/step3-experts.md) | 163 | Why the native GPU expert kernels refused both real files (no IQ3_XXS / Q5_K / Q6_K down kernels at 512) and what was added; `native_expert_parity` pairs at `--geom 2048x512`. |
| [step3-attn.md](../../qwen36-port/step3-attn.md) | 214 | Dense attention replacing QSA: identity ids, `G` templated at 8, no indexer, the dense state's layout, bounded split-K scratch, KV streaming refused; `qsa_dense_parity` and the decode cost before/after. |
| [step3-prefill.md](../../qwen36-port/step3-prefill.md) | 217 | The prompt path (`src/prefill/`) at Qwen3.6's geometry, site by site: GDN kernels at 32/16/8192, route at `k`, attention, fused native experts; `prefill_qwen36_test`, `gdn_rec_parity`. It cites a `step3-common.md` of shared rules that is not in this tree. |
| [step3-integration.md](../../qwen36-port/step3-integration.md) | 63 | How the five branches were combined, the ctest totals on both cards (89/92 and 87/92, three host failures), and the read-only review's findings (the `qsa_shapes` builder) carried into step 4. |

### Step 4: pack, drafter, runner

| File | Lines | Question it answers |
| --- | --- | --- |
| [step4-pack.md](../../qwen36-port/step4-pack.md) | 209 | `tools/iq_pack.py`'s `qwen35moe` form table, the pack's exact layout (every file), how `data/qwen36-expert-profile.bin` was made (section 3), `pack_load_check`, and the load order the engine follows. Its section 7 precedes section 6 in the file. |
| [step4-drafter.md](../../qwen36-port/step4-drafter.md) | 266 | The MTP draft block from `blk.40`: the draft pack `tools/mtp_qwen36_pack.py` writes (section 2), `mtp.cpp` on the plain graph, `tests/cuda/qwen36_mtp_test.cpp`, acceptance numbers, and the interface the runner calls. |
| [step4-runner.md](../../qwen36-port/step4-runner.md) | 195 | Lifting the step-2 gate: the engine produces tokens on Qwen3.6, held to the oracle (`qwen36_verify_test`, `qwen36_runner_check`) and to llama.cpp on the real files (`--spec-oracle --teacher-force`, `tools/qwen36_compare_logits.py`); tolerances; the speculation rate with the real drafter. |
| [step4-integration.md](../../qwen36-port/step4-integration.md) | 58 | What is on the `qwen36-step4` branch, commit by commit; why the first full ctest was not a result (another service held the 4090's VRAM); the numbers carried forward; what was owed to step 5. |

### Step 5: serving and measurement

| File | Lines | Question it answers |
| --- | --- | --- |
| [step5-serve.md](../../qwen36-port/step5-serve.md) | 521 | What broke when serving Qwen3.6 through `unbiased-strata` below full residency, and how each fault was fixed (the draft-vocab requirement, snapshots without an indexer, the pool's canonical widths, the multi-token CPU dispatch); what the 4090 serves at 262,144; the A5 serving checks on the 4090 (REGINA-110) and the 1080 Ti; why the same greedy request gives different tokens (REGINA-111: CPU-computed misses within 0.3 logits); the front end's `max_tokens` fix (REGINA-112); the Flash-Next regression check and both models on one 4090 as two processes. |

Steps 6 (the Pascal fat binary) and 7 (the README) of the study have no separate note: the Makefile's default
`CUDA_ARCHS=61;70;75;86;89;120` with the automatic SM60 switch (`pascal-build.md` section 5) and the current
README are their outcome.

## How these notes relate to `docs/codebase/`

| Topic | Code-level page | Port note with the reasoning and numbers |
| --- | --- | --- |
| model geometry, the planner | [../core/README.md](../core/README.md) | step2-notes.md, study 3.8 and 4.2 |
| the block, verify window, drafter | [../core/README.md](../core/README.md) | step3-block.md, step4-runner.md, step4-drafter.md |
| GPU kernels and parity | [../kernels/README.md](../kernels/README.md), [../kernels/parity/README.md](../kernels/parity/README.md) | step3-gdn.md, step3-attn.md, step3-experts.md |
| prompt path | [../prefill/README.md](../prefill/README.md) | step3-prefill.md |
| packs, profiles, tools | [../tools/README.md](../tools/README.md), [data-files.md](data-files.md) | step4-pack.md, UPSTREAM.md |
| HTTP front end | [../frontend/README.md](../frontend/README.md) | step5-serve.md |
| tests | [testing.md](testing.md) | every step's "Results" section |

## Gotchas when reading the notes

- `file:line` citations are against the tree of that day; lines have moved since. Search for the symbol.
- Paths such as `/models/work/...`, `/home/rk/...`, `~/Code/regina/...` are on the author's machines and VMs and are
  not in this repository; the "regina" repository holds the measurement harness and plots.
- "row N" means a row of `geometry-inventory.md`; "study 3.x" or "table 4.x" means `docs/QWEN36-35B-A3B.md`.
- Branch names (`qwen36-step3-attn`, ...) are historical; `qwen36-step4` was merged into `main` (commits `21b2aa2`, `e7ae2c8`).

## See also

- [README.md](README.md) - the guides overview.
- [glossary.md](glossary.md) - REGINA ids, rival readings, seams, and the other terms the notes use.
- [testing.md](testing.md) - the tests each note's results come from.
