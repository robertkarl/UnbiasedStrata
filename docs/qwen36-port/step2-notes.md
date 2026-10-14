# Qwen3.6 port, Phase 2 step 2: geometry, metadata, the VRAM planner

Working notes, kept as the step was done (2026-10-11, branch `qwen36-step2`, branched from `qwen36-oracle`).
The design is `docs/QWEN36-35B-A3B.md` sections 3.7 to 3.9 and 4.2; the plan is `docs/PLAN-qwen36-35b-a3b.md`
section 5 step 2. Every item below says what was decided, where it lives, and how it is checked.

## 1. What the engine does with a Qwen3.6 file at the end of this step

Started with `--native Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --pack <dir with native_experts.txt>` the engine:

1. reads every field of `ModelGeometry` from the file's `qwen35moe.*` keys and logs them;
2. refuses the two fused tensor layouts by name if the file had them (no shipped file does);
3. starts without `--ple-gguf` (the architecture has no PLE), with no PLE kernels selected;
4. checks the pack's `native_experts.txt` against the model's 2048 x 512 (every layer's ggml formats must make
   the pack's blob size);
5. makes the VRAM plan of section 4.2 from this card's free VRAM (clamped by `--vram-cap`) and the file's own
   census, prints the three tiers (VRAM slots, RAM copies, mapped experts), and
6. **stops at the step 2 gate**, exit 2, with this message (from `strata::core::qwen35moe_not_yet`):

> this is a Qwen3.6-35B-A3B file (qwen35moe, 40 layers, 2048 wide, 256 experts x top-8): Phase 2 step 2
> (geometry, metadata, the VRAM plan above) is done, but the engine cannot run it yet. Not done: step 3 (kernels
> and the block: the GDN gate SiLU(z) instead of sigmoid(z), dense attention at 16 heads with no indexer, the
> block and the verify window without hyper-connections, prefill at this geometry), step 4 (the native pack and
> the MTP drafter from blk.40, the expert profile at 40 x 256).

Why a gate and not a run: `layer.cpp` and `verify.cpp` still wrap every layer in the hyper-connection read/write
and require the `hc_*` and `indexer.*` tensors (`layout.cpp`'s `check_one`), the GDN kernels gate with sigmoid,
prefill refuses any geometry but 2560/4/320/top-10. Each of those is step 3. Failing before the first weight
moves, with the plan printed, is Maya's Phase 0 shape.

The real file's log on the Mac (`model_geometry_test --real`), both the IQ3_XXS and the Q4_K_XL file:

    arch qwen35moe, n_embd 2048, n_layers 40 (nextn 0 / 1), qsa_interval 4 (10 attention, 30 GDN),
    ssm state 128 / 16 k-heads / 32 v-heads / conv 4, conv channels 8192, value dim 4096,
    n_head 16, n_head_kv 2, head_dim 256, n_rot 64, sections 11,11,10,0, base 1e7, eps 1e-6,
    hc 1, n_expert 256, k 8, n_ff 512, shexp 512, no PLE, no indexer

## 2. Decisions

| Decision | Where | Why |
| --- | --- | --- |
| `ModelGeometry` carries `arch` (an enum), `nextn_layers`, `n_rot`, `rope_sections`, `rope_base`, `rms_eps`, `n_ff_shexp`, `n_expert_used`, `n_vocab`, `has_ple`; `derive()` computes conv channels and value dim | `include/strata/core/layout.hpp` | the stored conv channels / value dim were the row-2 trap: changing `ssm_v_heads` alone left them at Flash-Next's. An enum, not a string, so the struct stays trivially copyable for the `.cu` files that take it by value |
| The reader is header-only and CUDA-free | `include/strata/core/model_geometry.hpp` | the unit test links only the GGUF reader; the engine reads the geometry before any weight moves |
| `hc = 1`, `has_ple = false`, no indexer come from the architecture string, not a key | same | the file has no key for any of them; llama.cpp decides the same way |
| For `qwen4exp` the GDN, ff and interval keys are optional (defaults stay); for `qwen35moe` they are required | same | no Flash-Next file was on hand to confirm which of its keys are present; a required key that a shipped Flash-Next file lacks would have broken the working model. The six keys the guard always demanded stay required for both |
| `check_architecture` accepts `qwen35moe` with presence-only checks of llama.cpp's required key set | `include/strata/artifact/gguf_reader.hpp` | the file is the authority on its sizes; the guard's job is "is this a file llama.cpp's own loader would accept" |
| Fused forms refused by tensor name (`ffn_gate_up_exps`, `attn_qkv` in an attention layer) at the engine and in `iq_pack.py` | `model_geometry.hpp` (`check_model_tensor_forms`), `tools/iq_pack.py` (`expert_layout`) | `attn_qkv` is legitimate in a GDN layer (the conv input), so the check is per layer type |
| `expert_layout_load` takes `n_embd`, `n_ff` | `include/strata/kernels/cpu/expert_layout.hpp`, `src/kernels/cpu/expert_layout.cpp` | the one leak (row 108). A pack with no `native_experts.txt` (canonical Q2_0, which bakes 2560 x 640) is now refused for any other geometry with a message naming the packer |
| The geometry is read before the layout load in `generate.cpp`; `K` and the embedding dims come from it; the rope keys are read under the file's own prefix | `src/program/generate.cpp` (the block before `expert_layout_load`) | rows 7 to 11 |
| The draft (nextn) block is left out of the trunk's `native_experts.txt` | `tools/iq_pack.py` | the Q4_K_XL file carries `blk.40`; a layout with 41 layers would make the engine route layer 40 of a 40-layer model. The drafter's pack (step 4) takes it |
| `--ple-gguf` is not required when the architecture has no PLE; an explicit one is refused; the PLE kernels are not selected; the session carves no PLE history | `generate.cpp` (the `--native` block), `src/core/session.cpp` | 3.9: the kernels already ran without it, only the CLI insisted. The 360 KB `NG_HIST x NG_HC_DIM` buffer is skipped through `has_ple` |
| The planner is a pure function at the mean blob | `include/strata/plan/vram_plan.hpp` | testable without a GPU; the engine's per-layer sizing prices the same bytes at each profile entry's own blob and lands within a few slots |
| The context is lowered in steps of 4,096 toward 131,072, never below; below the floor the start refuses with every number and the IQ3 file's name | same | the plan's rule; KV streaming is not a lever for dense attention |
| The reserve in the Qwen3.6 plan is 300 MiB (`kSmallReserveMib`) unless `--vram-reserve-mib` was given | `generate.cpp` (the gate) | table 4.3 was derived at the small reserve; the Flash-Next auto sizing starts at 700 and shrinks to 300 only when the cache would be too small |
| The prompt buffers in the plan are the engine's own estimate (`160 + chunk * 680 / 1024` MiB: 1,520 at chunk 2048), not llama.cpp's 1,904 | same | it is this engine's prompt path that allocates them; the study's figure was the baseline's compute buffer. With a profile prefill borrows cache slots and the term is 0 |
| `--vram-cap MIB`: every `cudaMemGetInfo` in `generate.cpp` goes through `vram_info`, which clamps the free figure to `cap - used`; a blocker of `total - cap` is allocated before the weight arena | `generate.cpp` (`vram_info`, the cap block) | 4.2 item 4. `used` counts every process, so the cap is the card's size, not this engine's share |
| `--ram-cap GIB`: `host_available_memory` is clamped (so `pin_cache_complement` sizes the RAM tier as that machine would), the resident budget is clamped, the pinned arena's limit too | `src/core/expert_source.cpp` (`set_host_ram_cap`), `generate.cpp` | the page cache cannot be capped from inside the process (the 12 GiB VM does that physically) |
| Tier counters: the serve loop's `STATS` command answers one JSON line; the same line is logged at start; `/health` carries it as `tiers` | `generate.cpp` (`tier_stats`), `frontend/main.cpp` | VRAM slots and how many hold an expert, RAM copies (`complement_count`), mapped experts, tokens served, VRAM hits of lookups, RAM-copy reads, file reads, warm hints, and the major page faults from `/proc/self/stat` since start as the disk-read proxy. During a request `/health` serves the last reading marked `tiers_stale` |
| The Flash-Next path keeps its allocation order and auto sizing; it gains the cap and the counters | `generate.cpp` | its fixed set is allocated progressively with its own refusals; the lowering rule is for the dense-KV model, which is gated here anyway |

## 3. What a Qwen3.6 native pack contains (step 4 builds it; the layout half exists now)

`tools/iq_pack.py --gguf <file> --out <dir> --layout-only` writes `native_experts.txt` alone (v3: one line per
trunk layer with the gate/up and down ggml types, the blob size, and the three tensors' absolute offsets) and
prints the census. The two real files:

| File | Layers x experts | Expert bytes | Mean / largest blob | Per-layer formats |
| --- | --- | ---: | ---: | --- |
| `UD-IQ3_XXS` | 40 x 256 | 10,594 MiB (11,108,614,144 B) | 1.035 / 1.172 MiB | gate/up IQ2_S everywhere; down IQ3_XXS on 37 layers (blob 1,073,152 B), IQ4_XS on 34, 38, 39 (1,228,800 B) |
| `UD-Q4_K_XL` (MTP repo) | 40 x 256 (+ `blk.40`, the drafter's, left out) | 18,726 MiB (19,635,634,176 B) | 1.829 / 2.195 MiB | gate/up Q4_K, down Q5_K on 37 layers (1,900,544 B); Q4_K / Q6_K on 34 and 38 (2,039,808 B); Q5_K / Q6_K on 39 (2,301,952 B) |

These two files are `tests/data/native_experts/qwen36-{iq3_xxs,q4_k_xl}.txt` and `expert_layout_test` loads them
at 2048 x 512 (every layer's `NativeFmt` makes the blob) and sees them refused at 2560 x 640. The full pack
(step 4) adds `index.txt` (the dense projections served natively from the GGUF: Q6_K / Q8_0 `attn_*`, `ssm_out`,
`ffn_*_shexp`, `output`), `dense.bin` with the float tensors in the engine's forms (the norms F32 as stored;
the F32 routers, `ssm_alpha`, `ssm_beta` and the 1-D shexp gate need `--compat-bf16` today, which step 4 decides),
the tokenizer, and `expert-profile.bin` at 40 x 256 from the E0 trace.

## 4. Build and test commands

On the Mac (no CUDA; the artifact reader, the planner and the geometry test build and run):

    cmake -S . -B build-mac -G Ninja -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON
    cmake --build build-mac --target model_geometry_test vram_plan_test gguf_split_test gguf_reader_test
    (cd build-mac && ctest -R 'qwen36_synth|model_geometry|vram_plan|gguf' --output-on-failure)
    ./build-mac/model_geometry_test --real ~/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
    ./build-mac/vram_plan_test --at q4 11264
    python3 tools/iq_pack.py --gguf ~/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --out /tmp/p --layout-only
    python3 tools/test_iq_pack.py

Results on the Mac, 2026-10-11: `model_geometry_test` 32 checks pass on the synthetic files and the real-file
mode passes on both files; `vram_plan_test` 43 checks pass; `test_iq_pack.py` 23 tests pass.

On VM 211 (the 1080 Ti, sm_61; `docs/qwen36-port/pascal-build.md` has the build's outcome):

    rsync -a --delete --exclude build-mac --exclude .git --exclude build --exclude build-make --exclude logs \
          /Users/robertkarl/Code/UnbiasedStrata-step2/ rk@192.168.50.43:/home/rk/UnbiasedStrata-step2/
    # (--delete without the three VM-side excludes removed the VM's build/, build-make/ and logs/ once)
    cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON \
          -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
    cmake --build build -j 8 --target unbiased-strata
    cmake --build build -j 8   # every parity target the root CMakeLists declares
    (cd build && ctest --output-on-failure)
    ./build/expert_layout_test tests/data/native_experts
    ./build/model_geometry_test build/qwen36_synth
    ./build/unbiased-strata --engine --native /home/rk/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --pack /home/rk/packs/qwen36-iq3 \
        --tokens 1 --max-context 262144 --kv q4_0 --spec 4 --prefill 2048 --expert-profile /dev/null   # the gate

## 5. VM 211 results

Done 2026-10-11 after the sweep and its nvprof re-runs released the card (`docs/qwen36-port/pascal-build.md` has
every log line; the VM keeps the tree at `/home/rk/UnbiasedStrata-step2/` with `build/`, `build-make/` and `logs/`).

| Item | Result |
| --- | --- |
| sm_61 build | no source change needed: configure 4 s, `unbiased-strata` 173 s, every other target 43 s more, 0 errors; `cuobjdump --list-elf` shows `sm_61` only. Every sm_75+ site was already behind `__CUDA_ARCH__` (pascal-build.md section 1) |
| `make build CUDA_ARCHS=61` | end to end with no compiler override, 181 s, `sm_61` only; the Makefile's switch sets `STRATA_EXPERIMENTAL_SM60=ON` |
| ctest on the 1080 Ti | 75 tests: 69 pass, 4 fail, 2 skip (sm_80 fused prompt MoE). Of the failures one is Pascal-specific, `kv_hybrid_parity` [5/5], where `qsa_prompt_attn_batch` refuses `cc < 70` by design and the test counts the refusal as a failure (step 3). The other three are the host: `platform_memory_test` (mlock limit 8 MiB), `ple_parity` (hardcoded Flash-Next path), `expert_multi_test` (AVX-512 on Zen 1). Two of the branch's own tests had never been built (CUDA-gated): `qwen36_synth_fixtures` failed under cmake 3.31 (CMP0007, commit `cd2a73c`) and `expert_layout_test` carried a wrong Q4 total and a refusal-message expectation (commit `5798a46`); both pass now, and `model_geometry_test --real` passes on the VM against both real files |
| the pack | `native_experts.txt` only: 2,669 B in 0.35 s on the Mac for the IQ3_XXS file; 2,727 B in 0.6 s on the VM (vendored gguf-py) for the Q4_K_XL file |
| the gate | **not reached.** Both real files print the geometry block and are then refused by the native expert kernel's format whitelist, before the plan: `layer 0's experts are IQ2_S/IQ3_XXS (ggml types 22/18), which this engine has no GPU kernels for` (exit 1, 0.13 s), and `Q4_K/Q5_K (12/13)` for the Q4 file. The grouped kernels' *down* types are IQ4_NL / IQ4_XS / Q5_0 / Q5_1 / Q8_0 only; the IQ3_XXS file's down is IQ3_XXS (37 layers), the Q4_K_XL's Q5_K and Q6_K. A step 3 kernel item (or a step 4 repack of the down projections) the plan did not list; the gate message itself is exercised by `model_geometry_test --real` |

Correction to section 3's table: the Q4_K_XL expert total is 19,635,634,176 B (the pack header's figure, and the sum
of its blobs); the earlier 19,636,158,464 was 524,288 B high and was also in `expert_layout_test`.

### The planner on the 1080 Ti against the study's table 4.3 (2026-10-11, after `4a998ad`)

The engine's own plan (`--vram-cap`, prefill borrowing, `logs/gate-*-cap*.log` on the VM) at the caps an 11 GB card
can emulate; the study's "derived, borrow" column beside each. The engine's figures run a few percent above the
study's because the study's budgets carry a 300 MiB CUDA-context term that the engine's free figure already lacks,
less the 164 MiB this VM's display and driver hold. The 12, 16, 24 and 32 GB rungs cannot be emulated on this card
(`--vram-cap 12288` is "not below this card's 11165 MiB: no cap"); `vram_plan_test` covers them from the study's
inputs.

| Budget | cap (MiB) | VRAM budget | Q4 slots (study) | Q4 RAM copies / mapped | IQ3 slots (study) | IQ3 RAM copies / mapped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 GB | 5722 | 5557 | 598 (~525) | 2,894 / 6,748 | 1,486 (~1,350) | 5,172 / 3,582 |
| 8 GB | 7629 | 7463 | 1,641 (~1,570) | 2,855 / 5,744 | 3,328 (~3,200) | 5,115 / 1,797 |
| 8 GiB | 8192 | 8027 | 1,949 | 2,874 / 5,417 | 3,873 | 5,068 / 1,299 |
| 11 GB (decimal) | 10490 | 10325 | 3,206 (~3,130) | 2,959 / 4,075 | 6,094 (~5,960) | 4,146 / 0 |
| 1080 Ti, no cap | | 11001 | 3,575 (~3,550) | 2,929 / 3,736 | 6,748 (~6,700) | 3,492 / 0 |

The RAM tier is sized from this VM's ~9.4 GiB available minus the 4 GiB headroom (the `--resident-experts` rule),
so the Q4 file keeps a disk tier at every cap on this host and the IQ3 file none from 11 GB up, as 4.3 said.

### What the review and the Pascal run changed afterwards (2026-10-11, the same branch)

| Finding | Commit | What changed |
| --- | --- | --- |
| the down-type whitelist, missed by the study's section 3.7 | `30e3add` | 3.7 said "the IQ3_XXS file's 37/3 down split and the Q4_K_XL's Q6_K layers need no format change" of the *layout*; it did not check `STRATA_D_FMTS`, the grouped kernels' down types (IQ4_NL, IQ4_XS, Q5_0, Q5_1, Q8_0), against the files' IQ3_XXS / Q5_K / Q6_K down projections. The gate now runs before that whitelist and names the missing pairs per layer; the kernels are step 3's `qwen36-step3-experts` branch |
| `--vram-cap` subtracted the blocker twice | `2b4fcf1` | `vram_info` takes the blocker's own bytes out of `used`; the cap line prints what a card of the cap would show and refuses if the clamped figure is off by more than 64 MiB |
| the planner's 300 MiB CUDA-context term on top of a free figure already net of it | `4a998ad` | the engine passes 0; the unit test keeps the study's 300 MiB with the cards' sizes as budgets |
| `--ram-cap` set an explicit resident budget on the auto path | `76a437f` | only an explicit budget above the cap is clamped |
| `/health`'s last reading raced | `84794ce` | its own mutex, copies under it |
| the qwen4exp reader's optional keys untested | `afec2f1` | a fixture with the real Flash-Next key set equals the defaults field by field; the fixture writer gained f32 and u32-array values |
| the trace bound literal 48; the refused-profile test | `db4da2f`, `d010040` | `Drive::n_layers` from the geometry; `expert_profile_save_test` sees a 48 x 512 profile refused for 40 x 256 |
| `test_iq_pack.py` had no test of the step's additions | `b2ccff0` | `--layout-only`, the nextn exclusion, the fused refusal: 26 tests |
| `tools/make_profile.py` `N_LAYER, N_EXPERT = 48, 512` | (step 4) | the profile from the E0 trace at 40 x 256 is step 4's, with the trace format |

`model_geometry_test --real` on the Mac's `~/models/dwarfstar/Qwen3.8-Flash-Next-Q2.gguf` (147 GB, the only Flash-Next
file on the Mac or VM 211; the `ple_parity` shard path does not exist anywhere): the qwen4exp reader reads it
(48 trunk layers, `nextn_layers 1`, every other field the struct's default, separate tensor forms), but the
architecture guard refuses it with `qwen4exp.block_count = 49, expected 48`: the guard's qwen4exp branch wants the
literal 48 (unchanged by this step, the reviewer's byte-for-byte check), and this file is an MTP build with a draft
block. Whether that file ever ran under this engine is not this step's question; it is recorded here because the
guard, not the reader, is what refuses it.
