# Qwen3.6 port, Phase 2 step 4, area `drafter`: the MTP draft block from `blk.40`

Working notes, kept as the step was done (2026-10-11, branch `qwen36-step4-drafter`, from the step-4 integration
branch `qwen36-step4` at 1db8252).  The design is `docs/QWEN36-35B-A3B.md` 3.10 and section 6 (step 4); what the
block computes is `docs/qwen36-port/oracle-crosscheck.md` ("Nextn / MTP", 50 seams verified against llama.cpp) and
`gguf-metadata.md` ("The nextn block"); the trunk pack this one sits beside is `step4-pack.md`.  Section 2 is the
pack's layout, section 5 the numbers from both cards, section 7 the interface the runner consumes.

## 1. The numbers first

**The real file, RTX 4090 (VM 221).**  The draft pack cut from `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf`'s
`blk.40`, loaded by the engine (506 MiB of VRAM: experts 464, dense 39, the K/V aside), fed the trunk's final residual
of a 33-token prompt (llama.cpp's own `l_out-39` from `tools/qwen36_dump_llama.cpp` on that file), one row per
position, token k with the hidden of k-1:

| | |
| --- | --- |
| top-1 agreement with llama.cpp's MTP arm (its `graph_mtp` on the same pairs, `qwen36-dump ... mtp`) | **32 of 32** (the whole head); 31 of 32 with the shipped `data/draft_vocab.bin` subset (position 2's " fox", id 37550, is not in the subset) |
| top-1 equal to the prompt's next token (teacher-forced acceptance over 31 positions) | **engine 24 of 31 (77.4 %)**, llama.cpp's MTP 24 of 31 (77.4 %): the same 24 |
| seams vs llama.cpp's MTP graph (relative L2 per row, worst over 32 positions / relative L1 over all) | `mtp_eh_proj` 1.2e-3 / 5.6e-4, `mtp_attn_out` 1.2e-2 / 7.7e-3, `mtp_ffn_out` 2.5e-2 / 1.8e-2, `mtp_shared_head_norm` 2.2e-2 / 1.5e-2 |

llama.cpp's figures are its CPU graph with Q8_K / Q8_0 activation rounding; the engine's are q8_1 activations on
the GPU.  Those are the real-file numbers of the policy's Q8 class (`oracle-crosscheck.md`: `ffn_out` 4.5e-3 max per
forced layer; here the block's products compound) and the drafts agree at every position.  This is the B4 arm of
the plan taken as a per-position reference rather than a generation: the 64-token greedy acceptance under the
runner's verify window is the runner's measurement, after the gate is lifted.

**The synthetic fixture, both cards, all 37 checks** (section 5): every seam at every one of 75 positions within
the gates against the Q8 oracle (the first product `mtp_eh_proj` at 2.3e-4), the router's top-8 identical at
75 of 75 positions, the argmax identical at 74 and a near tie at the 75th, the three negatives 0.14 to 1.41 away.

## 2. The pack, exactly

`python3 -I tools/mtp_qwen36_pack.py --gguf <-MTP file>.gguf --out <dir> [--draft-vocab data/draft_vocab.bin | none]`
(4 s on the Mac, 17 s on VM 221; reads 505 MB).  Flash-Next's `tools/mtp_pack.py` / `mtp_rt.py` are untouched; the
file names are theirs, so `--mtp DIR` is the one CLI and `MtpDrafter::load` picks the path by the trunk's
architecture.  Six files:

| File | Contents |
| --- | --- |
| `dense.txt` | 17 lines `name kind rows cols offset bytes`; the name without `blk.40.`; `kind` = `f32` (the 7 norms), `bf16` (`ffn_gate_inp` 256 x 2048, the 1-D `ffn_gate_inp_shexp`), or the projection's ggml type in lowercase (`q8_0` for all 8 in both shipped files); rows = ne1, cols = ne0 |
| `dense.bin` | the tensors' bytes as stored, 256-aligned: **42,317,824 B** (Q4_K_XL-MTP: 7 x F32 norms 34 KiB, 2 x BF16 1.0 MiB, 8 x Q8_0 41.3 MiB) |
| `experts.bin` | 256 blobs of `[gate rows \| up rows \| down rows]`, raw GGUF blocks, in id order: **486,539,264 B** (blob 1,900,544 = Q4_K 589,824 + 589,824 + Q5_K 720,896; `native_expert_layout(12, 13, 2048, 512)`) |
| `native_experts.txt` | the v3 header and one line `0 12 13 0 1900544 0 589824 1179648` (the role offsets inside a blob; the header says so and names the source) |
| `mtp.json` | `format`, `arch`, `source`, `source_bytes`, `trunk_layers` / `draft_block` 40, `n_embd` 2048, `n_head` 16, `n_head_kv` 2, `head_dim` 256, `n_rot` 64, `n_expert` 256, `n_expert_used` 8, `n_ff` 512, `n_ff_shexp` 512, the expert types and blob bytes, the dense types by name, `draft_vocab` |
| `draft_vocab.bin` | copied from `--draft-vocab` (default `data/draft_vocab.bin`, 106,299 ids; `none` leaves it out: the whole head) |

Refused by name: a file without exactly one nextn block (the plain IQ3_XXS: "nextn_predict_layers is 0"), fused
`ffn_gate_up_exps` / `attn_qkv`, a projection type outside `iq_pack.py`'s `NATIVE_PROJECTION_TYPES`, an F32 router
that is not BF16-exact, and an expert pair the native kernels lack: **the IQ4_XS-MTP file's `blk.40` is Q3_K
gate/up with Q4_K down, neither of which is in `iq_kernels.cu`'s lists (gate/up 16, 17, 18, 21, 22, 23, 29, 42,
12, 13, 14, 6, 8; down 20, 23, 42, 7, 6, 8, 18, 13, 14)**, so only the Q4_K_XL-MTP file's draft block is served
today, and it serves the IQ3_XXS trunk too (the study's 3.10: the drafter is separate from the trunk file; the
float tensors are the same in every quant and the head and embedding come from the trunk's file).

The engine's view (`MtpDrafter::load`): `check_qwen36_pack` holds `mtp.json`'s nine geometry fields against the
trunk's `ModelGeometry` and refuses a mismatch by name ("was cut for n_expert 256 but the trunk has 512"), requires
the 17 names in the forms above (norms F32 of n_embd or head_dim elements, routers BF16, projections a type
`native_mmvq_supported` takes, `eh_proj` n_embd x 2 n_embd); `load_qwen36_experts` requires one layer line, the
header's `n_expert` equal to the trunk's, `native_expert_supported(gu, d, n_embd, n_ff)` and the blob and role
offsets equal to `native_expert_layout`'s, then uploads `experts.bin` whole.

## 3. The drafter on the plain graph

`src/core/mtp.cpp`, `record_forward_qwen36` (dispatched from `record_forward` when `g.arch == Qwen35Moe`; the
Flash-Next body is byte-for-byte what it was), T rows for the K/V-only pass, one row for the full layer, the same
step / position records, graphs, staging, window, K/V formats, coupled sampling and draft chain as Flash-Next:

    emb = token_embd[tok]                              (the native embedding table)
    e   = rms_norm(emb) * nextn.enorm
    hn  = rms_norm(R_in) * output_norm                 (the trunk's output norm: the caller hands the FINAL RESIDUAL, as
    h   = rms_norm(hn) * nextn.hnorm                    for Flash-Next; llama.cpp's h_nextn is the normed state)
    R   = eh_proj . concat(e, h)                       (embedding first; q8_1 x the pack's type through native_mmvq)
    x   = rms_norm(R) * attn_norm
    K   = rope(k_norm(attn_k . x)), V = attn_v . x     -> the drafter's own K/V at the row's cell  [the K/V-only pass stops here]
    q   = rope(q_norm(attn_q . x)[:, :256]), gate = attn_q . x[:, 256:]  per head
    a   = dense attention over the cache (qsa_decode_attn_batch, the window's ids) * sigmoid(gate)
    R  += attn_output . a
    x   = rms_norm(R) * post_attention_norm
    ids, w = top-8 of softmax(ffn_gate_inp . x), renormalised  (bf16_gemv_fp32_mmvf, router_top10 at 256 / 8)
    R  += sum_j w_j expert_ids[j](x)  +  sigmoid(ffn_gate_inp_shexp . x) * shexp(x)
          (moe_group_resident over experts_ at blob stride -> native_expert_grouped on q8_1 x; shared_expert native)
    y   = rms_norm(R) * nextn.shared_head_norm
    logits = head . y  (the draft subset or the whole head), argmax / the coupled sampler, map_ids

What changed against Flash-Next's drafter, item by item of the study's 3.10: the two input projections
(`fc_embedding`, `fc_hidden`) are the one `eh_proj` over the concatenation; the three hyper-connection groups are
the three norms (`attn_norm`, `post_attention_norm`, `shared_head_norm`) with `add_inplace` residuals; the draft
experts are 256 native blobs through `native_expert_grouped` (the kernels of step 3, `step3-experts.md`) instead of
the Q2_0 grouped kernel; the geometry is the trunk's.  What is the same: the dense attention over the drafter's own
ring at `qsa_shapes(g)` (dense: null indexer, cap = max_cells or the `--mtp-window`), the main head over
`draft_vocab.bin`'s subset, `token_embd` embeds, the pairing (cell i: the residual of position i with the token at
i+1, RoPE at i).

**The RoPE position.**  llama.cpp's MTP rotates the pair (token k, hidden k-1) at position k; the engine's drafter
(both models) rotates at the cell, k-1.  RoPE enters the scores only through position differences, so the two
conventions give the same attention; the harness runs the engine's convention against the oracle's dumps and the
seams agree to float rounding (`mtp_attn_pregate` 1.6e-3, the fp16 pool inside).  Nothing in `prefill` / `draft` /
`draft_kv` needed a Qwen branch for it.

**Buffers**: `cat_` (T x 2 n_embd) for the concatenation, `xq_` sized for the wider of `n_head x head_dim` and
2 n_embd, `hit_scratch_` the larger of the grouped kernel's and `native_expert_scratch_bytes(T x k, n_ff)`; the
rest of the carve is Flash-Next's at hc = 1.  The K/V: without `--kv-resident` (KV streaming does not apply to the
dense model, `step3-attn.md` item 6) `kv_plan` gives no ring, so the drafter's cache is `max_cells` whole: 2 KiB
per cell fp16 (512 MiB at 262,144 cells), 576 B q4_0 (144 MiB); `--mtp-window` still bounds what a row attends to.

**The harness's entry point**: `MtpDrafter::debug_row(R_row, token, cell, pos, DebugOpts, seam, &top1, err)` runs
the full layer for one row on the drafter's stream outside any graph and hands every seam to `seam` by llama.cpp's
`cb()` name after a stream sync (the list is in `mtp.hpp`); `DebugOpts{hidden_first, skip_hnorm}` are the rival
readings, set by nothing else.

**The prompt-side pass** (`src/prefill/prefill.cpp`, `Prefill::draft_kv`): the `qwen35moe` branch computes the
nextn block's K/V for the prompt's cells through the prompt path's GEMMs in the drafter's order (embedding, the
three norms, the concat, `eh_proj`, `attn_norm`, `attn_k` / `attn_v`, `k_norm`, RoPE at the cell, the append in the
state's format), MMQ q8_1 x the pack's type where `mmq::supported` and `fits` say so (Q8_0: yes), else the FP16
GEMM over dequantized weights; batches of 64 rows and up as for Flash-Next (`B < 64` declines and the drafter's own
`prefill` runs).  The harness checks it end to end: the positions after 64 `draft_kv` cells agree with the oracle
to the same figures as the drafter's own pass and draft the same tokens.

## 4. The harness: `tests/cuda/qwen36_mtp_test.cpp`

The fixture (`tests/python/qwen36_mtp_fixture.py`; CTest `qwen36_mtp_fixtures`, 3 min on VM 221, 11 s on the Mac
at 11 tokens) is the prompt-path harness's `full4` trunk (n_embd 2048, three DeltaNet layers, layer 3 attention
16 / 2 x 256, 16 experts top-8 of 512, shared 512) plus the nextn block `blk.4` from `tools/qwen36_synth_gguf.py`,
with every projection, `nextn.eh_proj`, the experts, the head and the embedding Q8_0 (the native paths' types) and
the routers BF16-exact; 76 tokens, so 75 nextn positions and a 64-row `draft_kv` batch.  The oracle runs twice
(`--mtp`: faithful, and `--act-quant q8`) and `tools/mtp_qwen36_pack.py` cuts the pack from the fixture itself.
`qwen36_mtp_test <dir>` (CTest `qwen36_mtp_test`, exit 77 without a device or the fixture):

1. Per position k = 1..75, `debug_row(l_out-3[k-1], tokens[k], cell k-1, pos k-1)`: 21 seams against the Q8
   oracle's row k-1 (relative L2 per row, the worst over positions; the F32 oracle's figure beside it from a
   second pass over its own trunk rows), `ffn_moe_topk` identical as a set, the draft equal to the oracle's argmax
   (a flip counts as a near tie when the oracle's gap is within 3x the row's largest logit error, and is printed).
2. The negatives, each more than 5 % relative L1 from the oracle: the concat with the hidden state first
   (`mtp_eh_proj`, `result_output`), hnorm skipped (`mtp_hnorm`, `mtp_eh_proj`), the pairing with position k
   instead of k-1 (`mtp_hnorm`, `result_output`).
3. `draft_first` (the captured round graph: the catch-up and the full layer at cell 0): draft 0 equals the
   uncaptured forward's draft at cell 0.
4. `Prefill::draft_kv` for cells 0..63 (a `Prefill` on the fixture's trunk), then positions 65..75 through the
   forward again: the five seams within the same gates and the drafts equal to the drafter's own pass's.

`qwen36_mtp_test --real --gguf F --pack D --dump D1 --mtp-dump D2 --tokens ...` is the real-file arm of section 1:
`D1` / `D2` from `qwen36-dump F D1 <tokens> 0 mtp` (CPU llama.cpp at the pinned commit, built under
`/models/work/llama-cpu` on VM 221 with gcc-13; 52 s for 33 tokens at 4 threads, 871 MB of dumps).

The gates, relative L2 per row against the Q8 oracle, and why: `mtp_tok_embd` 1e-6 (exact dequantization),
the norms of forced inputs 1e-5, `mtp_eh_proj` / `mtp_attn_norm` 1e-3 (one q8_1 x Q8_0 product: 2.3e-4 measured),
then **every product after the first sees an input 2.3e-4 away from the oracle's, and the oracle's own q8 rounding
of that input flips about 1 % of the int8 codes by a step (2e-4 x rms/amax x 127), which is 2e-3 of the output**:
the Q/K/V projections 5e-3 (2.0-2.2e-3 measured), `mtp_attn_pregate` 5e-3 (1.6e-3), `mtp_attn_out` 1e-2 (5.4e-3),
the residual / post norm / router logits 5e-3 (1.1-1.2e-3), the renormalised weights 1e-2 (1.7e-3), the shared
expert and `mtp_ffn_out` 3e-2 (1.1-1.2e-2: the moe_out class), `mtp_post_ffn` / `mtp_shared_head_norm` 1.5e-2
(5.3e-3: the l_out class), `result_output` 3e-2 (1.0e-2).  The exact checks are the router's picks and the argmax.

## 5. Results

### VM 221, the RTX 4090 (sm_89), 2026-10-11

Build: `/models/work/build-step4-drafter.sh` (the common configure at sm_89 with `-DSTRATA_MMQ_KQUANTS=ON`;
`qwen36_mtp_test` and `strata`, 2 min 34 s).  `./build/qwen36_mtp_test /models/work/qwen36-drafter-fx`
(`/models/work/logs/qwen36-drafter-synth.log`), 37 of 37:

| seam (75 positions) | vs Q8 oracle | gate | vs F32 oracle (reported) |
| --- | ---: | ---: | ---: |
| `mtp_tok_embd` | 0.0 | 1e-6 | 0.0 |
| `mtp_enorm` / `mtp_hnorm` / `mtp_concat` | 1.3e-7 / 1.3e-7 / 1.2e-7 | 1e-5 | 1.3e-7 / 1.3e-7 / 1.3e-7 |
| `mtp_eh_proj` / `mtp_attn_norm` | 2.30e-4 / 2.31e-4 | 1e-3 | 6.2e-3 / 6.2e-3 |
| `mtp_Vcur` / `mtp_Kcur_normed` / `mtp_Qcur_full` / `mtp_Qcur_normed` | 2.02e-3 / 2.08e-3 / 2.15e-3 / 2.22e-3 | 5e-3 | 9.0e-3 / 9.3e-3 / 8.6e-3 / 8.8e-3 |
| `mtp_attn_pregate` / `mtp_attn_out` | 1.64e-3 / 5.41e-3 | 5e-3 / 1e-2 | 1.3e-2 / 1.5e-2 |
| `mtp_attn_residual` / `mtp_attn_post_norm` / `ffn_moe_logits` | 1.09e-3 / 1.09e-3 / 1.19e-3 | 5e-3 | 7.4e-3 / 7.4e-3 / 1.2e-2 |
| `ffn_moe_topk` | identical at 75 of 75 | identical | |
| `ffn_moe_weights_norm` | 1.65e-3 | 1e-2 | 1.3e-2 |
| `mtp_ffn_shexp_gated` / `mtp_ffn_out` | 1.18e-2 / 1.08e-2 | 3e-2 | 2.6e-2 / 1.9e-2 |
| `mtp_post_ffn` / `mtp_shared_head_norm` | 5.32e-3 / 5.33e-3 | 1.5e-2 | 1.1e-2 / 1.1e-2 |
| `result_output` | 1.09e-2 | 3e-2 | 1.4e-2 |
| the draft = the oracle's argmax | 74 of 75; position 45 a near tie (oracle gap 1.2e-2, largest logit error 4.6e-2) | | 3 of 75 differ from the F32 argmax |

| negative case | relative L1 from the oracle | required |
| --- | ---: | ---: |
| concat(hnorm, enorm), the hidden state first, at `mtp_eh_proj` / `result_output` | 1.41 / 1.38 | > 0.05 |
| hnorm skipped, at `mtp_hnorm` / `mtp_eh_proj` | 0.165 / 0.143 | > 0.05 |
| paired with position k instead of k-1, at `mtp_hnorm` / `result_output` | 1.38 / 1.01 | > 0.05 |

`draft_first` at cell 0: drafts 65, 35, 16; the forward's draft at position 1: 65 (equal).  `Prefill::draft_kv`
took the 64 cells (batched, MMQ); positions 65..75 after it: `mtp_attn_pregate` 1.42e-3, `mtp_attn_out` 4.83e-3,
`mtp_ffn_out` 1.11e-2, `mtp_shared_head_norm` 5.22e-3, `result_output` 1.14e-2, the drafts equal to the drafter's
own pass's at 11 of 11.  `ctest -R qwen36_mtp` (the fixture through CMake and the test): 2 of 2 (the fixture 132.5 s, the test 0.6 s).

The real file: section 1 (`/models/work/logs/qwen36-drafter-real-qwen36-mtp-draft{,-full}.log`;
`/models/work/packs/qwen36-mtp-draft{,-full}`, the dumps in `/models/work/qwen36-drafter-real/`).

### VM 211, the GTX 1080 Ti (sm_61, Pascal), 2026-10-11, under `/home/rk/VM-BUSY`

The same tree (`/home/rk/qwen36-step4-drafter/`, the submodule copied from the step-4 pack tree), configured with
`-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`, `qwen36_mtp_test` and `strata` built in 3 min 52 s;
the fixture written there (`/home/rk/qwen36-drafter-fx`, 4 min 43 s) and the two packs cut from
`/home/rk/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (the MTP repo's file, 22 s); the real-file dumps copied from VM 221
(`l_out-39`, the MTP `result_output` and four seams).  The lock was held from 21:28 to 21:34 UTC (the build and the
fixture ran under it; the runs take seconds).  Logs in `/home/rk/logs/qwen36-drafter-*.log`.

`./build/qwen36_mtp_test /home/rk/qwen36-drafter-fx`: **37 of 37**, the same numbers as the 4090 to the printed
digits at every seam but four (`mtp_attn_out` 5.42e-3, `ffn_moe_logits` 1.23e-3, `ffn_moe_weights_norm` 1.61e-3,
`mtp_ffn_shexp_gated` 1.14e-2, `mtp_post_ffn` / `mtp_shared_head_norm` 5.75e-3 / 5.70e-3: the per-entry expert
kernels Pascal takes, `step3-experts.md`), the router's picks identical at 75 of 75, the argmax identical but the
same near tie at position 45, the negatives 1.41 / 1.38, 0.165 / 0.143, 1.38 / 1.01, `draft_first` 65 35 16 = 65,
`draft_kv` batched with the positions after it at 1.45e-3 / 4.75e-3 / 1.09e-2 / 5.34e-3 / 1.04e-2 and the drafts
equal.  The kernels the drafter adds to Pascal's path are the step-3 native expert kernels (dp4a), the MMVQ, the
rms norm, the dense attention's `pre75` kernel: nothing new was guarded.

The real file, both packs: **top-1 agreement with llama.cpp's MTP arm 32 of 32 (the whole head), 31 of 32 (the
subset: position 2 again); teacher-forced acceptance 24 of 31 = 77.4 % (23 with the subset), llama.cpp 24 of 31**;
`mtp_eh_proj` 1.18e-3 / 5.6e-4, `mtp_attn_out` 1.22e-2 / 7.7e-3 as the 4090, but `mtp_ffn_out` 1.54e-1 / 2.6e-2 and
`mtp_shared_head_norm` 1.42e-1 / 1.9e-2 (worst row / relative L1): **one position, 10, is 1.4e-1 from llama.cpp's
head-normed state on the 1080 Ti and 1.6e-2 on the 4090**, every other position 1.0e-2 to 2.3e-2 on both.  The
only mechanism that moves one row by 14 % while its neighbours stay at 1 % is a routed-expert flip at a near tie
(the real file's router sees percent-level differences, `oracle-crosscheck.md` "Tolerance policy"), from the
cards' different dot-product orders in the attention and expert kernels; the draft at that position (411) is the
same on both cards and equals llama.cpp's.  Not confirmed by dumping both cards' `ffn_moe_topk` for that row (the
lock was released); the runner's end-to-end parity on Pascal is where a flip rate per position is measured.

## 6. Not done, surprises, where the study was off

- **The 64-token greedy acceptance** (the brief's alternative to the llama.cpp arm) needs the trunk generating,
  which is the runner's (the step-2 gate, the verify window on qwen35moe); the per-position reference against
  llama.cpp's MTP arm is the stronger check and was cheap (the dumper exists since step 1).  `draft()` and
  `draft_first()` run on the real pack (the captured graphs are the same code as Flash-Next's); only `draft_first`
  is held to the forward here.
- **The VRAM plan's drafter line** (`generate.cpp`, `in.drafter_bytes = draft + 16 MiB` "its ring KV and 8 x n_vocab
  logits") undercounts: the weights are 506 MiB as planned, but the K/V is the whole context (no ring without
  `--kv-resident`: 512 MiB fp16 / 144 MiB q4_0 at 262,144 cells; `--mtp-window` does not shrink it), the draft head
  subset is `n_dvocab x row_bytes` (106,299 x 2,176 B = 221 MiB on the Q8_0 head of the Q4_K_XL file, 171 MiB on the
  IQ3_XXS file's Q6_K head), the logits 7.6 MiB, the identity ids T x max_cells x 4 (8 MiB at 256K).  The runner's
  area (the plan is the gate block's); the numbers are here for it.
- **The IQ4_XS-MTP file's draft block cannot be served** (Q3_K gate/up, Q4_K down: section 2).  The study's 3.10
  counted it ("Q3_K/Q4_K, 364 MiB in the IQ4_XS file"); the step-3 expert kernels never gained those two roles.
  The Q4_K_XL-MTP pack serves every trunk, so nothing is lost today; adding Q3_K gate/up and Q4_K down to
  `iq_kernels.cu`'s lists is a step-5 item if that file is wanted.
- **The data/draft_vocab.bin subset drops " fox"** (id 37550) and so one of 32 positions on the English prompt;
  the subset is Flash-Next's English-and-code selection over the same vocabulary and carries over as the study
  said, with that coverage.
- **The study's 3.10 said "IMRoPE"** for the draft block's attention: the sections reduce to plain NEOX rope for
  text (section 1 of the study), and the engine's `native_rope_apply` / `rope_neox_apply` at n_rot 64 are what the
  oracle's dumps match.
- The oracle's `mtp_ffn_moe_out` (the routed sum without the shared expert) is not a seam the engine materialises
  (`native_moe_combine` adds the shared expert in the same launch); `mtp_ffn_out` covers it.
- `tests/python/qwen36_mtp_fixture.py` imports the prompt-path fixture module for its geometry and the BF16 rounding;
  a change there changes this fixture's trunk too (intended: one trunk shape for both harnesses).

## 7. The interface the runner consumes

- `MtpDrafter` is unchanged for the runner: `load(rt_dir, g, ss, max_t, err, window)` with `g` the trunk's geometry
  (`generate.cpp`: `draft_geometry = g.arch == Qwen35Moe ? g : ModelGeometry{}`, the one `[shared]` hunk, lines
  3267-3271), `bind(wt, &native_head, window_R, err)` (needs `output_norm.weight` F32 in `wt`, `output.weight`'s row
  for n_vocab, `token_embd.weight`'s row and the native embedding set, as the trunk pack provides), `prefill(R_rows,
  next_tokens, n, cell0)`, `draft(T, tokens, p, a, drafts, ...)`, `draft_first(...)`, `kv_state()`, `vram_bytes()`,
  `bind_bytes(row_bytes, n_vocab)`.  `R_rows` / `window_R` are the trunk's FINAL RESIDUAL rows (hc = 1: n_embd each,
  `Verifier::final_R_all()`), not the normed state: the drafter applies `output_norm` itself.
- `Prefill::draft_kv(mtp, R_rows, next_tokens, n, cell0, err)`: the same call; true for the Qwen3.6 drafter when
  the native embedding is set and the batch is at least 64 rows, false (err empty) otherwise.
- New: `MtpDrafter::qwen36()`, `tensor_native(name, &type)`, `expert_gu_type()` / `expert_d_type()` /
  `expert_blob_bytes()`, `debug_row(...)` with `DebugOpts` and `SeamFn` (the harness's).
- `tools/mtp_qwen36_pack.py`: `--gguf`, `--out`, `--draft-vocab`; the layout of section 2.  The CLI's `--mtp DIR`
  takes the directory; `--draft-vocab en` style selection is whatever the pack holds as `draft_vocab.bin`.
- The commits on `qwen36-step4-drafter` past 1db8252: ad70f3d the pack tool, 6ae03ea the drafter, f0e5362 the
  prompt-side pass, e21be6c `[shared] generate` (the one hunk, `src/program/generate.cpp:3267-3271`), 41c568d the
  harness and the fixture, then this file.
