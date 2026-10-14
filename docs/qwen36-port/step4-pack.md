# Qwen3.6 port, Phase 2 step 4, area `pack`: the native pack

Working notes, kept as the step was done (2026-10-11, branch `qwen36-step4-pack`, from the step-3 integration
branch `qwen36-step3` at c6fa016).  The design is `docs/QWEN36-35B-A3B.md` 3.7, 3.8, 3.11 and section 6 (step 4);
what the engine reads is `docs/qwen36-port/step3-block.md`, `step3-gdn.md`, `step3-attn.md`, `step3-prefill.md`;
the census that decided the forms is `step3-gdn.md` section 3.  This file is the contract the drafter and runner
agents consume: section 2 is the pack's layout, exactly; section 5 is how the engine loads it and what was verified
on both cards.

## 1. What changed

| Where | What | Why |
| --- | --- | --- |
| `tools/iq_pack.py` | `FORM_QWEN35MOE` beside Flash-Next's `FORM` (`FORMS[arch]`, `form_of(name, arch)`); `NATIVE_PROJECTIONS_QWEN35MOE` / `NATIVE_PROJECTION_TYPES` / `NATIVE_EMBED_TYPES` with `native_type_problem`; the draft block (`blk.<trunk>` and up) left out of `index.txt` and `dense.bin` (`n_trunk`); the inexact-BF16 refusal carries the count; `--expert-profile` (default `data/qwen36-expert-profile.bin` for qwen35moe) copied in as `expert-profile.bin` after a header check | the study's step 4 line; "a second entry, not a rewrite": a toy Flash-Next model with every FORM tensor kind packs byte-identically with the step-3 and the step-4 packer (all six files, checked) |
| `tools/test_iq_pack.py` | `write_gguf(..., architecture=)`; `Qwen36PackTests`: the forms and `index.txt`'s contents, the count in the refusal and `--compat-bf16`'s rounding, the native types refused by name (projection, head, embedding each with its list), the nextn block left out, the profile checked and copied, the Flash-Next table untouched | the brief's fixtures |
| `tools/make_profile.py`, `test_make_profile.py` | `--n-layer` (the base's header decides when given), `--text-trace` for llama.cpp eval-callback route traces (`<layer> <e0> ... <e7>` lines, `#` headers, `.gz`), `profile_header` | `N_LAYER = 48` was the study's row; the E0 trace is text |
| `data/qwen36-expert-profile.bin` | the 40 x 256 ranking from the E0 trace (section 3) | shipped in every Qwen3.6 pack |
| `tools/strata_tokenizer.py` | `regex` imported if present; `extract` needs no pattern, `encode` refuses by name without it | neither VM's system python has `regex`, and the packer runs the export as a subprocess |
| `src/core/layout.cpp` (`[shared]`) | `check_one` skips the four `indexer.*` rows when `!g.has_indexer()` | they were still required of every attention layer at hc = 1 (step3-block.md section 7 said so): every Qwen3.6 pack would have been refused "missing blk.3.indexer.q_proj.weight" |
| `src/program/generate.cpp` | `native_dense.load(..., 0, g.n_layers)` | a -MTP file's `blk.40` carries the projection names; it is not in the trunk's pack and `NativeDense::load` refuses a name absent from the table.  A no-op for Flash-Next |
| `tests/core/pack_load_check.cpp`, `CMakeLists.txt` | the pack loaded as `generate` loads it after the gate (section 5) | the gate fires before any of those loads |

Not needed, and why: `--compat-bf16` for the real files (section 2: 140 of 140 conversions exact, max error 0);
any change to `weights.cpp` (index kinds 4 and 2 were already the raw-BF16 and F32 paths), `native_dense.cpp`
(every projection type of both files is in `native_mmvq_supported`), `native_head.cpp` or the embedding (Q6_K /
Q8_0 are in both lists), `expert_layout.cpp` (v3 at 2048 x 512 since step 2).

## 2. The pack, exactly

Built by `python3 -I tools/iq_pack.py --gguf <file>.gguf --out <dir>` from a single-shard qwen35moe GGUF
(both shipped files are single shards; a split would name its shards in `native_experts.txt` as for Flash-Next).
Six entries:

| File | Contents |
| --- | --- |
| `index.txt` | "strata pack index v3", `# align 64 pool <arena bytes> tensors 613`, then one row per non-expert trunk tensor: `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind` (19 fields; `file` 0 = dense.bin).  Kinds: **4** = raw BF16 in dense.bin (the loader's `Bf16InF32` form, copied), **2** = F32 copied, **0** with `code_bits 8 group_elems 32` and every byte count 0 = served natively from the GGUF (shape only; the loader refuses to read such a row, `NativeDense` / the head / the embedding attach it).  `dst_off` is the running 64-aligned arena offset; a native row carries the current offset with 0 bytes. |
| `dense.bin` | the kind-4 and kind-2 tensors at their `src_off`, each padded to 64 B: **54,610,432 B** for both files (byte-identical between IQ3_XXS, the plain Q4_K_XL and the MTP Q4_K_XL: md5 `096a0d6e140c3ac412b2f9542eb3c335`; the float tensors are the same in every quant, as they were for Flash-Next) |
| `native_experts.txt` | v3, one line per trunk layer `layer gu_type d_type offset blob_bytes gate_off up_off down_off` (absolute byte offsets of the three expert tensors in the GGUF), header `(n_expert 256, total <bytes>; absolute offsets in <file>)`; 40 lines, the nextn block excluded.  The IQ3_XXS table equals `tests/data/native_experts/qwen36-iq3_xxs.txt` byte for byte; the MTP Q4_K_XL table equals `qwen36-q4_k_xl.txt` up to the file name in the header. |
| `conversions.json` | schema 1: the 140 F32 -> BF16 records (`exact: true`, `max_abs_err 0.0`, the source bytes' SHA-256), `compat_bf16: false` |
| `expert-profile.bin` | `STRP`, version 1, 40 x 256, 10,240 ranked `(layer, expert)` u16 pairs then the 40 x 256 slot table: 81,944 B (section 3) |
| `tokenizer/` | `vocab.json` (248,320), `merges.txt` (247,587), `token_type.json`, `tokenizer.json` (`pre: qwen35`, the shipped pattern, `add_bos_token false`, eos 248046 / pad 248055 / bos 248044), `chat_template.jinja` (8,057 B, the file's own) |

The 613 rows, per tensor (the suffix after `blk.N.`; shapes as GGUF `ne0 x ne1`, ne0 contiguous):

| Tensor | Layers | Index kind, form in the arena | Shape | Engine site that reads it (form it demands) |
| --- | --- | --- | --- | --- |
| `attn_norm.weight`, `post_attention_norm.weight` | all 40 | 2, F32 (8,192 B) | 2048 | `plain_block_pre` / verify `plain_w` / prefill `rms_norm_images`: `const float*`, `WeightKind::F32`, `n_embd` elements |
| `ffn_gate_inp.weight` | all 40 | 4, BF16 (1,048,576 B) | 2048 x 256 | `moe_route` (`bf16_gemv_fp32_mmvf`), prefill `bf16_proj`: `Bf16InF32` |
| `ffn_gate_inp_shexp.weight` | all 40 | 4, BF16 (4,096 B) | 2048 (1-D) | `moe_shared`, verify, prefill: `Bf16InF32`, `n_embd` elements (the one 1-D BF16 tensor) |
| `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` | all 40 | 0, native (Q6_K in IQ3_XXS, Q8_0 in Q4_K_XL) | 2048 x 512, 2048 x 512, 512 x 2048 | `moe_shared` through `NativeSharedWeights` (`native_data`, `native_type`) |
| `attn_qkv.weight`, `attn_gate.weight`, `ssm_out.weight` | 30 GDN | 0, native (Q6_K / Q8_0) | 2048 x 8192, 2048 x 4096, 4096 x 2048 | `gdn_layer` (`native_mmvq` through `w.native_data`), verify, prefill |
| `ssm_alpha.weight`, `ssm_beta.weight` | 30 GDN | 4, BF16 (131,072 B each) | 2048 x 32 | `gdn_layer` (`fused_gdn_ab`), prefill `bf16_proj`: `Bf16InF32` |
| `ssm_conv1d.weight` | 30 GDN | 2, F32 (131,072 B) | 4 x 8192 | `gdn_layer`: `const float*` (`ssm_d_conv x ssm_conv_channels`) |
| `ssm_a`, `ssm_dt.bias` | 30 GDN | 2, F32 (128 B each) | 32 | `gdn_layer`: `const float*` (`ssm_v_heads`) |
| `ssm_norm.weight` | 30 GDN | 2, F32 (512 B) | 128 | `gdn_layer` / `gdn_out_norm`: `const float*` (`ssm_state_size`) |
| `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight` | 10 attention | 0, native (Q6_K / Q8_0) | 2048 x 8192, 2048 x 512, 2048 x 512, 4096 x 2048 | `qsa_layer` (`gemv_quantized` takes the native path), verify, prefill |
| `attn_q_norm.weight`, `attn_k_norm.weight` | 10 attention | 2, F32 (1,024 B) | 256 | `qsa_layer` `normalize_rotate`: `const float*` (`head_dim`) |
| `output_norm.weight` | once | 2, F32 (8,192 B) | 2048 | `lm_head_mix` / verify's head: `WeightKind::F32`, `n_embd` |
| `output.weight` | once | 0, native (Q6_K / Q8_0) | 2048 x 248320 | `NativeHead::load` from the GGUF (`native_mmvq`); the index row gives `n_vocab = ne1` |
| `token_embd.weight` | once | 0, native (Q6_K / Q8_0) | 2048 x 248320 | `NativeEmbed::load` from the GGUF into mapped host memory (`iq_embed_rows`) |

Counts: 252 native rows (30 x 3 + 10 x 4 + 40 x 3 + 2), 140 BF16 rows, 221 F32 rows; 613 in all.  The arena
(`dense.bin` as loaded, every tensor at its `dst_off`) is 54,610,432 B.

**The types the native paths accept** (`native_type_problem`): projections and the head, `native_mmvq_supported`:
Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, Q2_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S,
IQ1_M; the embedding, `embed_type_supported` (the `is_iq` set plus BF16): the same minus Q4_0, plus BF16.  Any
other type on one of those names (F32, F16, Q4_1, Q2_K, IQ1_S, ...) is refused before anything is written, naming
the tensor, the type, the path and the list.  The head also needs `n_embd % 256 == 0` (2048: yes).

**The BF16 rule.**  The 140 F32 tensors read as BF16 (40 routers, 40 shexp gates, 30 `ssm_alpha`, 30 `ssm_beta`)
have their low 16 bits zero in every element of both files, as `step3-gdn.md` found: written as BF16 exactly,
`conversions.json` records 140 exact / 0 rounded / max error 0, no `compat-bf16.json`.  A file where that fails is
refused with "`<name>` is F32 with N of M values that are not BF16 (low 16 bits set), but the engine reads it as
BF16; --compat-bf16 would round them to nearest-even (a lossy pack, recorded in compat-bf16.json)".  `ssm_conv1d`
is BF16-exact too but the engine reads it as F32, so it is copied.  `attn_norm` is not BF16-exact (17%) and is
F32.

**The nextn block.**  A -MTP file (`nextn_predict_layers 1`, 41 blocks) has `blk.40.*`: an attention layer plus
MoE (its routers BF16, not F32) and `blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`.  17 non-expert
tensors; none of them, and none of its three expert tensors, is in the trunk's pack.  The drafter's pack is the
drafter agent's (`tools/mtp_qwen36_pack.py` per the study).

## 3. The expert profile

`python3 -I tools/make_profile.py --no-base --n-layer 40 --n-expert 256 --text-trace
~/Code/regina/analysis/qwen36-35b-single-gpu-ladder/routes/routes.txt --out data/qwen36-expert-profile.bin`
(the Mac, 14.6 s).  The raw `routes.txt` (269 MB, also as `.gz`) was present and used: the E0 trace of the IQ3_XXS
file through llama.cpp 3cf0325 on the M3 Ultra, 214,940 routed tokens (143,091 prefill, 71,849 decode) over the
110-prompt corpus (`README.md`, `summary.json` beside it).  Every one of the 10,240 (layer, expert) pairs was routed
at least once (mean entropy 7.37 of 8 bits), so the ranking is wholly from the trace: 10,240 from the trace, 0
filled.  The ranking is by count, ties broken by the smaller pair.  `read_expert_profile` reads it back on both cards
(section 5).

## 4. Build and test commands

Python (the Mac, `STRATA_GGUF_PY` at the main tree's gguf-py; VM 221 and VM 211 with their own):

    python3 -m unittest tools.test_iq_pack tools.test_make_profile      # 41 tests: 33 + 8 new, all three hosts
    python3 -I tools/iq_pack.py --gguf /models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --out /models/work/packs/qwen36-iq3

The engine (both cards, the common configure with `-DSTRATA_MMQ_KQUANTS=ON`, the 1080 Ti with
`-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`):

    cmake --build build -j 8 --target strata pack_load_check expert_layout_test model_geometry_test
    ./build/pack_load_check --native <file>.gguf --pack <dir>                       # section 5
    ./build/strata --native <file>.gguf --pack <dir> --tokens 1 --max-context 262144 --kv q4_0 --spec 4 \
        --prefill 2048 --expert-profile <dir>/expert-profile.bin                    # to the step-2 gate, exit 2

## 5. Results

### The packs (both VMs, 2026-10-11)

| Host, file | Pack | Time | `dense.bin` | rows / native / BF16 exact | expert table |
| --- | --- | ---: | ---: | --- | --- |
| VM 221, `/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | `/models/work/packs/qwen36-iq3` | 1.9 s | 54,610,432 B | 613 / 252 / 140 of 140 | == `tests/data/native_experts/qwen36-iq3_xxs.txt`; 11,108,614,144 B of experts |
| VM 221, `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (the plain 40-block file: VM 221's Q4_K_XL is NOT the MTP repo's, see 7) | `/models/work/packs/qwen36-q4` | 2.6 s | same bytes | 613 / 252 / 140 of 140 | 19,671,285,760 B; Q6_K down on layers 1, 34, 38, 39 (the plain file has one more Q6_K layer than the MTP file's 34, 38, 39) |
| VM 221, `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf` (copied from the Mac, sha256 55983c5a..., 22,853,663,008 B, 41 blocks) | `/models/work/packs/qwen36-q4-mtp` | 2.4 s | same bytes | 613 / 252 / 140 of 140; 17 draft tensors left out | == `qwen36-q4_k_xl.txt` up to the file name; 19,635,634,176 B |
| VM 211, `/home/rk/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | `/home/rk/packs/qwen36-iq3` | 2.6 s | same bytes | 613 / 252 / 140 of 140 | == `qwen36-iq3_xxs.txt` |
| VM 211, `/home/rk/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (the MTP repo's file, 41 blocks) | `/home/rk/packs/qwen36-q4` | 2.5 s | same bytes | 613 / 252 / 140 of 140; 17 draft tensors left out | == `qwen36-q4_k_xl.txt` |

The pack is 65 MB (dense 52.1 MiB, tokenizer 9.4 MiB, the rest under 200 KB); nothing of the experts or the
projections is copied.  `python3 -m unittest tools.test_iq_pack tools.test_make_profile`: 41 of 41 on the Mac,
VM 221 (numpy 2.2.4) and VM 211.

### VM 221, the RTX 4090 (sm_89), 2026-10-11

Build (`/models/work/qwen36-step4-pack`, the four targets after a fresh configure): 2 min 14 s.
`./build/pack_load_check --native FILE --pack DIR` (`/models/work/logs/qwen36-step4-check-*.log`):

| Pack | arena | native projections | head | embedding (mapped) | check_all | profile | exit |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| qwen36-iq3 | 54,610,432 B, read 11 ms + upload 7 ms | 250 matrices, 1,152,614,400 B (1,099 MiB), 1.8 s | Q6_K 417,177,600 B | Q6_K 417,177,600 B | 40 layers (10 + 30) ok | 10,240 pairs | 0 (3.1 s) |
| qwen36-q4 | same | 250, 1,492,910,080 B (1,424 MiB), 3.2 s | Q8_0 540,344,320 B | Q8_0 540,344,320 B | ok | 10,240 | 0 (5.8 s) |
| qwen36-q4-mtp | same; 259 names in the skip set (the 7 of `blk.40` are not in the index: harmless) | 250 (blk.40's not loaded: the trunk bound), 1,424 MiB | Q8_0 | Q8_0 | ok | 10,240 | 0 (2.4 s) |

Every weight of section 2's table resolves on every layer with the form its site demands (the log lists layers 0
and 3 in full and reports any miss on the other 38: none).  The gate run,
`./build/strata --native FILE --pack DIR --tokens 1 --max-context 262144 --kv q4_0 --spec 4 --prefill 2048
--expert-profile DIR/expert-profile.bin` (`qwen36-step4-gate-*.log`): the geometry (31 lines), the VRAM plan
(IQ3_XXS: fixed set 4,019 MiB = weights 1,596 + drafter 520 + KV 1,440 + session 162 + reserve 300; 10,240 of 10,240
slots resident at 19,729 MiB; 9,135 MiB spare), the step-2 gate message, exit 2, for both the IQ3_XXS and the MTP
Q4_K_XL file; no "missing", "wrong form" or other refusal in the 42 lines.  `ctest -R "pack_load_check|
expert_layout_test"` with `STRATA_QWEN36_GGUF` / `STRATA_QWEN36_PACK` set: 2 of 2.

### VM 211, the GTX 1080 Ti (sm_61, Pascal), 2026-10-11, under `/home/rk/VM-BUSY`

The same tree (`/home/rk/qwen36-step4-pack`, the submodule copied from the step-3 tree), configured with
`-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`; the four targets built in 3 min 22 s.
`./build/pack_load_check` (`logs/check-*.log`):

| Pack | arena | native projections | head | embedding | check_all | profile | exit |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `/home/rk/packs/qwen36-iq3` | 54,610,432 B, read 54 ms + upload 12 ms | 250, 1,099 MiB, 1.7 s | Q6_K 417,177,600 B, 0.5 s | Q6_K, 0.8 s | ok | 10,240 | 0 (3.5 s) |
| `/home/rk/packs/qwen36-q4` (the MTP file) | same | 250, 1,424 MiB, 2.1 s | Q8_0 540,344,320 B, 0.9 s | Q8_0, 1.2 s | ok | 10,240 | 0 (4.7 s) |

The gate run (`logs/gate-*.log`), both packs: the geometry, the VRAM plan, the gate message, exit 2, no refusal.
The plan on the 11 GB card: IQ3_XXS fixed set 4,019 MiB, 6,744 of 10,240 slots (65%) resident at 6,977 MiB;
Q4_K_XL fixed set 4,461 MiB (weights 2,038), 3,573 slots (34%) at 6,535 MiB.  `expert_layout_test` all passed;
`ctest -R "pack_load_check|expert_layout_test"` 2 of 2.  The lock was held from 20:50 to 20:58 UTC (sync and
packs done beforehand without it: they touch neither the card nor the build cores for more than seconds).

## 7. Not done, surprises, what the brief and the study had differently

- **The gate is before the loads.**  `generate.cpp`'s step-2 gate (line 2253) returns before the arena (2510),
  the native projections (2520), the head (3282) and the embedding (2424) are touched, so "load every dense weight
  ... through to the gate" cannot be shown by the gate run alone: `pack_load_check` does those loads in the same
  order and is the evidence; the gate run shows the start reaches the gate with the pack in place.  The
  gate-lifting agent gets section 6's order, including `check_all`, which `generate.cpp` has never called (the
  per-layer shape check exists since round 194 and only the parity binaries call it).
- **VM 221's `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` is the plain 40-block file** (733 tensors, 22,360,456,160 B,
  no `nextn_predict_layers`), not the MTP repo's as the common file said; the MTP repo's Q4_K_XL (41 blocks, sha256
  `55983c5a...`, the file on VM 211 and the Mac, and the one `tests/data/native_experts/qwen36-q4_k_xl.txt` and the
  plan's "Q4 number" refer to) was copied to VM 221 as `/models/Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf` (rsync from the
  Mac, 3 min 40 s), so the drafter has `blk.40` on the development card.  The two files are different quantization
  runs: the plain one has Q6_K down on layers 1, 34, 38, 39 and a 35.6 MB larger expert total.
- **Neither VM's python has `regex`** (PEP 668 refuses `pip install` on VM 221; VM 211 has no pip), hence the lazy
  import in `strata_tokenizer.py`.  The export is exact without it; `encode` needs it.
- **`check_one` still demanded the indexer rows at hc = 1** (the attention agent's note said the four tensors "are
  required only with an indexer", which was true of `qsa_layer` and the window but not of `layout.cpp`).  Fixed
  here, `[shared]`.
- **The study's step-4 line** had the routers and the SSM gates "F32 to BF16 under `--compat-bf16` with
  `compat-bf16.json`"; the census (step3-gdn.md) made that moot and the packs carry no `compat-bf16.json`.
  `step2-notes.md` section 3 ("need `--compat-bf16` today, which step 4 decides") is superseded: decided, not needed.
- **`ModelGeometry::n_vocab`** is the Flash-Next default (248,320, which happens to be Qwen3.6's too); the head and
  the embedding are sized from `output.weight`'s index row, as `generate.cpp` does.
- Not mine, not done: lifting the gate, the verify window and the first token through this pack, the drafter's pack
  from `blk.40`, the chat template check, `generate.cpp` beyond the one-argument dense-load bound.

## 6. The interface other agents consume

- The pack layout of section 2, file by file and row by row.  A consumer resolves a tensor by name through
  `WeightTable::find`; a native row has `data == nullptr`, `resident == false` after the skip-set load, and
  `native_data` / `native_type` after `NativeDense::load`; `output.weight`'s row is shape only and the head is
  `NativeHead`; `token_embd.weight` is `NativeEmbed`.
- The load order the gate-lifting agent keeps (it is `generate.cpp`'s after the gate, and `pack_load_check.cpp`'s):
  `read_model_geometry` -> `check_model_tensor_forms` -> `expert_layout_load(pack, n_layers, n_expert, n_embd, n_ff)`
  -> `NativeDense::served_names(shards, false, skip)` + `output.weight` + `token_embd.weight` -> `pool_bytes(pack,
  &skip)` -> `WeightTable::load(pack, arena, pool, &skip)` -> `NativeDense::load(shards, wt, err, false, 0,
  n_layers)` -> `check_all(wt, g, err)` (generate.cpp does not call it today; it should, after the dense load) ->
  `NativeHead::load(shards, n_embd, wt.find("output.weight")->ne1)` -> `NativeEmbed::load(shards, n_embd, n_vocab)`
  -> `read_expert_profile(pack + "/expert-profile.bin", n_layers, n_expert, ...)`.
- `tools/iq_pack.py`: `form_of(name, arch)`, `needs_bf16(name, type, arch)`, `native_type_problem(name, type, arch)`,
  `layer_of(name)`, `profile_problem(path, n_layers, n_expert)`, `index_standalone(src, out, model, compat_bf16,
  n_trunk)`, `model_arch(model)`; `main` takes `--expert-profile`.
- `tools/make_profile.py`: `read_profile(path, n_expert, n_layer)`, `read_text_trace(path, n_expert, n_layer)`,
  `write_profile(path, ranked, n_expert, n_layer)`, `profile_header(path)`, `rank_profile(..., n_layer=)`.
- `tests/core/pack_load_check.cpp`: `pack_load_check --native FILE --pack DIR` (or `STRATA_QWEN36_GGUF` /
  `STRATA_QWEN36_PACK`), exit 0 / 1 / 77.
