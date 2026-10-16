[Codebase docs](../README.md) › [Guides](README.md) › Data files

# Data files: `data/` and `tests/data/native_experts/`

## Purpose

The repository ships six kinds of binary or text data that the engine, the pack tools or the tests read. None of them
is model weights. This page says, for each file: what it is, its exact format (taken from the code that reads and
writes it), which model it belongs to, how it was produced, and who consumes it.

| File | Bytes / lines | Model | What it is |
| --- | --- | --- | --- |
| [`data/draft_vocab.bin`](../../../data/draft_vocab.bin) | 425,196 B (106,299 ids) | both (same 248,320-token vocabulary) | the MTP draft head's token subset: English/code plus every CJK token |
| [`data/draft_vocab_en.bin`](../../../data/draft_vocab_en.bin) | 162,100 B (40,525 ids) | both | the smaller English/code-only draft subset |
| [`data/expert-profile.bin`](../../../data/expert-profile.bin) | 196,632 B (48 x 512) | Qwen3.8-Flash-Next | expert-cache ranking of all 24,576 (layer, expert) pairs |
| [`data/qwen36-expert-profile.bin`](../../../data/qwen36-expert-profile.bin) | 81,944 B (40 x 256) | Qwen3.6-35B-A3B | expert-cache ranking of all 10,240 pairs, from a llama.cpp route trace |
| [`tests/data/native_experts/*.txt`](../../../tests/data/native_experts) | 8 files, 41-49 lines each | Flash-Next family (6) and Qwen3.6 (2) | copies of real packs' `native_experts.txt` for `expert_layout_test` |

The SHA-256 of `draft_vocab.bin`, `expert-profile.bin` and `qwen36-expert-profile.bin` match the pack files listed in
[UPSTREAM.md](../../../UPSTREAM.md) (`mtp/draft_vocab.bin` and `qwen36-mtp-draft/draft_vocab.bin` = `b1e1d3a7...`,
`pack-iq3-xxs/expert-profile.bin` = `8f59b4aa...`, `qwen36-{q4,iq3}/expert-profile.bin` = `eb2ed6c8...`): the data
files are the sources those pack files are copied from. `draft_vocab_en.bin` is referenced by no tool default.

---

## 1. Draft vocabulary subsets (`draft_vocab.bin`, `draft_vocab_en.bin`)

### Format

A headerless array of little-endian `int32` token ids (`array("i", ids).tobytes()` in
`tools/draft_vocab.py:140`; `np.fromfile(dv, dtype="<i4")` in `tools/mtp_qwen36_pack.py:221`). The file size must be
a positive multiple of 4; the count is `size / 4`. The position of an id in the file is its index in the draft
head (subset index -> token id), so order matters.

| File | Ids | Order | Relation |
| --- | --- | --- | --- |
| `draft_vocab_en.bin` | 40,525 | ascending | the English/code subset upstream shipped before 0.1.27 |
| `draft_vocab.bin` | 106,299 | the 40,525 English/code ids first (identical prefix), then the added CJK ids in id order | `draft_vocab_en.bin` + `--add cjk` (checked: the first 40,525 entries are byte-identical, no duplicates) |

Both hold ids in `[0, 248319]`, the 248,320-token vocabulary that Flash-Next and Qwen3.6-35B-A3B share
(`tools/mtp_qwen36_pack.py:26-27`, `docs/qwen36-port/geometry-inventory.md` row 77).

### Reader (engine)

- `MtpDrafter::bind` (`src/core/mtp.cpp:561`, the subset block at `src/core/mtp.cpp:575-594`) reads
  `<--mtp dir>/draft_vocab.bin` when present. It checks only `size >= 4 && size % 4 == 0`; it does **not** check
  that the ids are inside the vocabulary. It uploads the ids, allocates `n_ids x row_bytes` of VRAM for a draft
  head and gathers those rows of the native output head into it (`gather_rows`), then logs
  `draft head over N tokens`. Draft logits are then computed over the subset only (the sampler maps subset index
  back to token id, `src/kernels/cuda/sampler.cu:766`).
- `MtpDrafter::bind_bytes` (`src/core/mtp.cpp:451`) prices the head for the VRAM planner from the file size
  (`n_ids x row_bytes + file size`); `src/program/generate.cpp:2224` also reads the file size for the plan.
- When the head does not fit, `draft_head_hint` (`src/core/mtp.cpp:526`) prints the smaller subsets upstream's
  setup offered (`cyrillic` 58,963, `en` 40,525) and tells the user to run `START-HERE.bat` / `setup.sh` with
  `--draft-vocab en`. Those installers are upstream's and are not in this fork (see Gotchas).
- No file means no subset: the drafter scores the whole vocabulary (more VRAM, slightly higher acceptance;
  UPSTREAM.md quotes "a few percent higher acceptance, 345 MiB more VRAM" for Qwen3.6).

The front end does not require the file (`frontend/main.cpp:666-669`; it used to, fault 1 of
`docs/qwen36-port/step5-serve.md`).

### Writers and how the shipped files were produced

- `tools/draft_vocab.py` (146 lines) builds subsets: `--base` an existing subset, `--add han|kana|hangul|cjk_punct|cjk|cyrillic`
  adds whole scripts (Unicode ranges at `tools/draft_vocab.py:35-43`), `--corpus FILES --coverage 0.99` adds the most
  frequent tokens of a text corpus (#597), `--stats` prints what a subset holds. It needs the model GGUF for the
  vocabulary (`tools/strata_tokenizer.py`). Base ids keep their order; added ids follow in id order.
- The shipped `draft_vocab.bin` came from upstream Strata (UPSTREAM.md copies `$U/data/draft_vocab.bin` into the
  Flash-Next `mtp/` directory); its docstring's example (`--base data/draft_vocab.bin --add cjk`) and DETAILS.md
  ("Since 0.1.27 the subset includes every Chinese, Japanese and Korean token (106,299 ids)") describe how it was made.
- `tools/mtp_qwen36_pack.py` (`--draft-vocab`, `tools/mtp_qwen36_pack.py:98-100, 214-228`) copies
  `data/draft_vocab.bin` into a Qwen3.6 draft pack by default, refuses ids outside `[0, n_vocab)`, records
  `{"file", "tokens"}` in `mtp.json`, and with `--draft-vocab none` leaves the file out.

### Model use

| Model | Where the file ends up | How |
| --- | --- | --- |
| Qwen3.8-Flash-Next | `mtp/draft_vocab.bin` beside the Q2_0 draft head | `cp data/draft_vocab.bin mtp/` (UPSTREAM.md) |
| Qwen3.6-35B-A3B | `qwen36-mtp-draft/draft_vocab.bin` | `tools/mtp_qwen36_pack.py` default |

`docs/qwen36-port/step4-drafter.md` measured the subset's cost on Qwen3.6: 31 of 32 top-1 agreements with llama.cpp's
MTP arm instead of 32 of 32 (" fox", id 37550, is not in the subset), and in `step4-integration.md` 0.72 vs 0.77
accepted per round.

### Tests

`tools/test_draft_vocab.py` (4 tests, `python -m unittest tools.test_draft_vocab`) covers the corpus-coverage mode
on synthetic counts; no test reads the shipped files.

---

## 2. Expert-cache profiles (`expert-profile.bin`, `qwen36-expert-profile.bin`)

### Format (`STRP` v1)

Little-endian, written by `write_profile` (`tools/make_profile.py:92`) and, byte for byte the same, by
`write_expert_profile` (`src/core/expert_cache.cpp:110`).

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `STRP` |
| 4 | 4 x `uint32` | `version` (1), `n_layers`, `n_expert`, `slots`, `n_ranked` (the tools write `slots = n_ranked`) |
| 24 | `n_ranked` x (`uint16` layer, `uint16` expert) | the ranked pairs, hottest first |
| 24 + 4 `n_ranked` | `n_layers` x `n_expert` x `int32` | each pair's rank (its index in the list), -1 when unranked |

Size = 24 + 4 `n_ranked` + 4 `n_layers` `n_expert`. Checked against the shipped files:

| File | Header | Size | First pairs |
| --- | --- | --- | --- |
| `expert-profile.bin` | v1, 48 x 512, slots 24,576, ranked 24,576 | 196,632 B | (28, 288), (46, 185), (47, 242), ... |
| `qwen36-expert-profile.bin` | v1, 40 x 256, slots 10,240, ranked 10,240 | 81,944 B | (38, 209), (28, 130), (27, 218), ... |

### Readers

- `read_expert_profile` (`src/core/expert_cache.cpp:23`) checks the magic, requires `n_layers x n_expert` to equal
  the model's ("a profile for a different artifact" otherwise), `n_ranked <= slots`, every pair in range, and
  returns the ranked list and `slots`. It ignores `version` and never reads the trailing table
  (`include/strata/core/expert_cache.hpp:37-52`).
- `src/program/generate.cpp:2984-3025` (`--expert-profile P`) loads it before the expert cache is filled; with
  `--expert-cache 0` the count becomes `auto` (the profile ranks every pair, issue #46). The cache then fills its
  slots in rank order; `STRATA_PEER_HOT` reorders the head of the list for a peer GPU (`generate.cpp:2999-3022`).
  `generate.cpp:2852-2858` reads it a second time for the multi-GPU `--split-skip-if-fits` decision.
- The front end always starts the engine with `--expert-profile <pack>/expert-profile.bin` and refuses to start if
  that file is missing (`frontend/main.cpp:665-669, 678`), so every pack served through `unbiased-strata` must carry one.
- Python: `profile_header` / `read_profile` (`tools/make_profile.py:38, 46`) and `profile_problem`
  (`tools/iq_pack.py:594`), which refuses a profile whose layers x experts differ from the model's trunk.

### Writers and how the shipped files were produced

- `tools/make_profile.py` (176 lines). Ranking (`rank_profile`, `tools/make_profile.py:104`): by default the base
  profile's order, then pairs from routing traces (most frequent first), then a fill of every missing pair
  interleaved across layers; `--reorder` puts the traces first; `--no-base` drops the base. Traces are binary
  `--dump-routing` files (`int32 layer, int32 k, k int32 ids, k float weights` records, `read_trace`) or llama.cpp
  eval-callback text (`<layer> <e0> ... <e(k-1)>` per token per layer, `.gz` accepted, `--text-trace`).
  `--n-layer` (default 48) and `--n-expert` (default 512) set the geometry; with `--base` the base header decides.
  `--out` is required, so a bare run cannot overwrite the pinned Flash-Next profile (`tools/make_profile.py:139`).
- `data/expert-profile.bin` (Flash-Next) came from upstream Strata unchanged (UPSTREAM.md: `cp $U/data/expert-profile.bin
  pack-iq3-xxs/`); the traces it was built from are not in this repository.
- `data/qwen36-expert-profile.bin` was made by
  `python3 -I tools/make_profile.py --no-base --n-layer 40 --n-expert 256 --text-trace .../routes.txt --out data/qwen36-expert-profile.bin`
  from the "E0" trace of the IQ3_XXS file through llama.cpp 3cf0325 on an M3 Ultra: 214,940 routed tokens over a
  110-prompt corpus, every one of the 10,240 pairs routed at least once, so the ranking is wholly from the trace,
  ties broken by the smaller pair (`docs/qwen36-port/step4-pack.md` section 3).
- `tools/iq_pack.py --expert-profile` (`tools/iq_pack.py:627-629, 655-727`) copies a profile into a pack as
  `expert-profile.bin`, defaulting to `data/qwen36-expert-profile.bin` for a `qwen35moe` model when it fits;
  UPSTREAM.md also copies it by hand.
- At run time `--expert-profile-save P` (`--serve`, #477) writes a learned profile in the same format:
  `rank_learned_profile` (`src/core/expert_cache.cpp:81`: resident pairs first, then by counted routing heat, then by
  the loaded profile's order, then by index) and `write_expert_profile` (atomic: `<path>.tmp` then rename).

### Tests

- `tests/core/expert_profile_save_test.cpp` - `rank_learned_profile` order and `write_expert_profile` bytes against
  make_profile.py's format, read back by `read_expert_profile` (CPU only).
- `tools/test_make_profile.py` (9 tests) - ranking modes, trace readers, `--n-layer`, round trip.
- `tools/test_iq_pack.py` - the Qwen3.6 pack copies a fitting profile and refuses one of another geometry
  (`tools/test_iq_pack.py:812-820`).

---

## 3. `native_experts.txt` fixtures (`tests/data/native_experts/`)

### What they are

Copies of the `native_experts.txt` that `tools/iq_pack.py` wrote for real model files. A native pack keeps every
routed expert in its GGUF encoding; this text file is the expert table: one line per layer giving the gate/up and
down ggml types, the layer's offset in the expert arena, the per-expert blob size (`[gate rows | up rows | down rows]`),
and the absolute byte offsets of the three tensors in the GGUF, plus (v3/v4) the shard that holds them.

### Format

```
# strata native experts vN: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert E, total T; ...)
<layer> <gu_type> <d_type> <offset> <blob_bytes> <gate_off> <up_off> <down_off> [<shard> | <gate>,<up>,<down>]
```

| Version | Adds | Writer condition |
| --- | --- | --- |
| v1 | `layer gu_type d_type offset blob_bytes` | old packs (no GGUF offsets; `src/core/expert_source.cpp:2523` asks for a repack) |
| v2 | the three absolute GGUF offsets | |
| v3 | an optional shard file name per layer (a file beside `--native`); `(n_expert E` in the header wins over the caller's count | `tools/iq_pack.py:571` |
| v4 | the shard column may be `gate,up,down` (empty = the `--native` shard) when a shard boundary falls inside a layer | `tools/iq_pack.py:565`, only when some layer is split |

Types are `ggml_type` ids (`tools/gguf_reader.py:19-29`): 12 Q4_K, 13 Q5_K, 14 Q6_K, 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS,
20 IQ4_NL, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 29 IQ1_M, 42 Q2_0. The reader is `expert_layout_load`
(`src/kernels/cpu/expert_layout.cpp:233-345`): it refuses a version above `kExpertLayoutVersion = 4`
(`include/strata/kernels/cpu/expert_layout.hpp:85`), asks `native_fmt` (`src/kernels/cpu/native_expert.cpp:32`) for the
blob size each type pair makes at the model's `n_embd x n_ff` and refuses a mismatch ("the pack is of another model"),
and requires the layers to be contiguous in the arena. A pack with no `native_experts.txt` is the canonical Q2_0
form, accepted only at Flash-Next's 2560 x 640. The Qwen3.6 draft pack writes its own one-line variant (offsets
inside `experts.bin`), read by `MtpDrafter::load_qwen36_experts` (`src/core/mtp.cpp:238`), not by these tests.

### The eight files

| File | Header | Source GGUF (named in the header) | Layers x experts | Expert type pairs (gate-up / down) | Arena total |
| --- | --- | --- | --- | --- | --- |
| `q2_0.txt` | v2 | Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 | 48 x 512 | Q2_0 / Q2_0 | 33,973,862,400 B |
| `iq2_xs.txt` | v2 | Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS | 48 x 512 | IQ2_XXS, IQ2_S, IQ1_M / Q2_0 | 35,454,976,000 B |
| `iq3_xxs.txt` | v2 | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS | 48 x 512 | IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S / IQ4_NL, Q2_0 | 42,912,972,800 B |
| `iq3_s.txt` | v3 | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S | 48 x 512 | IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS / IQ4_NL, Q2_0 | 50,292,326,400 B |
| `coder-iq1_m.txt` | v3 | Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M (the pruned "Coder") | 48 x 256 | IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS / IQ4_NL, Q2_0 | 25,146,163,200 B |
| `swift-iq2_xs.txt` | v3, 35 lines with a shard column | Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS (layers 13-47 in shard 2) | 48 x 512 | IQ2_XXS, IQ2_S, IQ1_M / Q2_0 | 35,454,976,000 B |
| `qwen36-iq3_xxs.txt` | v3 | Qwen3.6-35B-A3B-UD-IQ3_XXS | 40 x 256 | IQ2_S / IQ3_XXS (37 layers), IQ4_XS (34, 38, 39) | 11,108,614,144 B |
| `qwen36-q4_k_xl.txt` | v3 | Qwen3.6-35B-A3B-UD-Q4_K_XL | 40 x 256 | Q4_K / Q5_K (37), Q4_K / Q6_K (34, 38), Q5_K / Q6_K (39) | 19,635,634,176 B |

The Qwen3.6 pair was produced on 2026-10-11 by `tools/iq_pack.py --layout-only` over the real files
(`tests/core/expert_layout_test.cpp:110-115`, `docs/qwen36-port/step2-notes.md` section 3); the six Flash-Next-family
files are copies of upstream packs' tables (the test header lists them; their exact provenance is not stated).

### Consumers

- `expert_layout_test` (`tests/core/expert_layout_test.cpp`, registered at `CMakeLists.txt:476` with the directory as
  its argument): the six Flash-Next-family files parse at 48 x 512 x 2560 x 640 to exactly what the pre-v4 reader
  produced, field by field, with totals equal to the headers (`real_packs`); the two Qwen3.6 files load at
  40 x 256 x 2048 x 512 with the expected per-layer formats and blob sizes, and are refused at Flash-Next's
  2560 x 640 (`qwen36`). The rest of the test builds synthetic manifests and GGUFs.
- `tests/core/vram_plan_test.cpp` hard-codes the two Qwen3.6 totals (it does not read the files).
- `expert_layout_test --real PACK_DIR NATIVE_SHARD [N_LAYERS N_EXPERT N_EMBD N_FF]` checks a real pack against its
  GGUF headers (by hand; `tests/core/expert_layout_test.cpp:383-401`).

---

## Gotchas

- The draft-vocab reader trusts the ids: an id at or above `n_vocab` is not caught by the engine
  (`src/core/mtp.cpp:575-594`); `tools/mtp_qwen36_pack.py` is the only place that range-checks.
- `draft_head_hint` and DETAILS.md tell users to run `START-HERE.bat --draft-vocab en` or `./setup.sh --draft-vocab en`.
  This fork has no installer: to use the English subset, put `data/draft_vocab_en.bin` into the draft directory as
  `draft_vocab.bin` (for Qwen3.6, `tools/mtp_qwen36_pack.py --draft-vocab data/draft_vocab_en.bin`).
- The comment in `frontend/main.cpp:666-667` names a `tools/mtp_qwen36_pack.py --full-head` option; the tool's
  option is `--draft-vocab none`.
- `include/strata/core/expert_cache.hpp:39-40` calls the profile's trailing table "a frequency table"; both writers
  store each pair's rank there (`tools/make_profile.py:92-101`, `src/core/expert_cache.cpp:117-141`). No reader uses it.
- `tools/make_profile.py`'s docstring speaks of "all 24,576" pairs; that is Flash-Next's 48 x 512. Qwen3.6 needs
  `--n-layer 40 --n-expert 256` (or a 40 x 256 `--base`).
- A Flash-Next profile on a Qwen3.6 pack (or the reverse) is refused at start by `read_expert_profile` with
  "is 48x512 but this model is 40x256".

## See also

- [testing.md](testing.md) - where `expert_layout_test`, `expert_profile_save_test` and the Python tests run.
- [glossary.md](glossary.md) - pack, profile, slot, draft head, draft vocab.
- [../core/experts/README.md](../core/experts/README.md) - the expert cache and sources that consume the profile and the layout.
- [../tools/README.md](../tools/README.md) - `iq_pack.py`, `make_profile.py`, `draft_vocab.py`, `mtp_qwen36_pack.py`.
- [UPSTREAM.md](../../../UPSTREAM.md) - the pack-building commands and checksums.
- [docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md) section 3 and
  [step4-drafter.md](../../qwen36-port/step4-drafter.md) section 2 - how the Qwen3.6 files were made.
