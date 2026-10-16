[Codebase docs](../README.md) › [Tools](README.md) › Profile and draft vocabulary

# Expert profile and draft vocabulary builders

## Purpose

Two small data files steer the engine at run time and are built offline:

- `expert-profile.bin`: a ranking of every (layer, expert) pair. `--expert-cache auto` fills the VRAM expert
  slots in this order, so it decides which experts start resident.
- `draft_vocab.bin`: the token subset the MTP draft head may propose. A smaller subset is a smaller, faster head
  but tokens outside it are never drafted.

| File | Lines | Role |
| --- | --- | --- |
| [`tools/make_profile.py`](../../../tools/make_profile.py) | 176 | Writes `expert-profile.bin` from a base profile and routing traces |
| [`tools/draft_vocab.py`](../../../tools/draft_vocab.py) | 146 | Extends `draft_vocab.bin` by Unicode script or by corpus coverage |
| [`tools/test_make_profile.py`](../../../tools/test_make_profile.py) | 145 | Unit tests, no model |
| [`tools/test_draft_vocab.py`](../../../tools/test_draft_vocab.py) | 47 | Unit tests of the corpus coverage mode, no model |

Shipped data (`data/`): `expert-profile.bin` (Flash-Next, 48 x 512), `qwen36-expert-profile.bin` (Qwen3.6, 40 x 256),
`draft_vocab.bin` (106,299 ids), `draft_vocab_en.bin` (40,525 ids). Formats: [pack-format.md](pack-format.md)
sections 1.7 and 4.

## `tools/make_profile.py`

```
python3 tools/make_profile.py [TRACE ...] --out PATH
        [--base data/expert-profile.bin | --no-base] [--reorder]
        [--n-expert 512] [--n-layer N] [--text-trace ROUTES[.gz] ...]
```

| Function | What it does |
| --- | --- |
| `profile_header(path)` (`:38`) | `(version, n_layer, n_expert, slots, n_ranked)` from the 24-byte `STRP` header |
| `read_profile(path, n_expert, n_layer)` (`:46`) | the ranked `(layer, expert)` list; refuses another geometry |
| `read_trace(path, ...)` (`:56`) | binary `--dump-routing` traces: records of `int32 layer, int32 k, k x int32 ids, k x 4-byte weights`; counts `(layer, e)` hits in range |
| `read_text_trace(path, ...)` (`:70`) | llama.cpp eval-callback route traces, one line `<layer> <e0> ... <e(k-1)>` per token per layer, `#` lines skipped, `.gz` accepted; any out-of-range or non-integer line is an error |
| `rank_profile(base, freq, ...)` (`:104`) | default: base order, then trace pairs by descending count (ties: smaller pair first), then every missing pair layer-interleaved (`e` outer, layer inner). `--reorder` or `--no-base` put the trace pairs first |
| `write_profile(path, ranked, ...)` (`:92`) | `STRP`, header `(1, n_layer, n_expert, len, len)`, the `u16` pairs, then the `i32` slot table |

Rules in `main` (`:131-172`): `--out` is required (it no longer defaults to the shipped file, which UPSTREAM.md
pins by hash). With a base, the base header decides the layer count and a differing `--n-layer` or `--n-expert`
is an error; without one, `--n-layer` defaults to 48 (`N_LAYER`, Flash-Next). The output is read back and compared
before the tool reports success.

Because the shipped Flash-Next profile already ranks all 24,576 pairs, a trace added with the default mode changes
nothing; `--reorder` is how a workload trace moves pairs to the top.

**Qwen3.6:** `data/qwen36-expert-profile.bin` was made with
`--no-base --n-layer 40 --n-expert 256 --text-trace .../routes.txt` from a llama.cpp route trace of the IQ3_XXS file
(214,940 routed tokens; every pair was routed at least once, so the ranking is entirely from the trace)
([docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md) section 3). `tools/iq_pack.py` copies it into a
qwen35moe pack by default when its header matches.

**Engine side:** `read_expert_profile` (`src/core/expert_cache.cpp:23`) checks the geometry; the engine's
`--expert-profile-save` writes the same format from what the adaptive tier learned
(`src/core/expert_cache.cpp:120-145`, `src/program/generate.cpp` option `--expert-profile-save`).
`--dump-routing PATH` on the engine produces the binary traces (`src/program/generate.cpp:641`, `:4132-4140`).

## `tools/draft_vocab.py`

```
python3 tools/draft_vocab.py --gguf <model shard 1> [--base data/draft_vocab.bin]
        [--add han,kana,hangul,cjk_punct,cjk,cyrillic] [--corpus f1.txt ...] [--coverage 0.99]
        [--out new.bin] [--stats]
```

- Loads the vocabulary with `strata_tokenizer.Tokenizer.from_gguf` and classifies every token by the Unicode
  scripts its decoded text contains (`SCRIPTS`, `GROUPS`, `scripts_of`, `:46`; tokens that are not whole UTF-8
  characters are unclassified, `token_text`, `:56`).
- `--add` appends every token of the named scripts not already in the base, in id order; `cjk` expands to
  `han, kana, hangul, cjk_punct`.
- `--corpus` tokenizes UTF-8 files as plain text (`corpus_counts`, `:64`) and adds the most frequent tokens that
  together cover `--coverage` of the occurrences (`covering`, `:72`; ties take the lower id); it prints the share of
  occurrences outside the base and outside the new subset (`outside`, `:84`).
- The base ids keep their order; additions follow. `--stats` (or no `--out`) prints per-script counts.

Model notes: both models share the 248,320-token vocabulary, so the shipped subsets apply to Qwen3.6 too
(`tools/mtp_qwen36_pack.py` copies `data/draft_vocab.bin` by default). Measured trade-off for Qwen3.6, from
UPSTREAM.md: `--draft-vocab none` gives a few percent higher acceptance and costs 345 MiB more VRAM.

## Tests

```
python3 -m unittest tools.test_make_profile tools.test_draft_vocab
```

Neither needs a model, gguf-py or a GPU. `test_make_profile` covers the ordering modes, tie breaks, text traces
and the layer count from `--n-layer` or the base header; `test_draft_vocab` covers `covering`, `outside` and
multi-file corpora. `test_iq_pack.Qwen36PackTests.test_expert_profile_is_checked_and_copied` covers the
profile check in `iq_pack.py`.

## Gotchas

- `--corpus` and the script classification need the `regex` module (through `strata_tokenizer`).
- `make_profile.py` defaults to 48 layers and 512 experts; Qwen3.6 needs `--n-layer 40 --n-expert 256` (or a
  matching base).
- A trace of another model is rejected only by the range checks; `read_trace` silently skips out-of-range records
  while `read_text_trace` refuses them.

## See also

- [pack-format.md](pack-format.md), [pack-builders.md](pack-builders.md), [mtp-tools.md](mtp-tools.md)
- [docs/qwen36-port/step4-pack.md](../../qwen36-port/step4-pack.md),
  [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md)
