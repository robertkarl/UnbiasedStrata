# Upstream Strata tooling the Qwen3.6 port needs back

Surveyed 2026-10-11. Read-only references:

- `/Users/robertkarl/Code/strata-upstream-6f32ec0`: exact snapshot, exported with `git archive 12a7825` from this repository's first commit ("Import Strata at 6f32ec0, unmodified"). Use this one.
- `/Users/robertkarl/Code/strata-upstream`: a clone of `Niko1221/Strata` main (1859 commits). **Commit `6f32ec0` is not in the public history**: `git checkout` fails with "unable to read tree", so the pinned commit was force-rewritten or never public. Main still carries the same tooling and adds `bench/bench_vs_llama.py`.

Nothing was built. UnbiasedStrata was not modified.

## 1. The finding that changes the plan

Upstream's Python oracle is **not** published. `ref/` contains only `ref/load.py`, in the snapshot and in main, in every commit. The engine source cites `ref/gdn.py` (22 times), `ref/qsa.py` (26), `ref/moe.py` (12), `ref/gr.py` (11), `ref/ngram.py` (10), `ref/model.py` (9) and `ref/quant.py` (7), and `docs/semantics.md` and `docs/activation-contract.md` do not exist in any commit. The same holds for `bench/micro/` (the ggml-oracle tests), `tests/python/gen_ple_oracle.py`, `scripts/_gen_f16_oracle.py` and `tests/CMakeLists.txt`. The published CMake says so: "the test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out".

So the Maya recipe (`ref/glm.py` plus a synthetic GGUF and dumps) has no upstream Strata counterpart to restore for the mixers or the MoE. What is restorable is `ref/load.py`, the pack tools, the in-C++ parity tests, and the discipline. The Qwen3.6 oracle is new code. It is specified from the pinned llama.cpp graph `third_party/llama.cpp/src/models/qwen35moe.cpp`, as Maya specified `ref/glm.py` from `glm5-next.cpp`.

## 2. Inventory

### `ref/`

- `ref/load.py` (212 lines). `Gguf(path)`: a demand-paged view of a possibly sharded GGUF (shard discovery from the `-0000N-of-0000M` suffix, a directory, or a list). `tensor(name)` returns FP32 shaped (out, in[, experts]). `expert(name, e)` decodes one expert of a fused `[in, out, experts]` tensor without touching the others. `rows(name, idx)` gathers rows of a 2-D tensor (built for the 27 GB PLE table; Qwen3.6 has no PLE). One dequant dispatch `_dequant_flat`: F32, F16, BF16 by hand; Q2_0 through `tools/gguf_writer.dequantize_q2_0`; everything else through gguf-py `quants.dequantize` (the table GEOM holds 13 entries: F32 F16 BF16 Q4_0 Q5_0 Q8_0 IQ4_NL Q3_K Q4_K Q5_K Q6_K IQ4_XS Q2_0). It lacks Q4_1, Q5_1, Q2_K and the IQ1/IQ2/IQ3 types the engine itself serves; a Q4_K_M or UD-Q4_K_XL Qwen3.6 file needs only Q4_K, Q5_K, Q6_K, Q8_0, F32, BF16 or F16, all present. It imports `gguf` without adding it to `sys.path`, so `PYTHONPATH` or `STRATA_GGUF_PY` must point at gguf-py (see section 5).

### `tools/` (the ones a port uses)

- `tools/iq_pack.py` (602 lines). The native-pack builder, described in section 4. Hardcodes Flash-Next assumptions listed in section 6.
- `tools/gguf_reader.py` (142). A GGUF v3 header reader with no gguf-py dependency; knows ggml type ids to 42 (Q2_0). Used by `iq_pack`, the tokenizer exporter and `ref/load.py`. It is the Python twin of `src/artifact/gguf_reader.cpp`.
- `tools/gguf_writer.py` (234). A minimal GGUF v3 writer with a Q2_0 quantizer. Its self-test writes a synthetic file with `general.architecture = qwen4exp`. Not a model generator.
- `tools/iq_fixture.py` (137). Writes deterministic i-quant blocks plus gguf-py's dequantized reference for `iq_parity` (see section 3). It is the one fixture generator upstream ships.
- `tools/test_iq_pack.py` (528). The closest thing to a synthetic-GGUF generator upstream ships: `write_gguf()` builds tiny sharded GGUFs with gguf-py's `GGUFWriter` and `quants.quantize`, then asserts pack behaviour (BF16 rounding to even, refused inexact F32 routers, split shards, a layer split across shards, interrupted packs, stale `experts.bin`). Arch string is `qwen4exp`.
- `tools/make_profile.py` (119). Writes `expert-profile.bin`, a ranking of every (layer, expert) pair, from the base profile plus routing traces from `generate --dump-routing FILE`. Has `--n-expert` but `N_LAYER = 48` is hardcoded (line 26) and the output header carries layer and expert counts that the engine checks. Test: `tools/test_make_profile.py`. The shipped `data/expert-profile.bin` is for Flash-Next and is useless for Qwen3.6 (40 layers, 256 experts).
- `tools/mtp_fetch.py` (308). Range-reads the 31 `mtp.*` tensors out of the 131-shard BF16 checkpoint over HTTP and verifies SHA256s (`inventory`, `fetch`, `verify`). Pinned to Qwen3.8-Flash-Next.
- `tools/mtp_pack.py` (193). Quantizes those tensors into an MTP GGUF (`--experts q2_0|q4_0|q8_0`). Hardcodes the Flash-Next tensor names and the 512-expert fused shapes.
- `tools/mtp_rt.py` (111). Turns the MTP GGUF into the engine's runtime files: `experts.bin`, `dense.bin`, `dense.txt`. Applies Gemma-style `+1` to RMSNorm weights.
- `tools/draft_vocab.py` (+ `test_draft_vocab.py`). Extends `draft_vocab.bin`, the token subset the draft head may propose, by script (`--add han|kana|...`) or corpus (`--corpus`, `--coverage`). Data files `data/draft_vocab*.bin` are tied to the Flash-Next vocabulary of 248,320 tokens.
- `tools/strata_tokenizer.py` (331). Exports `tokenizer/` (vocab, merges, chat template) from GGUF metadata. Refuses pre-tokenizer schemes it has not transcribed; Qwen3.6 uses `qwen35`, same family as Flash-Next. Called by `iq_pack` when `<out>/tokenizer/` is missing. Needs `regex`.
- `tools/strata_pack.py`, `pack_layer.py`, `pack_index.py`, `canonical_xcheck.py`. The canonical Q2_0 pack (bakes in the 1,382,400-byte Flash-Next blob). The plan says Qwen3.6 rides the native pack instead, so these are reference only. `canonical_xcheck.py` shows the "bit-exact against gguf-py over every block" style.
- `tools/calibrate.py`, `needle_bench.py`, `batch_test.py`, `test_conversation_cache_*.py`. Server-level measurement and regression harnesses. `calibrate.py` tunes `--pcie-frac`, `--spec-min-p`, `--pool-workers` through an engine's per-request keys. `needle_bench.py` is recall at 32K to 262K through the HTTP API. `batch_test.py` checks batched against solo greedy tokens.
- `tools/_paths.py`. Finds gguf-py: `$STRATA_GGUF_PY`, then `<repo>/third_party/llama.cpp/gguf-py`. That second path exists in this fork's submodule, so the tools work unmodified.
- Not needed: `setup.py`/`test_setup_*.py` (the installer), `strata_mcp.py`, `embd_bf16_pack.py`, `ple_fp8_pack.py` (Flash-Next PLE), `hip/`, `vision/`.

### `tests/` and the parity sources

- `tests/` in the snapshot has 30 files and no `CMakeLists.txt`: `tests/core/*.cpp` (loader and cache tests, `gguf_fixture.hpp` is a C++ synthetic-GGUF writer with pattern-filled tensors), `tests/cuda/prefill_*_test.cpp`, `tests/hip/*`, `tests/data/native_experts/*.txt` (golden `native_experts.txt` files for q2_0, iq2_xs, iq3_s, iq3_xxs, coder-iq1_m, swift-iq2_xs).
- The root `CMakeLists.txt` references 23 `tests/` sources that are **absent in UnbiasedStrata** (the snapshot has 30 files there), and four `bench/micro/*` sources that upstream never published. Configuring this fork with `-DSTRATA_BUILD_TESTS=ON` would fail at `tests/core/gguf_split_test.cpp` (line 231). Restoring `tests/` fixes all but the `bench/micro` ones, which sit under the oracle option.
- The 36 `src/kernels/*_parity.cpp` files, `src/prefill/gdn_rec_parity.cu`, `src/prefill/gemm_bf16_parity.cu` and `src/kernels/cpu/*_parity.cpp` are **still in UnbiasedStrata**. Relevant to the port: `gdn_parity.cpp`, `router_top10_parity.cpp`, `shared_expert_parity.cpp`, `gr_parity.cpp` (the Flash-Next hyper-connection residual, which Qwen3.6 lacks), `rope_parity.cpp`, `qsa_parity.cpp` (indexer path, bypassed for dense attention), `native_expert_parity.cpp`, `native_grouped_parity.cpp`, `sampler_parity.cpp`, `kv_*_parity.cpp`, `iq_parity.cpp`, `gdn_rec_parity.cu`.

### Docs

- `docs/semantics.md`, `docs/activation-contract.md`: **do not exist upstream** (no commit). The semantics live in comments in the parity files and in the unpublished `ref/*.py`.
- `docs/ORCA.md` (95 lines) and `ORCA_Q4_K_S.md` (36). These are the precedent for a **different quantization of the same geometry** (512 experts), not for a different expert count: `iq_pack.py --compat-bf16` rounds the small projections the engine reads as BF16, records each in `compat-bf16.json`, and the pack carries its own tokenizer and `dense.bin`. The precedent for a **different expert count** is the Coder (256 of 512 experts): `iq_pack` reads `n_expert` from the router's shape, `make_profile.py --n-expert 256`, and `tests/data/native_experts/coder-iq1_m.txt`. `docs/MODELS.md` describes it.
- `docs/DETAILS.md` (1127 lines; this fork keeps it as `docs/upstream/DETAILS.md`), `docs/HOW_IT_WORKS.md`, `docs/OLDER_GPUS.md`, `docs/NVIDIA_V100.md`, `docs/paper/`.

### `bench/`

- Snapshot: results only (`bench/results/<date>-<topic>/` with `README.md`, `matrix.md`, `matrix.json`). Useful as the format for reporting. The speed matrix of `2026-09-28-speed-0114` records the flags the setup writes (`--prefill auto`, `--expert-cache auto`, `--kv-resident`, `--spec 4 --spec-min-p 0.5`).
- Main only: `bench/bench_vs_llama.py` drives Strata's and llama.cpp's OpenAI-compatible servers with the same prompts and GGUF, takes server-side prompt and decode tokens per second, peak VRAM from `nvidia-smi` polling, RSS from `psutil`, interleaves engines across rounds, and has `prompts`, `run`, `quality`, `summary` subcommands. Needs `psutil`. This is the closest match for the plan's baseline-versus-engine comparison, and it is not in the pinned snapshot.
- The engine itself prints a `strata decode GPU stages (ms/window)` line and takes `--dump-routing` (`src/program/generate.cpp:1310`).

## 3. How upstream produces oracle dumps for a parity test today

1. There is no dump pipeline for the model. No script in the published tree runs a forward pass of the model and writes per-seam tensors. Everything below is what exists.
2. Most parity tests are self-contained: the oracle is a **host C++ transcription of the spec**, inside the test file. `gdn_parity.cpp` has `ref_step()` and `ref_conv()` "a transcription of `ref/gdn.py`", in double precision, with boolean switches for the rival readings. `router_top10_parity.cpp` has `reference()` (softmax, stable sort, renormalise). Random inputs come from `std::mt19937(2024)`. Nothing is read from `data/`.
3. Registration: `add_test(NAME gdn_parity COMMAND gdn_parity --selftest)` in the root `CMakeLists.txt` under `if(STRATA_BUILD_TESTS)` and a CUDA or HIP backend. `STRATA_BUILD_TESTS` defaults ON only when `tests/CMakeLists.txt` and `bench/micro` both exist, so a published checkout builds none of it unless `-DSTRATA_BUILD_TESTS=ON` is given. Run with `ctest` or the binary with `--selftest`. Exit 0 pass, 1 fail, 77 or 3 skip (`SKIP_RETURN_CODE`).
4. Vector tests embed the oracle as a generated header: `ple_oracle_vectors.inc` ("GENERATED by tests/python/gen_ple_oracle.py") holds token windows and expected hash rows; `qsa_oracle_vectors.inc` holds float32-to-float16 bit pairs from `np.float16(np.float32(x))`. Both generators are unpublished. Regenerating means rewriting them.
5. Fixture tests generate files into the build directory at test time: `iq_parity_fixtures` runs `tools/iq_fixture.py --out <build>/iq_fixture`, which writes `<name>.bin` (int32 header: kernel type id, rows, cols, then raw block bytes) and `<name>.f32` (reference values from gguf-py). `iq_parity <dir>` reads them. CMake probes interpreters for numpy (`-DIQ_FIXTURE_PYTHON` overrides) and uses `FIXTURES_SETUP`/`FIXTURES_REQUIRED` for ordering. This is the only dump-like flow in the published tree, and only for dequantization.
6. Real-GGUF tests take a path from an environment variable or argument: `STRATA_PLE_GGUF` (default `../../Q2_0/...-00002-of-00002.gguf`) and `--pack`, plus binary captures `bench/micro/ple_in.bin` and `ple_out.bin` recorded from ggml's graph by an unpublished `ple_layer_xcheck.cpp`, which "records EVERY intermediate" so a failure is attributable to a stage.
7. The independent-oracle tests (`bench/micro/native_{mmvq,bf16,shared,gr_norm,moe,gdn,gr_postops,router,gdn_preprocess,gdn_pipeline,qsa,rope,ple_postops,flash_attn,qsa_indexer,qsa_score}_parity.cpp`) link a pinned llama.cpp CUDA build as the oracle: `-DSTRATA_ORACLE_SOURCE_DIR=<llama.cpp>` and `-DSTRATA_ORACLE_BUILD_DIR=<build>` (CMake lines 387 to 408; libraries `ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda` located with `find_library`). These sources are unpublished, so the 16 names are a list of what to rewrite, and the CMake foreach is the wiring to restore.
8. No dump file naming convention exists upstream. Maya's convention (llama.cpp `cb()` names, files under `data/glm5-synth-dumps/`) is the one to adopt.

**Negative-test discipline, from `gdn_parity.cpp`.** Before the kernel is judged, each wrong reading of the spec is computed with the same reference and must differ from the right one by more than 5 % (relative L1). Four traps are pinned: modulo versus interleaved head pairing, decay before versus after the rank-1 update, conv slide direction (state rows labelled so every position is identifiable), and `l2_norm`'s epsilon on the squared norm versus the mean. It prints `yes` or `*** NO ***` for "is the rival observable" and counts a failure when it is not. The first fixture for the decay trap showed only 3.25 % apart, so the gate and the input range were changed (decay in (0.0025, 0.135], beta in [0, 1)) until the trap was clearly visible. The state layout test fills the state with `0.001*(i + 2j + 3h) - 1` so a layout mix-up yields a different number, not a plausible one. `router_top10_parity.cpp` shows the same idea for selection: all-equal logits, a 12-way tie, a dominant expert, and an assertion that the 2^-14 renormalisation clamp is unreachable (top-k sum at least k/n). Comparisons are NaN-safe (`le`/`gt` demand finite values first) after a NaN fixture once printed a perfect score.

## 4. How a native pack is built from a GGUF

Entry point: `python tools/iq_pack.py --gguf <first-shard>.gguf --out <dir> [--compat-bf16] [--experts-bin] [--skip-experts] [--base <pack>]`. All sibling shards must exist (`Model.__init__` reads `-0000N-of-0000M`, a missing one is an error). `main()` runs `expert_layout()` first so a model that cannot be packed is refused before any file changes, then `index_standalone()`, then the tokenizer export, then writes `native_experts.txt` last.

- **`native_experts.txt`** is the completion marker and the expert map. One line per layer: `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]`. `gu_type` and `d_type` are ggml type ids (gate and up must match; down may differ, e.g. IQ3_XXS gate/up with Q2_0 down). `offset` is the layer's start in the virtual `experts.bin`. `blob_bytes` is the per-expert size of gate + up + down. The last three are absolute byte offsets of the three role tensors in their GGUF shard. Header `v3`; `v4` only if one layer's roles sit in different shards (column becomes `gate,up,down`). The engine reads the experts straight from the GGUF unless `experts.bin` exists. Golden examples: `tests/data/native_experts/*.txt`.
- **`index.txt`**: "strata pack index v3", one row per non-expert tensor: `name, ?, kind, offset, bytes, arena_offset, arena_bytes, ne0, ne1, ...`. Kind 4 = BF16, 5 = F16, 2 = F32, 3 = F32 narrowed to F16 by the loader. A quantized tensor the engine serves natively from the GGUF has zero offsets and only its shape. The arena is 64-byte aligned.
- **`dense.bin`**: the float tensors the engine reads from the pack, in the form it reads them (table `FORM`: routers, hyper-connection and indexer projections, SSM gates BF16; norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm` F32; PLE conv F16). Exact conversions are the default (F32 whose low 16 bits are zero to BF16; F16 or BF16 widened to F32). Rounding conversions need `--compat-bf16`. Anything the engine requires in a form the GGUF does not hold, without the flag, is a hard error.
- **`conversions.json`** (schema 1): every converted tensor: shard, source and destination types, method, exact or not, max absolute error, SHA-256 of the source bytes, plus the source shard names and sizes.
- Also: `experts.bin` plus `experts.bin.src.json` (optional, only with `--experts-bin`, reuse is checked by shard sizes, the hash of `native_experts.txt` and six sampled blobs), `tokenizer/`, and `compat-bf16.json` with the flag. All files are written under `.tmp` names and renamed, so an interrupted run leaves no completion marker.

## 5. Dependencies and GPU

- Pack tools and `ref/load.py`: Python 3.10+, `numpy`, gguf-py from llama.cpp (the submodule's `third_party/llama.cpp/gguf-py` works through `_paths.py`; `ref/load.py` needs `PYTHONPATH` set to it). `regex` for the tokenizer export. `requirements.txt` pins numpy, jinja2, regex, pyyaml, tqdm. `psutil` for `bench_vs_llama.py`. `mtp_fetch.py` uses the standard library's HTTP client.
- **No torch, no safetensors library, no GPU** anywhere in `tools/` or `ref/`. The oracle is NumPy on the CPU. Maya's `ref/glm.py` is also NumPy; the full-attention and GDN loops are slow but small models are fine, and the synthetic GGUF keeps them fast.
- GPU is needed only to run the C++ parity binaries and the engine.
- For cross-checking, the submodule has `examples/eval-callback`, `src/models/qwen35moe.cpp`, `src/models/delta-net-base.cpp` and `convert_hf_to_gguf.py`. llama.cpp's `cb()` names in `qwen35moe.cpp` include `conv_output_silu`, `gate_sigmoid`, `beta_sigmoid` and `shared_expert_gate_sigmoid`.

## 6. Hardcoded Flash-Next assumptions found while reading (each must be generalized or bypassed)

- `iq_pack.py`: `FORM`/`ROUTERS` name Flash-Next tensors (`hc_*`, `indexer.*`, `ple_*`); `is_expert()` matches any `*_exps.weight`; `expert_layout()` takes `n_expert` from `blk.0.ffn_gate_inp.weight` and demands every layer to have all three expert tensors (true for qwen35moe, which has no dense-lead layers); tensors with more than two dimensions outside experts are refused; the `FORM` table has no qwen35moe entries, so SSM and attention tensors are served natively from the GGUF where quantized and copied as stored where float. Whether the engine needs `ssm_alpha`/`ssm_beta`, the router and the shared-expert gate as BF16 for the GDN and router kernels is a step 3 and 4 question; the `FORM` table is where that gets decided. `ffn_gate_inp_shexp.weight` is already in `ROUTERS`.
- `make_profile.py`: `N_LAYER = 48`, `N_EXPERT = 512`.
- `mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`: Flash-Next tensor names, the 512-expert fused shapes, the revision pin. The Qwen3.6 MTP head comes from the MTP GGUF's `nextn` tensors per the plan.
- `gdn_parity.cpp` and `src/kernels/cuda/gdn.cu`: the out-norm is `rms_norm(o) * ssm_norm * sigmoid(z)` (`gdn.cu:148`), and the parity test's negative case asserts SiLU is the wrong reading. llama.cpp's `qwen35moe` `build_norm_gated` uses `ggml_silu(gate)`. The test's geometry is fixed at S = 128, h_k = 16, h_v = 48 against Qwen3.6's 32 v-heads. Re-pin both before reusing either.
- `tests/data/native_experts/` and `data/*.bin`: Flash-Next only.

## 7. What to restore and what to write

Restore into a dev checkout (copy from `/Users/robertkarl/Code/strata-upstream-6f32ec0`; a dev branch, or keep out of the fork's release tree):

1. `ref/load.py`.
2. `tools/`: `_paths.py`, `gguf_reader.py`, `gguf_writer.py`, `iq_pack.py`, `test_iq_pack.py`, `make_profile.py`, `test_make_profile.py`, `strata_tokenizer.py`, `mtp_pack.py`, `mtp_rt.py`, `mtp_fetch.py` (for reference), `draft_vocab.py`, `iq_fixture.py`, `needle_bench.py`, `calibrate.py`, `batch_test.py`.
3. `tests/` whole (30 files), so that `-DSTRATA_BUILD_TESTS=ON` configures. `bench/micro/` cannot be restored; the CMake foreach for the llama.cpp-oracle parity tests stays dormant.
4. `requirements.txt` (numpy, regex, jinja2).
5. From upstream main only: `bench/bench_vs_llama.py`.

Write new (nothing upstream to copy): `ref/qwen36.py`, `tools/qwen36_synth_gguf.py`, the oracle invariant tests, qwen35moe parity binaries (GDN at 32 v-heads with SiLU gating, the top-8 over 256 router with shared-expert gate, the gated full-attention layer), the layer composition test, and the streaming runner test.
