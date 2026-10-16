[Codebase docs](../README.md) › [Tools](README.md) › Qwen3.6 verification

# Qwen3.6 verification tooling

## Purpose

The Qwen3.6 port is held to two references: the pinned llama.cpp graph (`third_party/llama.cpp`, used through small
C++ programs linked against its `libllama`) and the numpy oracle [`ref/qwen36.py`](reference-model.md). These tools
generate synthetic models, run both references, compare their per-tensor dumps, and compare the engine's logits
with llama.cpp's on real files. Results and the tolerance policy they produced are in
[docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md).

| File | Lines | Role |
| --- | --- | --- |
| [`tools/qwen36_synth_gguf.py`](../../../tools/qwen36_synth_gguf.py) | 373 | Writes tiny deterministic `qwen35moe` GGUFs (F32 by default) |
| [`tools/qwen36_dump_llama.cpp`](../../../tools/qwen36_dump_llama.cpp) | 171 | llama.cpp's graph on a GGUF, every `cb()`-named tensor dumped in full (`qwen36-dump`) |
| [`tools/qwen36_compare_dumps.py`](../../../tools/qwen36_compare_dumps.py) | 85 | Seam-by-seam comparison of two dump directories |
| [`tools/qwen36_check_eval_callback.py`](../../../tools/qwen36_check_eval_callback.py) | 130 | Checks `llama-eval-callback`'s printed stats against oracle dumps |
| [`tools/qwen36_crosscheck.sh`](../../../tools/qwen36_crosscheck.sh) | 82 | One command: build llama.cpp (CPU), write fixtures, run all cases, compare |
| [`tools/qwen36_dequant_ggml.cpp`](../../../tools/qwen36_dequant_ggml.cpp) | 30 | ggml's own `to_float` on a slab of one tensor (`qwen36-dequant`) |
| [`tools/qwen36_check_dequant.py`](../../../tools/qwen36_check_dequant.py) | 55 | `ref/load.py` dequantization vs `qwen36-dequant` on a real file |
| [`tools/qwen36_spread.py`](../../../tools/qwen36_spread.py) | 37 | Per-seam-class relative error between oracle runs and a llama.cpp dump of a real file |
| [`tools/qwen36_router_bf16_flips.py`](../../../tools/qwen36_router_bf16_flips.py) | 111 | Measures how often BF16-rounding the router flips a top-k choice |
| [`tools/qwen36_teacher_llama.cpp`](../../../tools/qwen36_teacher_llama.cpp) | 199 | llama.cpp greedy continuation with per-position logits and probed routes (`qwen36-teacher`) |
| [`tools/qwen36_compare_logits.py`](../../../tools/qwen36_compare_logits.py) | 155 | Engine `--dump-logits` vs a reference: top-1, KL, drift, routes |
| [`tools/iq_fixture.py`](../../../tools/iq_fixture.py) | 137 | Deterministic i-quant blocks plus gguf-py reference values for `iq_parity` |
| [`tools/test_qwen36_oracle.py`](../../../tools/test_qwen36_oracle.py) | 419 | Invariant tests for the oracle and the synthetic fixture |

## Data flow

```
                          tools/qwen36_synth_gguf.py  (seed, geometry flags, --like-real REAL.gguf)
                                       │ toy.gguf (F32, 128 ASCII-byte vocabulary)
            ┌──────────────────────────┼─────────────────────────────┐
            ▼                          ▼                             ▼
  ref/qwen36.py --dump o/     qwen36-dump MODEL d/ T [CHUNK] [mtp]   llama-eval-callback -p "hello world"
  (+ o_mtp/ with --mtp)       (+ d_mtp/)                             (printed stats, eval.txt)
            │                          │                             │
            └──────► qwen36_compare_dumps.py o d ◄┘    qwen36_check_eval_callback.py o eval.txt
                       (exit 1 on a failing seam)

  real GGUF ──► qwen36_check_dequant.py + qwen36-dequant          (weights bit-identical?)
  real GGUF ──► ref/qwen36.py [--act-quant q8] [--force-topk d] [--force-inputs d] ──► qwen36_spread.py
  real GGUF ──► ref/qwen36.py --act-quant q8 ──► qwen36_router_bf16_flips.py
  real GGUF ──► qwen36-teacher ──► logits.f32, routes.txt ─┐
  strata --dump-logits (engine) ───────────────────────────┴► qwen36_compare_logits.py
```

## Dump format (shared by `ref/qwen36.py` and `qwen36_dump_llama.cpp`)

A directory with one `<stem>.f32` per named tensor (float32, ggml memory order, `ne0` fastest; I32 tensors such as
`ffn_moe_topk-N` stored as float) and `manifest.txt` with lines `<stem> <type> <ne0>,<ne1>,<ne2>,<ne3>` in execution
order. Stems are llama.cpp's `cb()` names (`attn_norm-0`, `ffn_moe_topk-3`, `result_output`); a name produced again
gets `#2`, `#3` (llama.cpp names `Kcur`, `Vcur` and `norm-<il>` several times per layer). The C++ dumper writes the
llama type name (`f32`, `i32`, `f16`); the oracle writes `f32` or `i32`.

## `tools/qwen36_synth_gguf.py`

`python3 tools/qwen36_synth_gguf.py OUT.gguf [flags]`. Writes through gguf-py's `GGUFWriter` every metadata key
llama.cpp's `qwen35moe` loader reads, a GPT-2 byte-level vocabulary of the 128 ASCII bytes (token id = ASCII code,
one dead merge), and tensors from `build_tensors` (`:151`) in llama.cpp's naming and shape rules.

`Cfg` (`:59`) default: `n_embd 64`, 4 trunk layers with interval 4 (layers 0-2 Gated DeltaNet, 3 full attention),
GDN 4 k-heads / 8 v-heads of 16, conv 4, attention 2 heads / 1 KV head of 32, `n_rot 8`, IMRoPE sections
`[2,1,1,0]`, rope base 1e4, 8 experts top-2, expert ff 32, shared ff 32, and a nextn block at index 4.

| Flag | Effect |
| --- | --- |
| `--seed N` | RNG seed (files are byte-deterministic per seed) |
| `--like-real REAL.gguf` | derive ratios, rope base, eps, interval, conv kernel and presence of nextn / fused tensors from a real file (`real_config`, `:113`) and compare tensor-name sets |
| `--heads N,KV` | e.g. `4,2` for non-degenerate GQA |
| `--head-dim D` | `n_rot = D/4`, sections in the real `[11,11,10,0]` proportion, rope base 1e7 (256 = the engine's kernel width) |
| `--no-nextn`, `--fused-gate-up`, `--fused-qkv` | layout variants |
| `--expert-weights-scale X` | write a stray key llama.cpp ignores |
| `--gdn K,V,S`, `--experts E,K`, `--shexp-ff N` | geometry for the engine's kernel fixtures (state 128 for the CUDA GDN kernels) |
| `--quant-shexp TYPE` | write `ffn_{gate,up,down}_shexp` quantized with gguf-py |
| `--bf16-exact` | round `ssm_alpha`, `ssm_beta`, `ffn_gate_inp`, `ffn_gate_inp_shexp` to BF16-representable F32 |

Callers: `qwen36_crosscheck.sh`, `test_qwen36_oracle.py`, `cmake/qwen36_synth_fixtures.cmake` (toy, no-nextn,
fused gate/up, fused qkv), `cmake/qwen36_oracle_dumps.cmake` (the kernel fixture `k.gguf`),
`cmake/qwen36_attn_fixtures.cmake` (head_dim 256, G = 8, 12, 2), and `tests/python/qwen36_*_fixture.py` (which reuse
`Cfg`, `build_tensors`, `byte_unicode` with their own writer).

## `tools/qwen36_dump_llama.cpp` (`qwen36-dump`)

`qwen36-dump MODEL.gguf OUTDIR tok,tok,... [CHUNK [mtp]]`. Installs a `cb_eval` callback (`:56`) on every node,
keeps named F32/I32/F16 tensors (skipping `node_*`, `leaf_*`, views and `cache_*`), and writes them in full.
CPU only, F32 KV cache, flash attention off, `n_ctx 256`, one thread unless `QWEN36_THREADS`. `CHUNK > 0` decodes in
chunks and dumps only the last call, so carried GDN state and KV are what is checked. `mtp` loads the nextn block,
then runs a second context of type `LLAMA_CONTEXT_TYPE_MTP` on `(tokens[1:], h_nextn[:-1])` at positions `1..` and
dumps it to `OUTDIR_mtp` (`:143-166`). Build commands are in the file header and in `qwen36_crosscheck.sh`.

## `tools/qwen36_compare_dumps.py`

`python3 tools/qwen36_compare_dumps.py ORACLE_DIR LLAMA_DIR [--tol 1e-4] [--quiet]`. For each stem in both
manifests (`main`, `:31`): shapes must match, except `kq` / `kq_soft_max`, whose KV axis llama.cpp pads to its cache
size (cropped; the softmax padding must be exactly zero); I32 seams must be identical; float seams fail when
`max|a-b| > tol * max(1, max|b|)` or when the non-finite pattern differs. `conv_state_update-N` (a view read before
its update) is ignored. Exit 1 on any failure. `qwen36_spread.py` imports its `load`.

## `tools/qwen36_check_eval_callback.py`

`python3 tools/qwen36_check_eval_callback.py ORA eval.txt [--tol 1e-4] [--min-seams 100] [--quiet] [--require-tail]`.
Parses `common_debug_cb_eval` blocks of `llama-eval-callback` output, reconstructs which elements it prints
(first and last 3 along each dimension, `printed_indices`, `:31`), compares them and the float32 running `sum` with
the oracle (`main`, `:71`). A block cut off by llama.cpp's asynchronous logger at exit is dropped; `--require-tail`
fails unless `result_output` was reached. `kq`/`kq_soft_max` are skipped.

## `tools/qwen36_crosscheck.sh`

`tools/qwen36_crosscheck.sh WORKDIR [REAL-MTP.gguf]`. Builds `llama-eval-callback` from `${LLAMA_CPP_DIR:-third_party/llama.cpp}`
into `WORKDIR/llbuild` (CPU, no Metal, BLAS, curl, tests or server), compiles `qwen36_dump_llama.cpp`, writes six
fixtures (seven with a real file), and runs the cases `base_11`, `base_40`, `base_ar` (one token per call),
`base_c7` (chunks of 7), `seed7`, `fused_gu`, `fused_qkv`, `gqa4_2`, `stray_scl`, and `like_real`; cases with a nextn
block also compare the MTP dumps. It then retries `llama-eval-callback` up to 8 times until the printout is complete
and checks it. Exit 1 if any seam fails. Honors `PYTHON`, `LLAMA_CPP_DIR`, `STRATA_GGUF_PY`.

## Real-file tools

- **`qwen36_check_dequant.py MODEL.gguf $B/qwen36-dequant`**: for `token_embd` rows (12345, 12346, 247000), one
  expert of each quantized 3-D tensor type and the first slab of each 2-D type, compares `ref.load.Gguf`
  dequantization with ggml's `to_float` (via `qwen36_dequant_ggml.cpp`); fails above `1e-6 * max|x|`. The cross-check
  doc reports bit-identical results on the UD-IQ3_XXS file.
- **`qwen36_spread.py LLAMA_DIR LABEL=ORACLE_DIR ...`**: prints a Markdown table of median / max relative L2
  error per seam class (`CLASSES`, `:14`: `attn_norm`, GDN in-projection, GDN output, attention output on layers
  3,7,...,39, `ffn_moe_out`, `ffn_out`, `l_out`, logits) and counts differing top-k slots. Its numbers are the
  per-seam tolerance table in oracle-crosscheck.md.
- **`qwen36_router_bf16_flips.py REAL.gguf DUMP [--k 8]`**: recomputes each layer's router logits from the dump's
  `attn_post_norm-N` three ways (F32 x F32, F32 x BF16(W), BF16(x) x BF16(W)) and reports changed top-k slots, sets
  and tokens, the smallest top-k boundary gap, and the share of BF16-exact weights. Its docstring records the
  finding that drove `iq_pack.py`'s exact BF16 rule: every router, shared-expert gate, `ssm_alpha`, `ssm_beta` and
  `ssm_conv1d` element in both shipped files is BF16-exact.
- **`qwen36_teacher_llama.cpp` (`qwen36-teacher`)**:
  `qwen36-teacher MODEL.gguf OUTDIR --prompt-file FILE | --prompt-ids FILE [--n-gen 64] [--ngl 99] [--ctx 8192]
  [--probe 0,3,20,39] [--force-ids FILE]`, or `qwen36-teacher MODEL.gguf --detok FILE`. Greedy continuation with an
  F16 KV cache; writes `prompt_ids.txt`, `continuation.txt`, `logits.f32` (one `n_vocab` row per generated
  position), `meta.txt`, `routes.txt` (`<row> <layer> e0..e7` for probed layers), `text.txt`. `--force-ids` teacher
  forces the decoded tokens. The header shows a static-library CUDA build command.
- **`qwen36_compare_logits.py ENGINE.f32 REFERENCE.f32 --n-vocab N`**: the engine file has a two-`int32` header
  (`n_vocab`, rows) unless `--no-engine-header`. Reports top-1 agreement, top-5 overlap, KL(ref || engine) per
  position, a drift slope and Spearman correlation of KL with position, and (with `--routes-engine`/`--routes-ref`)
  top-k set agreement per layer. Gates: `--min-top1`, `--max-mean-kl`, `--max-max-kl`, `--max-spearman` (exit 1);
  `--json OUT` for the summary.

## `tools/iq_fixture.py`

`python3 tools/iq_fixture.py --out DIR [--seed 7] [--rows 32] [--cols 256]`. For IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
IQ3_S, IQ1_M, IQ4_NL, IQ4_XS, Q2_0 and Q3_K writes `<name>.bin` (`int32` kernel type id, rows, cols, then raw blocks)
and `<name>.f32` (reference values). Blocks are seeded random bytes with their fp16 scale fields overwritten to 1.0
(`repair_scales`, `:68`), dequantized by gguf-py; Q2_0 uses `gguf_writer`'s codec. Exit 3 when numpy or the vendored
gguf-py is missing. Wired as CTest `iq_parity_fixtures` -> `iq_parity` (`CMakeLists.txt:704-711`).

## Configuration

| Name | Used by | Meaning |
| --- | --- | --- |
| `STRATA_GGUF_PY` | synth, oracle, check_dequant (through `_paths.py`) | gguf-py location |
| `QWEN36_THREADS` | `qwen36-dump` | thread count (default 1) |
| `LLAMA_CPP_DIR`, `PYTHON` | `qwen36_crosscheck.sh` | llama.cpp source dir, interpreter |
| `IQ_FIXTURE_PYTHON` | CMake | interpreter for `iq_fixture.py` |

## Tests

- `python3 -I tools/test_qwen36_oracle.py` (24 test methods; numpy and gguf-py only, no llama.cpp build): fixture
  determinism and the real tensor-name set, layer pattern, router renormalization and the ignored scale key, GDN
  state shapes, chunked-equals-sequential, causality, the SiLU output gate, every rival reading observable (> 5 %
  relative L1 at its seam), GQA mapping, l2-norm epsilon, attention gate and IMRoPE, Q8 activation rounding, the dump
  format, and the nextn block. See [python-tests.md](python-tests.md).
- `tools/qwen36_crosscheck.sh` is the cross-check itself (exit 1 on failure).

## Gotchas

- llama.cpp must be checked out in `third_party/llama.cpp` (or `LLAMA_CPP_DIR`) for every C++ tool here; in a
  checkout without the submodule only the Python tools run.
- `qwen36_router_bf16_flips.py` imports `ref/load.py` without calling `_paths.add_gguf_py()`, so on a quantized
  file gguf-py must already be importable (for example via `PYTHONPATH`).
- `iq_fixture.py` puts only `third_party/llama.cpp/gguf-py` on `sys.path`; it does not consult `STRATA_GGUF_PY`.
- `qwen36_spread.py` hard-codes the attention layers of a 40-layer model and calls `main()` at import.
- See oracle-crosscheck.md "Quirks of the comparison" (repeated names, views read before their copy, padded `kq`,
  the eval-callback logger).

## See also

- [reference-model.md](reference-model.md), [python-tests.md](python-tests.md)
- [docs/qwen36-port/oracle-crosscheck.md](../../qwen36-port/oracle-crosscheck.md),
  [docs/qwen36-port/step4-runner.md](../../qwen36-port/step4-runner.md) (teacher and compare_logits on real files),
  [docs/qwen36-port/upstream-tooling.md](../../qwen36-port/upstream-tooling.md) section 3
