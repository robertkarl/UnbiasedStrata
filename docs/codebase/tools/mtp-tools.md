[Codebase docs](../README.md) › [Tools](README.md) › MTP / draft-head tools

# MTP draft-head tooling

## Purpose

Speculative decoding in the engine uses a one-block MTP ("multi-token prediction", llama.cpp's `nextn`) draft head
loaded with `--mtp <dir>` (`MtpDrafter::load`, `src/core/mtp.cpp:279`). The two models get that directory from
different sources, so there are two tool chains:

| Model | Source of the draft block | Tools | Output directory (UPSTREAM.md) |
| --- | --- | --- | --- |
| Qwen3.8 Flash-Next | the 31 `mtp.*` tensors of the original BF16 safetensors checkpoint (the GSQ-RCO GGUF has none) | `mtp_fetch.py` -> `mtp_pack.py` -> `mtp_rt.py` | `mtp/` |
| Qwen3.6-35B-A3B | `blk.40` of unsloth's `-MTP` GGUF (`nextn_predict_layers 1`) | `mtp_qwen36_pack.py` | `qwen36-mtp-draft/` |

| File | Lines | Role |
| --- | --- | --- |
| [`tools/mtp_fetch.py`](../../../tools/mtp_fetch.py) | 308 | HTTP range reads of the `mtp.*` tensors from the Hugging Face checkpoint, SHA-256 checked |
| [`tools/mtp_pack.py`](../../../tools/mtp_pack.py) | 193 | Quantizes the fetched BF16 tensors into an MTP GGUF (`qwen4exp-mtp`) |
| [`tools/mtp_rt.py`](../../../tools/mtp_rt.py) | 111 | Turns that GGUF into the Flash-Next runtime files |
| [`tools/mtp_qwen36_pack.py`](../../../tools/mtp_qwen36_pack.py) | 246 | Cuts Qwen3.6's draft pack from a `-MTP` GGUF |
| [`tools/test_mtp_fetch.py`](../../../tools/test_mtp_fetch.py) | 201 | Unit tests of `mtp_fetch.py` with a mocked `urlopen` |

## Flash-Next chain

```
HF Qwen/Qwen3.8-Flash-Next @ PINNED_REVISION (safetensors, 131 shards)
   │  mtp_fetch.py inventory | fetch | verify --out mtp-bf16
   ▼
mtp-bf16/{mtp-inventory.json, mtp-inventory.md, tensors/<name>.bin, tensors/verified.json, mtp-manifest.json}
   │  mtp_pack.py --src mtp-bf16 --experts q2_0 --out mtp-bf16/mtp-q2_0.gguf
   ▼
mtp-q2_0.gguf (+ mtp-q2_0.gguf.report.json)
   │  mtp_rt.py --gguf mtp-q2_0.gguf --out mtp
   ▼
mtp/{experts.bin, dense.bin, dense.txt}   + cp data/draft_vocab.bin mtp/
```

### `tools/mtp_fetch.py`

Subcommands (`main`, `:292`): `inventory --out DIR`, `fetch --out DIR [--only SUBSTR]`, `verify --out DIR`.

- `inventory` (`:177`) reads `model.safetensors.index.json`, then only the 8-byte length and JSON header of each
  shard that holds an `mtp.*` tensor (`shard_header`, `:170`), and writes `mtp-inventory.json` (repo, total bytes,
  per tensor shard/dtype/shape/byte range) and a Markdown table `mtp-inventory.md`.
- `fetch` (`:207`) downloads each tensor's byte range in 64 MiB chunks into `tensors/<name>.bin`, resuming partial
  files when the inventory is for the same repo URL, and writes `mtp-manifest.json` (inventory rows plus `file` and
  `sha256`). A tensor whose hash differs from `SHA256` (`:44-107`, all 31 tensors at the pinned revision) is
  deleted and fetched once more, then refused.
- `verify` (`:259`) is offline: hashes `tensors/*.bin` against `SHA256`, caching verdicts by size and mtime in
  `tensors/verified.json`; exit code 3 (`BAD`) lists missing or corrupt tensors. It checks nothing when another
  revision is selected.
- `get` / `check_range` (`:110-137`) demand HTTP 206 with the exact `Content-Range` asked for (issue #327: a mirror
  that ignores `Range` would otherwise deliver the shard header as tensor data). Retries 4 times with backoff.
- `resolve_repo` (`:153`) falls back to `main` if the pinned revision returns 404.

Environment: `STRATA_MTP_REVISION` (default `PINNED_REVISION = de4b8e4d...`, `:32-33`) and `HF_ENDPOINT` (a mirror,
`:35`). Standard library only.

### `tools/mtp_pack.py`

`python3 tools/mtp_pack.py --src DIR --experts q2_0|q4_0|q8_0 --out OUT.gguf [--check-experts 8]`.
Writes a gguf-py `GGUFWriter` file with architecture string `qwen4exp-mtp` and keys `strata.mtp.source`,
`strata.mtp.source_sha256` (hash of the manifest's name-to-sha map), `strata.mtp.expert_format`,
`strata.mtp.expert_quantizer`. Per tensor (`main`, `:134-189`):

- the two fused expert tensors (`gate_up_proj` `[512, 1280, 2560]`, `down_proj` `[512, 2560, 640]`) are quantized
  one expert at a time from a memmap along the last axis: `q2_0` (`:58`) searches 17 scales between `amax/2` and
  `amax` per 64-block for minimum squared error over the `{-1,0,1,2}` grid (the ggml reference uses `amax` only);
  `q4_0` (`:85`) and `q8_0` (`:100`) follow ggml's reference rounding. A relative RMS error over the first experts is
  reported in `<out>.report.json`.
- 1-D tensors are written F32; other tensors stay BF16.

### `tools/mtp_rt.py`

`python3 tools/mtp_rt.py --gguf mtp-q2_0.gguf --out DIR`. Reads with `gguf.GGUFReader` and asserts the Q2_0 expert
shapes `(512, 1280, 720)` and `(512, 2560, 180)` in bytes (`:72`). Writes the canonical expert blob layout
(`blob_of`, `:48`), and `dense.bin`/`dense.txt` where the `Q8` set of ten large projections (`:29-31`) is
re-quantized to Q8_0 with ggml's reference rounding (`q8_0`, `:34`), other 2-D tensors stay BF16, and F32 (1-D norm)
weights get `+1` (Gemma-style RMSNorm, `:85-87`). Hard-coded Flash-Next geometry `H, FF, NE = 2560, 640, 512` (`:27`).
The `dense.txt` names drop `mtp.` and `layers.0.`.

## Qwen3.6: `tools/mtp_qwen36_pack.py`

```
python3 -I tools/mtp_qwen36_pack.py --gguf Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf --out qwen36-mtp-draft
        [--draft-vocab data/draft_vocab.bin | none]
```

`main` (`:94-242`):

1. Requires `general.architecture == qwen35moe` and `nextn_predict_layers == 1`; the draft block is
   `blk.<block_count - 1>`. Reads the geometry keys into `geo` (`n_embd`, heads, `head_dim`, `n_rot`, experts, `k`,
   `n_ff`, `n_ff_shexp`).
2. Refuses fused `ffn_gate_up_exps` or `attn_qkv` in the block, missing names from `NORMS + ROUTERS + PROJECTIONS +
   EXPERTS` (`:58-63`), and notes extra tensors it leaves out.
3. Expert types: gate and up equal and in `GU_TYPES`, down in `D_TYPES` (a 256-block down type needs
   `n_ff % 256 == 0`) (`:54-56`, `:143-151`). These lists mirror the native GPU expert kernels in
   `src/kernels/cuda/iq_kernels.cu`; the engine asks `native_expert_supported` again at load.
4. Dense tensors, in a fixed order, into `dense.bin`/`dense.txt` (`:157-195`): norms must be F32 vectors; routers
   BF16, or F32 that is BF16-exact (written as high halves; `bf16_exact_problem`, `:85`); projections must be in
   `iq_pack.NATIVE_PROJECTION_TYPES` and are copied as stored. `nextn.eh_proj.weight` must be
   `ne [2 n_embd, n_embd]`.
5. `experts.bin`, `native_experts.txt` (one line, in-blob role offsets), the draft vocabulary (validated against
   `output.weight`'s vocabulary size), and `mtp.json`.

The pack cut from the Q4_K_XL `-MTP` file also serves the IQ3_XXS trunk: the engine checks only the geometry in
`mtp.json` against the trunk ([docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md) section 2).
The IQ4_XS `-MTP` file's draft block (Q3_K gate/up, Q4_K down) is refused because the native kernels lack that pair.

## Model-specific summary

| | Flash-Next (`mtp_rt.py`) | Qwen3.6 (`mtp_qwen36_pack.py`) |
| --- | --- | --- |
| Tensor names in `dense.txt` | checkpoint names minus `mtp.` / `layers.0.` | llama.cpp names minus `blk.N.` |
| Projections | re-quantized to Q8_0 | copied as stored (any native MMVQ type) |
| Norm weights | F32 with `+1` | F32 as stored (llama.cpp multiplies by the stored weight) |
| Experts | canonical Q2_0 blob (gate/up interleaved, planes) | raw GGUF blocks `[gate | up | down]` |
| `native_experts.txt`, `mtp.json` | not written | written |

## Tests

- `python3 -m unittest tools.test_mtp_fetch` (no network: a fake safetensors checkpoint behind a mocked `urlopen`).
  Covers `check_range`, a mirror that ignores `Range`, re-fetch on wrong bytes, an inventory from another revision,
  `verify`'s cache, and that all 31 tensors have hashes.
- `tests/python/qwen36_mtp_fixture.py` runs `mtp_qwen36_pack.py --draft-vocab none` on a synthetic model with a
  nextn block; `tests/cuda/qwen36_mtp_test.cpp` (CTest `qwen36_mtp_test`) holds the engine's drafter to the
  oracle's `--mtp` dumps. See [python-tests.md](python-tests.md).
- `mtp_pack.py`, `mtp_rt.py` have no tests in this repository.

## Gotchas

- `test_mtp_fetch.Setup.test_setup_rebuilds_a_corrupt_install` imports upstream's `setup.py`, which this fork
  removed; that one test errors (`ModuleNotFoundError: No module named 'setup'`); the other tests pass.
- `mtp_fetch.py`, `mtp_pack.py` and `mtp_rt.py` are pinned to Flash-Next tensor names and 512-expert shapes.
- The UPSTREAM.md Flash-Next commands run these tools from an upstream checkout (`$U/tools/...`); the copies here
  are the same tools restored for the port (docs/qwen36-port/upstream-tooling.md section 7).

## See also

- [pack-format.md](pack-format.md) sections 2-4, [profile-and-draft-vocab.md](profile-and-draft-vocab.md),
  [reference-model.md](reference-model.md) (`mtp_forward`, the nextn oracle)
- [docs/qwen36-port/step4-drafter.md](../../qwen36-port/step4-drafter.md),
  [docs/upstream/DETAILS.md](../../upstream/DETAILS.md)
