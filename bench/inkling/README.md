# Inkling 2 Mini validation, 2026-11-22

## Generation speed

Cold starts on one RTX A6000 (48 GB, 300 W, driver 550.163.01, CUDA 12.4, i9-9900K). Both engines used the same SM86 build target, Unsloth UD-Q4_K_XL weights, exact input token IDs, greedy decoding, FP8 KV, 344,064 context and 2,048-token chunks. Each run produced 400 tokens. The engine times the 399 intervals after the first token. Every paired 400-token output matched. Each cell gives prefill / decode in tokens per second.

| Prompt tokens | Published Maya, RTX 3090 | Stock Maya, A6000 | UnbiasedStrata, A6000 |
|---:|---:|---:|---:|
| 47 | 187 / 128.7 | 196.5 / 115.21 | 200.9 / 115.80 |
| 4,096 | 2,742 / 124.2 | 2,549.6 / 111.93 | 2,666.6 / 111.87 |
| 63,981 | 2,361 / 105.6 | 2,528.9 / 92.43 | 2,527.3 / 93.17 |
| 127,910 | 2,079 / 90.6 | 2,230.3 / 80.14 | 2,222.4 / 79.25 |
| 340,000 | 1,473 / 62.3 | 1,589.4 / 58.01 | 1,553.8 / 57.78 |

Maya publishes the lengths and settings. Its source documents for the 63,981- and 127,910-token runs are unavailable here. The A6000 runs used a deterministic shipment ledger trimmed to each published length. The same-host columns each represent one timed run per depth. [Maya’s RTX 3090 measurements and commands](https://github.com/alesha-pro/maya-inkling-mini/blob/e819852f822479af783638a467bfdfe2ed6882ec/docs/INKLING2-MINI.md#numbers)

With Maya’s MTP drafter, context 262,144, draft context 65,536 and 384 generated tokens, decode varied with the prompt.

| Workload | Prompt tokens | Maya published, RTX 3090 | Stock Maya, A6000 | UnbiasedStrata, A6000 |
|---|---:|---:|---:|---:|
| Prose | 42 | 131.1 | 107.68 | 106.98 |
| New code | 51 | 148.7 | 175.05 | 173.85 |
| Edit | 512 | 211.6 | 215.92 | 215.31 |
| Copy code | 512 | 188.1 | 217.17 | 213.51 |
| Long document | 63,981 | 124.8 | 139.19 | 139.57 |

Generate the A6000 inputs and run either executable with the following commands. For the drafter, select a `draft-*.ids` input, set `IK2_DRAFT_CTX=65536`, use `--ctx 262144 --gen 384`, and add `--draft /models/mtp.safetensors`.

```sh
python3 bench/inkling/speed_inputs.py /models/tokenizer.json /tmp/inkling-inputs
IK2_KV=fp8 IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1 \
  build/bin/strata-ik2 --model /models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --ctx 344064 --chunk 2048 --tokens @/tmp/inkling-inputs/perf-4096.ids --gen 400
```

## Metal on Debra

Debra has an M3 Ultra, a 60-core GPU and 96 GiB unified memory. The SQ-20 pack has SHA-256 `0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97`. The shipping v0.2.19 executable identifies its build as `26350e4387b8`. The source import is the release tag at `c6f6427d9de9`. The model card lists **108.9 tok/s on M5 Max with SQ-20** and **27.6 tok/s on M4 Mac mini with SQ-10**. Both are single-request results with drafting. [Strata model card and benchmark conditions](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/blob/04996a1e2c45009b6a04f2af126cb6bdb2ba3587/README.md)

| Debra, upstream synthetic benchmark | Release prefill | Fork prefill | Release decode | Fork decode |
|---|---:|---:|---:|---:|
| Plain | 1,255.1 | 1,250.9 | 51.0 | 51.0 |
| DSpark | 1,223.7 | 1,235.2 | 49.9 | 49.9 |

```sh
build-frontend/bin/strata-inkling --bench --native "$PACK" --parallel 1 \
  --max-context 4096 --prompt-len 512 --gen 128 --draft off
```

In single-client serving, each pair received identical prompt IDs with thinking off and generated the same 384 tokens. Decode here counts the 383 intervals after the first emitted token. The code prompt requested a Python red-black tree with insertion, deletion and invariant checks.

| Debra serving workload | Release plain | Fork plain | Release DSpark | Fork DSpark |
|---|---:|---:|---:|---:|
| Python code, 46 input tokens | 48.64 | 49.37 | 76.62 | 76.97 |
| Museum conversation, 265 input tokens | 50.72 | 50.92 | 45.22 | 45.30 |

## Quality

The ledger and bilingual museum conversations and gold assistant continuations come from PR #32. The C++ HTTP tokenizer produced the same prompt IDs as Maya’s independent template renderer and checkpoint tokenizer. The ledger prompt has 231 tokens and the museum prompt has 265. All metrics score only assistant targets, over all 200,058 vocabulary entries. KL is oracle-to-fork in nats. PPL is the exponential of the mean negative log probability of the gold token. The pooled PPL weights each token equally. [PR #32](https://github.com/robertkarl/UnbiasedStrata/pull/32)

| Backend and weights | Case | Targets | Oracle PPL | Fork PPL | Mean / max KL | Top-1 agreement |
|---|---|---:|---:|---:|---:|---:|
| CUDA UD-Q4_K_XL | Ledger | 168 | 6.426442 | 6.426442 | 0 / 0 | 168/168 |
| CUDA UD-Q4_K_XL | Museum | 163 | 7.694422 | 7.694422 | 0 / 0 | 163/163 |
| Metal SQ-20 | Ledger | 168 | 6.336325 | 6.336325 | 0 / 0 | 168/168 |
| Metal SQ-20 | Museum | 163 | 7.696368 | 7.696368 | 0 / 0 | 163/163 |
| CUDA UD-Q4_K_XL | Image-conditioned color description | 16 | 1.140019 | 1.138518 | 0.00001483 / 0.00015031 | 16/16 |

Text logits were byte-identical for both same-backend pairs. Pooled text PPL was 7.022352 for CUDA and 6.973066 for Metal. Each engine predicted the gold token at 176/331 positions.

The CUDA oracle was Maya `e819852f`, built independently for SM86. The GGUF’s SHA-256 is `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`. For the image case, each projector independently encoded the same 165 × 93 PNG on the Linux host with four CPU threads. Both produced 15 rows of width 2,304. The relative L2 difference was 0.00639. The Maya projector used llama.cpp b12590. The real CUDA and Metal HTTP servers both identified the red rectangle and blue circle, changed their answer when the colors changed to green and yellow, and returned to the original answer when the first image was resent. Debra used the Metal image encoder via `--mmproj-gpu`.

### Reproduce the comparisons

`cases.json` contains the exact prompt and continuation IDs. The scripts require NumPy. `speed_inputs.py` also requires the Python `tokenizers` package.

```sh
python3 bench/inkling/parity.py --maya /path/to/maya-ik2 \
  --strata build/bin/strata-ik2 --model "$MODEL"
python3 bench/inkling/metal_parity.py --oracle /path/to/reference/strata-inkling \
  --candidate build-frontend/bin/strata-inkling --model "$PACK"
```

For CUDA images, apply `maya-oracle-image.patch` to a separate Maya checkout. It exposes the server’s existing image-row loader to the diagnostic CLI. Encode `shapes.png` with each projector and pass `--maya-image-engine`, `--maya-picture` and `--strata-picture` to `parity.py`. The text comparison uses the unchanged Maya CLI and its single-token forced-decode path.

For the Metal oracle, apply `strata-oracle-dump.patch` to a separate v0.2.19 checkout and build `engine/src/inkling2` in Release mode. The released Metal runner has workspace for 64 output rows. Its original `--dump-ref` asks for hundreds in one call and leaves some output rows zero. The diagnostic patch sends up to 64 rows per call to the unchanged upstream backend. On a separate 64-position check the fork matched the shipped macOS executable byte for byte. The speed comparisons use the unmodified release executable.
