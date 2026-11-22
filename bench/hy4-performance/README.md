# Hy4 CUDA performance trials

The [stock measurements](RESULTS.md) compare checkpoint sizes. The [UnbiasedStrata measurements](UNBIASEDSTRATA.md) compare the native and HTTP paths with stock on the same cards.

All measurements concern one request at a time. The stock engine is the unmodified Linux x86-64 CUDA 12 package from Strata v0.2.18, engine v0.1.64, source `6e38f588a646`, built with CUDA 12.9. Its executable SHA-256 is `02cffdc5d456506c2b0dbd91cdc31f2a4adf97f813f3173f87f3c160a0ece7d1`. [Release](https://github.com/Niko1221/Strata/releases/tag/v0.2.18)

The reference uses SQ-64, DFly2, two 32 GB Tesla V100s and a 4K context. Strata reports 1,068 prompt tokens/s and 41.8 decode tokens/s. The card does not specify its prompt text, generation length, sampling settings, or statistic for this row. Our host has a 24 GB RTX 4090, 48 GB RTX A6000, i9-9900K and 114 GiB guest RAM. The comparison measures speed on different hardware. [Model card](https://huggingface.co/strata-engine/Hy4-Flash-Strata#performance-with-strata)

## Method

`measure.py` runs the published binary via its native token protocol. It checks the process's GPU UUIDs, disables prompt-prefix reuse, and saves each engine-reported prefill and decode time. The synthetic inventory prompt contains high-effort chat framing and requests a long engineering design. Each output directory contains a command and environment manifest, prompt IDs, generated IDs, native placement logs, warm-up and measured rows. Top-k 64 is explicit because this native release limits its sampled distribution to 64 candidates. The sampled trials use temperature 0.9 and top-p 1.

The earlier SQ-128 inventory runs used 4,750 input tokens, a 131,072-token context and 1,024-token chunks. They establish the larger pack's performance separately from the SQ-64 reference. The fork sample used unrestricted top-k.

| Engine | Pack | GPUs | Input / actual output | Prompt tokens/s | Decode tokens/s | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Fork 978b012f | SQ-128 | 4090 + A6000 | 4,750 / 4,878 | 144.0 | 20.87 | Cold prefix |
| Fork 978b012f | SQ-128 | 4090 + A6000 | 4,750 / 4,358 | 144.8 | 18.95 | Cold prefix |
| Fork 978b012f | SQ-128 | 4090 | 4,750 / 2,048 | 65.49 | 9.15 | 218 decode samples with zero reported clock-throttle flags |
| Fork 978b012f | SQ-128 | A6000 | 4,750 / interrupted | — | — | Operator switched to base Strata during generation |

SQ-128 dual placement put 43.56 GB of expert weights on the CPU. The single-card loader backed all 105.76 GB of expert weights in host RAM and selected adaptive GPU slots. During 4090 decode, the host disk-read median was 1.9 MB/s and the maximum was 4.0 MB/s. Every recorded decode sample had zero major faults and zero swap activity. The transient high disk read rate during A6000 loading accompanied the 113.65 GB checkpoint read.

The model packs and Q8 Engram file are archived and checksum-verified on the SMB model share. NVMe is the host-side cache. Raw benchmark outputs are kept under the GPU host's `hy4-ring-tests` directory and in Regina's `results_hy4_sq128_single_card_20261121` directory. The SQ-128 BBW run was stopped when single-GPU performance became the priority.

## SQ-64 search order

First test the dual-GPU layout with each card as primary and Strata's automatic prefill chunk size. Then test explicit 1,024- and 2,048-token chunks and each card alone. A 128-token warm-up precedes the 256-token screening request. A configuration clearing both speed floors gets a new engine process, a warm-up and three long requests. The initial search uses sampled generation. Greedy generation is recorded as its own configuration if needed. Data are retained for failed trials and slower configurations.

The SQ-128 upstream runs use the native `--serve` protocol, one fresh process per placement, with `--ctx 131072 --chunk 1024 --draft dfly2 --session-reuse 1`. Each first request submits 4,750 IDs and requests 2,048 new tokens at temperature 0.9, top-p 1 and top-k 64. `CUDA_VISIBLE_DEVICES` restricts the process to the intended UUIDs. The recorded GPU process list verifies placement. The release recommends NVIDIA driver 575.51 or newer in its component manifest; this host runs 550.163.01. No driver was changed for this test.

## Reproduction

The SQ-64 experiments use Python 3 and Hugging Face Tokenizers. Export the embedded tokenizer once, then run the same binary in `measure.py`. Obtain the ordered physical GPU UUIDs from `nvidia-smi -L`.

```sh
strata-hy4 --export-tokenizer /models/Hy4-Flash-SQ-64.strata ./tokenizer
python3 bench/hy4-performance/measure.py \
    --engine /path/to/published/strata-hy4 \
    --model /models/Hy4-Flash-SQ-64.strata \
    --engram /models/Hy4-Flash-Engram-Q8.strata \
    --tokenizer ./tokenizer/tokenizer.hf.json --gpus "$GPU_UUIDS" \
    --context 4096 --prompt 2048 --generate 2048 --trials 3 \
    --out ./hy4-result
```
