# Hy4 CUDA performance trials

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
