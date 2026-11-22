# Stock Strata measurements

Each SQ-128 row used a fresh cache, a 4,750-token input and a 131,072-token context. Speeds are native tokens/s.

| GPU placement | Output tokens | Prompt | Decode |
| --- | ---: | ---: | ---: |
| 4090 | 2048 | 65.54 | 9.74 |
| a6000 | 2048 | 88.77 | 12.52 |
| dual | 2048 | 144.43 | 19.87 |

## SQ-64 at 4K context

Dual-V100 reference 1,068 prompt and 41.8 decode tokens/s. The 80% floors are 854.4 and 33.44. Automatic chunks are denoted by 0.

| Stage | GPU order | Chunk | Temperature | Output tokens per trial | Prompt | Decode |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| screen | 4090-a6000 | 0 | 0.9 | 256 | 2192.72 | 90.94 |
| confirm | 4090-a6000 | 0 | 0.9 | 2047, 2047, 2047 | 2175.71 | 77.10 |

## Confirmation on November 21, 2026

The 4090 was primary and the A6000 was the second device. With SQ-64 and Q8 Engram, all 20,992 experts fit across the two GPUs. The 4090 held 13.60 GB of experts and the A6000 held 42.44 GB in the 4K run. Strata selected a 4,096-token prefill chunk. DFly2 and temperature 0.9 sampling were active.

| Measured request | Uncached input | Output | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 2,047 | 2,190.84 | 77.10 |
| 2 | 2,048 | 2,047 | 2,175.71 | 75.76 |
| 3 | 2,048 | 2,047 | 2,171.79 | 85.68 |
| Median | 2,048 | 2,047 | 2,175.71 | 77.10 |

Each request asked for 2,048 output tokens and reached the engine's length limit at 2,047. Median prefill was 2.04× and median decode was 1.84× the published dual-V100 speeds. The card gives the context capacity and hardware. Its prompt, output length, and sampling settings are unspecified. [Strata model card](https://huggingface.co/strata-engine/Hy4-Flash-Strata#performance-with-strata)

## Same-prompt comparison at 131,072 context capacity

Both checkpoints ran in fresh stock processes with the same 4,750 input token IDs, 2,048 output tokens, zero prefix reuse, 1,024-token chunks, DFly2 and Q8 Engram. Both used the 4090 as primary and A6000 as peer.

| Pack | Experts on CPU | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: |
| SQ-128 | 43.56 GB | 144.43 | 19.87 |
| SQ-64 | 0 GB | 1,774.18 | 79.69 |

On this request SQ-64 delivered 12.28× the prefill rate and 4.01× the decode rate. The two packs use different weight quantization. These throughput measurements used the published Strata binary. The numerical oracle tests for UnbiasedStrata are recorded separately in [the integration results](../../docs/HY4.md).

The [verification record](verified.json) includes checkpoint hashes, prompt digests, device order, native timings, placement and shutdown status. Both prompt digests for the matched input can be reproduced from its saved IDs. The older harness used comma-separated IDs and the new harness saved space-separated IDs followed by a newline.
