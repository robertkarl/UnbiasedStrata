# Hy4-Flash text pack measurements

The CUDA host was an i9-9900K with an RTX 4090 (24 GiB), an RTX A6000 (48 GiB), and 114 GiB of host RAM. The Metal host was an M3 Ultra with 96 GiB of unified memory. Every text pack used the same Q8 Engram. The first GPU in a pair is primary. The A6000-primary results use cuBLAS 12.9.2 for both programs. [Ampere controls](ampere-cublas.json)

## Sustained decode

Each cell gives UnbiasedStrata HTTP / stock native in output tokens per second. A fresh process ran a 128-token warm-up and generated 2,048 tokens on the refund-ledger conversation. The HTTP timer runs from the first streamed tokens to the last. The stock timer runs from its first token to the completion message. Both exclude loading and prefill.

| Text pack | 4090 | A6000 | 4090 + A6000 | A6000 + 4090 | Metal auto | Metal 32 GB cache | Metal 12 GB cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| SQ-64 | 21.1 / 22.8 | 31.8 / 36.9 | 88.9 / 95.7 | 72.1 / 79.2 | 47.6 / 48.1 | 20.8 / 19.8 | 19.7 / 19.8 |
| SQ-96 | 14.0 / 13.6 | 18.9 / 20.2 | 39.1 / 38.6 | 35.9 / 38.5 | 18.3 / 18.4 * | 11.8 / 11.8 | 12.3 / 12.4 |
| SQ-128 | 9.6 / 10.7 | 13.0 / 14.6 | 20.5 / 21.5 | 19.8 / 19.5 | 10.2 / 10.2 * | 6.4 / 5.5 | 5.5 / 5.9 |

* The original fork and stock both exceeded the Metal buffer limit on these packs. The marked cells use automatic placement in the patched fork and cap the stock cache at the measured device limit of 62,620,631,040 bytes.

## Prompt processing

Each cell gives UnbiasedStrata HTTP / stock native in input tokens per second. The 231-token ledger and 261-token museum prompts each ran after a 16-token warm-up, with zero prefix reuse. Rates divide total input tokens by total time to first token. The continuation length for these short tests was 256 tokens.

| Text pack | 4090 | A6000 | 4090 + A6000 | A6000 + 4090 | Metal auto | Metal 32 GB cache | Metal 12 GB cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| SQ-64 | 37 / 37 | 86 / 84 | 1096 / 1063 | 1032 / 1014 | 181 / 181 | 29 / 27 | 78 / 82 |
| SQ-96 | 23 / 24 | 38 / 39 | 106 / 105 | 100 / 100 | 31 / 31 | 17 / 17 | 43 / 45 |
| SQ-128 | 18 / 18 | 25 / 25 | 43 / 43 | 42 / 42 | 14 / 14 | 13 / 13 | 13 / 13 |

## Output distributions

21 of 21 scored placements produced bit-identical logits across both conversations. Each comparison covers all 120,832 logits at 317 teacher-forced assistant targets. The scoring runs disabled drafting and used one full-sequence chunk per conversation (393 tokens for the ledger and 414 for the museum). Accuracy is next-token accuracy on the two continuations. The paired result file includes per-conversation KL, total variation, perplexity, accuracy, and logit hashes. [All measurements](pack-matrix.json)

The table below compares weight quantization within stock Strata, using SQ-128 as the reference.

| Platform | Compared pack | Mean KL (nats) | Top-1 agreement | SQ-128 perplexity | Compared pack perplexity | SQ-128 accuracy | Compared pack accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| 4090-a6000 | SQ-64 | 0.21027 | 75.7% | 8.456 | 7.925 | 47.9% | 49.2% |
| 4090-a6000 | SQ-96 | 0.06077 | 87.1% | 8.456 | 8.479 | 47.9% | 49.2% |
| metal-cache12 | SQ-64 | 0.20951 | 77.6% | 8.431 | 7.841 | 48.9% | 50.2% |
| metal-cache12 | SQ-96 | 0.06048 | 86.8% | 8.431 | 8.384 | 48.9% | 49.8% |

## Held-out text quality

The 4090-primary pair scored the same 2,048 next tokens from the WikiText-2 raw test split on each checkpoint. From the 58 articles with at least 640 tokens, take articles 0, 14, 29, and 43 in corpus order. Use the first 128 tokens of each as context and score the next 512. The runner scored each article with drafting disabled and a 639-token input chunk. The JSON lists the article titles, token digests and per-article results. [Pinned dataset](https://huggingface.co/datasets/ggml-org/ci/tree/927b3642933080f1b0e811e2f916e14c292992f9)

| Text pack | Perplexity | Next-token accuracy | Mean KL from SQ-128 | Top-1 agreement with SQ-128 |
|---|---:|---:|---:|---:|
| SQ-64 | 12.906 | 52.34% | 0.8174 | 70.31% |
| SQ-96 | 15.898 | 52.39% | 0.2767 | 82.57% |
| SQ-128 | 16.368 | 53.08% | 0.0000 | 100.00% |

## Serving on the RTX pair

For speed-first serving, use SQ-64 with the 4090 primary and the A6000 secondary. This order sustained 88.9 tok/s, compared with 39.1 on SQ-96 and 20.5 on SQ-128. WikiText next-token accuracy differed by one token between SQ-64 and SQ-96 (1,072 versus 1,073 correct out of 2,048). SQ-128 predicted 1,087 correctly. SQ-64 and SQ-96 each predicted 156 of the 317 fixed conversation targets, compared with 152 for SQ-128.

SQ-96 is the closer approximation when preserving the larger pack’s token choices matters. On WikiText it agreed with SQ-128 on 82.57% of top choices, compared with 70.31% for SQ-64. On the publisher’s BF16 tool-prompt evaluation, mean KL was 0.237 for SQ-96 versus 0.518 for SQ-64. These distribution checks and the separately scored Regina tool tasks measure different aspects of quality. [Publisher quality comparisons](https://huggingface.co/strata-engine/Hy4-Flash-Strata#how-close-is-it-to-the-original) · [Regina results](UNBIASEDSTRATA.md#regina-serving-checks)

## Conditions

For throughput, both programs used a 4,096-token context, DFly2, seed 41, temperature 0.9, top-k 64, top-p 1, and disabled session reuse. The CUDA planner chose the chunk size. Metal used 128-token chunks for throughput. Each sustained rate comes from one generation per program and configuration. The Metal cache sizes are decimal GB on the 96 GiB machine. The 12 GB cache exercises the runner’s path for smaller Macs. [Conversations](conversations.json) · [Published text packs](https://huggingface.co/strata-engine/Hy4-Flash-Strata/tree/995bac227ab7e0d221772be641e2b5954a9fac06)
