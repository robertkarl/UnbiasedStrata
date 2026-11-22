# UnbiasedStrata on the 4090 and A6000

The stock baseline and the fork used the same SQ-64 text pack, Q8 Engram table and physical GPU order. The [stock receipts](verified.json) give the artifact hashes. The fork builds the imported Strata runner from source and serves it through the UnbiasedStrata front end.

## Experiment log

1. The fork's existing native runner placed all 20,992 experts in VRAM. Three warmed, uncached 2,048-token prompts at 4,096 context capacity generated 2,047 tokens each. Median prefill was 2,211.42 tokens/s and median decode was 77.14. Stock delivered 2,175.71 and 77.10. The upstream placement, DFly2 and automatic chunk planner already transferred with the source import.
2. The first stock CUDA logit process hit an allocation error during model loading as the throughput benchmark was finishing. Its results were discarded. The rerun began with 2 MiB in use on each GPU and ran the two engines serially. The full-vocabulary comparison covered 381 observed next tokens, including 127 past the sparse-index boundary. Both engines produced perplexity 8.584705 and accuracy 230/381. Mean KL was zero and top-1 agreement was 381/381.
3. Source inspection found an HTTP context-limit mismatch. The front end kept eight positions free for every backend. Upstream Hy4's CUDA and Metal serve loops each keep one. The fork now applies the Hy4 limit when serving a Hy4 pack. The 2,048-input, 2,048-requested-output case exercises this path.
4. Code review reproduced an HTTP token-accounting error using a Unicode character encoded as two tokens. The original stream sent only the last token ID when its text became valid UTF-8. The front end now sends every requested token ID and buffers incomplete text until it can be emitted. A regression exercises the five-token reply “Hello 𐍈 world” and checks both the IDs and the reconstructed text. Native benchmark trials now reject cancellation, nonfinite results and invalid timings.
5. The first 4K HTTP build with the corrected context limit measured 2,187.57 prefill and 72.19 decode tokens/s. After the UTF-8 fix, we rebuilt, passed the Linux HTTP suite and repeated the 4K test from a fresh process.

## Sustained throughput

| Interface and configuration | Context capacity | Uncached input | Actual output per turn | Trials | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock native, top-k 64 | 4,096 | 2,048 | 2,047 | 3 | 2,175.71 | 77.10 |
| UnbiasedStrata native, top-k 64 | 4,096 | 2,048 | 2,047 | 3 | 2,211.42 | 77.14 |
| UnbiasedStrata HTTP, top-k 64 | 4,096 | 2,048 | 2,047 | 3 | 2,181.36 | 73.07 |
| Stock native, top-k 64 | 131,072 | 4,750 | 2,048 | 1 | 1,774.18 | 79.69 |
| UnbiasedStrata HTTP, top-k 64 | 131,072 | 4,750 | 2,048 | 1 | 1,782.47 | 78.41 |
| UnbiasedStrata HTTP, unrestricted top-k | 131,072 | 4,750 | 2,048 | 1 | 1,804.10 | 65.22 |

Native rows use the engine's timers. HTTP rows use time to first streamed token for prefill and elapsed time from first to last token for decode. The 4K rows are medians after one 128-token warm-up. At 131K, top-k 64 was the first request in a fresh process. The unrestricted request followed it with session reuse disabled. The released binary caps sampled generation at 64 candidates. For each matched top-k 64 comparison, both prompt and decode speeds also cleared the 80% floor when timed entirely at the client on both sides. The [per-request results](unbiasedstrata-cuda.json) include both sets of measurements, binary and logit hashes, and observed GPU placement.

## Reproduce through HTTP

Start the verified two-GPU server using [the serving command](../../docs/HY4.md#serving-sq-64-on-a-4090-and-a6000). Use `measure.py` to produce the stock reference and its `prompt.ids`, then send those same token IDs to UnbiasedStrata.

```sh
python3 bench/hy4-performance/measure_http.py \
  --url http://127.0.0.1:8080/completion --prompt-ids ./hy4-result/prompt.ids \
  --generate 2048 --trials 3 --warmup 128 --out ./hy4-http-result
```

For a 131,072-token context capacity, pass the saved 4,750-token prompt, set `--warmup 0 --trials 1`, and launch the server with a 1,024-token chunk. Each HTTP result saves the prompt digest, full token counts, client timings and server timings.
