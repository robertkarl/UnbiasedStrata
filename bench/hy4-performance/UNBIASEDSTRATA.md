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

## Regina serving checks

The reviewed SQ-64 HTTP build passed the inventory and tool-call preflight at 131,072 context capacity using Hy4's unrestricted production sampler. Both 4,750-token inventory prompts returned the verified count for record 179. They produced 3,517 and 2,939 tokens at 64.76 and 62.64 decode tokens/s. The assistant requested the correct SKU and incorporated the returned stock count in its answer.

The four pilot tasks scored 1/1 on write-hello, 1/1 on write-hello-word, 10/10 on comfyui-sd and 5/5 on api-contract-drift. The subsequent file-ring and file-ring-traverse runs scored 3/3 and 6/6 with Regina's pinned tooling. The full 86-task BBW remains paused.

## Multi-turn conversation oracle

The [fixtures](conversations.json) contain two independent four-message conversations and fixed assistant continuations. One asks for a PostgreSQL design for partial refunds. The other asks for bilingual museum labels about objects with uncertain provenance. Render each conversation with Hy4's chat template. Verify that the checkpoint tokenizer and the HTTP tokenizer produce identical prompt IDs.

For each assistant token, score the row immediately preceding it. The CUDA runners receive exactly the same prompt and teacher-forced prefix. The scorer computes forward KL from stock to fork and total variation over all 120,832 next-token probabilities, followed by total variation after applying temperature 0.9 and top-k 64. Perplexity and accuracy refer only to the written assistant continuations. Sampling has its own behavioral check, using a fresh prefix and the same 96 nonzero seeds per engine. The speed comparison uses seeds 41–43, a 128-token warm-up, no prefix reuse, a 4,096-token context and a 2,048-token generation cap.

```sh
bench=bench/hy4-performance
python3 "$bench/conversation_oracle.py" --out ./conversations prepare \
  --url http://127.0.0.1:8080 --tokenizer /path/to/tokenizer.hf.json
# Stop the serving process to release both GPUs before loading either native engine.
python3 "$bench/conversation_oracle.py" --out ./conversations compare \
  --oracle /path/to/stock/strata-hy4 --candidate build-hy4/bin/strata-hy4 \
  --model /models/hy4/Hy4-Flash-SQ-64.strata --engram /models/hy4/Hy4-Flash-Engram-Q8.strata
```

Set the same CUDA device order as in the serving example. Use `sample_conversations.py` once per native executable with separate output directories for the 96 paired first-token draws. For timing, pass each case's saved `prompt.ids` and `--seed 41` to `measure.py` and `measure_http.py`. The native tool also requires `--prompt` set to the number of IDs. Stock and HTTP elapsed time both start before sending the request and end after the completion arrives. Model loading and warm-ups are excluded.

| Conversation | Assistant targets | Mean / maximum KL in nats | Maximum full / top-k TV | Perplexity, both engines | Stock / fork top-1 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refund ledger | 163 | 0 / 0 | 0 / 0 | 7.430197 | 85/163 on both |
| Bilingual museum labels | 154 | 0 / 0 | 0 / 0 | 8.483685 | 71/154 on both |

The full logit arrays were bit-identical. All 96 paired first-token draws matched for each conversation. Code review found that a NumPy partial sort could select the wrong token IDs when logits tied at rank 64. The final scorer follows the native lower-token-ID tie break. Rescoring the saved arrays and comparing native draws yielded the values above.

| Client-observed configuration | Ledger prefill | Ledger decode | Museum prefill | Museum decode | Both conversations, end-to-end output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock native | 993.44 | 94.95 | 1,074.04 | 81.98 | 87.03 |
| UnbiasedStrata HTTP | 1,008.76 | 87.74 | 1,080.07 | 73.54 | 78.97 |

Each arm generated 12,288 tokens over six requests. Stock took 141.19 seconds and UnbiasedStrata took 155.60 seconds, including prefill. The fork delivered 90.74% of stock end-to-end throughput. The median time to first token was 236 ms for HTTP and 238 ms for stock. See the [per-seed results and first-token counts](conversation-results.json).

The six long, seeded DFly2 replies had different token hashes between stock and fork. We repeated seed 41 in fresh processes to distinguish sampler behavior from HTTP delivery. With DFly2 off, all 256 tokens matched stock on each conversation. With DFly2 on, stock's own 2,048-token replay first diverged at token 5 on both conversations. The fork's native and HTTP paths returned identical 2,048-token sequences on both. Upstream's draft scheduler uses measured iteration times to change the verification window and switch to plain decode. These decisions change which random draws are consumed. The fork fixes that scheduling choice for explicitly seeded requests. The [replay controls](conversation-results.json) include each hash and the matching prefix lengths.

A second complete set of 96 one-token draws per conversation verified the engine-reported prompt length on every request. All results matched the first set on each engine.
