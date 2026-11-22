# UnbiasedStrata on the 4090 and A6000

The stock baseline and the fork used the same SQ-64 text pack, Q8 Engram table and physical GPU order. The [stock receipts](verified.json) give the artifact hashes. The fork builds the imported Strata runner from source and serves it through the UnbiasedStrata front end.

## Experiment log

1. The fork's existing native runner placed all 20,992 experts in VRAM. Three warmed, uncached 2,048-token prompts at 4,096 context capacity generated 2,047 tokens each. Median prefill was 2,211.42 tokens/s and median decode was 77.14. Stock delivered 2,175.71 and 77.10. The upstream placement, DFly2 and automatic chunk planner already transferred with the source import.
2. The first stock CUDA logit process hit an allocation error during model loading as the throughput benchmark was finishing. Its results were discarded. The rerun began with 2 MiB in use on each GPU and ran the two engines serially. The full-vocabulary comparison covered 381 observed next tokens, including 127 past the sparse-index boundary. Both engines produced perplexity 8.584705 and accuracy 230/381. Mean KL was zero and top-1 agreement was 381/381.
3. Source inspection found an HTTP context-limit mismatch. The front end kept eight positions free for every backend. Upstream Hy4's CUDA and Metal serve loops each keep one. The fork now applies the Hy4 limit when serving a Hy4 pack. The 2,048-input, 2,048-requested-output case exercises this path.
