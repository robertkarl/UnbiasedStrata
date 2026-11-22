## Hy4-Flash text inference

- [x] Build a native Hy4 runner and serve a completion through unbiased-strata using the vocabulary and chat template from the same .strata checkpoint.
- [x] Compare identical teacher-forced positions against a separately built, pinned Strata runner on the same checkpoint with Engram enabled. Target mean KL ≤ 0.001 nats, relative perplexity difference ≤ 1%, and top-1 agreement ≥ 99%.
- [x] Measure prompt processing and greedy generation separately with the same token IDs, device, context, draft setting and warm-up. Target median throughput within 20% of Strata.
- [x] Record merge difficulty for PolyStrata, the size and dependencies of Niko’s changes, and the scope of a kernel-only implementation.

The measured results, hardware, checkpoint coverage and reproduction commands are in [docs/HY4.md](docs/HY4.md).

## CUDA integration on the 4090 and A6000

- [x] Serve SQ-128 with its verified Engram-Q8 table on the rebased branch. Confirm nonzero expert placement on both physical GPUs and record the host spill.
- [x] Run Regina’s file-ring and file-ring-traverse prompts through the HTTP tool interface for Hy4 and Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS. Use fresh workspaces, the pinned Pi images and each model’s declared production sampling. Require 3/3 and 6/6 from the supplied evaluators for each model.
- [ ] Use Regina’s substantial text and tool-call preflight, four-task pilot and frozen 86-task BBW. Record prompt and decode throughput and the usable context for each scored arm.
- [x] Honor unrestricted top-k on Hy4 for both plain and speculative sampling, including reduced draft vocabularies and rejection sampling.

## Published Strata CUDA performance

- Record every stock-engine attempt on the performance branch. Include the binary, model, device order, context, prompt length, output length, sampling, and placement.
- Compare prompt processing and decode separately with the published SQ-64 dual-V100 reference of 1,068 and 41.8 tokens/s at a 4,096-token context. The corresponding 80% floors are 854.4 and 33.44 tokens/s. Report faster results as exceeding the reference.
- Verify a qualifying configuration on the same two-card host using three measured requests after warm-up, fresh 2,048-token prompts, and approximately 2,048 generated tokens per request. Report actual output lengths, hardware differences and available upstream methodology.

## UnbiasedStrata CUDA parity with stock SQ-64

- Use the released Strata CUDA binary as the oracle on the same RTX 4090 primary and RTX A6000 peer. Compare the checkpoint, Q8 Engram, prompt IDs, context, DFly2, sampling, prefix reuse and expert placement.
- Require at least 80% of stock median prefill and decode in the fork's native runner and through the ordinary HTTP server on the 2,048-token prompt. Warm each process and measure three requests of approximately 2,048 generated tokens. Check the matched 4,750-token prompt at 131,072 context capacity.
- Compare full-vocabulary CUDA logits on short prompts and past the sparse-index boundary using the existing KL, perplexity and top-1 thresholds. Check HTTP chat and tokenizer behavior. Record failed attempts and actual output lengths alongside the upstream results.

- Repeat CUDA verification on two independent rendered conversations. Score every supplied assistant target against the full 120,832-token distribution in stock and the fork. Require mean forward KL ≤ 0.001 nats, relative perplexity difference ≤ 1%, top-1 agreement ≥ 99%, and mean total-variation distance ≤ 0.01 for the temperature-0.9, top-k-64 target. Check the actual sampler with 96 paired seeds per conversation. Measure three warmed, uncached generations per conversation through HTTP and stock with the same prompts, seeds, and 2,048-token output cap.
