## Hy4-Flash text inference

- [x] Build a native Hy4 runner and serve a completion through unbiased-strata using the vocabulary and chat template from the same .strata checkpoint.
- [x] Compare identical teacher-forced positions against a separately built, pinned Strata runner on the same checkpoint with Engram enabled. Target mean KL ≤ 0.001 nats, relative perplexity difference ≤ 1%, and top-1 agreement ≥ 99%.
- [x] Measure prompt processing and greedy generation separately with the same token IDs, device, context, draft setting and warm-up. Target median throughput within 20% of Strata.
- [x] Record merge difficulty for PolyStrata, the size and dependencies of Niko’s changes, and the scope of a kernel-only implementation.

The measured results, hardware, checkpoint coverage and reproduction commands are in [docs/HY4.md](docs/HY4.md).

## CUDA integration on the 4090 and A6000

- [x] Serve SQ-128 with its verified Engram-Q8 table on the rebased branch. Confirm nonzero expert placement on both physical GPUs and record the host spill.
- [ ] Run Regina’s file-ring and file-ring-traverse prompts through the HTTP tool interface for Hy4 and Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS. Use fresh workspaces, the pinned Pi images and each model’s declared production sampling. Require 3/3 and 6/6 from the supplied evaluators for each model.
- [ ] Use Regina’s substantial text and tool-call preflight, four-task pilot and frozen 86-task BBW. Record prompt and decode throughput and the usable context for each scored arm.
- [x] Honor unrestricted top-k on Hy4 for both plain and speculative sampling, including reduced draft vocabularies and rejection sampling.
