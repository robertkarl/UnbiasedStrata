## Hy4-Flash text inference

- [x] Build a native Hy4 runner and serve a completion through unbiased-strata using the vocabulary and chat template from the same .strata checkpoint.
- [x] Compare identical teacher-forced positions against a separately built, pinned Strata runner on the same checkpoint with Engram enabled. Target mean KL ≤ 0.001 nats, relative perplexity difference ≤ 1%, and top-1 agreement ≥ 99%.
- [x] Measure prompt processing and greedy generation separately with the same token IDs, device, context, draft setting and warm-up. Target median throughput within 20% of Strata.
- [x] Record merge difficulty for PolyStrata, the size and dependencies of Niko’s changes, and the scope of a kernel-only implementation.

The measured results, hardware, checkpoint coverage and reproduction commands are in [docs/HY4.md](docs/HY4.md).
