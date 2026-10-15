## Spec, 2026-10-15, Muse Spark Open

**Goal:** Serve Meta Muse Spark Open 816B-A42B through UnbiasedStrata, starting
with Unsloth UD-IQ2_XXS. Establish correctness on the provided H200 machine
first, then measure the RTX A6000 path. Preserve the existing Qwen paths.

### Acceptance Criteria

- [x] The six-shard UD-IQ2_XXS release loads and generates coherent text on one
  H200 with host memory backing without exhausting VRAM or host RAM; record the command, model revision,
  memory consumption, disk traffic, and measured prompt/decode speed.
- [x] Model geometry and tensor forms are taken from the GGUF, including the
  metadata-only first shard, 8192-wide residual stream, 4096-wide expert latent
  space, sigmoid top-8 routing, and the explicit sliding/global attention pattern.
- [x] Teacher-forced logits agree with the pinned upstream Muse implementation
  on multiple prompts; record top-1 agreement and numerical error before declaring
  inference support complete.
- [x] `/v1/chat/completions` serves text, streaming, reasoning, and tool calls
  using the model's template; invalid reasoning effort is rejected explicitly.
- [x] A missing shard or unsupported tensor form fails with a useful diagnostic
  before inference rather than silently using Qwen defaults.
- [x] Existing frontend tests pass (`make test-frontend`), and existing Qwen
  configurations retain their model-specific behavior.
- [x] Build instructions pin all upstream source dependencies and require no
  network during compilation; the model source, revision, and shard hashes are
  recorded with reproducible download and run commands.

Initial scope is text inference. Multimodal encoders, full-million-token context,
speculative decoding, multi-GPU native inference, and A6000 performance tuning
are not prerequisites for this first correctness milestone.
