# Muse Spark Open native port

Native text inference has passed the short-context correctness checks below.
This is an initial implementation, with performance and feature limitations.
The existing Qwen build is unchanged.

The initial target is Unsloth's six-shard **UD-IQ2_XXS** release of
Muse Spark Open 816B-A42B. [The manifest](../data/muse-spark-iq2-xxs.json)
pins the model revision, all six sizes and SHA-256 values, and the reference
implementation from [llama.cpp PR 30616](https://github.com/ggml-org/llama.cpp/pull/30616).
The files total 225,568,569,304 bytes. Shard one contains only metadata.

## Build and run

The native path has a separate build directory and a separate pinned submodule
(`third_party/llama-muse`). The original Qwen dependency is not upgraded.

```sh
git submodule update --init
make build-muse CUDA_ARCHS=90 JOBS=16   # H200; use 86 for A6000, 100 for B200
```

The build reads local source only. Model preparation requires no pack or draft head:

```sh
hf download unsloth/Muse-Spark-Open-816B-A42B-GGUF \
  --revision 19d0b1b3a1b1974a48439e11e0639c0dad164fc8 \
  --include 'UD-IQ2_XXS/*.gguf'

CUDA_VISIBLE_DEVICES=0 ./build-muse/unbiased-strata \
  --model /path/to/UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf \
  --ctx 4096 --host 127.0.0.1 --port 8080
```

All six shards must be beside the first. Set `HF_HOME` before downloading to
choose the cache location; use local storage for the working weights. The native engine
uses one GPU. The reference runner can use both H200s.

This first path processes one token at a time, including prompt tokens. Dense
weights and FP32 KV stay on the GPU. Quantized experts are cached in VRAM with
LRU eviction; misses read the mapped GGUF, with host caching managed by the OS.
The automatic expert budget leaves 1 GiB free after dense weights and KV.
Every cache miss also checks physical free VRAM, so CUDA allocation rounding
cannot overfill the cache even when its tensor payloads fit the nominal budget.
Use `-- --cache-mib N` to limit the expert cache. This is an expert-cache budget,
not a limit on total VRAM. There is no pinned host expert tier or CPU expert
execution in this initial path.

The HTTP frontend uses the new submodule's tokenizer, template, and Muse parser.
Defaults are temperature 1, top-p 1, and top-k disabled; set temperature 0 for
greedy output. Supported reasoning efforts are `low`, `medium`, `high`, and
`xhigh`. Images, MTP, prompt checkpoints, multi-GPU native execution, and the
Qwen engine's RAM/VRAM cap flags are not implemented here.

## Correctness checks

`muse-check` writes raw little-endian float32 logits, one vocabulary-sized row
per processed token, and a `.tokens` sidecar. Run native and reference separately
so they do not compete for GPU memory:

```sh
./build-muse/muse-check --model "$MODEL" --out reference.f32 --reference \
  --ctx 4096 --prompt 'The capital of France is'
CUDA_VISIBLE_DEVICES=0 ./build-muse/muse-check --model "$MODEL" --out native.f32 \
  --ctx 4096 --prompt 'The capital of France is'
python3 tools/muse_compare_logits.py reference.f32 native.f32
```

The reference uses llama.cpp's graph, FP32 KV, and flash attention disabled to
match the initial native attention representation. The comparison rejects
different input-token sequences, non-finite values, and inconsistent shapes.
These checks compare final output logits, not intermediate layer activations,
and do not establish bitwise equality. Layer-by-layer activation parity remains
unverified. Its default gates are at least 95% greedy top-1 agreement and maximum per-token
KL divergence of 0.01. It also reports maximum absolute and RMS logit error.
`--generate N` is a smoke test; do not compare independently generated sequences
as teacher-forced logits.

Synthetic checks retain the real tokenizer but replace all weights. A four-token
sliding window exercises wraparound with short token sequences. The Python checks
require NumPy; the GGUF writer comes from the pinned submodule.

```sh
PYTHONPATH=third_party/llama-muse/gguf-py python3 tools/muse_synth_gguf.py "$MODEL" toy.gguf
PYTHONPATH=third_party/llama-muse/gguf-py python3 tools/muse_synth_gguf.py "$MODEL" toy-iq2.gguf \
  --expert-width 4096 --iq2-experts --mixed-dense
# Compare --reference against native on identical --tokens CSV, using --ctx 64.
# Use --cache-mib 1 for the small Q8 model, 16 for the wide IQ2 model, to force eviction.
ctest --test-dir build-muse --output-on-failure
./build-muse/muse-chat-test "$MODEL"
python3 tools/muse_http_smoke.py http://127.0.0.1:8080 http-results.json
```

## Results

- The real metadata-only first shard is parsed successfully: 62 layers, residual
  width 8192, expert width 4096, vocabulary 202048.
- The GGUF split tests pass, including a new unpadded metadata-only shard case.
- Existing frontend process/protocol tests pass.
- H200 CUDA operation checks against independent CPU calculations pass. Maximum
  absolute error was 2.50e-6 for paired RoPE and 5.97e-7 for grouped-query attention;
  residual, gating, SwiGLU, accumulation, and logit softcap checks also pass.
- Synthetic full-model comparisons pass on H200, with forced cache eviction and
  sliding-window wraparound. Both the small Q8 fixture and the 4096-wide IQ2_XXS
  expert fixture agree on all 12 top-1 predictions. Maximum KL divergences are
  2.16e-6 and 6.37e-6 respectively; RMS logit errors are 0.00111 and 0.00260.
  A third fixture combining Q4_K embeddings, Q6_K projections, Q8_0 projections,
  and 4096-wide IQ2_XXS experts also agrees on all 12 predictions (maximum KL
  1.21e-5, RMS error 0.00309).
- The serving adapter passes real-template tests for reasoning, answer text,
  tool arguments, streaming, tool-result history, and invalid reasoning effort.
  Synthetic HTTP requests also pass normal and streaming response checks.
  Closing a streaming client and then reusing the engine also passes.
- Real UD-IQ2_XXS teacher-forced comparisons pass on H200 and B200, using the
  prompts `The capital of France is`, `2 + 2 =`, and `def add(a, b):`.
  Both devices agree with their reference run on all 19 top-1 predictions.
  Maximum KL divergence is 0.00320 on H200 and 0.00147 on B200.
  [Machine-readable results](../data/muse-spark-validation.json) record each case.
- Real API text, separate reasoning, streaming, tool calls, and the tool-result
  round trip pass on one B200 at context 2048. It answered `4`, called
  `weather({"city":"Paris"})`, and then reported the supplied sunny/20 C result.
  Streaming reproduced the non-streaming content and reasoning exactly.
- A6000 build and CUDA operation checks pass (CUDA 12.4, architecture 86).
  Its real-template chat-parser checks also pass.
  Native model inference there remains pending: the existing Qwen service leaves
  too little free VRAM for the Muse loader's 1 GiB safety reserve, even with the
  small fixture. The current service was left running.

The B200 API smoke tests measured 10.0–16.9 generated tokens/s. The first 157-token
prompt ran at 4.89 tokens/s with a cold expert cache; subsequent prompts ran at
12.8–17.5 tokens/s. These are short functional tests, not a general benchmark.
The longer tool-result request used 774 prompt tokens and generated 41 tokens.
Peak observed GPU usage was 181618 MiB on GPU 0, with 1025 MiB free once the cache
filled; GPU 1 was idle during serving. Peak engine RSS was 170.51 GiB, and peak
container memory including file cache was 301.61 GiB. The first short B200
native check reported 11812944 512-byte OS input blocks (about 5.63 GiB); later
short checks were served from host cache. Expert host-to-device transfers are
reported separately in the engine log.

The tool test initially exposed a CUDA allocation-rounding OOM. Enforcing
physical VRAM headroom on every cache miss fixed it; the entire API suite then
passed, including prompts longer than the original failure point. The current
server runs on the B200 pod at `127.0.0.1:8080`, with logs under
`/workspace/muse-spark/b200/results/`.

The cloud machines' `/workspace` is object-backed FUSE storage. Direct
mapped reads were too slow for practical validation. The validation setup stages
the pinned weights in `/dev/shm` and verifies their SHA-256 hashes, retaining the
persistent download in `/workspace`. This requires at least 226 GB available in
the RAM filesystem, plus process and OS headroom; it is not an A6000 requirement.
The B200 validation machine instead uses its 300 GB local disk at `/opt/muse-model`.
All six shards on both machines passed the hashes in the pinned manifest.
HF cache symlinks appeared as empty files on the second FUSE mount; copying the
underlying `blobs/<sha256>` files into ordinary, correctly named GGUF files avoided
that problem. Prefer a local filesystem for the working weights.
