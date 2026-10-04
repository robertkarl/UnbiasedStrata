# UnbiasedStrata

A cut-down fork of [Strata](https://github.com/Niko1221/Strata), an inference engine for the
Qwen3.8-Flash-Next model. The engine source is unchanged from upstream. Everything around it has been
removed or is being replaced, so that building and running work the way they do for llama.cpp: clone,
build, copy one binary, run. Nothing is downloaded during the build or at start-up.

## Status

Work in progress. What works today, checked on an RTX 3090 pod:

- Clone, `git submodule update --init`, `make build`, with no network access during the build.
- One binary, `build/unbiased-strata`, that loads the model and serves an OpenAI-style chat API. It answered
  streaming and non-streaming requests correctly, with the model's reasoning separated from its answer and
  draft tokens being accepted.

Not done yet:

- **No speed is claimed.** The only end-to-end run so far was on a pod whose network volume could not be
  read normally by the engine, so it ran in a slow fallback mode. A run on a machine with a local disk is
  still to do.
- The prepared model files (below) are not published yet.
- The engine source is still upstream's in full. Cutting it down to what this one configuration needs is
  the next piece of work.
- Tool calling and image input are not supported.

## What it supports

One configuration, on purpose:

| | |
| --- | --- |
| Model | Qwen3.8-Flash-Next, GSQ-RCO IQ3_XXS, with the draft (MTP) head |
| GPU | One NVIDIA card with 24 GB or more, RTX 30 series or newer |
| RAM | 128 GB |
| CPU | x86-64 with AVX2 |
| OS | Linux |

Tested so far:

| Machine | What was checked |
| --- | --- |
| RTX 3090 pod, CUDA 12.8, g++ 13.3 | `git submodule update --init` and `make build` complete with network access blocked for the build (144 s); the binary starts. The model has not been run. |

## Build

Needs CMake 3.24 or newer, a C++20 compiler and the CUDA toolkit (12.8 or newer; 13.0 or newer for RTX 50
cards).

```
git clone <this repo>
cd UnbiasedStrata
git submodule update --init
make build
```

The result is `build/unbiased-strata`, a single file you can copy to another machine. It needs the NVIDIA
driver and the CUDA runtime libraries there, as llama.cpp does.

## Run

```
./unbiased-strata --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
                  --pack /models/pack-iq3-xxs --mtp /models/mtp
```

It listens on `127.0.0.1:8080` and serves `POST /v1/chat/completions` (with streaming), `GET /v1/models` and
`GET /health`. Loading takes a few minutes. Options: `--ctx N` (default 32768), `--threads N`, `--host`,
`--port`. Inside a container set `--threads` yourself: the default is every physical core of the host.

Requests are greedy unless they set a `temperature` above zero. The model thinks before it answers; send
`"reasoning_effort": "none"` to turn that off.

## Model files

The engine needs three things on disk before it starts. It never fetches them.

1. The two IQ3_XXS GGUF shards from
   [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
   side by side in one directory.
2. The prepared "pack" for those shards (about 1.5 GB): `index.txt`, `dense.bin`, `native_experts.txt`,
   `conversions.json`, `expert-profile.bin`.
3. The prepared draft head (about 0.8 GB): `experts.bin`, `dense.bin`, `dense.txt`, `draft_vocab.bin`.

Items 2 and 3 are made once with upstream Strata's tools and will be published; see [UPSTREAM.md](UPSTREAM.md).

## Credits and licence

MIT, as upstream. The engine is the work of Niko1221 and the Strata contributors; see [LICENSE](LICENSE)
and [UPSTREAM.md](UPSTREAM.md). llama.cpp and ggml (MIT) are used as a pinned submodule. The model files
have their own licences.
