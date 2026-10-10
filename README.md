# UnbiasedStrata

A cut-down fork of [Strata](https://github.com/Niko1221/Strata), an inference engine for the
Qwen3.8-Flash-Next model. The engine is updated to upstream 0.1.42 with additional engine performance/correctness ports; see [upgrade scope and validation](docs/UPGRADE.md). Everything around it has been
removed or is being replaced, so that building and running work the way they do for llama.cpp: clone,
build, copy one binary, run. Nothing is downloaded during the build or at start-up.

## Status

Work in progress. The frontend/model workflow below was checked on RTX 3090 pods before this upgrade.
The current native upgrade build and fixture results are in [UPGRADE_VALIDATION.md](docs/UPGRADE_VALIDATION.md).

- Clone, `git submodule update --init`, `make build`, with no network access during the build.
- One binary, `build/unbiased-strata`, that loads the model and serves an OpenAI-style chat API: streaming
  and non-streaming replies, the model's reasoning separated from its answer, and tool calls.
- A coding agent (pi) completed a multi-step task against it: it wrote a program, ran it through its tools
  and reported the result.

Not done yet:

- **No speed is claimed.** The only end-to-end runs so far were on a pod whose network volume the engine
  could not load experts from in its normal way (see Known problems), so it ran in a fallback mode. In that
  mode it generated at 21 to 61 tokens per second. A run in the normal mode on a machine with a local disk
  is still to do.
- The prepared model files (below) are not published yet.
- The engine source remains broadly imported, with the additional ports recorded separately. Removing unused
  source, and then the multi-GPU, AMD, Windows and other-model code, remains future work.
- Image input is not supported.

## Known problems

- **RunPod network volumes.** With the model on a RunPod network volume (`/workspace`), the engine fails at
  start with "short read or unreadable shard": once it has pinned its expert arena, reads from the volume
  return "cannot allocate memory". The cause is not known. Keeping the model on local disk avoids it. On a
  pod without enough local disk, a working fallback was shard 1 in `/dev/shm`, shard 2 on local disk, and
  `-- --mmap-experts` at the end of the command line.

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
| RTX 3090 pods, CUDA 12.8, g++ 13.3 | `git submodule update --init` and `make build` complete with network access blocked for the build. The model loads and answers chat requests, in the fallback mode described under Known problems. |

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
`"reasoning_effort": "none"` to turn that off. Tool calling follows the OpenAI format (`tools`, `tool_calls`).

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
