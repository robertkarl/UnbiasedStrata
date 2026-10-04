# UnbiasedStrata

A cut-down fork of [Strata](https://github.com/Niko1221/Strata), an inference engine for the
Qwen3.8-Flash-Next model. The engine source is unchanged from upstream. Everything around it has been
removed or is being replaced, so that building and running work the way they do for llama.cpp: clone,
build, copy one binary, run. Nothing is downloaded during the build or at start-up.

## Status

Work in progress. What exists today:

- The upstream engine, building from a pinned llama.cpp submodule with no network access.
- `make build`.

Not done yet:

- The C++ front end (tokenizer, chat template, HTTP API). Upstream does these in Python, which this fork
  removed. Until it exists, the binary is upstream's engine and takes token ids, not text.
- Published model files (see below).
- Any benchmark of this fork. No speed is claimed.

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

The result is `build/strata`. It links the CUDA runtime libraries dynamically, as llama.cpp does.

## Model files

The engine needs three things on disk before it starts. It never fetches them.

1. The two IQ3_XXS GGUF shards from
   [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
2. The prepared "pack" for those shards.
3. The prepared draft head.

Items 2 and 3 are made once with upstream Strata's tools and will be published; see [UPSTREAM.md](UPSTREAM.md).

## Credits and licence

MIT, as upstream. The engine is the work of Niko1221 and the Strata contributors; see [LICENSE](LICENSE)
and [UPSTREAM.md](UPSTREAM.md). llama.cpp and ggml (MIT) are used as a pinned submodule. The model files
have their own licences.
