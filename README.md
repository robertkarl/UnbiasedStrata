# UnbiasedStrata

[Strata](https://github.com/Niko1221/Strata)'s inference engine and llama.cpp's tokenizer, chat-template and
HTTP code, built into one binary that serves Qwen3.8-Flash-Next (one 24 GB+ card, 128 GB RAM) and
Qwen3.6-35B-A3B (any single NVIDIA card from 6 to 32 GB, Pascal or newer) over an OpenAI-style API. No
network is needed to build or run it: rsync the source, run `make build`, start the binary. No Python, no
installer.

## Requirements

| | Build | Run |
| --- | --- | --- |
| Network access | no | no |
| CMake 3.24+, C++20 compiler (tested: g++ 13.3) | yes | no |
| CUDA | toolkit 12.x (tested: 12.4 and 12.8; 12.8+ for RTX 50; CUDA 13 dropped Pascal and Volta) | `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` and the NVIDIA driver |

Hardware: Linux, x86-64 with AVX2, one NVIDIA card.

| Model | Card | RAM |
| --- | --- | --- |
| Qwen3.8-Flash-Next | 24 GB or more (RTX 30, 40 or 50) | 128 GB |
| Qwen3.6-35B-A3B | 6 to 32 GB, Pascal (GTX 10 series, sm_61) or newer, one card | enough to hold the experts the card does not: 32 GB for Q4_K_XL on an 11 GB card, 16 GB for IQ3_XXS (the table under Sizing) |

The model can be on a local disk or on a network volume; what the card does not hold is read into RAM at
start, and what RAM does not hold is read from the file on a miss, so a slow store costs start-up time and,
on the small cards, decode speed.

## Build

```
rsync -a UnbiasedStrata/ host:UnbiasedStrata/   # or: git clone, then git submodule update --init
ssh host 'cd UnbiasedStrata && make build'      # produces build/unbiased-strata
```

The tree is 6 MB plus the pinned llama.cpp submodule in `third_party/` (about 170 MB checked out). The
build fetches nothing. `make build JOBS=N` sets the job count; `CUDA_ARCHS` defaults to
`61;70;75;86;89;120`, one binary for every card from Pascal up, minus any architecture the installed
`nvcc` cannot compile (12.4 has no 120, 13 has no 61 or 70; the build says what it dropped). `make build
CUDA_ARCHS=61` builds for one card in about 3 minutes. An architecture below 75 in the list turns on
`STRATA_EXPERIMENTAL_SM60`, the Pascal/Volta build (FP32 and dp4a paths; no tensor cores). The
chat-parser patch in `patches/` is applied to the submodule at configure time.

## Run

```
# Qwen3.8-Flash-Next
./unbiased-strata --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
                  --pack /models/pack-iq3-xxs --mtp /models/mtp
# Qwen3.6-35B-A3B at the full 262,144-token context (the engine sizes itself to the card; see Sizing)
./unbiased-strata --model /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
                  --pack /models/qwen36-q4 --mtp /models/qwen36-mtp-draft --ctx 262144 -- --kv q4_0
```

| | |
| --- | --- |
| Endpoints | `POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `GET /health`, on `127.0.0.1:8080` |
| Options | `--ctx N` (32768), `--threads N` (default: physical cores, or half the CPU quota in a container), `--host`, `--port`, `--mmproj PATH`, `--vision-tokens N` (1024); engine flags after `--`: `--kv fp16\|q4_0\|int8`, `--vram-cap MiB` and `--ram-cap GiB` (plan as a smaller card or host), `--expert-cache N`, `--prompt-cache N` (checkpoints kept between requests) |
| Sampling | greedy unless `temperature` > 0 |
| Reasoning | returned as `reasoning_content`; `"reasoning_effort": "none"` turns it off |
| Tools | OpenAI format (`tools`, `tool_calls`) |
| Images | with `--mmproj`, OpenAI `image_url` parts holding a base64 `data:image/...` URL (PNG, JPEG, BMP or GIF), several per message |
| Not supported | batch requests (remote-expert mode and the web UI stay upstream-only) |

Images: `--mmproj /models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` loads the model's own vision encoder (from the same
Hugging Face repository as the GGUF; 0.9 GB). The encoder runs on the CPU, so the card keeps its expert cache:
about 25 s and 1,000 prompt tokens for a 1-megapixel screenshot on 8 threads, 0.2 s for a small image; a repeated
image is served from a cache. `GET /v1/models` then lists `image` under `input_modalities`. Without `--mmproj` an
image request is refused with a 400.

The engine runs as a child process. When it exits or its stall watchdog fires, the front end exits with a failure
status, for a supervisor such as systemd (`Restart=on-failure`) to restart; `/health` and `/v1/models` answer 503
while the engine is down. Reasoning streams as it is generated; content and tool calls are sent once the whole
message is parsed, so a partial tool-call JSON never changes arguments already sent. `make test-frontend` checks
the child handling and the protocol without a model; `make test-chat-parser MODEL=<shard 1>` checks the chat
template and tool-call parsing against the real GGUF.

## The supported models

### Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (125B parameters, mixture of experts), the full model at IQ3_XXS quantization, from
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
directory `IQ3_XXS`, revision `ed59f920`. No other model, quantization or pruned variant is supported. The
draft (MTP) head is required.

Everything below must be on disk before start. Nothing is fetched.

| What | Size | SHA-256 or source |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (weights) | 47.04 GB | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (n-gram table), beside shard 1 | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| Pack: `index.txt`, `dense.bin`, `native_experts.txt`, `conversions.json`, `expert-profile.bin` | 1.5 GB | made once with upstream's tools; not published yet |
| Draft head: `experts.bin`, `dense.bin`, `dense.txt`, `draft_vocab.bin` | 0.8 GB | made once with upstream's tools; not published yet |

About 78 GB in total. The commands that make the pack and draft head, and the SHA-256 of every file they
produce, are in [UPSTREAM.md](UPSTREAM.md).

### Qwen3.6-35B-A3B

Qwen3.6-35B-A3B (35B parameters, 3B active: 40 layers, 256 experts of which 8 route per token, 30 Gated
DeltaNet layers and 10 full-attention layers, 262,144-token context), unsloth's UD quantizations from
[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) and
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF). Two
quantizations, with the pack and draft pack made by this repository's own tools (`tools/iq_pack.py`,
`tools/make_profile.py`, `tools/mtp_qwen36_pack.py`; commands and every file's SHA-256 in
[UPSTREAM.md](UPSTREAM.md)):

| What | Size | SHA-256 |
| --- | --- | --- |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, **the default** (K-quants: the fast AVX2 path for the experts the CPU computes) | 22.36 GB | `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450` |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` from the **-MTP** repository (the same trunk plus the draft block `blk.40`; serves as the model file too) | 22.85 GB | `55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1` |
| `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, for 6 and 8 GB cards | 13.21 GB | `9c964e657212fea1f24905dd7b0a89b82fd807d19fab0b41da14251b07b88fbe` |
| Pack `qwen36-q4/` or `qwen36-iq3/` (`index.txt`, `dense.bin`, `native_experts.txt`, `conversions.json`, `expert-profile.bin`, `tokenizer/`) | 65 MB | per file in UPSTREAM.md |
| Draft pack `qwen36-mtp-draft/` (`experts.bin`, `dense.bin`, `dense.txt`, `native_experts.txt`, `mtp.json`, `draft_vocab.bin`), cut from the -MTP file's draft block; serves both quantizations | 0.5 GB | per file in UPSTREAM.md |

No other Qwen3.6 file is accepted: the IQ4_XS-MTP file's draft block (Q3_K experts) and the 27B dense model
are refused by name. Images work with `--mmproj` and the model's own encoder (`mmproj-F16.gguf` from the -MTP repository,
899 MB); the encoder runs on the CPU, so the first use of a full-screen image costs 24 s on an i9-9900K and
about 70 s on a Ryzen 7 1700, and repeats are served from a cache.

#### Sizing

At start the engine prints its plan for the card it finds: the fixed set for the requested context (the
dense weights, the drafter, the KV cache, the session, a reserve), then every remaining megabyte as expert
slots, filled by the route-trace profile's rank. If the fixed set does not fit, the KV cache is streamed
to RAM above a resident window, then the context is lowered toward 131,072, then the engine refuses with
the numbers and suggests the IQ3_XXS file. It never silently picks a shorter context. The experts that do
not fit in VRAM are read from pinned RAM when the host has it, else from the file on a miss; `/health`
reports the VRAM hits, RAM reads and file reads per request. `--vram-cap` and `--ram-cap` make a big card
plan like a small one.

| VRAM | Q4_K_XL at 262,144, q4_0 KV: fixed set, experts resident | IQ3_XXS: fixed set, experts resident |
| --- | --- | --- |
| 24 GB (RTX 4090) | 4,823 MiB; 9,980 of 10,240 (97%) | 4,331 MiB; all 10,240, 8.8 GB spare |
| 11 GB (GTX 1080 Ti) | 4,461 MiB; 3,575 (35%) | 4,019 MiB; 6,748 (66%) |
| 8 GB | 1,949 (19%) | 3,873 (38%) |
| 6 GB | 598 (6%) | 1,486 (15%) |

The 24 GB row is measured; the others are the planner's figures from the same files under `--vram-cap`
(docs/qwen36-port/step2-notes.md section 5, step4-pack.md section 5).

**RAM is a requirement, not a tier.** The engine assumes the card and the host RAM together hold the expert set;
every expert the card does not hold is pinned in RAM. What the host must hold is the expert bytes minus the card's
share, plus the engine's own 3 to 4 GB and the OS:

| VRAM | Q4_K_XL experts off the card | RAM to have | IQ3_XXS experts off the card | RAM to have |
| --- | --- | --- | --- | --- |
| 24 GB | none | 8 GB | none | 8 GB |
| 16 GB | 8 GB | 16 GB | none | 8 GB |
| 11 GB | 13 GB | 32 GB (24 is tight) | 4 GB | 12 GB |
| 8 GB | 15 GB | 32 GB | 7 GB | 16 GB |
| 6 GB | 17 GB | 32 GB | 9 GB | 16 GB |

Below those figures the engine still starts, maps the rest from the file and says so in its plan, but that is a
fallback, not a supported configuration: measured once on the GTX 1080 Ti at Q4_K_XL under a 12 GiB memory bound,
decode fell from 52 tokens/s to 12 to 15, the NVMe's rate, with about 12 experts per token read from the file.

## Status

Checked on one machine: RTX 3090, 32 vCPUs of an EPYC 7763, 125 GB RAM, CUDA 12.8, in a container.

| | |
| --- | --- |
| Build | from an rsynced tree with no git metadata; the binary holds no download code |
| Serving | chat, streaming, reasoning, tool calls; a coding agent (pi) wrote and ran a program through it |
| RTX 4090 (24 GB), 82 GB RAM, Debian, bare metal | serving with tool calls and images, through pi (file attachments and clipboard paste) and curl, over several days of use |
| Start from local disk | 19 s with the files in the OS file cache |
| Start from a network volume | 12 min for 76 GB (S3-backed RunPod volume, 120 to 330 MB/s). The volume refused reads twice with "cannot allocate memory"; the loader retried and went on |
| Not tested | RTX 50 cards; a cold start from local disk |

Qwen3.6-35B-A3B, checked on the RTX 4090 (VM, 16 vCPU i9-9900K, 62 GB RAM, CUDA 12.4):

| | |
| --- | --- |
| Correctness | the engine's logits held to a numpy reference and to llama.cpp at every position of the real files (greedy top-1 64/64 on three prompts at IQ3_XXS, 61-64/64 at Q4_K_XL, mean KL 1e-3 to 7e-3); the expert-miss path (what every card below 24 GB runs on) held with the same prompts under `--vram-cap` |
| Serving | chat, streaming of reasoning and answer per token, tool calls streamed and not, multi-turn prompt checkpoints at 262,144 context, at 24 GB, under an 11 GB cap (Q4_K_XL) and a 6 GB cap (IQ3_XXS); a coding agent (pi) wrote and ran a program through it |
| Pascal | built and unit-tested on a GTX 1080 Ti (sm_61: every kernel parity test, the pack, the drafter, the verify window); the expert-miss path held to llama.cpp's logits there too |

Speed against llama.cpp on the same card (RTX 4090, `--vram-cap` for the smaller budgets, 262,144 context,
no RAM cap, drafter on; llama.cpp with `--n-cpu-moe N` chosen to fit the same budget, no speculation). Decode
is a 1,200-token answer to a 2.4K prompt; prefill is a 32K prompt. Tokens/s, engine / llama.cpp:

| Budget | Q4_K_XL decode | Q4_K_XL prefill | IQ3_XXS decode | IQ3_XXS prefill |
| --- | --- | --- | --- | --- |
| 24 GB | 251 / 159 (1.6x) | 4,319 / 8,604 (0.50x) | 318 / 167 (1.9x) | 4,426 / 8,245 (0.54x) |
| 16 GB | 180 / 41 (4.4x) | 4,232 / 1,798 (2.4x) | 319 / 44 (7.3x) | 4,401 / 2,637 (1.7x) |
| 12 GB | 122 / 38 (3.3x) | 3,877 / 1,475 (2.6x) | 256 / 40 (6.4x) | 4,381 / 2,223 (2.0x) |
| 11 GB | 104 / 37 (2.8x) | 3,723 / 1,420 (2.6x) | 234 / 39 (6.1x) | 4,375 / 2,137 (2.1x) |
| 8 GB | 64 / 33 (2.0x) | 3,003 / 1,212 (2.5x) | 135 / 34 (4.0x) | 4,320 / 1,854 (2.3x) |
| 6 GB | 45 / 30 (1.5x) | 517 / 1,108 (0.47x) | 81 / 32 (2.5x) | 1,121 / 1,713 (0.65x) |

The llama.cpp 24 GB Q4_K_XL row is at 131,072 context; it does not load at 262,144 with every layer on the card.
Prompt processing is the weak side: with every expert resident llama.cpp is twice as fast, and at 6 GB the engine
spends the prompt copying experts to the card. Full tables, profiler windows and plots are in the regina
repository's `analysis/qwen36-35b-single-gpu-ladder/engine-4090/`.

On the GTX 1080 Ti (VM, 14 vCPU of a Ryzen 7 1700, 52 GB RAM), against llama.cpp on the same card and host
(`--n-cpu-moe N` at the lowest N that fits each budget, 8 threads, no speculation, the model file in RAM).
262,144 context on both sides. Tokens/s, engine / llama.cpp:

| Budget | Q4_K_XL decode | Q4_K_XL prefill 3.1K | Q4_K_XL prefill 32K | IQ3_XXS decode | IQ3_XXS prefill 3.1K | IQ3_XXS prefill 32K |
| --- | --- | --- | --- | --- | --- | --- |
| 11 GB (the card) | 66 / 28 (2.4x) | 941 / 570 (1.7x) | 462 / 545 (0.85x) | 81 / 33 (2.5x) | 910 / 904 (1.0x) | 453 / 714 (0.63x) |
| 8 GB (`--vram-cap 8192`) | 54 / 25 (2.1x) | 911 / 488 (1.9x) | 442 / 487 (0.91x) | 69 / 28 (2.5x) | 900 / 747 (1.2x) | 453 / 637 (0.71x) |
| 6 GB (`--vram-cap 6144`) | 42 / 24 (1.8x) | 361 / 452 (0.80x) | 265 / 460 (0.58x) | 55 / 24 (2.3x) | 748 / 614 (1.2x) | 437 / 562 (0.78x) |

Decode is 1.8 to 2.5 times llama.cpp; the GPU computes 85% of the decode window against llama.cpp's 39 to 45%.
A long prompt is slower than llama.cpp at every budget on this card. The draft head is worth a third of the decode
speed (49 tokens/s without drafts at Q4_K_XL on the full card). Tables and plots: regina
`analysis/qwen36-35b-single-gpu-ladder/engine-1080ti/` and `llama-1080ti-52g/`.

Speed, one run of each, same machine and model files, greedy sampling, reasoning off. Generation is the mean
of 8 different prompts of 400 tokens; prompt speed is the mean of 2 different prompts of about 20,000 tokens.

| | Generation, tokens/s | Prompt, tokens/s |
| --- | --- | --- |
| This fork | 107 (87 to 116) | 2,483 |
| Upstream, with `--pool-workers 15 --ple-io ram` | 78 (61 to 93) | 2,439 |
| Upstream, arguments as its installer writes them | 20 (16 to 21) | 2,368 |

Upstream as installed started 127 CPU workers on the container's 32 vCPUs and read the n-gram table from
disk per token; the middle row removes both. The remaining gap to the fork is not explained yet: the two
builds differ in compile flags and in how the n-gram table is held in RAM, and each row is a single run.

## Known problems

- Text in a message that looks like a control token (`<think>`, `<tool_call>`) is tokenized as that control
  token. Upstream's tokenizer keeps it as plain text.
- Replies are not reproducible token for token between runs or between this fork and upstream, even with
  greedy sampling, so "same output" cannot be checked by comparing text.
  For Qwen3.6-35B-A3B the cause is traced: an expert the card does not hold is computed by a different
  kernel on the CPU, and the cache keeps changing which experts those are for the first few requests; replies
  diverge only where two tokens are within about 0.3 logits, and every state holds to llama.cpp's logits.
  `--adapt-every 0 --suffix-draft 0` makes a request repeat token for token. `--pcie-frac 1.0` does too
  only while a layer misses fewer than 16 experts per window; past that the rest are computed on the CPU.
- Qwen3.6-35B-A3B: between user turns the prompt cache reuses the previous prompt but not the previous reply,
  because the model's template re-renders an earlier reply without its reasoning. Tool loops reuse nearly
  everything (1,867 of 1,948 tokens in a pi session); `chat_template_kwargs.preserve_thinking` with
  `reasoning_content` sent back reuses the reply too, at the cost of keeping old reasoning in context.
  A tool call arrives as one delta, not streamed. Images are checked on one screenshot only, and not
  against llama.cpp; the RTX 50 and Volta builds have not
  run on a card. Prompt processing is slower than llama.cpp with every expert on the card, at 6 GB, and for long
  prompts on a GTX 1080 Ti.

## Compared with upstream Strata (6f32ec0)

| | Upstream | This fork |
| --- | --- | --- |
| Engine source (`src/`, `include/`) | 272 files | 58 files changed for the Qwen3.6 port (the model geometry read from the GGUF, the plain-residual block, dense attention, the SiLU-gated DeltaNet, the expert kernels for the UD quantizations, the VRAM planner, the drafter pack) plus retrying reads from slow stores; docs/qwen36-port/ has the per-step notes |
| Build | clones llama.cpp during CMake | pinned submodule, fetches nothing; one patch, applied at configure time: tool-call arguments in any order (see UPSTREAM.md) |
| Setup | Python installer downloads the model and builds the pack and draft head | none; files must already exist |
| Server | Python | C++, in the same binary (`frontend/`), with the image encoder from llama.cpp's `mtmd` |
| Python | 33,745 lines | 0 |
| C, C++ and CUDA | 208,254 lines (with tests and the Intel port) | 84,086 lines |
| Tracked files | 851, 27 MB | 292, 6 MB |
| Configurations | several models, NVIDIA, AMD and Intel, Linux and Windows, multi-GPU | two models, one NVIDIA card (Pascal to RTX 50) or one card per tier, Linux |

## Credits and licence

MIT, as upstream. The engine is the work of Niko1221 and the Strata contributors; see [LICENSE](LICENSE) and
[UPSTREAM.md](UPSTREAM.md). llama.cpp and ggml are MIT. The model files have their own licences.
