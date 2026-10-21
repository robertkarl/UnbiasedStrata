## Credits and license

The engines are PolyStrata's, by VecSzn and contributors, built on [Strata](https://github.com/Niko1221/Strata) by
Niko1221. The front end is built from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)'s server, common and
mtmd code. Qwen3.8-Flash-Next and Qwen3.6-35B-A3B are by the Qwen team; the IQ3_XXS GSQ-RCO file is
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)'s, the Qwen3.6 files are
[Unsloth](https://huggingface.co/unsloth)'s. Upstream's full credits: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
MIT License ([LICENSE](LICENSE)); some parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

# UnbiasedStrata 2 - note from a human.


Issues with Strata / PolyStrata:

* piping instructions from markdown into agent == bad idea
* replacing components at build-time or runtime isn't great
* desired: clone; submodule init; make; situation

UnbiasedStrata aims to be the suckless (dwm) variant of Strata.

It's been vetted in a few ways:
* agents went through and verified kernels to the difference between CPU/GPU within llama.cpp
* full a run of Qwen 3.8 Flash Next over Bob-bench wide test set with a high score.


This wouldn't exist without llama.cpp.

without Niko's Strata.

Without PolyStrata either.

Use at your own risk. Works on my machine.


# AI generate nonsense from here on out



A fork of [PolyStrata](https://github.com/VecSzn/PolyStrata) v0.1.45-poly.1 that keeps its two inference engines
unchanged and replaces everything around them: no installer, no downloads at build or start, no Python between a
request and a token. One `make build` on a Linux box with an NVIDIA card produces the engines and a C++ front end,
`unbiased-strata`, that serves an OpenAI-style chat API from llama.cpp's own server code. The target is one NVIDIA GPU,
Pascal or newer, CUDA 12.x, Linux x86-64, on a machine that may have no network at all. What was changed and why is in
[UPSTREAM.md](UPSTREAM.md); the plan this follows is [PLAN.md](PLAN.md).

```
git clone <this repo> && cd UnbiasedStrata
git submodule update --init          # shallow, third_party/llama.cpp @ robertkarl/llama.cpp UnbiasedStrata (a6de9af3)
make build                           # build/bin/{strata,strata-poly,libstrata_kq.so,unbiased-strata}
make test                            # the front end against a fake engine; no GPU, no model
```

For AMD/HIP (tested on RX 6700 XT only), install ROCm with HIP/hipBLAS and select your GPU architecture:

```sh
make build BACKEND=hip HIP_ARCHS=gfx1031 ROCM_PATH=/opt/rocm BUILD_DIR=build-hip
```
`HIP_CXX` overrides the HIP compiler. CUDA remains the default; use separate build directories for each backend.

## Run

```
unbiased-strata --engine <strata|strata-poly|path> -m <model.gguf> [llama-server options] -- <engine options>
```

`-m` names the GGUF the vocabulary and chat template are read from; no tensors are loaded from it by the front end.
`--engine` takes a path, or a bare name looked up beside the program. Everything before `--` is a `llama-server`
option (`--host`, `--port`, `--api-key`, `--alias`, `--temp`, `--top-k`, `--top-p`, `--min-p`,
`--reasoning-format`, `--chat-template-kwargs`, `--sse-ping-interval`, ...); everything after `--` goes to the
engine, which is started as a child process with `--serve` added and spoken to over its line protocol
([docs/BACKENDS.md](docs/BACKENDS.md)). Where the model ends up (GPU cache, pinned RAM, SSD) and the three expert
modes: [docs/MEMORY_LAYOUT.md](docs/MEMORY_LAYOUT.md). The server answers `/v1/chat/completions` (streaming and not), `/v1/models`,
`/health`, `/props`, `/tokenize`, `/detokenize`, as llama.cpp's server does, one request at a time.

```
build/bin/unbiased-strata --engine build/bin/strata-poly \
  -m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 127.0.0.1 --port 18091 --alias qwen3.6-35b-a3b-ud-q3_k_m \
  -- --model /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --max-context 32768 \
     --expert-profile /models/expert-profile-q3km.txt --tail-skip 6
```

With pictures, add `--mmproj /models/mmproj-F16.gguf` (the projector from the same `-MTP` repository) and
`--threads N` for the encoder before the `--`. The front end checks the projector's width against the model's at
start, encodes each picture with llama.cpp's mtmd and hands the rows to the engine over upstream's `GENI`
protocol. A picture is at most 300 rows by default (upstream's installer's CPU setting; `--image-max-tokens`
changes it), which read a 1920x1080 dashboard correctly in 15 s of encoding on a Ryzen 7 1700.
Flash-Next through `strata` takes the same `--mmproj` (its own BF16 projector) and also needs `--vision` after the
`--`, or the engine refuses the rows ("this engine was started without --vision"); checked on the 4090, 2026-10-20.

**Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS** (`strata`; the two GGUF shards, the pack directory made from shard 1, the MTP
draft directory). On the 24 GB card:

```
build/bin/unbiased-strata --engine build/bin/strata \
  -m /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --host 127.0.0.1 --port 18090 --alias qwen3.8-flash-next-iq3_xxs --sse-ping-interval 10 \
  -- --pack /models/pack-iq3-xxs \
     --native /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
     --ple-gguf /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
     --expert-profile /models/pack-iq3-xxs/expert-profile.bin \
     --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /models/mtp \
     --max-context 65536 --kv int8 --ple-io ram --pool-workers 15 --kv-resident 32768
```


### Preparing a Flash-Next pack

`strata-poly` reads a GGUF as it is. `strata` needs a pack directory and an MTP draft directory made once, offline
except for the draft download, with upstream's tools (`tools/requirements.txt`: numpy, regex; `gguf-py` comes from the
submodule). This is what upstream's installer ran. Re-run in this fork on 2026-10-20 on three hosts; the MTP directory
came out byte-identical each time, so it is published, with the commands and hashes, at
https://huggingface.co/robertkarl/UnbiasedStrataPacks: copy it instead of fetching and quantizing.

```
python tools/iq_pack.py --gguf <shard1> --out <pack>             # --compat-bf16 for Unsloth files
STRATA_SOURCE=huggingface python tools/mtp_fetch.py fetch --out <mtp>
python tools/mtp_pack.py --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf
python tools/mtp_rt.py --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt
cp data/draft_vocab.bin <mtp>/rt/draft_vocab.bin
```

Model files themselves are downloaded on a machine with network access and copied to the inference box; the fork does
not fetch them. The files that have been served, with sizes and SHA-256, are in
[docs/MODEL-FILES.md](docs/MODEL-FILES.md).

## Measured

Each number came from a run directory with the command, the raw client output, the server log and the model files'
checksums; the per-run records were removed from the tree on 2026-10-19 and are in git history at commit 7551f138 (`git checkout 7551f138 -- bench/unbiased bench/results`). What stays in `bench/` is
reusable tooling only: upstream's `bench_vs_llama.py` and its corpus, the kernel probes, and `bench/correctness/`, the
teacher-forced harness of the correctness checks. Tokens per second, median of 3 runs. Upstream is PolyStrata v0.1.45-poly.1 with its Python
server, on the same card, the same files, the same client, back to back.

**Upstream versus this fork, RTX 3090, 2026-10-18** (the step-9 record, in history at 7551f138; the client is
upstream's `tools/strata_bench.py --runs 3 --long` plus a three-document sustained supplement):

| Model | Workload | upstream | fork |
| --- | --- | --- | --- |
| Flash-Next IQ3_XXS, 64K ctx | chat decode | 87.1 | 103.1 |
| | sustained decode, 512 tokens | 105.1 | 119.1 |
| | prefill 4K / 16K / 32K | 1924 / 2342 / 2393 | 2065 / 2531 / 2616 |
| | agent turn decode | 110.6 | 116.8 |
| Qwen3.6 UD-Q3_K_M, 32K ctx | chat decode | 220.8 | 213.5 |
| | sustained decode, 512 tokens | 207.6 | 200.0 |
| | prefill 4K / 16K | 3652 / 3570 | 3822 / 3486 |
| | agent turn decode | 194.0 | 186.9 |

Flash-Next: the fork is ahead on every workload. Qwen3.6: decode is within the run-to-run range; 16K prefill is 2.3 %
slower with non-overlapping ranges and is recorded as an open finding. The two engines were built separately in the
two trees, so an engine-build difference is not excluded.

**pi on a fixed coding task** (pi 0.85.1, Flash-Next on the 3090, three runs a side, `…-step9/row3/`): every run
completed the task. Generated tokens per wall-clock second, the whole run: upstream 72.1, fork 76.6 (medians); the raw
wall times (36.3 vs 44.7 s) follow the length of each sampled trajectory.

**Short-prompt decode through the fork, first end-to-end runs** (`…-step6/`, 2026-10-17; ranges over the smoke
requests, not medians of a benchmark):

| Model | RTX 3090 | RTX A6000 |
| --- | --- | --- |
| Qwen3.6 UD-Q3_K_M | 211 to 227 | 214 to 235 |
| Qwen3.6 UD-IQ4_XS | 205 to 245 | 207 to 215 |

Greedy answers to the same prompt were identical on the two boxes and identical to upstream on the pod for both
Flash-Next and Qwen3.6 (text and token count). Against llama.cpp at the pinned commit (PLAN.md step 12,
harness `bench/correctness/`, record in history at 7551f138, 2026-10-18 on the A6000): teacher-forced on 16 sequences per model, both engines' per-step
distributions are as close to llama.cpp's GPU run as llama.cpp's own CPU run is (mean KL 0.6 to 1.2x that spread on every
set and arm), and the greedy token agrees at every position with a 0.5-nat margin except a handful of near ties per arm
(Qwen3.6: 4 of 2,700; Flash-Next: 0 to 4 of about 2,880 depending on the arm). The findings, each with a ticket, are in
that directory's README: upstream's own `poly_expert_parity` exceeds its bound on one expert; speculation is not bitwise
inert above 4-token windows (it passes the KL gate); `strata`'s bits depend on the free VRAM at start when not every
expert fits the card. Extended 2026-10-19 to DeepSeek V4 Flash UD-IQ2_XXS and GLM 5.3 Flash 3.0bit on a 2x RTX PRO 6000 pod (section 8 of that record): T2 holds on every set and arm, 0.69 to 1.13x llama.cpp's own CPU-vs-GPU spread; a handful of 0.5-nat-margin flips per model where both llama.cpp runs agree (USTRAT-10); on GLM the draft block's VRAM moves 52 experts to the CPU, so its answers are not bitwise the no-draft ones (USTRAT-9); and llama.cpp master's CPU run differs from the pinned commit's by as much as the engine does, so the DeepSeek numbers are against 3cf03257 specifically (USTRAT-11).

**Model acceptance (PLAN.md step 13, 2026-10-19).** A coding agent (pi 0.85.1, in a container with tools, sampled as
the model card asks) completed four work items through the fork's server with every served model: a Markdown
table-of-contents CLI with tests, an HTTP key-value store with persistence, three planted bugs, and a browser
Minesweeper that a headless browser can play; Minesweeper again at temperature 1.0. Flash-Next IQ3_XXS and Qwen3.6
UD-Q3_K_M on an RTX 3090 and Qwen3.6 UD-IQ4_XS on the A6000 pass as is (the IQ4_XS temperature-1.0 page threw a
`ReferenceError` of the agent's own making); DeepSeek V4 Flash UD-IQ2_XXS and GLM 5.3 Flash 3.0bit on a B200 pass
after the two fixes under "Known limits" (argument order; GLM's architecture name in the vocabulary load). The
API-only checks (`frontend/tests/acceptance-real-engine.sh`) hold decode and prompt reading within 15 % of step 9 on
the 3090 (Flash-Next with the installer's `--pool-workers 15`; 24 costs 14 %) and record the first B200 figures
(72 and 74 tok/s decode on one card). Record: the step-13 directory, in history at 7551f138. pi's compaction defaults (reserve 16K, keep the last 20K tokens) assume a window well above 64K: against a server at `--max-context 32768` a long pi session compacts to no effect and then sends `max_tokens: 1` ("Response was truncated before completion"); serve 128K or more to pi, or lower pi's `compaction` settings (USTRAT-8).


