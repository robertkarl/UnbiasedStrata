# Relationship to upstream

## Sources

| What | Where | Version |
| --- | --- | --- |
| Engine | https://github.com/Niko1221/Strata | commit `6f32ec070f23ced9f50e704d854d775da52591ab`, engine 0.1.39 |
| ggml (CPU backend, CUDA matrix kernels) | https://github.com/ggml-org/llama.cpp, as the submodule `third_party/llama.cpp` | commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the one upstream Strata pins |

The first commit of this repository is the upstream snapshot as published. One file was not imported:
`bench/results/2026-09-30-community-rtx-5090/engine.log`, which upstream's own ignore rules exclude.
`git diff <first commit> HEAD -- src include CMakeLists.txt` shows every change to the engine and its build.

## What this fork changed

- **Build:** ggml is taken from the submodule. Upstream cloned llama.cpp during the build.
- **Removed:** the Python installer and launchers, the Python server and web app, the Python tools, the
  Intel SYCL port, Docker files, benchmark results, the test suite, translations, and all docs except
  `docs/upstream/DETAILS.md` and the paper.
- **Added:** `frontend/`, the C++ front end (tokenizer, chat template, output parser and HTTP API, using
  llama.cpp's code from the submodule), and a build target that links it with the engine into one binary.
- **ggml / llama.cpp common (`third_party/llama.cpp`):** one change, kept as
  `patches/llama.cpp-qwen3-coder-any-argument-order.patch`, applied to the submodule's working tree at CMake
  configure time (and by `make build`); Qwen3.6's template routes to the same parser:
  the Qwen3-Coder tool-call grammar accepts a call's arguments in any order. Upstream requires the
  required arguments first; Qwen3.8 Flash-Next often writes the optional ones first (`limit`, `offset`,
  `path`), and because the grammar's tool-call group is optional and has no end anchor, such a call was
  dropped silently: the client got `finish_reason=stop` and an empty answer (2026-10-08).
- **Engine source (`src/`, `include/`):** three changes, all for loading from a slow or remote store. The
  engine's own program is compiled into the binary with its entry point renamed; the front end starts it
  as a child process.
  - `include/strata/platform/robust_read.hpp` (new): positional reads that retry an error a network store
    gives for a moment, and lower the number of parallel readers while it does.
  - `src/core/expert_source.cpp`: the expert loader reads through it, reports progress, and names the file,
    offset and error when it fails.
  - `src/kernels/ngram.cpp`: `--ple-io ram` reads the n-gram table into memory of its own with 16 readers,
    where upstream locked the file mapping (which fails under a container's `ulimit -l`).

Code paths for other models, AMD cards and Windows are still present in the engine source. They are not
built by `make build`, not tested and not supported here.

## Prepared model files

The pack and the draft head are produced once with upstream Strata's tools at the commit above. These are
the commands that were used, with `U` an upstream checkout, `G` this repository's
`third_party/llama.cpp/gguf-py`, and Python 3 with `numpy` and `regex`:

```
export STRATA_GGUF_PY=$G
# the pack, from the model's first shard (the second must be beside it)
python $U/tools/iq_pack.py --gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out pack-iq3-xxs
cp $U/data/expert-profile.bin pack-iq3-xxs/

# the draft head, from the original checkpoint's 31 draft-head tensors (about 5 GB, fetched over HTTP)
python $U/tools/mtp_fetch.py fetch  --out mtp-bf16
python $U/tools/mtp_fetch.py verify --out mtp-bf16
python $U/tools/mtp_pack.py --src mtp-bf16 --experts q2_0 --out mtp-bf16/mtp-q2_0.gguf
python $U/tools/mtp_rt.py --gguf mtp-bf16/mtp-q2_0.gguf --out mtp
cp $U/data/draft_vocab.bin mtp/
```

The model shards used were revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` of
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, verified against their published SHA-256 checksums. The
prepared files have not been published yet. Building them twice gave the same bytes; their SHA-256:

```
68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267  pack-iq3-xxs/conversions.json
d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13  pack-iq3-xxs/dense.bin
8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf  pack-iq3-xxs/expert-profile.bin
1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148  pack-iq3-xxs/index.txt
ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee  pack-iq3-xxs/native_experts.txt
c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a  mtp/dense.bin
991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd  mtp/dense.txt
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  mtp/draft_vocab.bin
09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f  mtp/experts.bin
```

## Prepared files for Qwen3.6-35B-A3B

Made with this repository's own tools (Python 3 with `numpy`; `STRATA_GGUF_PY` at `third_party/llama.cpp/gguf-py`),
from unsloth's files: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` and `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` of
[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), and the `-MTP` repository's
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (41 blocks; the draft block is `blk.40`) for the draft pack.

```
export STRATA_GGUF_PY=$PWD/third_party/llama.cpp/gguf-py
# the pack, one per model file (the tokenizer, the dense weights' index, the expert table, the route profile)
python3 -I tools/iq_pack.py --gguf Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --out qwen36-q4
python3 -I tools/iq_pack.py --gguf Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --out qwen36-iq3
cp data/qwen36-expert-profile.bin qwen36-q4/expert-profile.bin        # tools/make_profile.py from the route trace
cp data/qwen36-expert-profile.bin qwen36-iq3/expert-profile.bin
# the draft pack, from the -MTP file's draft block; serves either trunk
python3 -I tools/mtp_qwen36_pack.py --gguf Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf --out qwen36-mtp-draft
```

`--draft-vocab none` leaves `draft_vocab.bin` out (the drafter then scores the whole vocabulary: a few percent
higher acceptance, 345 MiB more VRAM for the head). Building twice gives the same bytes. SHA-256:

```
9c964e657212fea1f24905dd7b0a89b82fd807d19fab0b41da14251b07b88fbe  Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf (the -MTP repository's file)
7c20e11993ff853fe5698843d610b2a87927b04f9a4607551f4e023a39bc71ba  qwen36-q4/conversions.json
53467f9b49bbc57d570eb8802125144dd4b4d7fd6a658d685ef72005d75a5f4f  qwen36-q4/dense.bin
eb2ed6c8d86ebe0c16bb7a09af6c24fc4059627cc95e35def07c41568a30283c  qwen36-q4/expert-profile.bin
1a4725f13e24691a3d6fd10dba4bb6cb217c4831ff8d5529865fc01f39413c79  qwen36-q4/index.txt
257d84e2d36a2e6572cd0dd8382283defe28bc07c2c97ece22ab6c1bd2f5bff4  qwen36-q4/native_experts.txt
39bf397e3fb8591ab84cb9142dbd43f665c3bff456e9772c47dc04db5ca28195  qwen36-iq3/conversions.json
53467f9b49bbc57d570eb8802125144dd4b4d7fd6a658d685ef72005d75a5f4f  qwen36-iq3/dense.bin
eb2ed6c8d86ebe0c16bb7a09af6c24fc4059627cc95e35def07c41568a30283c  qwen36-iq3/expert-profile.bin
685d475659736b415a5801413f116a4371970d3cb24b16e53625b98019895c73  qwen36-iq3/index.txt
820053214b08d3eab1378705fb7c345a724ed0fe16544df07b2f519423bb10e8  qwen36-iq3/native_experts.txt
4a93092f5716d289367b97ce107d8b08b2236a8e1954bc7f746881875c73d03a  qwen36-mtp-draft/dense.bin
dd5990d0ef0926252cad59090fa1d00f138db2e3f316eb96691db83940cde339  qwen36-mtp-draft/dense.txt
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  qwen36-mtp-draft/draft_vocab.bin
9126741d7e98f0e470cd04bf7ce3c91a89c08c116af02aa2d8d666c4195fdb15  qwen36-mtp-draft/experts.bin
1e5ad8c088196bc95b49534ce68db9c278635c8a68f7b605ef1776589dea29d7  qwen36-mtp-draft/mtp.json
048604044d93d7dd4e4e1140deba8ccf681b5c7c11e330e9434a570af42bcfa2  qwen36-mtp-draft/native_experts.txt
```

(`dense.bin` is byte-identical between the two packs: the dense weights unsloth keeps unquantized are the same in
both files. The tokenizer directory's files are listed in `qwen36-*/index.txt`.)
