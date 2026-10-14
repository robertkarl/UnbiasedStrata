#!/usr/bin/env python3
"""The Qwen3.6 runner harness's fixture (docs/qwen36-port/step4-runner.md): the synthetic qwen35moe model at the real
per-layer geometry (tests/python/qwen36_prefill_fixture.py's `full4`: 4 layers, three DeltaNet and one full attention,
n_embd 2048, 16 experts top-8 of 512), written as a file a NATIVE PACK can be built from, the pack itself
(tools/iq_pack.py, with a 4 x 16 expert profile ranking every pair), and the oracle's dumps for the three sequences the
verify window and the streaming runner are held to:

    o8     8 tokens  "hello wo"                          the verify window's rows (one window)
    o11    11 tokens "hello world"                       two windows, 8 + 3 (a commit between them)
    o40    40 tokens "the quick brown fox jumps ..."     every position

each with --all-logits, as the F32 oracle (oN) and in its Q8 mode (oN_q8, --act-quant q8: the activations rounded to
Q8 before every quantized matmul, as ggml and the engine's int8 paths do - the policy's reference for such a path).

The file differs from the prompt-path fixture in one tensor: `token_embd.weight` is Q8_0 (a pack's embedding must be
a type `iq_embed_rows` serves; the oracle dequantizes it with gguf-py, bit-identical to ggml).  Without numpy or
gguf-py it writes <out>/SKIP with the reason and exits 0; the tests exit 77 on seeing it.

    python3 -I tests/python/qwen36_runner_fixture.py --out DIR [--seed 0] [--no-oracle]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

try:
    import qwen36_prefill_fixture as pf      # the writer, the geometry, the 40 tokens; imports numpy and gguf-py
    import make_profile
except ImportError as e:
    print(f"qwen36_runner_fixture: {e}", file=sys.stderr)
    out = None
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--out" and i < len(sys.argv) - 1:
            out = sys.argv[i + 1]
        elif a.startswith("--out="):
            out = a[len("--out="):]
    if out:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(out) / "SKIP").write_text(f"{e}\n")
    sys.exit(0)

CASES = {
    "o8": [ord(c) for c in "hello wo"],
    "o11": [ord(c) for c in "hello world"],
    "o40": pf.TOKENS,
}


def oracle(gguf: str, dump: str, tokens: list[int], extra: list[str]) -> None:
    cmd = [sys.executable, "-I", str(ROOT / "ref" / "qwen36.py"), gguf, "--tokens", ",".join(map(str, tokens)),
           "--dump", dump] + extra
    env = dict(os.environ)
    env.setdefault("STRATA_GGUF_PY", pf._paths.gguf_py())
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-oracle", action="store_true", help="write the GGUF and the pack only")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SKIP").unlink(missing_ok=True)
    c = pf.cfg("full4", a.seed)
    gguf = out / "full4.gguf"
    n, nq = pf.write(str(gguf), c, q8_embd=True)
    print(f"wrote {gguf}: {n} tensors ({nq} Q8_0, token_embd included), {c.n_layer} layers, {c.n_expert} experts "
          f"top-{c.n_expert_used}")
    # the expert cache's ranking: every (layer, expert) pair, layer-major, so a cache of 64 slots holds the whole model
    profile = out / "profile.bin"
    ranked = [(l, e) for l in range(c.n_layer) for e in range(c.n_expert)]
    make_profile.write_profile(str(profile), ranked, n_expert=c.n_expert, n_layer=c.n_layer)
    # the pack: the tokenizer export is for real files (the synthetic vocabulary is the 128 bytes and has no chat
    # template), so its two marker files are written here and iq_pack.py skips the export
    pack = out / "pack"
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    (pack / "tokenizer" / "vocab.json").write_text(json.dumps({chr(i): i for i in range(c.n_vocab)}))
    (pack / "tokenizer" / "chat_template.jinja").write_text("{{ messages[0].content }}")
    env = dict(os.environ)
    env.setdefault("STRATA_GGUF_PY", pf._paths.gguf_py())
    subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(gguf), "--out", str(pack),
                    "--expert-profile", str(profile)], check=True, env=env)
    for name in ("index.txt", "dense.bin", "native_experts.txt", "expert-profile.bin"):
        if not (pack / name).exists():
            raise SystemExit(f"the pack lacks {name}")
    print(f"pack in {pack}")
    if a.no_oracle:
        return 0
    for name, toks in CASES.items():
        oracle(str(gguf), str(out / name), toks, ["--all-logits"])
        oracle(str(gguf), str(out / (name + "_q8")), toks, ["--all-logits", "--act-quant", "q8"])
        print(f"  oracle dumps {out / name} and {out / (name + '_q8')} (activations rounded to Q8 as ggml does): "
              f"{len(toks)} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
