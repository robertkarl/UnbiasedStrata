#!/usr/bin/env python3
"""The Qwen3.6 drafter harness's fixture: a synthetic qwen35moe GGUF with a nextn block at the model's per-layer
geometry, the oracle's `--mtp` dumps for it, and the draft pack cut from it (docs/qwen36-port/step4-drafter.md).

    python3 -I tests/python/qwen36_mtp_fixture.py --out <dir>

The drafter's kernels are built for the real widths (n_embd 2048, attention 16 heads / 2 KV heads of 256, experts of
512), so this is tests/python/qwen36_prefill_fixture.py's `full4` trunk (three DeltaNet layers, layer 3 full attention,
16 experts top-8) plus the nextn block `blk.4` (tools/qwen36_synth_gguf.py's: an attention + MoE block, `nextn.eh_proj`
[2 n_embd, n_embd], `nextn.enorm` / `hnorm` / `shared_head_norm`).  Every projection, `nextn.eh_proj`, the experts, the
head and the token embedding are Q8_0 (what the engine's native paths take: MMVQ, the native expert kernels, the native
head and embedding); the routers, `ssm_alpha` / `ssm_beta` and the shared-expert gates are F32 values rounded to BF16
(the pack writes them as BF16 exactly); the norms, conv, dt bias and ssm_a F32.  The oracle (ref/qwen36.py) dequantizes
with gguf-py, bit-identical to ggml, so every difference the harness sees is the engine's arithmetic.

Writes <out>/mtp5.gguf; the oracle's dumps for 76 tokens in <out>/o and <out>/o_mtp (the faithful
reading: the trunk, then the nextn block on (tokens[1:], h_nextn[:-1]) at positions 1..75), <out>/o_q8 and
<out>/o_q8_mtp (`--act-quant q8`: the activations rounded as ggml does before a quantized matmul); and the draft pack
in <out>/pack (tools/mtp_qwen36_pack.py, no draft vocabulary: the whole 128-token head).  Without numpy or gguf-py it
writes <out>/SKIP with the reason and exits 0, and the test exits 77 on seeing it.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))

try:
    import numpy as np
    import _paths
    import qwen36_synth_gguf as qs          # adds gguf-py to sys.path on import (tools/_paths.py)
    import qwen36_prefill_fixture as pf     # the per-layer geometry, the BF16 rounding, the Q8_0 suffix list
    from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
except ImportError as e:                    # a trimmed third_party or no numpy: a skip, not a failure
    print(f"qwen36_mtp_fixture: {e}", file=sys.stderr)
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

# 76 tokens (the ASCII codes of this text: the fixture's vocabulary is the 128 byte tokens): 75 nextn positions, so the
# prompt-side pass (Prefill::draft_kv, batches of at least 64 rows) is exercised and the positions after it are checked
TEXT = "the quick brown fox jumps over the lazy dog while the cat sleeps on the mat "
TOKENS = [ord(c) for c in TEXT]
assert len(TOKENS) == 76, len(TOKENS)
Q8_SUFFIXES = pf.Q8_SUFFIXES + (".nextn.eh_proj.weight",)


def cfg(seed: int) -> qs.Cfg:
    c = pf.cfg("full4", seed)
    c.nextn = True
    return c


def write(path: str, c: qs.Cfg) -> tuple[int, int]:
    """qwen36_prefill_fixture.write with the nextn block declared and the embedding Q8_0."""
    w = GGUFWriter(path, qs.ARCH)
    k = lambda s: f"{qs.ARCH}.{s}"
    w.add_name("qwen36-synth drafter-geometry")
    w.add_uint32(k("context_length"), c.n_ctx_train)
    w.add_uint32(k("embedding_length"), c.n_embd)
    w.add_uint32(k("block_count"), c.n_layer + 1)
    w.add_uint32(k("nextn_predict_layers"), 1)
    w.add_uint32(k("attention.head_count"), c.n_head)
    w.add_uint32(k("attention.head_count_kv"), c.n_head_kv)
    w.add_uint32(k("attention.key_length"), c.head_dim)
    w.add_uint32(k("attention.value_length"), c.head_dim)
    w.add_float32(k("attention.layer_norm_rms_epsilon"), c.rms_eps)
    w.add_uint32(k("rope.dimension_count"), c.n_rot)
    w.add_array(k("rope.dimension_sections"), list(c.rope_sections))
    w.add_float32(k("rope.freq_base"), c.rope_base)
    w.add_uint32(k("expert_count"), c.n_expert)
    w.add_uint32(k("expert_used_count"), c.n_expert_used)
    w.add_uint32(k("expert_feed_forward_length"), c.n_ff_exp)
    w.add_uint32(k("expert_shared_feed_forward_length"), c.n_ff_shexp)
    w.add_uint32(k("ssm.conv_kernel"), c.ssm_conv)
    w.add_uint32(k("ssm.state_size"), c.ssm_state)
    w.add_uint32(k("ssm.group_count"), c.ssm_k_heads)
    w.add_uint32(k("ssm.time_step_rank"), c.ssm_v_heads)
    w.add_uint32(k("ssm.inner_size"), c.value_dim)
    w.add_uint32(k("full_attention_interval"), c.interval)
    w.add_file_type(7)                                     # LLAMA_FTYPE_MOSTLY_Q8_0
    toks = qs.byte_unicode()
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types([1] * c.n_vocab)
    w.add_token_merges(["Ā ā"])
    w.add_bos_token_id(0)
    w.add_eos_token_id(3)
    w.add_add_bos_token(False)
    n, nq = 0, 0
    for name, arr in qs.build_tensors(c):
        n += 1
        if name.endswith(Q8_SUFFIXES) or name in ("output.weight", "token_embd.weight"):
            w.add_tensor(name, quants.quantize(arr, Q.Q8_0), raw_dtype=Q.Q8_0)
            nq += 1
        elif name.endswith(pf.BF16_SUFFIXES):
            if name.endswith(".ffn_gate_inp.weight"):   # half the toy's router gain, as the prompt-path fixture
                arr = (arr * 0.5).astype(np.float32)
            w.add_tensor(name, pf.to_bf16_exact(arr))
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n, nq


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("STRATA_GGUF_PY", _paths.gguf_py())
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
    c = cfg(a.seed)
    gguf = str(out / "mtp5.gguf")
    n, nq = write(gguf, c)
    print(f"wrote {gguf}: {n} tensors ({nq} Q8_0), {c.n_layer} trunk layers + the nextn block, {c.n_expert} experts "
          f"top-{c.n_expert_used}")
    run([sys.executable, "-I", str(ROOT / "tools" / "mtp_qwen36_pack.py"), "--gguf", gguf, "--out", str(out / "pack"),
         "--draft-vocab", "none"])
    print(f"  pack in {out / 'pack'}")
    if a.no_oracle:
        return 0
    toks = ",".join(map(str, TOKENS))
    oracle = [sys.executable, "-I", str(ROOT / "ref" / "qwen36.py"), gguf, "--tokens", toks, "--mtp"]
    run(oracle + ["--dump", str(out / "o")])
    run(oracle + ["--dump", str(out / "o_q8"), "--act-quant", "q8"])
    print(f"  oracle dumps in {out}: o, o_mtp, o_q8, o_q8_mtp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
