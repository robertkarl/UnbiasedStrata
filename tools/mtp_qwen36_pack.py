"""tools/mtp_qwen36_pack.py - Qwen3.6's draft block (`blk.<trunk>`, the nextn block of a -MTP GGUF) as the engine's
drafter pack (docs/QWEN36-35B-A3B.md 3.10; docs/qwen36-port/step4-drafter.md).

    python3 -I tools/mtp_qwen36_pack.py --gguf Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --out <dir> [--draft-vocab data/draft_vocab.bin]

The pack is separate from the trunk's (tools/iq_pack.py leaves `blk.40` out of it), so the pack cut from the Q4_K_XL-MTP
file's draft block serves the plain IQ3_XXS trunk too: the geometry it was cut for is recorded and the engine checks it
against the trunk it drafts for.  Flash-Next's drafter keeps tools/mtp_pack.py / mtp_rt.py; this tool writes the same
file names with llama.cpp's tensor names, so `MtpDrafter::load` picks the path by the trunk's architecture.

Writes
  dense.txt     one line per dense tensor: `name kind rows cols offset bytes`, the name without its `blk.N.` prefix
                (`nextn.eh_proj.weight`, `attn_norm.weight`, `attn_q.weight`, `ffn_gate_inp.weight`, ...).  kind is
                `f32` (every norm, read as `const float*`), `bf16` (the router `ffn_gate_inp` and the 1-D shared-expert gate
                `ffn_gate_inp_shexp`: BF16 in the shipped -MTP files, F32 values that are BF16-exact in a synthetic one), or
                the ggml type of a projection as stored, lowercase (`q8_0` in both shipped files), served through the
                engine's native MMVQ (the types tools/iq_pack.py's NATIVE_PROJECTION_TYPES lists).  rows = ne1 (the
                output width), cols = ne0 (the contiguous input width); 1-D tensors are `1 n`.
  dense.bin     the tensors' bytes as stored in the GGUF, each at a 256-aligned offset
  experts.bin   n_expert blobs of [gate rows | up rows | down rows], the GGUF's blocks as stored: the native GPU expert
                kernels' blob (strata/kernels/iq_kernels.hpp `native_expert_layout`), all of them resident in VRAM
  native_experts.txt   the trunk pack's v3 line form for the one layer (`layer gu_type d_type offset blob_bytes gate_off
                up_off down_off`; the three role offsets are inside a blob, the header says so)
  mtp.json      the geometry the block was cut for (n_embd, heads, head_dim, n_rot, experts, k, n_ff, the shared width),
                the types per tensor, the source file, and which draft vocabulary was copied
  draft_vocab.bin   the draft head's token subset (tools/draft_vocab.py; the same 248,320-token vocabulary as Flash-Next,
                so the shipped data/draft_vocab*.bin carry over), copied from --draft-vocab unless `none`

Refused by name: a file without exactly one nextn block, fused expert tensors, a projection or head type outside
NATIVE_PROJECTION_TYPES, an expert type pair the native kernels lack (the IQ4_XS-MTP file's Q3_K gate/up with Q4_K down:
only the Q4_K_XL-MTP file's Q4_K / Q5_K block is served today), an F32 router that is not BF16-exact.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from iq_pack import NATIVE_PROJECTION_TYPES  # noqa: E402

ARCH = "qwen35moe"
# the native GPU expert kernels' type pairs (src/kernels/cuda/iq_kernels.cu, the three STRATA_*_FMTS lists; the gate/up
# and down lists of docs/qwen36-port/step3-experts.md section 2).  The engine asks `native_expert_supported` again at
# load; this is the early refusal with the tensor named.
GU_TYPES = {16, 17, 18, 21, 22, 23, 29, 42, 12, 13, 14, 6, 8}
D_TYPES = {20, 23, 42, 7, 6, 8, 18, 13, 14}
D_TYPES_256 = {18, 13, 14, 23}          # 256-value blocks: need n_ff % 256 == 0 as a down projection
# the dense tensors of the draft block, by role (names without the blk.N. prefix)
NORMS = ("attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
         "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight")
ROUTERS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")
PROJECTIONS = ("nextn.eh_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
               "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")
EXPERTS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
ALIGN = 256


def meta_int(r: gguf.GGUFReader, key: str, default=None):
    f = r.fields.get(key)
    if f is None:
        if default is None:
            sys.exit("%s: missing" % key)
        return default
    return int(f.parts[f.data[0]][0])


def raw_bytes(t) -> np.ndarray:
    """A tensor's bytes in ggml memory order (ne0 contiguous), whatever gguf-py typed its view as."""
    a = np.ascontiguousarray(np.asarray(t.data))
    b = a.view(np.uint8).reshape(-1)
    if b.size != int(t.n_bytes):
        sys.exit("%s: %d bytes read, %d expected" % (t.name, b.size, t.n_bytes))
    return b


def bf16_exact_problem(name: str, f32: np.ndarray):
    u = f32.view(np.uint32)
    bad = int(np.count_nonzero(u & np.uint32(0xFFFF)))
    if bad:
        return ("%s is F32 with %d of %d values that are not BF16 (low 16 bits set), but the engine reads it as BF16"
                % (name, bad, u.size))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="a qwen35moe -MTP GGUF (nextn_predict_layers 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft-vocab", default=None,
                    help="the draft head's token subset to copy in as draft_vocab.bin (default: data/draft_vocab.bin "
                         "beside this tool when it exists; `none` for the whole vocabulary)")
    a = ap.parse_args()
    src = pathlib.Path(a.gguf)
    out = pathlib.Path(a.out)
    r = gguf.GGUFReader(str(src))
    arch_f = r.fields.get("general.architecture")
    arch = bytes(arch_f.parts[arch_f.data[0]]).decode("utf-8") if arch_f else ""
    if arch != ARCH:
        sys.exit("%s: general.architecture is %r, this tool packs %s" % (src.name, arch, ARCH))
    k = lambda s: "%s.%s" % (ARCH, s)
    n_blocks = meta_int(r, k("block_count"))
    nextn = meta_int(r, k("nextn_predict_layers"), 0)
    if nextn != 1:
        sys.exit("%s: nextn_predict_layers is %d; the drafter is one nextn block (the plain file has none: use the "
                 "-MTP build of the same model)" % (src.name, nextn))
    trunk = n_blocks - nextn
    p = "blk.%d." % trunk
    geo = dict(n_embd=meta_int(r, k("embedding_length")), n_head=meta_int(r, k("attention.head_count")),
               n_head_kv=meta_int(r, k("attention.head_count_kv")), head_dim=meta_int(r, k("attention.key_length")),
               n_rot=meta_int(r, k("rope.dimension_count")), n_expert=meta_int(r, k("expert_count")),
               n_expert_used=meta_int(r, k("expert_used_count")), n_ff=meta_int(r, k("expert_feed_forward_length")),
               n_ff_shexp=meta_int(r, k("expert_shared_feed_forward_length")), trunk_layers=trunk, draft_block=trunk)
    T = {t.name[len(p):]: t for t in r.tensors if t.name.startswith(p)}
    if not T:
        sys.exit("%s: no blk.%d.* tensors" % (src.name, trunk))
    if "ffn_gate_up_exps.weight" in T or "attn_qkv.weight" in T:
        sys.exit("%s: blk.%d carries fused tensors (%s); the drafter reads separate ffn_gate_exps / ffn_up_exps and "
                 "attn_q / attn_k / attn_v" % (src.name, trunk,
                                                ", ".join(n for n in ("ffn_gate_up_exps.weight", "attn_qkv.weight") if n in T)))
    missing = [n for n in NORMS + ROUTERS + PROJECTIONS + EXPERTS if n not in T]
    if missing:
        sys.exit("%s: blk.%d is missing %s" % (src.name, trunk, ", ".join(missing)))
    stray = sorted(set(T) - set(NORMS + ROUTERS + PROJECTIONS + EXPERTS))
    if stray:
        print("note: %d tensor(s) of blk.%d the drafter does not read, left out: %s" % (len(stray), trunk, ", ".join(stray)))

    # ---- the expert blob: [gate | up | down] per expert, types the native kernels take
    ge, ue, de = (T[n] for n in EXPERTS)
    gt, ut, dt = int(ge.tensor_type), int(ue.tensor_type), int(de.tensor_type)
    E, FF, N = geo["n_expert"], geo["n_ff"], geo["n_embd"]
    for t, shape in ((ge, (N, FF, E)), (ue, (N, FF, E)), (de, (FF, N, E))):
        if tuple(int(x) for x in t.shape) != shape:
            sys.exit("%s%s: shape %s, expected ne %s" % (p, t.name[len(p):], [int(x) for x in t.shape], list(shape)))
    if gt != ut:
        sys.exit("%s: gate (%s) and up (%s) differ in type" % (p, ge.tensor_type.name, ue.tensor_type.name))
    bad = []
    if gt not in GU_TYPES:
        bad.append("%sffn_gate_exps / ffn_up_exps are %s" % (p, ge.tensor_type.name))
    if dt not in D_TYPES or (dt in D_TYPES_256 and FF % 256):
        bad.append("%sffn_down_exps is %s" % (p, de.tensor_type.name))
    if bad:
        sys.exit("%s: the native GPU expert kernels (iq_kernels.cu) have no kernel for this pair: %s" % (src.name, "; ".join(bad)))
    gu_mat, d_mat = int(ge.n_bytes) // E, int(de.n_bytes) // E
    if gu_mat * E != int(ge.n_bytes) or d_mat * E != int(de.n_bytes) or int(ue.n_bytes) != int(ge.n_bytes):
        sys.exit("%s: expert tensor bytes are not whole experts" % p)
    blob = 2 * gu_mat + d_mat

    # ---- the dense tensors
    out.mkdir(parents=True, exist_ok=True)
    lines, types, off = [], {}, 0
    with open(out / "dense.bin", "wb") as f:
        for name in NORMS + ROUTERS + PROJECTIONS:
            t = T[name]
            tn = t.tensor_type.name
            ne = [int(x) for x in t.shape]
            if name in NORMS:
                if tn != "F32" or len(ne) != 1:
                    sys.exit("%s%s is %s %s; the drafter reads its norms as F32 vectors" % (p, name, tn, ne))
                raw, kind, rows, cols = raw_bytes(t), "f32", 1, ne[0]
            elif name in ROUTERS:
                if tn == "BF16":
                    raw = raw_bytes(t)
                elif tn == "F32":
                    f32 = np.ascontiguousarray(np.asarray(t.data), dtype=np.float32).reshape(-1)
                    why = bf16_exact_problem(p + name, f32)
                    if why:
                        sys.exit(why)
                    raw = (f32.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)   # the high halves, exact
                else:
                    sys.exit("%s%s is %s; the drafter reads it as BF16" % (p, name, tn))
                kind, rows, cols = "bf16", (ne[1] if len(ne) > 1 else 1), ne[0]
            else:
                if tn not in NATIVE_PROJECTION_TYPES:
                    sys.exit("%s%s is %s, which the engine's native MMVQ has no kernel for; it serves %s"
                             % (p, name, tn, ", ".join(sorted(NATIVE_PROJECTION_TYPES))))
                if len(ne) != 2:
                    sys.exit("%s%s: a projection is 2-D, this one is %s" % (p, name, ne))
                raw, kind, rows, cols = raw_bytes(t), tn.lower(), ne[1], ne[0]
            pad = (-off) % ALIGN
            f.write(b"\0" * pad)
            off += pad
            f.write(raw.tobytes())
            lines.append("%s %s %d %d %d %d" % (name, kind, rows, cols, off, raw.size))
            types[name] = tn
            off += int(raw.size)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the eh_proj's shape is the one place the concatenation's width is checked before the engine runs it
    eh = T["nextn.eh_proj.weight"]
    if [int(x) for x in eh.shape] != [2 * N, N]:
        sys.exit("%snextn.eh_proj.weight is ne %s, expected [%d, %d] (concat(enorm, hnorm) -> n_embd)"
                 % (p, [int(x) for x in eh.shape], 2 * N, N))

    # ---- the experts, one blob each, in id order
    g_raw, u_raw, d_raw = (raw_bytes(t).reshape(E, -1) for t in (ge, ue, de))
    with open(out / "experts.bin", "wb") as f:
        for e in range(E):
            f.write(g_raw[e].tobytes())
            f.write(u_raw[e].tobytes())
            f.write(d_raw[e].tobytes())
    (out / "native_experts.txt").write_text(
        "# strata native experts v3 (the drafter's one block): layer gu_type d_type offset blob_bytes gate_off up_off "
        "down_off (n_expert %d, total %d; `offset` in experts.bin, the three role offsets inside a blob; from %s blk.%d)\n"
        "0 %d %d 0 %d 0 %d %d\n" % (E, blob * E, src.name, trunk, gt, dt, blob, gu_mat, 2 * gu_mat), encoding="utf-8")

    # ---- the draft vocabulary
    dv = a.draft_vocab
    if dv is None:
        cand = pathlib.Path(__file__).resolve().parents[1] / "data" / "draft_vocab.bin"
        dv = str(cand) if cand.is_file() else "none"
    vocab = None
    if dv != "none":
        ids = np.fromfile(dv, dtype="<i4")
        n_vocab = int(r.tensors[[t.name for t in r.tensors].index("output.weight")].shape[1])
        if ids.size == 0 or ids.min() < 0 or ids.max() >= n_vocab:
            sys.exit("%s: %d ids, not all inside the vocabulary (%d)" % (dv, ids.size, n_vocab))
        shutil.copyfile(dv, out / "draft_vocab.bin")
        vocab = {"file": pathlib.Path(dv).name, "tokens": int(ids.size)}
    else:
        (out / "draft_vocab.bin").unlink(missing_ok=True)

    info = {"format": "strata mtp pack (qwen35moe) v1", "arch": ARCH, "source": src.name,
            "source_bytes": src.stat().st_size, **geo,
            "experts": {"gu_type": gt, "gu_type_name": ge.tensor_type.name, "d_type": dt,
                        "d_type_name": de.tensor_type.name, "blob_bytes": blob, "gu_matrix_bytes": gu_mat,
                        "down_matrix_bytes": d_mat, "bytes": blob * E},
            "dense": types, "dense_bytes": off, "draft_vocab": vocab}
    (out / "mtp.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print("wrote %s: dense.bin %d B (%d tensors), experts.bin %d B (%d x %d: gate/up %s, down %s), draft_vocab %s"
          % (out, off, len(lines), blob * E, E, blob, ge.tensor_type.name, de.tensor_type.name,
             vocab["tokens"] if vocab else "none (the whole head)"))
    for line in lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
