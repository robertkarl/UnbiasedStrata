"""Focused pack compatibility tests; run .venv/bin/python -m unittest discover -s tools -p test_iq_pack.py."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
import iq_pack


def write_gguf(path, tensors, split=None, arch=True, kv=None, architecture="qwen4exp"):
    """`split` = (no, count, tensors): llama.cpp's gguf-split keys (0-based split.no); `kv` = extra u32 keys;
    `architecture` = the general.architecture string (qwen4exp, or qwen35moe for the Qwen3.6 fixtures)."""
    writer = GGUFWriter(path, architecture)
    if not arch:                                   # a later shard of a split: no model metadata
        writer.kv_data[0].pop("general.architecture", None)
    for key, value in (kv or {}).items():
        writer.add_uint32(key, value)
    if split is not None:
        writer.add_uint16("split.no", split[0])
        writer.add_uint16("split.count", split[1])
        writer.add_int32("split.tensors.count", split[2])
    for name, values, kind in tensors:
        writer.add_tensor(name, quants.quantize(values, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class CompatibilityTests(unittest.TestCase):
    def test_bf16_halfway_rounds_to_even(self):
        values = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32)
        got = np.frombuffer(iq_pack.bf16_bytes(values.view(np.uint8), "F32"), dtype=np.uint16)
        np.testing.assert_array_equal(got, [0x3F80, 0x3F82, 0xBF80, 0xBF82])

    def test_nonfinite_conversion_refused(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            iq_pack.bf16_bytes(np.array([np.nan], dtype=np.float32).view(np.uint8), "F32")

    def test_projection_conversion_and_native_bytes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
            router = np.full((2, 128), 0.10001, dtype=np.float32)
            names = ["blk.0.hc_attn_down.weight", "output_hc_up.weight",
                     "blk.0.ssm_alpha.weight", "blk.3.indexer.q_proj.weight", "blk.1.ple_value.weight"]
            write_gguf(source, [(n, values, Q.Q8_0) for n in names] + [
                ("blk.1.ple_key.weight", values, Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", router, Q.F32),
                ("blk.0.attn_qkv.weight", values, Q.Q8_0),
                ("per_layer_token_embd.weight", values, Q.Q8_0),
            ])
            before = source.read_bytes()
            model = iq_pack.Model(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, model, True), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            dense = (root / "dense.bin").read_bytes()
            # A scalar oracle independent of the BF16 encoder: choose the nearest BF16 value,
            # using the even representation at exact ties.
            decoded = quants.dequantize(quants.quantize(values, Q.Q8_0), Q.Q8_0).ravel()
            def rounded(x):
                bits = int(np.float32(x).view(np.uint32))
                return (bits >> 16) + ((bits & 65535) > 32768 or
                                      ((bits & 65535) == 32768 and (bits >> 16) & 1))
            expected = np.array([rounded(x) for x in decoded], dtype=np.uint16)
            for name in names:
                row = rows[name]
                self.assertEqual(row[2], "4")
                offset, size = int(row[3]), int(row[4])
                np.testing.assert_array_equal(np.frombuffer(dense[offset:offset + size], dtype=np.uint16), expected)
            for name in ["blk.0.attn_qkv.weight", "blk.1.ple_key.weight"]:   # a Q8_0 PLE key stays native
                native = rows[name]
                self.assertEqual(native[2], "0")
                self.assertEqual(native[4], "0")
                self.assertEqual(native[9], "8")
            self.assertNotIn("per_layer_token_embd.weight", rows)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(json.loads((root / "compat-bf16.json").read_text())["tensors"]), 6)
            conv = json.loads((root / "conversions.json").read_text())
            self.assertEqual(sorted(r["name"] for r in conv["tensors"]), sorted(names + ["blk.0.ffn_gate_inp.weight"]))

    def test_default_still_refuses_inexact_f32_router(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.ffn_gate_inp.weight", np.full((2, 32), 0.10001, np.float32), Q.F32)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)

    def test_existing_bf16_pack_remains_byte_identical(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-2, 2, 256, dtype=np.float32).reshape(2, 128)
            write_gguf(source, [("blk.0.hc_attn_down.weight", values, Q.BF16)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 0)
            self.assertEqual((root / "dense.bin").read_bytes(), quants.quantize(values, Q.BF16).tobytes())

    def test_quantized_control_weights_need_explicit_conversion(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0)])
            (root / "dense.bin").write_bytes(b"previous pack")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)
            self.assertEqual((root / "dense.bin").read_bytes(), b"previous pack")

    def test_q2_ple_and_large_tensors_stay_native(self):
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q2_0"))
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q8_0"))
        self.assertTrue(iq_pack.needs_bf16("blk.1.ple_key.weight", "IQ3_XXS"))
        for name in ["blk.0.ffn_gate_exps.weight", "token_embd.weight", "output.weight",
                     "per_layer_token_embd.weight", "blk.0.attn_qkv.weight"]:
            self.assertFalse(iq_pack.needs_bf16(name, "IQ3_XXS"))

    def test_experts_bin_is_kept_only_with_this_ggufs_blobs(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ["vocab.json", "chat_template.jinja"]:
                (out / "tokenizer" / name).touch()

            def model(name, scale):
                path = root / name
                write_gguf(path, [("blk.%d.hc_attn_down.weight" % l, np.ones((2, 32), np.float32), Q.BF16)
                                  for l in range(3)] +
                                 [("blk.%d.ffn_gate_inp.weight" % l, np.ones((4, 32), np.float32), Q.BF16)
                                  for l in range(3)] +
                                 [("blk.%d.ffn_%s_exps.weight" % (l, r),
                                   scale * np.arange(4 * 2 * 32, dtype=np.float32).reshape(4, 2, 32) / 256, Q.Q8_0)
                                  for l in range(3) for r in ("gate", "up", "down")])
                return path

            def pack(path):
                log = io.StringIO()
                argv = ["iq_pack.py", "--gguf", str(path), "--out", str(out), "--experts-bin"]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(log):
                    self.assertEqual(iq_pack.main(), 0)
                return log.getvalue()

            first = model("a.gguf", 1)
            pack(first)
            written = (out / "experts.bin").read_bytes()
            self.assertIn("not rewritten", pack(first))
            self.assertEqual((out / "experts.bin").read_bytes(), written)
            # the same geometry with other weights under the same file name: the size and the sidecar (shard names
            # and sizes, native_experts.txt) match, the contents do not
            second = model("a.gguf", -1)
            self.assertIn("not this GGUF's expert blobs", pack(second))
            rewritten = (out / "experts.bin").read_bytes()
            self.assertEqual(len(rewritten), len(written))
            self.assertNotEqual(rewritten, written)
            self.assertIn("not rewritten", pack(second))

    def test_layout_only_writes_the_table_and_the_census(self):
        """--layout-only (the Qwen3.6 port's step 2): native_experts.txt alone, the census printed, nothing else."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            out = root / "pack"
            path = root / "m.gguf"
            write_gguf(path, [("blk.%d.ffn_gate_inp.weight" % l, np.ones((4, 32), np.float32), Q.F32)
                              for l in range(2)] +
                             [("blk.%d.ffn_%s_exps.weight" % (l, r),
                               np.arange(4 * 2 * 32, dtype=np.float32).reshape(4, 2, 32) / 256, Q.Q8_0)
                              for l in range(2) for r in ("gate", "up", "down")])
            log = io.StringIO()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(path), "--out", str(out), "--layout-only"]), \
                    contextlib.redirect_stdout(log):
                self.assertEqual(iq_pack.main(), 0)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["native_experts.txt"])
            text = (out / "native_experts.txt").read_text()
            rows = [line.split() for line in text.splitlines() if line and not line.startswith("#")]
            self.assertEqual([r[0] for r in rows], ["0", "1"])
            blob = 3 * 2 * 32 * 34 // 32          # gate | up | down, Q8_0 rows of 32 values = 34 bytes
            self.assertEqual([int(r[4]) for r in rows], [blob, blob])
            self.assertIn("(n_expert 4, total %d;" % (2 * 4 * blob), text.splitlines()[0])
            self.assertIn("2 layers x 4 experts", log.getvalue())
            self.assertIn("gate/up Q8_0     down Q8_0     blob %9d B: layers [0, 1]" % blob, log.getvalue())
            self.assertIn("wrote %s" % (out / "native_experts.txt"), log.getvalue())

    def test_nextn_block_is_left_out_of_the_layout(self):
        """nextn_predict_layers 1 (the -MTP files): the last block's experts belong to the drafter's pack."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            out = root / "pack"
            path = root / "mtp.gguf"
            write_gguf(path, [("blk.%d.ffn_gate_inp.weight" % l, np.ones((4, 32), np.float32), Q.F32)
                              for l in range(3)] +
                             [("blk.%d.ffn_%s_exps.weight" % (l, r),
                               np.arange(4 * 2 * 32, dtype=np.float32).reshape(4, 2, 32) / 256, Q.Q8_0)
                              for l in range(3) for r in ("gate", "up", "down")],
                       kv={"qwen4exp.nextn_predict_layers": 1})
            log = io.StringIO()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(path), "--out", str(out), "--layout-only"]), \
                    contextlib.redirect_stdout(log):
                self.assertEqual(iq_pack.main(), 0)
            rows = [line.split() for line in (out / "native_experts.txt").read_text().splitlines()
                    if line and not line.startswith("#")]
            self.assertEqual([r[0] for r in rows], ["0", "1"])
            self.assertIn("1 draft (nextn) block(s) at blk.2 and up are left to the MTP pack; 2 trunk layers", log.getvalue())
            self.assertIn("2 layers x 4 experts", log.getvalue())

    def test_fused_gate_up_experts_are_refused_by_name(self):
        """llama.cpp's fused ffn_gate_up_exps form: refused before anything is written."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            out = root / "pack"
            path = root / "fused.gguf"
            write_gguf(path, [("blk.0.ffn_gate_inp.weight", np.ones((4, 32), np.float32), Q.F32),
                              ("blk.0.ffn_gate_up_exps.weight",
                               np.arange(4 * 4 * 32, dtype=np.float32).reshape(4, 4, 32) / 256, Q.Q8_0),
                              ("blk.0.ffn_down_exps.weight",
                               np.arange(4 * 2 * 32, dtype=np.float32).reshape(4, 2, 32) / 256, Q.Q8_0)])
            log = io.StringIO()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(path), "--out", str(out), "--layout-only"]), \
                    contextlib.redirect_stdout(log):
                self.assertEqual(iq_pack.main(), 1)
            self.assertIn("blk.0.ffn_gate_up_exps.weight: the experts are fused", log.getvalue())
            self.assertFalse((out / "native_experts.txt").exists())

    def test_snapshot_symlinks_keep_split_discovery(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            snap = root / "snapshot"
            snap.mkdir()
            first = snap / "model-00001-of-00002.gguf"
            second = snap / "model-00002-of-00002.gguf"
            write_gguf(root / "blob1", [
                ("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", np.ones((512, 32), np.float32), Q.BF16),
            ], split=(0, 2, 5))
            write_gguf(root / "blob2", [(f"blk.0.ffn_{r}_exps.weight", np.ones((512, 2, 32), np.float32), Q.Q8_0)
                                       for r in ("gate", "up", "down")], split=(1, 2, 5), arch=False)
            try:
                first.symlink_to(root / "blob1")
            except OSError:                              # Windows without Developer Mode or admin rights
                self.skipTest("symlinks are not available here")
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards"):
                iq_pack.Model(first)
            second.symlink_to(root / "blob2")
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ["vocab.json", "chat_template.jinja"]:
                (out / "tokenizer" / name).touch()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), "--compat-bf16"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(iq_pack.main(), 0)
            expert_index = (out / "native_experts.txt").read_text()
            self.assertIn(second.name, expert_index)
            self.assertIn("n_expert 512,", expert_index)
            (root / "blob2").write_bytes((root / "blob2").read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(first)



def bits(u32):
    return np.array(u32, dtype=np.uint32).view(np.float32)


def pack_one(root, tensors, *flags):
    """A one-file model of `tensors` packed into `root` (index only): (rc, printed, rows, dense, conversions)."""
    source = root / "model.gguf"
    write_gguf(source, tensors)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = iq_pack.index_standalone(source, root, iq_pack.Model(source), *flags)
    if rc:
        return rc, buf.getvalue(), None, None, None
    _, rows = iq_pack.read_index(root / "index.txt")
    conv = json.loads((root / "conversions.json").read_text())
    return rc, buf.getvalue(), rows, (root / "dense.bin").read_bytes(), conv


def stored(rows, dense, name, dtype):
    row = rows[name]
    return np.frombuffer(dense[int(row[3]):int(row[3]) + int(row[4])], dtype=dtype)


class ConversionTests(unittest.TestCase):
    """CS2: FORM decides the engine's form; exact F32->BF16 by default, rounding only with --compat-bf16, the PLE
    conv F32 -> F16 by the loader (index kind 3), non-finite values refused, every conversion in conversions.json."""

    def test_exact_f32_to_bf16_by_default(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            x = bits(np.arange(64, dtype=np.uint32).reshape(2, 32) << 16 | 0x3F000000)   # BF16 values held as F32
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.0.hc_attn_inject.weight", x, Q.F32),
                                                              ("blk.0.ssm_beta.weight", x, Q.F32),
                                                              ("blk.0.ssm_norm.weight", x, Q.F32)])
            self.assertEqual(rc, 0, log)
            for name in ["blk.0.hc_attn_inject.weight", "blk.0.ssm_beta.weight"]:
                self.assertEqual(rows[name][2], "4")
                np.testing.assert_array_equal(stored(rows, dense, name, "<u2"), (x.view(np.uint32) >> 16).ravel())
            self.assertEqual(rows["blk.0.ssm_norm.weight"][2], "2")               # F32 form: as stored
            recs = {r["name"]: r for r in conv["tensors"]}
            self.assertEqual(set(recs), {"blk.0.hc_attn_inject.weight", "blk.0.ssm_beta.weight"})
            r = recs["blk.0.ssm_beta.weight"]
            self.assertEqual((r["src_type"], r["dst_type"], r["exact"], r["max_abs_err"], r["shard"]),
                             ("F32", "BF16", True, 0.0, "model.gguf"))
            self.assertEqual(r["source_sha256"], hashlib.sha256(x.tobytes()).hexdigest())
            self.assertEqual(conv["schema"], 1)
            self.assertEqual(conv["source_shards"], [{"name": "model.gguf",
                                                      "size": (Path(tmp) / "model.gguf").stat().st_size}])

    def test_inexact_f32_rounds_only_with_compat(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32) + np.float32(1e-4)
            rc, log, *_ = pack_one(Path(tmp), [("blk.0.hc_ffn_inject.weight", x, Q.F32)])
            self.assertEqual(rc, 1)
            self.assertIn("not BF16", log)
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.0.hc_ffn_inject.weight", x, Q.F32)], True)
            self.assertEqual(rc, 0, log)
            got = stored(rows, dense, "blk.0.hc_ffn_inject.weight", "<u2")
            np.testing.assert_array_equal(got, np.frombuffer(iq_pack.bf16_bytes(x.view(np.uint8).ravel(), "F32"),
                                                             dtype="<u2"))
            r = conv["tensors"][0]
            back = (got.astype(np.uint32) << 16).view(np.float32)
            self.assertFalse(r["exact"])
            self.assertAlmostEqual(r["max_abs_err"], float(np.max(np.abs(back - x.ravel()))))
            self.assertLessEqual(r["max_abs_err"], 2.0 ** -8)
            self.assertIn("--compat-bf16", r["method"])

    def test_non_finite_is_refused(self):
        for bad in (0x7FC00000, 0x7F800000, 0xFF800000):          # NaN and +-Inf: their low 16 bits are zero too
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                x = bits([[0x3F800000] * 31 + [bad]] * 2)
                (Path(tmp) / "dense.bin").write_bytes(b"previous pack")
                for flags in ((), (True,)):
                    rc, log, *_ = pack_one(Path(tmp), [("blk.0.ssm_alpha.weight", x, Q.F32)], *flags)
                    self.assertEqual(rc, 1, (hex(bad), flags))
                    self.assertIn("non-finite", log)
                self.assertEqual((Path(tmp) / "dense.bin").read_bytes(), b"previous pack")
                self.assertFalse((Path(tmp) / "dense.bin.tmp").exists())
                rc, log, *_ = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", x, Q.F32)])
                self.assertEqual(rc, 1)
                self.assertIn("non-finite", log)

    def test_ple_conv1d_f32_is_narrowed_by_the_loader(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            conv1d = np.random.default_rng(3).normal(0, 0.0062, (32, 4)).astype(np.float32)
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", conv1d, Q.F32)])
            self.assertEqual(rc, 0, log)                                          # no --compat-bf16 needed
            row = rows["blk.1.ple_conv1d.weight"]
            self.assertEqual((row[2], row[4], row[6]), ("3", str(conv1d.nbytes), str(conv1d.nbytes // 2)))
            np.testing.assert_array_equal(stored(rows, dense, "blk.1.ple_conv1d.weight", "<f4"), conv1d.ravel())
            r = conv["tensors"][0]
            self.assertEqual((r["src_type"], r["dst_type"]), ("F32", "F16"))
            self.assertIn("index kind 3", r["method"])
            h = conv1d.astype(np.float16).astype(np.float32)
            self.assertEqual(r["exact"], bool(np.array_equal(h, conv1d)))
            self.assertAlmostEqual(r["max_abs_err"], float(np.max(np.abs(h - conv1d))))
            self.assertIn("narrowed to F16 by the loader", log)
            # an F16 conv (the GSQ-RCO files) stays as it is: index kind 5, nothing recorded
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", conv1d, Q.F16)])
            self.assertEqual((rows["blk.1.ple_conv1d.weight"][2], conv["tensors"]), ("5", []))
            # beyond F16's range: refused, not saturated
            rc, log, *_ = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", conv1d * 1e7, Q.F32)])
            self.assertEqual(rc, 1)
            self.assertIn("beyond F16", log)

    def test_q8_0_projections_need_compat_and_are_recorded(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            v = np.random.default_rng(4).normal(0, 0.05, (4, 64)).astype(np.float32)
            tensors = [("blk.1.ple_value.weight", v, Q.Q8_0), ("blk.2.hc_attn_up.weight", v, Q.Q8_0),
                       ("blk.1.ple_key.weight", v, Q.Q8_0)]
            rc, log, *_ = pack_one(Path(tmp), tensors)
            self.assertEqual(rc, 1)
            self.assertIn("use --compat-bf16", log)
            self.assertNotIn("ple_key", log)                                      # the Q8_0 key stays native
            rc, log, rows, dense, conv = pack_one(Path(tmp), tensors, True)
            self.assertEqual(rc, 0, log)
            self.assertEqual(rows["blk.1.ple_key.weight"][2], "0")
            recs = {r["name"]: r for r in conv["tensors"]}
            self.assertEqual(set(recs), {"blk.1.ple_value.weight", "blk.2.hc_attn_up.weight"})
            deq = quants.dequantize(quants.quantize(v, Q.Q8_0), Q.Q8_0).ravel()
            got = (stored(rows, dense, "blk.1.ple_value.weight", "<u2").astype(np.uint32) << 16).view(np.float32)
            r = recs["blk.1.ple_value.weight"]
            self.assertEqual((r["src_type"], r["dst_type"]), ("Q8_0", "BF16"))
            self.assertAlmostEqual(r["max_abs_err"], float(np.max(np.abs(got - deq))))
            self.assertEqual(conv["compat_bf16"], True)


# ---- split artifacts: Unsloth's UD-Q4_K_XL in miniature.  Shard 1 holds the metadata and no tensor; shard 2 the
# routers, layer 0 and layer 1's down; shard 3 layer 1's gate and up (a shard boundary inside a layer, like its
# layer 11); shard 4 one float tensor.  4 experts of 32 x 32, Q8_0 and F32.
N_EXPERT = 4


def expert(seed, shape):
    return np.random.default_rng(seed).standard_normal((N_EXPERT, *shape)).astype(np.float32)


def split_model(root, name="model", split_layer=True, dup=False, split_no=None, gate1=Q.Q8_0, seed=0):
    """Writes the 4 shards; returns their paths."""
    gu, dn = (32, 32), (32, 32)
    router = [(f"blk.{l}.ffn_gate_inp.weight", np.full((N_EXPERT, 32), 0.5, np.float32), Q.F32) for l in (0, 1)]
    l0 = [(f"blk.0.ffn_{r}_exps.weight", expert(seed + 10 + i, gu if r != "down" else dn), Q.Q8_0)
          for i, r in enumerate(("gate", "up", "down"))]
    l1_gu = [("blk.1.ffn_gate_exps.weight", expert(seed + 20, gu), gate1),
             ("blk.1.ffn_up_exps.weight", expert(seed + 21, gu), Q.Q8_0)]
    l1_d = [("blk.1.ffn_down_exps.weight", expert(seed + 22, dn), Q.F32)]
    shards = [[], router + l0 + (l1_d if split_layer else []), l1_gu + ([] if split_layer else l1_d),
              [("output_norm.weight", np.ones(32, np.float32), Q.F32)]]
    if dup:
        shards[3].append(l1_gu[1])
    total = sum(len(s) for s in shards)
    paths = []
    for i, tensors in enumerate(shards):
        p = root / f"{name}-{i + 1:05d}-of-00004.gguf"
        no = split_no.get(i, i) if split_no else i
        write_gguf(p, tensors, split=(no, 4, total), arch=i == 0)
        paths.append(p)
    return paths


def run_pack(first, out, *extra):
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for n in ["vocab.json", "chat_template.jinja"]:
        (out / "tokenizer" / n).touch()
    buf = io.StringIO()
    with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), *extra]):
        with contextlib.redirect_stdout(buf):
            rc = iq_pack.main()
    return rc, buf.getvalue()


def expected_experts(model):
    """experts.bin from first principles: per layer, per expert, gate | up | down slices."""
    parts = []
    for l in (0, 1):
        ts = [model.bytes(f"blk.{l}.ffn_{r}_exps.weight").reshape(N_EXPERT, -1) for r in ("gate", "up", "down")]
        parts.append(np.concatenate(ts, axis=1).tobytes())
    return b"".join(parts)


class SplitArtifactTests(unittest.TestCase):
    def test_four_shards_and_a_layer_split_per_role(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            rc, log = run_pack(paths[0], root / "pack", "--experts-bin")
            self.assertEqual(rc, 0, log)
            text = (root / "pack" / "native_experts.txt").read_text()
            head, line0, line1 = text.splitlines()
            self.assertTrue(head.startswith("# strata native experts v4:"), head)
            self.assertIn("(n_expert 4,", head)
            s2, s3 = paths[1].name, paths[2].name
            self.assertEqual(line0.split()[8:], [s2])
            self.assertEqual(line1.split()[8:], [f"{s3},{s3},{s2}"])
            # every role's offset is absolute in ITS shard: shard 3's data_start for gate/up, shard 2's for down
            m = iq_pack.Model(paths[0])
            for line, l in ((line0, 0), (line1, 1)):
                f = line.split()
                for r, role in enumerate(("gate", "up", "down")):
                    g, t, _, _ = m.where[f"blk.{l}.ffn_{role}_exps.weight"]
                    self.assertEqual(int(f[5 + r]), g.data_start + t.offset)
            self.assertNotEqual(m.where["blk.1.ffn_gate_exps.weight"][0].data_start,
                                m.where["blk.1.ffn_down_exps.weight"][0].data_start)
            self.assertEqual((root / "pack" / "experts.bin").read_bytes(), expected_experts(m))
            side = json.loads((root / "pack" / "experts.bin.src.json").read_text())
            self.assertEqual([s["name"] for s in side["shards"]], [p.name for p in paths])
            self.assertFalse(list((root / "pack").glob("*.tmp")))
            # the index: routers as BF16, nothing of the experts
            _, rows = iq_pack.read_index(root / "pack" / "index.txt")
            self.assertNotIn("blk.1.ffn_gate_exps.weight", rows)
            self.assertEqual(rows["blk.1.ffn_gate_inp.weight"][2], "4")

    def test_unsplit_layers_keep_the_v3_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root, split_layer=False)
            rc, log = run_pack(paths[0], root / "pack")
            self.assertEqual(rc, 0, log)
            head, line0, line1 = (root / "pack" / "native_experts.txt").read_text().splitlines()
            self.assertTrue(head.startswith("# strata native experts v3: layer gu_type d_type offset blob_bytes "
                                            "gate_off up_off down_off [shard] (n_expert 4, total "), head)
            self.assertEqual(line1.split()[8:], [paths[2].name])

    def test_missing_shard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp))
            paths[3].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards.*00004-of-00004"):
                iq_pack.Model(paths[0])

    def test_duplicate_tensor_across_shards(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp), dup=True)
            with self.assertRaisesRegex(ValueError, "duplicate tensor blk.1.ffn_up_exps.weight"):
                iq_pack.Model(paths[0])

    def test_shard_of_another_split(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp), split_no={2: 1})
            with self.assertRaisesRegex(ValueError, "does not declare itself shard 3 of 4"):
                iq_pack.Model(paths[0])

    def test_truncated_shard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp))
            paths[2].write_bytes(paths[2].read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(paths[0])

    def test_wrong_type_or_shape_leaves_the_pack_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root, gate1=Q.F32)            # layer 1: gate F32, up Q8_0
            pack = root / "pack"
            pack.mkdir()
            (pack / "native_experts.txt").write_text("previous pack")
            rc, log = run_pack(paths[0], pack)
            self.assertEqual(rc, 1)
            self.assertIn("layer 1: gate and up differ in type", log)
            self.assertEqual((pack / "native_experts.txt").read_text(), "previous pack")
            self.assertFalse((pack / "dense.bin").exists())

    def test_interrupted_pack(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            pack = root / "pack"
            real = np.concatenate
            calls = []

            def dies_on_layer_1(parts, axis):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt("stopped part-way")
                return real(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", dies_on_layer_1):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack, "--experts-bin")
            self.assertFalse((pack / "experts.bin").exists())
            self.assertFalse((pack / "experts.bin.src.json").exists())
            # the index stops part-way too: the old pack's files stay as they were
            before = {n: (pack / n).read_bytes() for n in ("index.txt", "dense.bin", "native_experts.txt")}
            with patch.object(iq_pack, "write_index", side_effect=KeyboardInterrupt("stopped")):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack)
            self.assertEqual(before, {n: (pack / n).read_bytes() for n in before})
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertEqual((pack / "experts.bin").read_bytes(), expected_experts(iq_pack.Model(paths[0])))

    def test_stale_same_size_experts_bin(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            other = split_model(root / "a", name="other", seed=100)
            paths = split_model(root / "b")
            pack = root / "pack"
            rc, log = run_pack(other[0], pack, "--experts-bin")    # another model, the same geometry
            self.assertEqual(rc, 0, log)
            stale = (pack / "experts.bin").read_bytes()
            want = expected_experts(iq_pack.Model(paths[0]))
            self.assertEqual(len(stale), len(want))
            self.assertNotEqual(stale, want)
            # without --experts-bin the engine would read it: refused, nothing rewritten
            rc, log = run_pack(paths[0], pack)
            self.assertEqual(rc, 1)
            self.assertIn("another source", log)
            self.assertEqual((pack / "experts.bin").read_bytes(), stale)
            # with it: rewritten, not reused by its size
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertEqual((pack / "experts.bin").read_bytes(), want)
            # and now reused
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertIn("not rewritten", log)
            # an experts.bin without its sidecar (an older packer) is rewritten too
            (pack / "experts.bin.src.json").unlink()
            (pack / "experts.bin").write_bytes(stale)
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual((pack / "experts.bin").read_bytes(), want)


# ---- Qwen3.6-35B-A3B (qwen35moe) in miniature: hidden 64, 2 trunk layers (0 GDN, 1 attention), 4 experts of 32,
# the shared expert 32 wide, plus (with nextn) a draft block blk.2 of the -MTP files' shape.  The float tensors are
# stored as the shipped files store them: the routers, ssm_alpha / ssm_beta and the 1-D shexp gate F32 holding BF16
# values; the norms, ssm_a, ssm_dt.bias and ssm_conv1d F32; every projection quantized (Q8_0 here, Q6_K / Q8_0 in
# the files).
Q36_H, Q36_E, Q36_FF, Q36_HV, Q36_C, Q36_V, Q36_VOCAB = 64, 4, 32, 2, 64, 32, 32


def bf16_exact(shape, seed):
    """F32 values whose low 16 bits are zero: BF16 values held as F32, as the shipped files hold them."""
    x = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def q36_trunk_layer(l, gdn, router=None, proj_type=Q.Q8_0, proj_values=None):
    r = np.random.default_rng(100 + l)
    f32 = lambda *shape: r.standard_normal(shape).astype(np.float32)
    q = lambda *shape: (proj_values if proj_values is not None else f32(*shape))
    b = "blk.%d." % l
    t = [(b + "attn_norm.weight", f32(Q36_H), Q.F32), (b + "post_attention_norm.weight", f32(Q36_H), Q.F32),
         (b + "ffn_gate_inp.weight", router if router is not None else bf16_exact((Q36_E, Q36_H), l), Q.F32),
         (b + "ffn_gate_inp_shexp.weight", bf16_exact((Q36_H,), 50 + l), Q.F32),
         (b + "ffn_gate_shexp.weight", f32(Q36_FF, Q36_H), proj_type),
         (b + "ffn_up_shexp.weight", f32(Q36_FF, Q36_H), proj_type),
         (b + "ffn_down_shexp.weight", f32(Q36_H, Q36_FF), proj_type)]
    t += [(b + "ffn_%s_exps.weight" % role, f32(Q36_E, Q36_FF, Q36_H) if role != "down" else f32(Q36_E, Q36_H, Q36_FF),
           Q.Q8_0) for role in ("gate", "up", "down")]
    if gdn:
        t += [(b + "attn_qkv.weight", q(Q36_C, Q36_H), proj_type), (b + "attn_gate.weight", q(Q36_V, Q36_H), proj_type),
              (b + "ssm_out.weight", q(Q36_H, Q36_V), proj_type),
              (b + "ssm_conv1d.weight", f32(Q36_C, 4), Q.F32),
              (b + "ssm_alpha.weight", bf16_exact((Q36_HV, Q36_H), 60 + l), Q.F32),
              (b + "ssm_beta.weight", bf16_exact((Q36_HV, Q36_H), 70 + l), Q.F32),
              (b + "ssm_a", -np.abs(f32(Q36_HV)), Q.F32), (b + "ssm_dt.bias", f32(Q36_HV), Q.F32),
              (b + "ssm_norm.weight", f32(16), Q.F32)]
    else:
        t += [(b + "attn_q.weight", q(128, Q36_H), proj_type), (b + "attn_k.weight", q(32, Q36_H), proj_type),
              (b + "attn_v.weight", q(32, Q36_H), proj_type), (b + "attn_output.weight", q(Q36_H, Q36_H), proj_type),
              (b + "attn_q_norm.weight", f32(32), Q.F32), (b + "attn_k_norm.weight", f32(32), Q.F32)]
    return t


def q36_model(path, nextn=False, **layer_kw):
    r = np.random.default_rng(7)
    f32 = lambda *shape: r.standard_normal(shape).astype(np.float32)
    tensors = [("token_embd.weight", f32(Q36_VOCAB, Q36_H), Q.Q8_0), ("output.weight", f32(Q36_VOCAB, Q36_H), Q.Q8_0),
               ("output_norm.weight", f32(Q36_H), Q.F32)]
    tensors += q36_trunk_layer(0, True, **layer_kw) + q36_trunk_layer(1, False)
    kv = {}
    if nextn:
        # the draft block: an attention layer with MoE, its routers BF16 (as in the -MTP files), plus the nextn set
        draft = [(n, v, Q.BF16 if n.endswith(("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")) else k)
                 for n, v, k in q36_trunk_layer(2, False)]
        draft += [("blk.2.nextn.eh_proj.weight", f32(Q36_H, 2 * Q36_H), Q.Q8_0),
                  ("blk.2.nextn.enorm.weight", f32(Q36_H), Q.F32), ("blk.2.nextn.hnorm.weight", f32(Q36_H), Q.F32),
                  ("blk.2.nextn.shared_head_norm.weight", f32(Q36_H), Q.F32)]
        tensors += draft
        kv["qwen35moe.nextn_predict_layers"] = 1
    write_gguf(path, tensors, kv=kv, architecture="qwen35moe")
    return tensors


def q36_pack(root, out, *extra, nextn=False, **layer_kw):
    """Packs a toy model at root/q36.gguf into `out` through main(): (rc, log, tensors)."""
    path = root / "q36.gguf"
    tensors = q36_model(path, nextn=nextn, **layer_kw)
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for n in ["vocab.json", "chat_template.jinja"]:
        (out / "tokenizer" / n).touch()
    buf = io.StringIO()
    with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(path), "--out", str(out), *extra]):
        with contextlib.redirect_stdout(buf):
            rc = iq_pack.main()
    return rc, buf.getvalue(), tensors


class Qwen36PackTests(unittest.TestCase):
    """The qwen35moe entry (docs/qwen36-port/step4-pack.md): the forms, the BF16 rule's refusal with its count, the
    native projection types refused by name, the nextn block left out, index.txt's contents, expert-profile.bin."""

    def test_forms_and_index_contents(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root, out = Path(tmp), Path(tmp) / "pack"
            rc, log, tensors = q36_pack(root, out)
            self.assertEqual(rc, 0, log)
            header, rows = iq_pack.read_index(out / "index.txt")
            dense = (out / "dense.bin").read_bytes()
            by_name = {n: (v, k) for n, v, k in tensors}
            # every non-expert tensor has a row, nothing else does, and the header counts them
            expect = {n for n in by_name if not n.endswith("_exps.weight")}
            self.assertEqual(set(rows), expect)
            self.assertIn("tensors %d" % len(expect), header[1])
            bf16 = {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ssm_alpha.weight", "ssm_beta.weight"}
            f32 = {"attn_norm.weight", "post_attention_norm.weight", "output_norm.weight", "attn_q_norm.weight",
                   "attn_k_norm.weight", "ssm_norm.weight", "ssm_a", "ssm_dt.bias", "ssm_conv1d.weight"}
            native = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight", "attn_k.weight",
                      "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                      "ffn_down_shexp.weight", "output.weight", "token_embd.weight"}
            at = 0
            for name, row in rows.items():
                suffix = iq_pack.re.sub(r"^blk\.\d+\.", "", name)
                v, k = by_name[name]
                ne0, ne1 = int(row[7]), int(row[8])
                self.assertEqual((ne0, ne1), (v.shape[-1], v.shape[0] if v.ndim == 2 else 0), name)
                if suffix in bf16:
                    self.assertEqual(row[2], "4", name)                            # index kind 4: raw BF16
                    self.assertEqual(int(row[4]), v.size * 2, name)
                    np.testing.assert_array_equal(stored(rows, dense, name, "<u2"), (v.view(np.uint32) >> 16).ravel())
                elif suffix in f32:
                    self.assertEqual(row[2], "2", name)                            # F32 as stored
                    np.testing.assert_array_equal(stored(rows, dense, name, "<f4"), v.ravel())
                else:
                    self.assertIn(suffix, native, name)
                    self.assertEqual(row[1:5] + [row[6]], ["0"] * 5, name)         # shape only: no bytes, no source
                    self.assertEqual(int(row[5]), at, name)                        # the running arena offset, 0 B
                    self.assertEqual((row[9], row[11]), ("8", "32"), name)         # marked quantized: served natively
                if row[2] != "0":
                    self.assertEqual(int(row[5]), at, name)                        # 64-aligned arena offsets
                    self.assertEqual(int(row[3]) % 64, 0, name)
                    at += (int(row[6]) + 63) // 64 * 64
            self.assertIn("pool %d" % at, header[1])
            # conversions.json: the four BF16 forms per GDN layer, two per attention layer, all exact, nothing lossy
            conv = json.loads((out / "conversions.json").read_text())
            self.assertEqual(sorted(r["name"] for r in conv["tensors"]),
                             sorted(n for n in rows if iq_pack.re.sub(r"^blk\.\d+\.", "", n) in bf16))
            self.assertTrue(all(r["exact"] and r["max_abs_err"] == 0.0 for r in conv["tensors"]))
            self.assertFalse(conv["compat_bf16"])
            self.assertFalse((out / "compat-bf16.json").exists())
            # the expert table: two trunk layers
            lines = [l for l in (out / "native_experts.txt").read_text().splitlines() if not l.startswith("#")]
            self.assertEqual([l.split()[0] for l in lines], ["0", "1"])
            self.assertIn("no expert profile shipped", log)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["conversions.json", "dense.bin", "index.txt", "native_experts.txt", "tokenizer"])

    def test_inexact_router_is_refused_with_the_count(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root, out = Path(tmp), Path(tmp) / "pack"
            router = bf16_exact((Q36_E, Q36_H), 1)
            router[0, :3] += np.float32(1e-3)                    # three values with low bits set
            out.mkdir()
            (out / "dense.bin").write_bytes(b"previous pack")
            (out / "native_experts.txt").write_bytes(b"previous pack")
            rc, log, _ = q36_pack(root, out, router=router)
            self.assertEqual(rc, 1)
            self.assertIn("blk.0.ffn_gate_inp.weight is F32 with 3 of %d values that are not BF16" % router.size, log)
            self.assertIn("--compat-bf16 would round them", log)
            self.assertEqual((out / "dense.bin").read_bytes(), b"previous pack")
            self.assertEqual((out / "native_experts.txt").read_bytes(), b"previous pack")
            self.assertFalse((out / "index.txt").exists())
            rc, log, _ = q36_pack(root, out, "--compat-bf16", router=router)
            self.assertEqual(rc, 0, log)
            compat = json.loads((out / "compat-bf16.json").read_text())
            self.assertEqual([t["name"] for t in compat["tensors"]], ["blk.0.ffn_gate_inp.weight"])
            conv = {r["name"]: r for r in json.loads((out / "conversions.json").read_text())["tensors"]}
            self.assertFalse(conv["blk.0.ffn_gate_inp.weight"]["exact"])
            self.assertTrue(conv["blk.1.ffn_gate_inp.weight"]["exact"])

    def test_projection_types_are_refused_by_name(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root, out = Path(tmp), Path(tmp) / "pack"
            # an F32 GDN projection: the engine has no F32 GEMV
            rc, log, _ = q36_pack(root, out, proj_type=Q.F32)
            self.assertEqual(rc, 1)
            self.assertIn("blk.0.attn_qkv.weight is F32, which the engine's native projection path", log)
            self.assertIn("Q6_K", log)
            self.assertIn("Q8_0", log)
            self.assertFalse((out / "index.txt").exists())
            self.assertFalse((out / "native_experts.txt").exists())
            # the head and the embedding have their own lists
            path = root / "q36.gguf"
            tensors = [(n, v, k) for n, v, k in q36_model(path)]
            tensors = [(n, v, Q.F32 if n == "output.weight" else Q.F16 if n == "token_embd.weight" else k)
                       for n, v, k in tensors]
            write_gguf(path, tensors, architecture="qwen35moe")
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = iq_pack.index_standalone(path, out, iq_pack.Model(path), False, 2)
            self.assertEqual(rc, 1)
            self.assertIn("output.weight is F32, which the engine's native head", buf.getvalue())
            self.assertIn("token_embd.weight is F16, which the engine's native embedding", buf.getvalue())
            self.assertIn("BF16", buf.getvalue())                 # the embedding takes BF16, the head does not

    def test_nextn_block_is_left_to_the_drafter(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root, out = Path(tmp), Path(tmp) / "pack"
            rc, log, tensors = q36_pack(root, out, nextn=True)
            self.assertEqual(rc, 0, log)
            _, rows = iq_pack.read_index(out / "index.txt")
            self.assertFalse([n for n in rows if n.startswith("blk.2.")], "the draft block is in the index")
            self.assertFalse([n for n in rows if "nextn" in n])
            self.assertEqual(len(rows), len([n for n, _, _ in tensors
                                             if not n.endswith("_exps.weight") and not n.startswith("blk.2.")]))
            draft_floats = sum(1 for n, _, k in tensors if n.startswith("blk.2.") and not n.endswith("_exps.weight"))
            self.assertIn("%d tensors of the draft (nextn) block(s) blk.2 and up are left to the drafter's pack"
                          % draft_floats, log)
            self.assertIn("1 draft (nextn) block(s) at blk.2 and up are left to the MTP pack; 2 trunk layers", log)
            lines = [l for l in (out / "native_experts.txt").read_text().splitlines() if not l.startswith("#")]
            self.assertEqual([l.split()[0] for l in lines], ["0", "1"])
            # dense.bin holds exactly the trunk's float tensors: its size is their 64-aligned sum
            size = 0
            for n, v, k in tensors:
                if n.startswith("blk.2.") or n.endswith("_exps.weight") or k != Q.F32:
                    continue
                suffix = iq_pack.re.sub(r"^blk\.\d+\.", "", n)
                nbytes = v.size * (2 if iq_pack.FORM_QWEN35MOE.get(suffix) == "BF16" else 4)
                size += (nbytes + 63) // 64 * 64
            self.assertEqual((out / "dense.bin").stat().st_size, size)

    def test_expert_profile_is_checked_and_copied(self):
        import make_profile as MP
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root, out = Path(tmp), Path(tmp) / "pack"
            ranked = [(l, e) for l in range(2) for e in range(Q36_E)]
            MP.write_profile(root / "fits.bin", ranked, Q36_E, 2)
            MP.write_profile(root / "other.bin", [(l, e) for l in range(3) for e in range(Q36_E)], Q36_E, 3)
            out.mkdir()
            (out / "native_experts.txt").write_bytes(b"previous pack")
            rc, log, _ = q36_pack(root, out, "--expert-profile", str(root / "other.bin"))
            self.assertEqual(rc, 1)
            self.assertIn("ranks 3 x 4 (layers x experts), this model's trunk is 2 x 4", log)
            self.assertEqual((out / "native_experts.txt").read_bytes(), b"previous pack")
            self.assertFalse((out / "expert-profile.bin").exists())
            rc, log, _ = q36_pack(root, out, "--expert-profile", str(root / "fits.bin"))
            self.assertEqual(rc, 0, log)
            self.assertEqual((out / "expert-profile.bin").read_bytes(), (root / "fits.bin").read_bytes())
            self.assertIn("expert-profile.bin: %s (2 x 4)" % (root / "fits.bin"), log)
            self.assertEqual(MP.read_profile(out / "expert-profile.bin", Q36_E, 2), ranked)
            self.assertFalse(list(out.glob("*.tmp")))

    def test_flash_next_forms_do_not_see_the_qwen36_table(self):
        """The qwen4exp FORM table is the one it was: the Qwen3.6 names resolve to nothing there, so a Flash-Next
        tensor by one of those names (none exists) would be written as stored, as before."""
        for name in ("blk.0.attn_norm.weight", "blk.0.ssm_a", "blk.0.attn_q_norm.weight"):
            self.assertEqual(iq_pack.form_of(name, "qwen35moe"), "F32")
        self.assertEqual(iq_pack.form_of("blk.0.ssm_alpha.weight"), "BF16")
        self.assertEqual(iq_pack.form_of("blk.0.hc_attn_down.weight", "qwen35moe"), None)
        self.assertEqual(iq_pack.form_of("blk.0.attn_norm.weight"), None)
        self.assertTrue(iq_pack.needs_bf16("blk.0.ssm_alpha.weight", "F32", "qwen35moe"))
        self.assertIsNone(iq_pack.native_type_problem("blk.0.attn_qkv.weight", "F32", "qwen4exp"))
        self.assertIsNone(iq_pack.native_type_problem("blk.0.attn_qkv.weight", "Q6_K", "qwen35moe"))
        self.assertIsNone(iq_pack.native_type_problem("blk.0.ffn_gate_inp.weight", "F32", "qwen35moe"))


if __name__ == "__main__":
    unittest.main()
