#!/usr/bin/env python3
"""Invariant tests for ref/qwen36.py and tools/qwen36_synth_gguf.py.  No llama.cpp build needed.

    python3 -I tools/test_qwen36_oracle.py            # numpy and gguf-py (STRATA_GGUF_PY or the submodule) only

These pin what the cross-check in docs/qwen36-port/oracle-crosscheck.md established against llama.cpp, so the oracle
cannot drift, and each plausible wrong reading ("rival") must change the answer by more than 5 % (relative L1) on the
synthetic fixture - the negative-test discipline of src/kernels/gdn_parity.cpp: a fixture that cannot tell the right
reading from the wrong one proves nothing.
"""
import hashlib
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tools")):            # -I drops the script directory from sys.path
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                    # noqa: E402

import qwen36_synth_gguf as synth                     # noqa: E402
from ref import qwen36 as q                           # noqa: E402

F32 = np.float32
TOKENS = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]      # "hello world!"
TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
ROOT = pathlib.Path(TMP.name)
_models: dict = {}


def fixture(name="base", **cfg_kw) -> pathlib.Path:
    path = ROOT / f"{name}.gguf"
    if not path.exists():
        synth.write(str(path), synth.Cfg(**cfg_kw))
    return path


def model(name="base", **cfg_kw) -> q.Model:
    key = name + repr(sorted(cfg_kw.items()))
    if key not in _models:
        _models[key] = q.Model(fixture(name, **cfg_kw))
    return _models[key]


def fresh(name="base", **cfg_kw) -> q.Model:
    """A Model with default (faithful) options, safe to mutate."""
    m = q.Model(fixture(name, **cfg_kw))
    return m


def run(m, tokens=TOKENS, dump_dir=None):
    d = q.Dump(dump_dir)
    logits, h, st = m.forward(tokens, d)
    d.close()
    return logits, h, st


def read_dump(dirname):
    """{stem: (kind, ne, array)} for a dump directory."""
    out = {}
    for line in (pathlib.Path(dirname) / "manifest.txt").read_text().splitlines():
        stem, kind, ne = line.split()
        ne = tuple(int(x) for x in ne.split(","))
        a = np.fromfile(pathlib.Path(dirname) / f"{stem}.f32", dtype=np.float32)
        trim = list(ne)
        while len(trim) > 1 and trim[-1] == 1:        # numpy natural shape: the writer padded ne to 4 with ones
            trim.pop()
        out[stem] = (kind, ne, a.reshape(tuple(reversed(trim))))
    return out


def rel_l1(a, b):
    return float(np.abs(a - b).sum() / np.abs(b).sum())


class FixtureTests(unittest.TestCase):
    def test_deterministic_and_seeded(self):
        a, b, c = ROOT / "s0a.gguf", ROOT / "s0b.gguf", ROOT / "s1.gguf"
        synth.write(str(a), synth.Cfg(seed=0))
        synth.write(str(b), synth.Cfg(seed=0))
        synth.write(str(c), synth.Cfg(seed=1))
        h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(h(a), h(b))
        self.assertNotEqual(h(a), h(c))

    def test_tensor_names_are_the_real_pattern_set(self):
        # the 32 normalised names of the real MTP build (docs/qwen36-port/gguf-metadata.md)
        real = """output.weight output_norm.weight token_embd.weight blk.N.attn_gate.weight blk.N.attn_norm.weight
        blk.N.attn_qkv.weight blk.N.ffn_down_exps.weight blk.N.ffn_down_shexp.weight blk.N.ffn_gate_exps.weight
        blk.N.ffn_gate_inp.weight blk.N.ffn_gate_inp_shexp.weight blk.N.ffn_gate_shexp.weight blk.N.ffn_up_exps.weight
        blk.N.ffn_up_shexp.weight blk.N.post_attention_norm.weight blk.N.ssm_a blk.N.ssm_alpha.weight blk.N.ssm_beta.weight
        blk.N.ssm_conv1d.weight blk.N.ssm_dt.bias blk.N.ssm_norm.weight blk.N.ssm_out.weight blk.N.attn_k.weight
        blk.N.attn_k_norm.weight blk.N.attn_output.weight blk.N.attn_q.weight blk.N.attn_q_norm.weight blk.N.attn_v.weight
        blk.N.nextn.eh_proj.weight blk.N.nextn.enorm.weight blk.N.nextn.hnorm.weight
        blk.N.nextn.shared_head_norm.weight""".split()
        names = {synth.norm_name(n) for n, _ in synth.build_tensors(synth.Cfg())}
        self.assertEqual(names, set(real))

    def test_shape_rules(self):
        c = synth.Cfg()
        t = dict(synth.build_tensors(c))
        self.assertEqual(t["blk.0.attn_qkv.weight"].shape, (c.conv_dim, c.n_embd))          # GDN: 2*key_dim + value_dim
        self.assertEqual(c.conv_dim, 2 * c.ssm_k_heads * c.ssm_state + c.ssm_v_heads * c.ssm_state)
        self.assertEqual(t["blk.0.ssm_conv1d.weight"].shape, (c.conv_dim, c.ssm_conv))
        self.assertEqual(t["blk.3.attn_q.weight"].shape, (2 * c.n_head * c.head_dim, c.n_embd))   # query + gate per head
        self.assertEqual(t["blk.3.attn_k.weight"].shape, (c.n_head_kv * c.head_dim, c.n_embd))
        self.assertEqual(t["blk.1.ffn_gate_exps.weight"].shape, (c.n_expert, c.n_ff_exp, c.n_embd))
        self.assertEqual(t["blk.1.ffn_down_exps.weight"].shape, (c.n_expert, c.n_embd, c.n_ff_exp))
        self.assertEqual(t["blk.1.ffn_gate_inp_shexp.weight"].shape, (c.n_embd,))              # 1-D, one scalar per token
        self.assertEqual(t["blk.4.nextn.eh_proj.weight"].shape, (c.n_embd, 2 * c.n_embd))
        self.assertLess(t["blk.0.ssm_a"].max(), 0.0)                                           # already -exp(A_log)
        self.assertNotIn("blk.3.ssm_a", t)                                                     # attention layer: no GDN tensors
        self.assertNotIn("blk.0.attn_q.weight", t)


class LayerPatternTests(unittest.TestCase):
    def test_full_attention_layers_are_3_mod_4(self):
        hp = model().hp
        self.assertEqual(hp.n_layer, 4)
        self.assertEqual(hp.attn_layers(), [3])
        hp8 = fresh("l8", n_layer=8, nextn=False).hp
        self.assertEqual(hp8.attn_layers(), [3, 7])
        self.assertEqual([il for il in range(8) if hp8.is_recr(il)], [0, 1, 2, 4, 5, 6])

    def test_real_geometry_is_30_gdn_plus_10_attention(self):
        md = {"general.architecture": "qwen35moe", "qwen35moe.block_count": 40, "qwen35moe.embedding_length": 2048,
              "qwen35moe.attention.head_count": 16, "qwen35moe.attention.head_count_kv": 2,
              "qwen35moe.attention.key_length": 256, "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
              "qwen35moe.rope.dimension_count": 64, "qwen35moe.rope.dimension_sections": [11, 11, 10, 0],
              "qwen35moe.expert_count": 256, "qwen35moe.expert_used_count": 8,
              "qwen35moe.expert_feed_forward_length": 512, "qwen35moe.expert_shared_feed_forward_length": 512,
              "qwen35moe.ssm.conv_kernel": 4, "qwen35moe.ssm.inner_size": 4096, "qwen35moe.ssm.state_size": 128,
              "qwen35moe.ssm.group_count": 16, "qwen35moe.ssm.time_step_rank": 32, "qwen35moe.full_attention_interval": 4}
        hp = q.read_hparams(md)
        self.assertEqual(hp.attn_layers(), list(range(3, 40, 4)))
        self.assertEqual(sum(hp.is_recr(i) for i in range(40)), 30)
        self.assertEqual((hp.conv_dim, hp.value_dim, hp.key_dim, hp.head_v_dim), (8192, 4096, 2048, 128))
        self.assertEqual(sum(hp.sections) * 2, hp.n_rot)                      # IMRoPE sections cover the rotated dims
        self.assertTrue(hp.expert_weights_norm)
        self.assertEqual(hp.expert_weights_scale, 0.0)
        md.update({"qwen35moe.block_count": 41, "qwen35moe.nextn_predict_layers": 1})
        hp = q.read_hparams(md)
        self.assertEqual(hp.n_layer, 40)
        self.assertFalse(hp.is_recr(40), "the nextn block is attention-only")
        self.assertEqual(hp.attn_layers(), list(range(3, 40, 4)), "and not part of the trunk")

    def test_nextn_block_is_not_run_by_the_trunk(self):
        with_n = model()
        without = model("nonextn", nextn=False)
        a, _, _ = run(with_n)
        b, _, _ = run(without)                      # same seed, the nextn tensors are drawn last: trunk weights identical
        np.testing.assert_array_equal(a, b)


class RouterTests(unittest.TestCase):
    def test_topk_weights_renormalize(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            run(model(), dump_dir=d)
            dump = read_dump(d)
        for il in range(4):
            _, _, w = dump[f"ffn_moe_weights-{il}"]            # raw softmax probabilities of the chosen experts (T, K, 1)
            _, _, wn = dump[f"ffn_moe_weights_norm-{il}"]      # (T, K)
            _, _, probs = dump[f"ffn_moe_probs-{il}"]
            _, _, top = dump[f"ffn_moe_topk-{il}"]
            np.testing.assert_allclose(wn.sum(axis=1), 1.0, atol=2e-6)
            self.assertTrue((wn > 0).all())
            self.assertTrue((w[:, :, 0].sum(axis=1) < 0.999).any(), "raw top-2 of 8 must not already sum to 1")
            np.testing.assert_allclose(wn, w[:, :, 0] / w[:, :, 0].sum(axis=1, keepdims=True), rtol=2e-6)
            kth = np.sort(probs, axis=1)[:, -2]                # top-2 are exactly the two largest probabilities
            chosen = np.take_along_axis(probs, top.astype(int), 1)
            self.assertTrue((chosen.min(axis=1) >= kth - 1e-9).all())
            self.assertTrue((np.diff(chosen, axis=1) <= 1e-9).all(), "top-k order is descending")

    def test_stray_scale_key_is_ignored_like_llama_cpp(self):
        m = fresh("scale", expert_weights_scale=2.5)
        self.assertEqual(m.hp.expert_weights_scale, 0.0)       # llama-hparams.h:124; the qwen35moe loader never reads the key
        self.assertIn("qwen35moe.expert_weights_scale", m.md)
        a, _, _ = run(m)
        b, _, _ = run(fresh("scale2"))                          # same seed, no key
        np.testing.assert_array_equal(a, b)

    def test_scale_branch_exists_for_other_archs_and_is_observable(self):
        m = fresh("scale3")
        m.hp.expert_weights_scale = 2.5
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            run(m, dump_dir=d)
            dump = read_dump(d)
        np.testing.assert_allclose(dump["ffn_moe_weights_scaled-0"][2].sum(axis=1), 2.5, atol=1e-5)

    def test_without_renormalization_the_answer_changes(self):
        m = fresh("rn")
        base, _, _ = run(m)
        m.opts["router_norm"] = False
        other, _, _ = run(m)
        self.assertGreater(rel_l1(other, base), 0.05)


class GdnTests(unittest.TestCase):
    def test_state_shapes_and_manifest(self):
        m = model()
        hp = m.hp
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            _, _, st = run(m, dump_dir=d)
            dump = read_dump(d)
        S, Hv, Hk, K = hp.ssm_state, hp.ssm_v_heads, hp.ssm_k_heads, hp.ssm_conv
        for il in (0, 1, 2):
            ls = st.layers[il]
            self.assertEqual(ls.ssm.shape, (Hv, S, S))             # [v head, v index, k index]
            self.assertEqual(ls.conv.shape, (hp.conv_dim, K - 1))
            self.assertEqual(dump[f"new_state-{il}"][1], (S, S, Hv, 1))
            self.assertEqual(dump[f"conv_states_reshaped-{il}"][1], (K - 1, hp.conv_dim, 1, 1))
            self.assertEqual(dump[f"q_conv_predelta-{il}"][1], (S, Hk, len(TOKENS), 1))   # not repeated to Hv heads
            self.assertEqual(dump[f"v_conv_predelta-{il}"][1], (S, Hv, len(TOKENS), 1))
            self.assertEqual(dump[f"attn_output-{il}"][1], (S, Hv, len(TOKENS), 1))
            self.assertEqual(dump[f"beta_sigmoid-{il}"][1], (1, Hv, len(TOKENS), 1))
        self.assertIsNone(st.layers[3].ssm)                         # the attention layer carries KV, not GDN state
        self.assertEqual(st.layers[3].k.shape, (len(TOKENS), hp.n_head_kv, hp.head_dim))
        self.assertTrue((dump["state_predelta-0"][2] == 0).all(), "a fresh sequence starts from zero state")
        self.assertTrue((dump["gate-0"][2] < 0).all(), "gate = softplus(.) * ssm_a is a log-decay, so negative")

    def test_chunked_equals_one_token_at_a_time(self):
        m = model()
        full, h_full, st_full = run(m)
        st = None
        for t in TOKENS:
            logits, _, st = m.forward([t], q.Dump(), st)
        np.testing.assert_allclose(logits, full, atol=2e-4)
        for il in (0, 1, 2):
            np.testing.assert_allclose(st.layers[il].ssm, st_full.layers[il].ssm, atol=2e-4)
            np.testing.assert_allclose(st.layers[il].conv, st_full.layers[il].conv, atol=1e-4)
        np.testing.assert_allclose(st.layers[3].k, st_full.layers[3].k, atol=1e-5)

    def test_causality(self):
        m = model()
        _, h1, _ = run(m, TOKENS)
        _, h2, _ = run(m, TOKENS[:8] + [97, 98, 99, 100])
        np.testing.assert_allclose(h1[:8], h2[:8], atol=1e-5)
        self.assertGreater(np.abs(h1[8:] - h2[8:]).max(), 1e-2)

    def test_output_gate_is_silu_of_z_not_sigmoid(self):
        m = fresh("silu")
        base, _, _ = run(m)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            run(m, dump_dir=d)
            dump = read_dump(d)
        z, o, fo = dump["z-0"][2], dump["attn_output-0"][2], dump["final_output-0"][2]
        T, Hv, S = o.shape
        w = m.t("blk.0.ssm_norm.weight")
        normed = dump["norm-0#2"][2] * w                              # rms_norm(o) * ssm_norm
        zz = z.reshape(T, Hv, S)
        silu = zz / (1 + np.exp(-zz))
        sig = 1 / (1 + np.exp(-zz))
        np.testing.assert_allclose(fo.reshape(T, Hv, S), normed * silu, atol=1e-5)
        self.assertGreater(rel_l1(normed * sig, normed * silu), 0.05, "the fixture must tell SiLU(z) from sigmoid(z)")
        m.opts["gdn_gate"] = "sigmoid"                                 # the Flash-Next engine's reading
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            other, _, _ = run(m, dump_dir=d)
            dump2 = read_dump(d)
        self.assertGreater(rel_l1(dump2["final_output-0"][2], fo), 0.05)
        self.assertGreater(rel_l1(other, base), 0.05, "and the difference must reach the logits")

    def test_rival_readings_are_observable(self):
        # (option, rival, the seam where the rival must first differ by > 5 %): a wrong reading has to be visible at the
        # layer that makes it, not only after 4 layers of averaging
        for key, rival, seam in (("head_pairing", "interleaved", "attn_output-0"), ("decay", "after", "attn_output-0"),
                                 ("conv", "flipped", "conv_output_raw-0"), ("rope_pairs", "adjacent", "kqv-3"),
                                 ("gdn_gate", "sigmoid", "final_output-0"), ("attn_gate", False, "attn_gated-3"),
                                 ("router_norm", False, "ffn_moe_weighted-0")):
            with self.subTest(rival=f"{key}={rival}"):
                m = fresh(f"rv_{key}")
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                    base, _, _ = run(m, dump_dir=d)
                    ref = read_dump(d)[seam][2]
                m.opts[key] = rival
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                    other, _, _ = run(m, dump_dir=d)
                    got = read_dump(d)[seam][2]
                self.assertGreater(rel_l1(got, ref), 0.05, f"{seam} cannot tell {key}={rival} from the faithful reading")
                self.assertGreater(rel_l1(other, base), 5e-3, "and it must reach the logits")

    def test_gqa_mapping_needs_more_than_one_kv_head(self):
        m = fresh("gqa", n_head=4, n_head_kv=2, seed=11)
        self.assertEqual((m.hp.n_head, m.hp.n_head_kv), (4, 2))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            base, _, _ = run(m, dump_dir=d)
            ref = read_dump(d)["kqv-3"][2]
        m.opts["gqa_map"] = "interleaved"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            other, _, _ = run(m, dump_dir=d)
            got = read_dump(d)["kqv-3"][2]
        self.assertGreater(rel_l1(got, ref), 0.05)
        self.assertGreater(rel_l1(other, base), 5e-3)
        # with one KV head every mapping is the same: the default fixtures cannot see this rival
        m1 = fresh("gqa1")
        a, _, _ = run(m1)
        m1.opts["gqa_map"] = "interleaved"
        b, _, _ = run(m1)
        np.testing.assert_array_equal(a, b)

    def test_l2_norm_puts_eps_inside_the_sum(self):
        # build_gdn_l2_norm (models.h:14): x / sqrt(sum(x^2) + eps), NOT x / max(sqrt(sum), eps)
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((6, 16)) * 3e-4).astype(F32)       # small enough that eps (1e-6) matters
        eps = 1e-6
        ss = (x.astype(np.float64) ** 2).sum(-1, keepdims=True)
        want = x / np.sqrt(ss + eps)
        rival = x / np.maximum(np.sqrt(ss), eps)
        np.testing.assert_allclose(q.l2_norm_gdn(x, eps), want, rtol=1e-4)
        self.assertGreater(rel_l1(rival, want), 0.05)
        big = rng.standard_normal((6, 16)).astype(F32)
        np.testing.assert_allclose(np.linalg.norm(q.l2_norm_gdn(big, eps), axis=-1), 1.0, rtol=1e-5)


class AttentionTests(unittest.TestCase):
    def test_output_gate_is_applied(self):
        m = fresh("og")
        base, _, _ = run(m)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            run(m, dump_dir=d)
            dump = read_dump(d)
        pre, g, gs, gated = (dump[f"{n}-3"][2] for n in ("attn_pregate", "gate_reshaped", "gate_sigmoid", "attn_gated"))
        np.testing.assert_allclose(gs, 1 / (1 + np.exp(-g)), atol=1e-6)
        np.testing.assert_allclose(gated, pre * gs, atol=1e-6)
        self.assertTrue(((gs > 0) & (gs < 1)).all())
        self.assertGreater(rel_l1(pre, gated), 0.05, "gating must matter on the fixture")
        # the gate is the SECOND half of each head's slice of the joint q projection: [q (hd) | gate (hd)]
        hp = m.hp
        qfull = dump["Qcur_full-3"][2].reshape(len(TOKENS), hp.n_head, 2, hp.head_dim)
        np.testing.assert_array_equal(qfull[:, :, 1, :].reshape(len(TOKENS), -1), g)
        np.testing.assert_array_equal(qfull[:, :, 0, :], dump["Qcur_reshaped-3"][2])
        m.opts["attn_gate"] = False
        other, _, _ = run(m)
        self.assertGreater(rel_l1(other, base), 0.05)

    def test_rope_is_partial_imrope_with_text_positions(self):
        rng = np.random.default_rng(1)
        T, H, D, n_rot = 5, 3, 32, 8
        x = rng.standard_normal((T, H, D)).astype(F32)
        pos = np.arange(T)
        pos4 = np.stack([pos, pos, pos, np.zeros_like(pos)])
        sec = (2, 1, 1, 0)
        y = q.rope_multi(x, pos4, n_rot, sec, 10000.0)
        np.testing.assert_array_equal(y[:, :, n_rot:], x[:, :, n_rot:])                 # only 25 % is rotated
        np.testing.assert_allclose(y[0], x[0], atol=1e-6)                                # position 0 is the identity
        np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)
        # rotation by pos * base^(-2p/n_rot) on the pair (p, p + n_rot/2)
        p = 1
        th = 3 * 10000.0 ** (-2 * p / n_rot)
        want0 = x[3, 0, p] * np.cos(th) - x[3, 0, p + n_rot // 2] * np.sin(th)
        self.assertAlmostEqual(float(y[3, 0, p]), float(want0), places=5)
        # with text, t = h = w = position and e = 0: a sector that falls to the "e" slot is NOT rotated.
        # sections (2,1,0,1): sector 2 has 2 % 3 == 2 and 2 >= 3 * sections[2] = 0, so it takes theta_e = 0.
        y2 = q.rope_multi(x, pos4, n_rot, (2, 1, 0, 1), 10000.0)
        np.testing.assert_allclose(y2[:, :, 2], x[:, :, 2], atol=1e-6)
        np.testing.assert_allclose(y2[:, :, 2 + n_rot // 2], x[:, :, 2 + n_rot // 2], atol=1e-6)
        self.assertGreater(np.abs(y2[1:, :, 0] - x[1:, :, 0]).max(), 1e-3)


class ActivationQuantTests(unittest.TestCase):
    def test_q8_k_and_q8_0_rounding(self):
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((4, 512)) * 2).astype(F32)
        for wt, bs in (("Q6_K", 256), ("IQ2_S", 256), ("Q8_0", 32)):
            y = q.quantize_act(x, wt)
            amax = np.abs(x.reshape(-1, bs)).max(axis=1)
            err = np.abs((x - y).reshape(-1, bs)).max(axis=1)
            self.assertTrue((err <= amax / 127 * 0.57 + 1e-7).all(), wt)     # half a step of the block scale (+ the fp16 scale of Q8_0)
            self.assertGreater(err.max(), 0)                                       # and it is not the identity
            np.testing.assert_allclose(q.quantize_act(y, wt), y, rtol=1e-5, atol=1e-6)   # idempotent
            # at most 255 distinct levels per block
            self.assertLessEqual(len(np.unique(y.reshape(-1, bs)[0])), 255)
        np.testing.assert_array_equal(q.quantize_act(x, "F32"), x)                 # F32 weights (router, ssm_alpha/beta): untouched

    def test_q8_mode_changes_little_on_the_f32_fixture(self):
        m = fresh("q8m")
        base, _, _ = run(m)
        m.opts["act_quant"] = "q8"                                                  # fixture weights are F32: no effect at all
        np.testing.assert_array_equal(run(m)[0], base)


class DumpFormatTests(unittest.TestCase):
    def test_manifest_and_file_sizes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            run(model(), dump_dir=d)
            dump = read_dump(d)
            names = (pathlib.Path(d) / "manifest.txt").read_text().splitlines()
        T, n_embd = len(TOKENS), model().hp.n_embd
        self.assertEqual(dump["attn_norm-0"][1], (n_embd, T, 1, 1))                   # ggml ne order: ne0 = features
        self.assertEqual(dump["ffn_moe_topk-0"][0], "i32")
        self.assertEqual(dump["ffn_moe_topk-0"][1], (2, T, 1, 1))
        self.assertEqual(dump["result_output"][1], (128, 1, 1, 1))                    # last token only, as llama.cpp
        self.assertEqual(dump["Kcur-3"][1], (32, T, 1, 1))
        self.assertEqual(dump["Kcur-3#2"][1], (32, 1, T, 1))                          # llama.cpp names Kcur twice: proj, then rope'd
        self.assertEqual(dump["Vcur-3#2"][1], (32, 1, T, 1))
        self.assertIn("norm-3#4", dump)                                               # four RMS norms in the attention block
        self.assertEqual(len(names), len({n.split()[0] for n in names}))


class MtpTests(unittest.TestCase):
    def test_nextn_block_shapes_and_causality(self):
        m = model()
        _, h, _ = run(m)
        toks = TOKENS[1:]
        logits, st = m.mtp_forward(toks, h[:-1], pos0=1)
        self.assertEqual(logits.shape, (len(toks), 128))
        self.assertEqual(st.layers[4].k.shape[0], len(toks))
        logits2, _ = m.mtp_forward(toks[:5], h[:5], pos0=1)
        np.testing.assert_allclose(logits2, logits[:5], atol=1e-4)                    # causal over its own KV

    def test_no_nextn_means_no_draft(self):
        with self.assertRaises(AssertionError):
            model("nonextn", nextn=False).mtp_forward([1], np.zeros((1, 64), F32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
