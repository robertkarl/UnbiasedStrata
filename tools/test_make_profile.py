"""Tests for tools/make_profile.py, without a model: synthetic profiles and routing traces.

    python -m unittest tools.test_make_profile
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402

NE = 4   # a tiny "experts per layer"; the layer count is fixed at MP.N_LAYER


def write_trace(path, records):
    """One record per (layer, [expert ids]); the reader skips k weights after the k ids."""
    with open(path, "wb") as f:
        for layer, experts in records:
            f.write(struct.pack("<ii", layer, len(experts)))
            f.write(struct.pack("<%di" % len(experts), *experts))
            f.write(struct.pack("<%df" % len(experts), *([1.0] * len(experts))))


def full_base():
    """Every pair, in an order that differs from rank_profile's fill (e-major then layer)."""
    return [(layer, e) for layer in range(MP.N_LAYER) for e in range(NE)]


class RankProfile(unittest.TestCase):
    def test_a_complete_base_hides_the_trace(self):
        # the shipped profile's situation: it already ranks every pair, so the trace cannot move anything
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked, base)
        self.assertEqual(counts["trace"], 0)
        self.assertEqual(counts["base"], MP.N_LAYER * NE)

    def test_reorder_puts_the_traces_first(self):
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE, reorder=True)
        self.assertEqual(ranked[:2], [(5, 1), (6, 2)])
        self.assertEqual(set(ranked), set(base))                 # still every pair, once
        self.assertEqual(len(ranked), len(base))
        self.assertEqual(counts["trace"], 2)
        self.assertEqual(counts["base"], MP.N_LAYER * NE - 2)    # the base fills the rest, minus the two

    def test_frequency_and_tie_break_order_the_trace(self):
        freq = defaultdict(int, {(1, 0): 1, (2, 0): 3, (3, 0): 3})
        ranked, _ = MP.rank_profile([], freq, NE, no_base=True)  # (2,0)/(3,0) tie -> the smaller pair first
        self.assertEqual(ranked[:3], [(2, 0), (3, 0), (1, 0)])

    def test_default_appends_only(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(0, 0): 5, (1, 3): 9})
        ranked, _ = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked[:3], [(0, 0), (0, 1), (1, 3)])   # base first; (0,0) already there, not moved

    def test_no_base_uses_the_traces_then_the_fill(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(1, 3): 9, (0, 0): 5})
        ranked, counts = MP.rank_profile(base, freq, NE, no_base=True)
        self.assertEqual(ranked[:2], [(1, 3), (0, 0)])
        self.assertEqual(counts["base"], 0)
        self.assertEqual(len(ranked), MP.N_LAYER * NE)

    def test_reorder_without_traces_is_the_default(self):
        base = full_base()
        self.assertEqual(MP.rank_profile(base, defaultdict(int), NE, reorder=True)[0], base)


class Main(unittest.TestCase):
    def run_main(self, argv):
        old, sys.argv = sys.argv, ["make_profile.py"] + argv
        try:
            MP.main()
        finally:
            sys.argv = old

    def test_end_to_end_reorder_over_a_complete_base(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.bin"
            out = Path(d) / "out.bin"
            trace = Path(d) / "t.bin"
            MP.write_profile(str(base), full_base(), NE)
            write_trace(str(trace), [(5, [1]), (6, [2]), (6, [2]), (6, [2])])
            self.run_main([str(trace), "--base", str(base), "--reorder", "--n-expert", str(NE), "--out", str(out)])
            ranked = MP.read_profile(str(out), NE)
            self.assertEqual(ranked[:2], [(6, 2), (5, 1)])       # most frequent first
            self.assertEqual(set(ranked), set(full_base()))
            # and without --reorder it is the base's order, untouched
            self.run_main([str(trace), "--base", str(base), "--n-expert", str(NE), "--out", str(out)])
            self.assertEqual(MP.read_profile(str(out), NE), full_base())

    def test_text_trace_and_the_layer_count_from_the_flag(self):
        """Qwen3.6 (step 4): a llama.cpp route trace (`<layer> <e0> ...` lines, `#` headers) at --n-layer 40."""
        import gzip
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.bin"
            routes = Path(d) / "routes.txt"
            routes.write_text("# prompt a phase prefill\n3 1 2\n3 1 0\n\n# prompt a phase decode\n39 3 1\n3 1 2\n")
            self.run_main(["--no-base", "--n-layer", "40", "--n-expert", str(NE), "--text-trace", str(routes),
                           "--out", str(out)])
            self.assertEqual(MP.profile_header(str(out))[1:3], (40, NE))
            ranked = MP.read_profile(str(out), NE, 40)
            self.assertEqual(ranked[:4], [(3, 1), (3, 2), (3, 0), (39, 1)])      # by count, then the smaller pair
            self.assertEqual(len(ranked), 40 * NE)
            self.assertEqual(len(set(ranked)), 40 * NE)
            # a gzipped trace reads the same; a line outside the geometry is an error, not a skipped pair
            gz = Path(d) / "routes.txt.gz"
            with gzip.open(gz, "wt") as f:
                f.write(routes.read_text())
            self.assertEqual(MP.read_text_trace(gz, NE, 40), MP.read_text_trace(routes, NE, 40))
            routes.write_text("40 1 2\n")
            with self.assertRaisesRegex(SystemExit, "layer 40"):
                MP.read_text_trace(routes, NE, 40)
            routes.write_text("3 1 %d\n" % NE)
            with self.assertRaisesRegex(SystemExit, "outside 40 x %d" % NE):
                MP.read_text_trace(routes, NE, 40)
            routes.write_text("3 one 2\n")
            with self.assertRaisesRegex(SystemExit, "not a route line"):
                MP.read_text_trace(routes, NE, 40)

    def test_base_header_decides_the_layer_count(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.bin"
            out = Path(d) / "out.bin"
            MP.write_profile(str(base), [(l, e) for l in range(40) for e in range(NE)], NE, 40)
            self.run_main(["--base", str(base), "--n-expert", str(NE), "--out", str(out)])
            self.assertEqual(MP.profile_header(str(out))[1:3], (40, NE))
            with self.assertRaisesRegex(SystemExit, "ranks 40 layers, --n-layer says 48"):
                self.run_main(["--base", str(base), "--n-layer", "48", "--n-expert", str(NE), "--out", str(out)])
            with self.assertRaisesRegex(SystemExit, "ranks %d experts a layer, --n-expert says 512" % NE):
                self.run_main(["--base", str(base), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
