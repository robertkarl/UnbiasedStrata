"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order: the base profile's ranking (default: the shipped data/expert-profile.bin), then the pairs your routing
traces used, most frequent first, then every pair still missing, interleaved across the layers.  The base is only
*appended to*, never reordered - and the shipped profile already ranks all 24,576 pairs, so with it a trace is a
no-op.  `--reorder` ranks the traces first and lets the base fill the rest, which is how a workload's trace decides
the top; `--no-base` drops the base entirely.

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--reorder] [--out PATH]
                                 [--n-expert 256]      (a pruned model: GSQ-RCO Coder keeps 256 of 512)
                                 [--n-layer 40]        (Qwen3.6-35B-A3B: 40 trunk layers x 256 experts)
                                 [--text-trace routes.txt ...]

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.

The layer count: `--n-layer` (default 48, Flash-Next's); with `--base`, the base profile's header decides and a
differing `--n-layer` is an error.  `--text-trace` reads llama.cpp eval-callback route traces (one line per token
per layer, `<layer> <e0> ... <e(k-1)>`, `#` lines skipped, `.gz` accepted): the Qwen3.6 E0 trace of
regina/analysis/qwen36-35b-single-gpu-ladder/routes/ is that format.
"""
import argparse
import gzip
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
MAGIC, VERSION = b"STRP", 1


def profile_header(path):
    """(version, n_layer, n_expert, slots, n_ranked) of a profile file."""
    blob = Path(path).read_bytes()[:24]
    if blob[:4] != MAGIC or len(blob) < 24:
        raise SystemExit(f"{path}: not a Strata profile")
    return struct.unpack_from("<5I", blob, 4)


def read_profile(path, n_expert=N_EXPERT, n_layer=N_LAYER):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (n_layer, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {n_layer}x{n_expert}")
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def read_trace(path, n_expert=N_EXPERT, n_layer=N_LAYER):
    """(layer, k, k expert ids, k weights) records, as `--dump-routing` writes them."""
    blob = Path(path).read_bytes()
    off, freq = 0, defaultdict(int)
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < n_layer and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k                                    # the ids and the weights
    return freq


def read_text_trace(path, n_expert=N_EXPERT, n_layer=N_LAYER):
    """llama.cpp eval-callback route traces: `<layer> <e0> ... <e(k-1)>` per token per layer; `#` headers skipped.
    A line whose layer or expert is out of range, or that is not integers, is an error (a trace of another model
    would otherwise rank a plausible subset of this one's pairs)."""
    freq = defaultdict(int)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            try:
                ids = [int(x) for x in line.split()]
            except ValueError:
                raise SystemExit(f"{path}:{n}: not a route line: {line.rstrip()!r}")
            layer, experts = ids[0], ids[1:]
            if not experts or not 0 <= layer < n_layer or any(not 0 <= e < n_expert for e in experts):
                raise SystemExit(f"{path}:{n}: layer {layer} / experts {experts} outside {n_layer} x {n_expert}")
            for e in experts:
                freq[(layer, e)] += 1
    return freq


def write_profile(path, ranked, n_expert=N_EXPERT, n_layer=N_LAYER):
    table = [[-1] * n_expert for _ in range(n_layer)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layer, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layer):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def rank_profile(base_pairs, trace_freq, n_expert=N_EXPERT, no_base=False, reorder=False, n_layer=N_LAYER):
    """The ranked (layer, expert) list and the counts it was built from (`base`, `trace`, `fill`).

    Default: the base's order, then the traces' pairs most frequent first, then the fill.  With `reorder` (or
    `no_base`) the traces' pairs come first, so a base that already ranks every pair no longer hides them; the base
    (unless `no_base`) and then the fill follow.  A base is never reordered against itself - it only ever fills."""
    ranked, seen, counts = [], set(), defaultdict(int)

    def take(pairs, key):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)
                counts[key] += 1

    trace_pairs = [p for p, _ in sorted(trace_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    if no_base or reorder:
        take(trace_pairs, "trace")
    if not no_base:
        take(base_pairs, "base")
    if not (no_base or reorder):
        take(trace_pairs, "trace")
    take(((layer, e) for e in range(n_expert) for layer in range(n_layer)), "fill")
    return ranked, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"), help="ranking to keep first")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--reorder", action="store_true",
                    help="rank the traces' pairs before the base's (the base, then the fill, follow); needed with a "
                         "base that already ranks every pair, where --base alone cannot change the order")
    # no default: the old one was data/expert-profile.bin, the Flash-Next profile UPSTREAM.md pins by hash, so a
    # bare run overwrote it
    ap.add_argument("--out", required=True, help="where to write the profile")
    ap.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default 512)")
    ap.add_argument("--n-layer", type=int, default=None,
                    help="trunk layers (default 48; with --base, the base's header decides and this must agree)")
    ap.add_argument("--text-trace", action="append", default=[], metavar="ROUTES",
                    help="a llama.cpp eval-callback route trace (`<layer> <e0> ... <e7>` lines; .gz accepted)")
    a = ap.parse_args()

    ne = a.n_expert
    nl = a.n_layer
    if not a.no_base:
        _, base_nl, base_ne, _, _ = profile_header(a.base)
        if nl is not None and nl != base_nl:
            raise SystemExit(f"{a.base} ranks {base_nl} layers, --n-layer says {nl}")
        if base_ne != ne:
            raise SystemExit(f"{a.base} ranks {base_ne} experts a layer, --n-expert says {ne}")
        nl = base_nl
    if nl is None:
        nl = N_LAYER
    freq = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, ne, nl).items():
            freq[p] += c
    for t in a.text_trace:
        for p, c in read_text_trace(t, ne, nl).items():
            freq[p] += c
    base_pairs = [] if a.no_base else read_profile(a.base, ne, nl)
    ranked, counts = rank_profile(base_pairs, freq, ne, no_base=a.no_base, reorder=a.reorder, n_layer=nl)
    write_profile(a.out, ranked, ne, nl)
    assert read_profile(a.out, ne, nl) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {nl} x {ne}, {len(ranked)} ranked pairs ({counts['base']} from the base, "
          f"{counts['trace']} from the traces, {counts['fill']} filled in)")


if __name__ == "__main__":
    main()
