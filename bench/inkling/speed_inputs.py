"""Generate fixed comma-separated token IDs for the upstream Inkling timing commands."""
import argparse
from pathlib import Path
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    corpus = encode("\n".join(f"Record {i}: the shipment of {i*7%311} crates left depot {i%53} "
                              f"at hour {i%24} and arrived intact." for i in range(16000)))
    def save(name, ids):
        (args.directory / (name + ".ids")).write_text(",".join(map(str, ids)) + "\n")

    suffix = encode("Write a short guide to auditing this ledger, with worked examples.")
    for depth in (47, 4096, 63981, 127910, 340000):
        save(f"perf-{depth}", corpus[:depth - len(suffix)] + suffix)

    code = encode("\n".join(f"def scale_{i}(value):\n    return value * {i+1}\n" for i in range(80)))
    for name, depth, prefix, question in (
        ("prose", 42, corpus, "Write a story about a lighthouse keeper who finds a message in a bottle."),
        ("code", 51, code, "Write a Python red-black tree with insertion, deletion and invariant checks."),
        ("edit", 512, corpus, "Rewrite the shipment records above, changing each occurrence of crates to boxes."),
        ("copy", 512, code, "Copy the functions above exactly, preserving their indentation."),
        ("deep", 63981, corpus, "Summarize the shipment ledger with a Python function."),
    ):
        end = encode(question)
        save("draft-" + name, prefix[:depth-len(end)] + end)


if __name__ == "__main__":
    main()
