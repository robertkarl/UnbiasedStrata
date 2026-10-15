"""Small deterministic Muse model for native/reference parity, including SWA wraparound.

Usage: PYTHONPATH=third_party/llama-muse/gguf-py python tools/muse_synth_gguf.py METADATA.gguf OUT.gguf
The real tokenizer is retained; weights are random, not extracted from the model.
"""
import argparse
import pathlib

import gguf
import numpy as np

from gguf_reader import GGUFFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=pathlib.Path)
    parser.add_argument("output")
    parser.add_argument("--expert-width", type=int, default=256)
    parser.add_argument("--iq2-experts", action="store_true")
    parser.add_argument("--mixed-dense", action="store_true", help="exercise Q4_K embeddings and Q6_K projections")
    args = parser.parse_args()
    width = args.expert_width
    if width < 256 or width % 256:
        parser.error("expert width must be a positive multiple of 256")
    source = GGUFFile(args.metadata)
    writer = gguf.GGUFWriter(args.output, "muse-spark")
    # Preserve tokenizer types and special IDs from the real release.
    for key, value in source.metadata.items():
        if not key.startswith("tokenizer."):
            continue
        if isinstance(value, list):
            subtype = gguf.GGUFValueType.get_type(value[0])
            writer.add_key_value(key, value, gguf.GGUFValueType.ARRAY, subtype)
        elif isinstance(value, int) and not isinstance(value, bool):
            writer.add_uint32(key, value)
        else:
            writer.add_key_value(key, value, gguf.GGUFValueType.get_type(value))
    values = {
        "block_count": 4, "context_length": 2048, "embedding_length": 512,
        "feed_forward_length": width, "expert_feed_forward_length": width,
        "moe_latent_size": width, "expert_count": 8, "expert_used_count": 2,
        "attention.head_count": 8, "attention.head_count_kv": 2,
        "attention.key_length": 64, "attention.value_length": 64,
        "attention.sliding_window": 4, "expert_gating_func": 2,
    }
    for k, v in values.items():
        writer.add_uint32("muse-spark." + k, v)
    for k, v in {
        "attention.layer_norm_rms_epsilon": 1e-5, "post_norm_epsilon": 1e-8,
        "rope.freq_base": 500000., "rope.freq_base_swa": 500000.,
        "logit_scale": 0.1767766923, "final_logit_softcapping": 20., "expert_weights_scale": 1.,
    }.items():
        writer.add_float32("muse-spark." + k, v)
    writer.add_bool("muse-spark.expert_weights_norm", True)
    writer.add_array("muse-spark.attention.sliding_window_pattern", [True, False, True, True])
    rng = np.random.default_rng(1729)

    def tensor(name, shape, quantized=False, bias=False):
        if args.mixed_dense and (name == "token_embd.weight" or name == "output.weight" or
                                 name.endswith(("attn_q.weight", "attn_output.weight"))):
            embedding = name == "token_embd.weight"
            block_bytes = 144 if embedding else 210
            raw = rng.integers(0, 256, (*shape[:-1], shape[-1] // 256 * block_bytes), dtype=np.uint8)
            blocks = raw.reshape(-1, block_bytes)
            half = np.frombuffer(np.float16(.0001 if embedding else .00005).tobytes(), dtype=np.uint8)
            if embedding:
                blocks[:, :2] = half
                blocks[:, 2:4] = half
            else:
                blocks[:, 208:210] = half
            writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.Q4_K if embedding else gguf.GGMLQuantizationType.Q6_K)
            return
        if args.iq2_experts and name.endswith("_exps.weight"):
            # Valid random IQ2 blocks exercise the real decoder without relying on
            # a Python IQ2 quantizer (gguf-py only implements its dequantizer).
            raw = rng.integers(0, 256, (*shape[:-1], shape[-1] // 256 * 66), dtype=np.uint8)
            raw.reshape(-1, 66)[:, :2] = np.frombuffer(np.float16(.001).tobytes(), dtype=np.uint8)
            writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.IQ2_XXS)
            return
        if len(shape) == 1:
            x = rng.normal(0 if bias else 1, .1, shape).astype(np.float32)
        else:
            x = (rng.standard_normal(shape, dtype=np.float32) / np.sqrt(shape[-1])).astype(np.float32)
        if quantized:
            writer.add_tensor(name, gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q8_0), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, x)

    vocab = len(source.metadata["tokenizer.ggml.tokens"])
    tensor("token_embd.weight", (vocab, 512), True)
    tensor("output.weight", (vocab, 512), True)
    tensor("output_norm.weight", (512,))
    for layer in range(4):
        p = f"blk.{layer}."
        for name in ("attn_norm", "attn_resid_alpha", "attn_resid_beta", "ffn_norm", "post_ffw_norm", "ffn_resid_alpha", "ffn_resid_beta"):
            tensor(p + name + ".weight", (512,))
        for name in ("attn_q_norm", "attn_k_norm"):
            tensor(p + name + ".weight", (64,))
        for name in ("attn_q", "attn_output", "attn_gate"):
            tensor(p + name + ".weight", (512, 512), True)
        for name in ("attn_k", "attn_v"):
            tensor(p + name + ".weight", (128, 512), True)
        tensor(p + "ffn_gate_inp.weight", (8, 512))
        tensor(p + "exp_probs_b.bias", (8,), bias=True)
        tensor(p + "ffn_latent_down.weight", (width, 512), True)
        tensor(p + "ffn_latent_up.weight", (512, width), True)
        for name in ("ffn_pre_exp_norm", "ffn_post_exp_norm"):
            tensor(p + name + ".weight", (width,))
        for name in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tensor(p + name + ".weight", (8, width, width), True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
