#pragma once

namespace strata::kernels {

// All arrays are device F32. Operations enqueue on the supplied CUDA stream.
void muse_mul(float* x, const float* weight, int n, void* stream);
void muse_gate(float* x, const float* gate, int n, void* stream);
void muse_residual(float* residual, const float* branch, const float* alpha,
                   const float* beta, int n, void* stream);
void muse_swiglu(float* gate, const float* up, int n, void* stream);
void muse_accumulate(float* sum, const float* x, float weight, int n, void* stream);
void muse_rope(float* x, int heads, int width, int pos, float base, void* stream);
void muse_logits(float* x, int n, float scale, float cap, void* stream);
// KV is a ring of `capacity` rows, each kv_heads*width. `count` rows are valid.
// Attention is independent of their physical order, provided K and V share it.
void muse_attention(const float* q, const float* k, const float* v, float* scores,
                    float* out, int heads, int kv_heads, int width, int count,
                    int capacity, void* stream);

} // namespace strata::kernels
