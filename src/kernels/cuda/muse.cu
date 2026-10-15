#include "strata/kernels/muse.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace strata::kernels {
namespace {
__global__ void multiply(float* x, const float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= w[i];
}
__global__ void gate_output(float* x, const float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}
__global__ void residual_add(float* x, const float* y, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(__fmul_rn(x[i], a[i]), __fmul_rn(y[i], b[i]));
}
__global__ void swiglu(float* g, const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}
__global__ void accumulate(float* sum, const float* x, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sum[i] += w * x[i];
}
__global__ void rope(float* x, int n, int width, int pos, float base) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= n / 2) return;
    int i = pair * 2;
    float theta = pos * powf(base, -float(i % width) / width);
    float c = cosf(theta), s = sinf(theta), a = x[i], b = x[i + 1];
    x[i] = a * c - b * s;
    x[i + 1] = a * s + b * c;
}
__global__ void logit_scale(float* x, int n, float scale, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] * scale / cap);
}
__global__ void attention_scores(const float* q, const float* k, float* scores,
                                  int kv_heads, int width, int count, int group) {
    int h = blockIdx.y, t = blockIdx.x, lane = threadIdx.x;
    float dot = 0;
    for (int d = lane; d < width; d += 32)
        dot += q[h * width + d] * k[(t * kv_heads + h / group) * width + d];
    for (int shift = 16; shift; shift >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, shift);
    if (!lane) scores[h * count + t] = dot / sqrtf(float(width));
}
__global__ void softmax(float* scores, int count) {
    __shared__ float scratch[256];
    float* row = scores + blockIdx.x * count;
    int tid = threadIdx.x;
    float m = -INFINITY;
    for (int i = tid; i < count; i += 256) m = fmaxf(m, row[i]);
    scratch[tid] = m;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    m = scratch[0];
    float sum = 0;
    for (int i = tid; i < count; i += 256) {
        row[i] = expf(row[i] - m);
        sum += row[i];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    sum = scratch[0];
    for (int i = tid; i < count; i += 256) row[i] /= sum;
}
__global__ void attention_values(const float* scores, const float* v, float* out,
                                  int heads, int kv_heads, int width, int count) {
    int h = blockIdx.x, d = threadIdx.x;
    if (d >= width) return;
    int kh = h / (heads / kv_heads);
    float sum = 0;
    for (int t = 0; t < count; ++t)
        sum += scores[h * count + t] * v[(t * kv_heads + kh) * width + d];
    out[h * width + d] = sum;
}
} // namespace

void muse_mul(float* x, const float* w, int n, void* s) {
    multiply<<<(n+255)/256,256,0,(cudaStream_t)s>>>(x,w,n);
}
void muse_gate(float* x, const float* g, int n, void* s) {
    gate_output<<<(n+255)/256,256,0,(cudaStream_t)s>>>(x,g,n);
}
void muse_residual(float* x, const float* y, const float* a, const float* b, int n, void* s) {
    residual_add<<<(n+255)/256,256,0,(cudaStream_t)s>>>(x,y,a,b,n);
}
void muse_swiglu(float* g, const float* u, int n, void* s) {
    swiglu<<<(n+255)/256,256,0,(cudaStream_t)s>>>(g,u,n);
}
void muse_accumulate(float* sum, const float* x, float w, int n, void* s) {
    accumulate<<<(n+255)/256,256,0,(cudaStream_t)s>>>(sum,x,w,n);
}
void muse_rope(float* x, int heads, int width, int pos, float base, void* s) {
    rope<<<(heads*width/2+255)/256,256,0,(cudaStream_t)s>>>(x,heads*width,width,pos,base);
}
void muse_logits(float* x, int n, float scale, float cap, void* s) {
    logit_scale<<<(n+255)/256,256,0,(cudaStream_t)s>>>(x,n,scale,cap);
}
void muse_attention(const float* q, const float* k, const float* v, float* scores,
                    float* out, int heads, int kv_heads, int width, int count,
                    int capacity, void* s) {
    (void) capacity;
    attention_scores<<<dim3(count,heads),32,0,(cudaStream_t)s>>>(q,k,scores,kv_heads,width,count,heads/kv_heads);
    softmax<<<heads,256,0,(cudaStream_t)s>>>(scores,count);
    attention_values<<<heads,((width+31)/32)*32,0,(cudaStream_t)s>>>(scores,v,out,heads,kv_heads,width,count);
}
} // namespace strata::kernels
