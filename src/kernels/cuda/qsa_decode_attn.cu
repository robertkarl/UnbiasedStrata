// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int CHUNK = 64;        // cells per SUB-CHUNK: the unit of the score / softmax / value phases below
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
// `G`, the query heads per KV head, is a template parameter of every kernel: 12 for Flash-Next (24 / 2), 8 for
// Qwen3.6 (16 / 2), 2 for the synthetic fixtures (4 / 2); `qsa_attn_group` picks it from the geometry.

// THE SPLIT-K BOUND (docs/QWEN36-35B-A3B.md 3.3 item 5).  A block used to own exactly one 64-cell chunk, so the
// partials grew with the capacity: 64.5 MiB per query row at 262,144 cells and 16 heads.  Now the grid has at
// most kMaxChunks blocks per KV head and a block walks its SPAN of the selection in 64-cell sub-chunks with the
// usual running (m, l, acc) rescale, so the partials are bounded (kMaxChunks x n_head x 258 floats: 2.0 MiB per
// row at 16 heads) whatever the capacity.  The span is computed ON THE DEVICE from this token's width, not from
// the capacity: a short selection in a long state still spreads over one chunk per 64 cells (kMaxChunks x 64 =
// 8,192 cells before any block takes a second sub-chunk), so decode at a short depth keeps its parallelism, and a
// capacity up to 8,192 is the old kernel exactly - one sub-chunk per block, the same grid, the same arithmetic.
constexpr int kMaxChunks = 128;

// The bound applies to DENSE shapes only.  A sparse (Flash-Next) shape keeps one block per 64 cells at any
// capacity - its states cap at 2,051, but the MTP drafter's ring runs this kernel at its --mtp-window (32,768 by
// default, 512 blocks) and must keep its previous grid, scratch and summation order bit for bit.
__host__ __device__ __forceinline__ int chunks_for(long long cap, bool dense) {
    const long long c = (cap + CHUNK - 1) / CHUNK;
    return (int) (dense && c > kMaxChunks ? kMaxChunks : c);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__device__ __forceinline__ void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(base);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = __half2float(__ushort_as_half(blk->d));
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE, int G>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    if (ids != nullptr) ids += (size_t) blockIdx.z * (size_t) cap;   // null: the identity (dense attention)
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB at G 12: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    __shared__ float srun_m[G], srun_l[G], sfac[G];   // the running max / sum per head, and this sub-chunk's rescale
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    // this block's span: the selection in ceil(n_ids / 64) sub-chunks, spread over the n_chunks blocks
    const int span = ((((n_ids + CHUNK - 1) / CHUNK) + n_chunks - 1) / n_chunks) * CHUNK;
    const int begin = chunk * span;
    const int end = min(begin + span, n_ids);
    const int slot = kvh * n_chunks + chunk;
    if (begin >= end) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < G) { srun_m[t] = -FLT_MAX; srun_l[t] = 0.0f; }
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c0 = begin; c0 < end; c0 += CHUNK) {
        const int n_here = min(CHUNK, end - c0);
        __syncthreads();   // the previous sub-chunk's value loop is done with sp and srow
        if (t < CHUNK) {
            long long r = -1;
            if (t < n_here) {
                const int cell = ids != nullptr ? ids[c0 + t] : c0 + t;
                const long long page = (long long) p.page_table[cell / page_size];
                // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
                // (score -FLT_MAX, weight 0) instead of being read from before the pool.
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            srow[t] = r;
        }
        __syncthreads();
        // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
        for (int c = warp; c < CHUNK; c += WARPS) {
            if (c >= n_here || srow[c] < 0) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8];
            load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                          k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                s = warp_sum(s);
                if (lane == 0) sp[h][c] = s * scale;
            }
        }
        __syncthreads();
        // per-head sub-chunk max and exp-sum, folded into the running pair: warp w handles heads w and w+8.
        // On the first sub-chunk the running max is -FLT_MAX, the rescale is 0 and the probabilities are
        // exp(s - m_chunk): the single-chunk kernel's arithmetic, bit for bit.
        for (int h = warp; h < G; h += WARPS) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float mc = warp_max(fmaxf(a, b));
            const float mr = srun_m[h];
            const float m = fmaxf(mr, mc);
            const float f = mr == -FLT_MAX ? 0.0f : __expf(mr - m);
            const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
            const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
            sp[h][lane] = ea;
            sp[h][lane + 32] = eb;
            const float l = warp_sum(ea + eb);
            if (lane == 0) { srun_m[h] = m; srun_l[h] = fmaf(srun_l[h], f, l); sfac[h] = f; }
        }
        __syncthreads();
        // values: thread t owns dimension t for all G heads.
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= sfac[h];
        for (int c = 0; c < n_here; ++c) {
            if (srow[c] < 0) continue;   // masked above, weight 0
            float v;
            if constexpr (KV_MODE == 0) {
                v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
            } else if constexpr (KV_MODE == 1) {
                const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
                v = (float) p.v_q[srow[c] * HD + t] * sc;
            } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
                constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
                const int b = t / QK4_0;
                const int rem = t % QK4_0;
                const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
                const float d = __half2float(__ushort_as_half(blk->d));
                const int j = rem < 16 ? rem : (rem - 16);
                const uint8_t byte = blk->qs[j];
                const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
                v = (float) nibble * d;
            }
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
        }
    }
    __syncthreads();
    if (t < G) { part_m[slot * G + t] = srun_m[t]; part_l[slot * G + t] = srun_l[t]; }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

// PR #540 (sskver): the same attention with fewer shuffles and less shared-memory traffic, bit for bit the kernel
// above (same operands, same order); +7% prompt speed on a V100, where this kernel reads every prompt chunk.  It
// uses more registers (80 vs 38 on sm_70), so it runs on cards below sm_75 only, and exists only in the experimental
// build (-DSTRATA_EXPERIMENTAL_SM60=ON): the ready-made engine keeps exactly the kernel above.
#if defined(STRATA_EXPERIMENTAL_SM60)
// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5.  Lane l ends holding head head_of_lane(l); lanes l and l^1 agree.
__device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own thread index), per KV format.
template <int KV_MODE>
__device__ __forceinline__ float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + d]));
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = __half2float(__ushort_as_half(blk->d));
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

template <int KV_MODE, int G>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    static_assert(G <= 16, "reduce12 folds 16 lanes' sums");
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    if (ids != nullptr) ids += (size_t) blockIdx.z * (size_t) cap;   // null: the identity (dense attention)
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB at G 12: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's G as float4s.  They used to alias `sq` (dead once
    // the scores were done); a block that walks several sub-chunks reads `sq` again, so they have their own 3-4 KB.
    __shared__ __align__(16) float spt[CHUNK][G];
    __shared__ float srun_m[G], srun_l[G], sfac[G];
    static_assert(G % 4 == 0 || G == 2, "the value loop reads the weights as float4 (or one float2 at G 2)");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int span = ((((n_ids + CHUNK - 1) / CHUNK) + n_chunks - 1) / n_chunks) * CHUNK;
    const int begin = chunk * span;
    const int end = min(begin + span, n_ids);
    const int slot = kvh * n_chunks + chunk;
    if (begin >= end) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < G) { srun_m[t] = -FLT_MAX; srun_l[t] = 0.0f; }
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    const int hd = head_of_lane(lane);
    for (int c0 = begin; c0 < end; c0 += CHUNK) {
        const int n_here = min(CHUNK, end - c0);
        __syncthreads();   // the previous sub-chunk's value loop is done with spt and srow
        if (t < CHUNK) {
            long long r = -1;
            if (t < n_here) {
                const int cell = ids != nullptr ? ids[c0 + t] : c0 + t;
                const long long page = (long long) p.page_table[cell / page_size];
                // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
                // (score -FLT_MAX, weight 0) instead of being read from before the pool.
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            srow[t] = r;
        }
        __syncthreads();
        // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per
        // step (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts
        // the traffic that bounds this phase (G x 16 B per lane per cell, against the FMAs) by NC; every dot
        // product is the same expression as for a single cell, so the scores are unchanged.
        constexpr int NC = 2;   // 4 measured slower on a V100 (the registers cost occupancy)
        for (int i = 0; i < CHUNK / WARPS; i += NC) {
            int cc[NC];
            bool all_ok = true;
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                cc[j] = warp + WARPS * (i + j);
                all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
            }
            if (all_ok) {
                float kk[NC][8];
#pragma unroll
                for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
                float pp[NC][16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                    for (int j = 0; j < NC; ++j)
                        pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                                   kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
#pragma unroll
                    for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                    const float sj = reduce12(pp[j], lane);
                    if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
                }
            } else {
                for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                    const int c = cc[k];
                    if (c >= n_here || srow[c] < 0) {
                        if (lane < G) sp[lane][c] = -FLT_MAX;
                        continue;
                    }
                    float k8[8];
                    load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                    float part[16];
#pragma unroll
                    for (int h = 0; h < G; ++h) {
                        const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                        const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                        part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                                  k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                    }
#pragma unroll
                    for (int h = G; h < 16; ++h) part[h] = 0.0f;
                    const float s = reduce12(part, lane);
                    if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
                }
            }
        }
        __syncthreads();
        // per-head sub-chunk max and exp-sum into the running pair (see attn_chunk_kernel): warp w, heads w, w+8.
        for (int h = warp; h < G; h += WARPS) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float mc = warp_max(fmaxf(a, b));
            const float mr = srun_m[h];
            const float m = fmaxf(mr, mc);
            const float f = mr == -FLT_MAX ? 0.0f : __expf(mr - m);
            const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
            const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
            spt[lane][h] = ea;
            spt[lane + 32][h] = eb;
            const float l = warp_sum(ea + eb);
            if (lane == 0) { srun_m[h] = m; srun_l[h] = fmaf(srun_l[h], f, l); sfac[h] = f; }
        }
        __syncthreads();
        // values: thread t owns dimension t for all G heads.
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= sfac[h];
        // One cell's G weights (broadcast float4 loads) into the G accumulators; the cells are folded in
        // ascending order, so the sums are the plain loop's, bit for bit.
        const auto fold = [&](int c, float v) {
            if constexpr (G == 2) {
                const float2 w = *reinterpret_cast<const float2*>(spt[c]);
                acc[0] = fmaf(w.x, v, acc[0]);  acc[1] = fmaf(w.y, v, acc[1]);
            } else {
                const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
#pragma unroll
                for (int g4 = 0; g4 < G / 4; ++g4) {
                    const float4 w = w4[g4];
                    acc[4 * g4] = fmaf(w.x, v, acc[4 * g4]);          acc[4 * g4 + 1] = fmaf(w.y, v, acc[4 * g4 + 1]);
                    acc[4 * g4 + 2] = fmaf(w.z, v, acc[4 * g4 + 2]);  acc[4 * g4 + 3] = fmaf(w.w, v, acc[4 * g4 + 3]);
                }
            }
        };
        int c = 0;
        // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
        // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
        for (; c + 4 <= n_here; c += 4) {
            const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
            if ((r0 | r1 | r2 | r3) < 0) break;
            const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
            const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
            fold(c, v0);
            fold(c + 1, v1);
            fold(c + 2, v2);
            fold(c + 3, v3);
        }
        for (; c < n_here; ++c) {
            if (srow[c] < 0) continue;   // masked above, weight 0
            fold(c, load_v1<KV_MODE>(p, srow[c], t));
        }
    }
    __syncthreads();
    if (t < G) { part_m[slot * G + t] = srun_m[t]; part_l[slot * G + t] = srun_l[t]; }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
// The single-chunk form of the kernel above: PR #540's body as it was, with G and the null-ids identity.  Every
// sparse (Flash-Next) shape runs THIS one at any capacity, and a dense state of at most kMaxChunks x 64 = 8,192
// cells too, so Flash-Next's Pascal path - the QSA layers at cap 2,051 and the MTP drafter's ring at its window -
// is the old kernel to the instruction (the looped form costs 16-28 registers on sm_61 and 15% on the Q4_0 pool at
// 2,051 cells; capping the registers spilled and cost more).  The looped form above runs only for the dense model's
// states past 8,192 cells.
template <int KV_MODE, int G>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75_single(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    if (ids != nullptr) ids += (size_t) blockIdx.z * (size_t) cap;   // null: the identity (dense attention)
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    static_assert(G == 2 || G % 4 == 0, "the value loop reads the weights as float4 (one float2 at G 2)");
    __shared__ __align__(16) float sq[G][HD];     // 12 KB at G 12: this KV head's query heads (dead once the scores are done)
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's 12 as three float4.  They are first written after the
    // __syncthreads that ends the score phase, the last reader of sq, so they share its storage (64 x 12 of its 256 x 12
    // floats) and the block's shared memory stays what it was before the cell-major layout.
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids != nullptr ? ids[c0 + t] : c0 + t;
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per step
    // (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts the traffic
    // that bounds this phase (24 x 16 B per lane per cell, against the FMAs) by NC; every dot product is the same
    // expression as for a single cell, so the scores are unchanged.
    constexpr int NC = 2;   // 4 measured slower on a V100 (the registers cost occupancy)
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all G heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    // One cell's G weights (broadcast float4 loads) into the G accumulators; the cells are folded in
    // ascending order, so the sums are the plain loop's, bit for bit.
    const auto fold = [&](int c, float v) {
        if constexpr (G == 2) {
            const float2 w = *reinterpret_cast<const float2*>(spt[c]);
            acc[0] = fmaf(w.x, v, acc[0]);  acc[1] = fmaf(w.y, v, acc[1]);
        } else {
            const float4* w4 = reinterpret_cast<const float4*>(spt[c]);
#pragma unroll
            for (int g4 = 0; g4 < G / 4; ++g4) {
                const float4 w = w4[g4];
                acc[4 * g4] = fmaf(w.x, v, acc[4 * g4]);          acc[4 * g4 + 1] = fmaf(w.y, v, acc[4 * g4 + 1]);
                acc[4 * g4 + 2] = fmaf(w.z, v, acc[4 * g4 + 2]);  acc[4 * g4 + 3] = fmaf(w.w, v, acc[4 * g4 + 3]);
            }
        }
    };
    int c = 0;
    // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
    // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

#endif  // STRATA_EXPERIMENTAL_SM60

template <int G>
__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

#if defined(STRATA_EXPERIMENTAL_SM60)
// the current device is below sm_75 (per device: a layer split can mix cards); STRATA_ATTN_PRE75=0 turns PR #540's
// kernel off (A/B)
bool pre75_attn() {
    static const bool off = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e != nullptr && e[0] == '0';
    }();
    static int cc[64] = {};
    int dev = 0;
    if (off) return false;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
}
#define STRATA_ATTN_CHUNK(M, G) (pre75_attn() ? (!dense || n_chunks < kMaxChunks ? attn_chunk_kernel_pre75_single<M, G> : attn_chunk_kernel_pre75<M, G>) : attn_chunk_kernel<M, G>)
#else
#define STRATA_ATTN_CHUNK(M, G) attn_chunk_kernel<M, G>
#endif

int kv_mode_of(const QsaAttnPools& pools) {
    return pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0));
}

template <int G>
void launch_group(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* scratch, float* attn, int64_t n_q, cudaStream_t st) {
    const int kv_mode = kv_mode_of(pools);
    const bool dense = qsa_dense(s);
    const int n_chunks = chunks_for(cap, dense);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK(3, G)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK(2, G)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK(1, G)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
        STRATA_ATTN_CHUNK(0, G)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    attn_merge_kernel<G><<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks,
                                                                                     attn, stride);
}

void refuse(const char* who, const char* why) {
    std::fprintf(stderr, "%s: %s\n", who, why);
    std::exit(1);
}

void run(const char* who, const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
         int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    char why[256];
    const int G = qsa_attn_group(s, why, sizeof why);
    if (G == 0) refuse(who, why);
    if (s.head_dim != HD || cap <= 0 || !scratch || !steps || !pools.page_table || n_q > 65535)
        refuse(who, "unsupported geometry or missing buffers (head_dim 256, cap > 0, scratch, steps, page_table, n_q <= 65535)");
    const int kv_mode = kv_mode_of(pools);
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool))))
        refuse(who, "incomplete KV pools");
    cudaStream_t st = (cudaStream_t) stream;
    if (G == 12) launch_group<12>(q, pools, ids, steps, cap, s, scratch, attn, n_q, st);
    else if (G == 8) launch_group<8>(q, pools, ids, steps, cap, s, scratch, attn, n_q, st);
    else launch_group<2>(q, pools, ids, steps, cap, s, scratch, attn, n_q, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) refuse(who, cudaGetErrorString(e));
}

__global__ void identity_ids_kernel(int32_t* __restrict__ ids, const int32_t* __restrict__ steps, long long cap) {
    const int q = blockIdx.y;
    const int width = __ldg(steps + (size_t) q * kStepCount + kStepWidth);
    for (long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x; j < width; j += (long long) gridDim.x * blockDim.x)
        ids[(size_t) q * cap + j] = (int32_t) j;
}

}  // namespace

int qsa_attn_group(const QsaShapes& s, char* why, size_t why_len) {
    const int64_t G = s.n_head_kv > 0 ? s.n_head / s.n_head_kv : 0;
    if (s.n_head_kv > 0 && G * s.n_head_kv == s.n_head && (G == 2 || G == 8 || G == 12)) return (int) G;
    if (why && why_len)
        std::snprintf(why, why_len, "n_head %lld over n_head_kv %lld is %lld query heads per KV head; the decode "
                      "attention is built for 2, 8 (Qwen3.6: 16 / 2) and 12 (Flash-Next: 24 / 2)",
                      (long long) s.n_head, (long long) s.n_head_kv, (long long) G);
    return 0;
}

bool qsa_attn_group_check(const QsaShapes& s, int G, char* why, size_t why_len) {
    if (G != 2 && G != 8 && G != 12) {
        if (why && why_len) std::snprintf(why, why_len, "G = %d: the decode attention is built for 2, 8 and 12", G);
        return false;
    }
    if (s.n_head != (int64_t) G * s.n_head_kv) {
        if (why && why_len)
            std::snprintf(why, why_len, "G = %d refused: n_head %lld is not %d x n_head_kv %lld (that would be %lld "
                          "query heads per KV head)", G, (long long) s.n_head, G, (long long) s.n_head_kv,
                          s.n_head_kv > 0 ? (long long) (s.n_head / s.n_head_kv) : 0ll);
        return false;
    }
    return true;
}

int64_t qsa_decode_attn_chunks(int64_t cap, const QsaShapes& s) { return chunks_for(cap, qsa_dense(s)); }

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    return (uint64_t) chunks_for(cap, qsa_dense(s)) * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    run("qsa_decode_attn_batch", q, pools, ids, steps, cap, s, scratch, attn, n_q, stream);
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    run("qsa_decode_attn", q, pools, ids, step, cap, s, scratch, attn, 1, stream);
}

void qsa_identity_ids(int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, void* stream) {
    if (n_q <= 0 || cap <= 0) return;
    if (!ids || !steps || n_q > 65535) {
        std::fprintf(stderr, "qsa_identity_ids: missing buffers or n_q > 65535\n");
        std::exit(1);
    }
    const unsigned bx = (unsigned) std::min<int64_t>((cap + 255) / 256, 64);
    identity_ids_kernel<<<dim3(bx, (unsigned) n_q), 256, 0, (cudaStream_t) stream>>>(ids, steps, (long long) cap);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_identity_ids: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
