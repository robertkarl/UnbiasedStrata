#include "model.hpp"
#include "strata/core/muse_geometry.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/muse.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <list>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace strata::muse {
namespace {
namespace K = strata::kernels;
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    size_t bytes = 0;
    Buffer() = default;
    explicit Buffer(size_t n) : bytes(n) { if (n) cuda_check(cudaMalloc(&p, n)); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(Buffer&& b) noexcept : p(std::exchange(b.p, nullptr)), bytes(b.bytes) {}
    Buffer& operator=(Buffer&& b) noexcept {
        if (this != &b) { if (p) cudaFree(p); p = std::exchange(b.p, nullptr); bytes = b.bytes; }
        return *this;
    }
    Buffer(const Buffer&) = delete;
    float* f() const { return static_cast<float*>(p); }
};
struct Tensor {
    const TensorInfo* info = nullptr;
    const uint8_t* source = nullptr;
    Buffer gpu;
    int type = 0;
    int in = 0, out = 0;
};
struct CachedExpert {
    Buffer gpu;
    size_t up_offset, down_offset;
    std::list<int>::iterator age;
};
}

struct Model::Impl {
    GgufModel files;
    core::MuseGeometry g;
    int context;
    int next_position = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::map<std::string, Tensor> tensors;
    std::vector<Buffer> keys, values;
    Buffer residual, normed, q, k, v, gate, attention, branch, latent, expert_gate,
           expert_up, expert_out, sum, route, logits, scores, quant;
    std::map<int, CachedExpert> cache;
    std::list<int> lru;
    size_t cache_limit = 0, cache_used = 0;
    uint64_t hits = 0, misses = 0, copied = 0;

    Impl(const std::string& path, int ctx, size_t cache_mib)
        : files(GgufModel::open(path)), g(core::MuseGeometry::read(files.meta())), context(ctx) {
        if (ctx < 1 || ctx > g.context || g.head_dim > 1024 || g.embedding % 256 ||
            g.latent % 256 || g.feed_forward % 256 || g.heads > 65535)
            throw std::runtime_error("Muse: unsupported context or kernel dimensions");
        // Validate every tensor before allocating VRAM or touching its payload.
        bind("token_embd.weight", {g.embedding, g.vocab});
        bind("output.weight", {g.embedding, g.vocab});
        bind("output_norm.weight", {g.embedding});
        for (int l = 0; l < g.layers; ++l) {
            const auto p = prefix(l);
            for (const char* s : {"attn_norm.weight", "attn_resid_alpha.weight", "attn_resid_beta.weight",
                                  "ffn_norm.weight", "post_ffw_norm.weight", "ffn_resid_alpha.weight", "ffn_resid_beta.weight"})
                bind(p+s, {g.embedding});
            bind(p+"attn_q_norm.weight", {g.head_dim});
            bind(p+"attn_k_norm.weight", {g.head_dim});
            bind(p+"attn_q.weight", {g.embedding, g.heads*g.head_dim});
            bind(p+"attn_k.weight", {g.embedding, g.kv_heads*g.head_dim});
            bind(p+"attn_v.weight", {g.embedding, g.kv_heads*g.head_dim});
            bind(p+"attn_output.weight", {g.heads*g.head_dim, g.embedding});
            bind(p+"attn_gate.weight", {g.embedding, g.heads*g.head_dim});
            bind(p+"ffn_gate_inp.weight", {g.embedding, g.experts});
            bind(p+"exp_probs_b.bias", {g.experts});
            bind(p+"ffn_latent_down.weight", {g.embedding, g.latent});
            bind(p+"ffn_latent_up.weight", {g.latent, g.embedding});
            bind(p+"ffn_pre_exp_norm.weight", {g.latent});
            bind(p+"ffn_post_exp_norm.weight", {g.latent});
            bind(p+"ffn_gate_exps.weight", {g.latent, g.feed_forward, g.experts});
            bind(p+"ffn_up_exps.weight", {g.latent, g.feed_forward, g.experts});
            bind(p+"ffn_down_exps.weight", {g.feed_forward, g.latent, g.experts});
        }
        cuda_check(cudaStreamCreate(&stream));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Muse: cublasCreate failed");
        cublasSetStream(blas, stream);
        for (auto& [name, t] : tensors) {
            if (t.info->shape.size() == 3 || name == "token_embd.weight") continue;
            if (t.info->shape.size() == 1 || t.type == GGML_TYPE_F16 || t.type == GGML_TYPE_BF16) {
                auto host = floats(t);
                t.type = GGML_TYPE_F32;
                t.gpu = Buffer(host.size()*sizeof(float));
                cuda_check(cudaMemcpy(t.gpu.p, host.data(), t.gpu.bytes, cudaMemcpyHostToDevice));
            } else {
                t.gpu = Buffer(tensor_payload_bytes(*t.info));
                cuda_check(cudaMemcpy(t.gpu.p, t.source, t.gpu.bytes, cudaMemcpyHostToDevice));
            }
        }
        auto f = [](int n) { return Buffer(size_t(n)*sizeof(float)); };
        residual=f(g.embedding); normed=f(g.embedding); q=f(g.heads*g.head_dim);
        k=f(g.kv_heads*g.head_dim); v=f(g.kv_heads*g.head_dim); gate=f(g.heads*g.head_dim);
        attention=f(g.heads*g.head_dim); branch=f(g.embedding); latent=f(g.latent);
        expert_gate=f(g.feed_forward); expert_up=f(g.feed_forward); expert_out=f(g.latent);
        sum=f(g.latent); route=f(g.experts); logits=f(g.vocab);
        scores=Buffer(size_t(g.heads)*context*sizeof(float));
        quant=Buffer(size_t(std::max({g.embedding,g.latent,g.feed_forward,g.heads*g.head_dim}))/32*36);
        for (int l=0; l<g.layers; ++l) {
            size_t n=size_t(g.sliding[l] ? std::min(g.window,context) : context)*g.kv_heads*g.head_dim;
            keys.emplace_back(n*sizeof(float)); values.emplace_back(n*sizeof(float));
        }
        size_t free=0, total=0;
        cuda_check(cudaMemGetInfo(&free,&total));
        const size_t reserve=1024ULL*1024*1024;
        cache_limit=free > reserve ? free-reserve : 0;
        if (cache_mib) cache_limit=std::min(cache_limit,cache_mib*1024*1024);
        size_t largest=0;
        for (int l=0;l<g.layers;++l) largest=std::max(largest,expert_bytes(l));
        if (cache_limit < largest) throw std::runtime_error("Muse: not enough VRAM for one expert after dense weights and KV");
        std::fprintf(stderr,"Muse native: %d layers, residual %d, latent %d, context %d; expert cache %.2f GiB\n",
                     g.layers,g.embedding,g.latent,context,double(cache_limit)/(1ULL<<30));
    }
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
    static std::string prefix(int l) { return "blk."+std::to_string(l)+"."; }
    void bind(const std::string& name, std::initializer_list<int> dims) {
        size_t shard=0;
        const auto* info=files.find(name,&shard);
        if (!info) throw std::runtime_error("Muse: missing tensor "+name);
        std::vector<uint64_t> expected;
        for (int d:dims) expected.push_back(d);
        if (info->shape != expected || !files.in_bounds(*info,shard))
            throw std::runtime_error("Muse: invalid tensor shape or payload: "+name);
        if (info->type != GGML_TYPE_F32 && info->type != GGML_TYPE_F16 && info->type != GGML_TYPE_BF16 &&
            !K::iq_supported(info->type)) throw std::runtime_error("Muse: unsupported tensor type: "+name+" "+info->type_name());
        if (dims.size()==3 && !K::iq_supported(info->type))
            throw std::runtime_error("Muse: experts require supported quantized weights: "+name);
        Tensor t;
        t.info=info; t.source=files.shard(shard).tensor_data(*info); t.type=info->type;
        t.in=expected[0]; t.out=expected.size()>1 ? expected[1] : 1;
        tensors.emplace(name,std::move(t));
    }
    std::vector<float> floats(const Tensor& t) const {
        std::vector<float> host(t.info->elements());
        if (t.info->type==GGML_TYPE_F32) std::memcpy(host.data(),t.source,host.size()*sizeof(float));
        else {
            const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(t.info->type));
            if (!traits->to_float) throw std::runtime_error("Muse: no dequantizer for "+t.info->name);
            traits->to_float(t.source,host.data(),host.size());
        }
        return host;
    }
    float* weight(const std::string& name) { return tensors.at(name).gpu.f(); }
    void copy(float* dst,const float* src,int n) {
        cuda_check(cudaMemcpyAsync(dst,src,size_t(n)*4,cudaMemcpyDeviceToDevice,stream));
    }
    void norm(float* x,const float* w,int width,float eps,int rows=1) {
        K::rms_norm_weighted(x,w,rows,width,eps,stream);
    }
    void mat(const Tensor& t,const void* data,const float* x,float* out) {
        if (t.type==GGML_TYPE_F32) {
            const float one=1,zero=0;
            if (cublasSgemv(blas,CUBLAS_OP_T,t.in,t.out,&one,static_cast<const float*>(data),t.in,x,1,&zero,out,1)!=CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Muse: GEMV failed: "+t.info->name);
        } else {
            K::quantize_q8_1_rows(x,1,t.in,quant.p,stream);
            K::iq_mmvq(t.type,data,quant.p,out,t.in,t.out,1,stream);
        }
    }
    void mat(const std::string& name,const float* x,float* out) {
        const auto& t=tensors.at(name); mat(t,t.gpu.p,x,out);
    }
    size_t slice_bytes(const Tensor& t) const { return tensor_payload_bytes(*t.info)/g.experts; }
    size_t expert_bytes(int l) const {
        size_t n=0;
        for (const char* s:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"})
            n+=slice_bytes(tensors.at(prefix(l)+s));
        return n;
    }
    CachedExpert& expert(int l,int e) {
        const int id=l*g.experts+e;
        auto it=cache.find(id);
        if (it!=cache.end()) {
            ++hits; lru.splice(lru.begin(),lru,it->second.age); return it->second;
        }
        ++misses;
        const size_t bytes=expert_bytes(l);
        // Previous kernels must finish before a slot is freed or overwritten.
        cuda_check(cudaStreamSynchronize(stream));
        // cudaMalloc can charge more VRAM than the tensor payload (allocation
        // granularity). Enforce physical headroom as well as the payload budget.
        size_t free=0,total=0;
        cuda_check(cudaMemGetInfo(&free,&total));
        constexpr size_t reserve=1024ULL*1024*1024;
        while ((cache_used+bytes>cache_limit || free<bytes+reserve) && !lru.empty()) {
            int old=lru.back(); cache_used-=cache.at(old).gpu.bytes; cache.erase(old); lru.pop_back();
            cuda_check(cudaMemGetInfo(&free,&total));
        }
        if (free<bytes+reserve) throw std::runtime_error("Muse: insufficient free VRAM for an expert and safety reserve");
        CachedExpert entry{Buffer(bytes),0,0,{}};
        size_t offset=0;
        for (const char* s:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) {
            const auto& t=tensors.at(prefix(l)+s);
            size_t n=slice_bytes(t);
            if (std::string(s)=="ffn_up_exps.weight") entry.up_offset=offset;
            if (std::string(s)=="ffn_down_exps.weight") entry.down_offset=offset;
            cuda_check(cudaMemcpy(static_cast<uint8_t*>(entry.gpu.p)+offset,t.source+size_t(e)*n,n,cudaMemcpyHostToDevice));
            offset+=n;
        }
        copied+=bytes; cache_used+=bytes; lru.push_front(id); entry.age=lru.begin();
        return cache.emplace(id,std::move(entry)).first->second;
    }
    std::vector<float> forward(int token,int pos) {
        if (token<0 || token>=g.vocab || pos<0 || pos>=context || (pos!=0 && pos!=next_position))
            throw std::runtime_error("Muse: invalid token or non-contiguous position");
        const auto& emb=tensors.at("token_embd.weight");
        std::vector<float> row(g.embedding);
        size_t rowbytes=tensor_payload_bytes(*emb.info)/g.vocab;
        const auto* src=emb.source+size_t(token)*rowbytes;
        if (emb.type==GGML_TYPE_F32) std::memcpy(row.data(),src,rowbytes);
        else ggml_get_type_traits(static_cast<ggml_type>(emb.type))->to_float(src,row.data(),g.embedding);
        cuda_check(cudaMemcpyAsync(residual.p,row.data(),row.size()*4,cudaMemcpyHostToDevice,stream));
        norm(residual.f(),nullptr,g.embedding,g.rms_eps);
        for (int l=0;l<g.layers;++l) {
            const auto p=prefix(l);
            copy(normed.f(),residual.f(),g.embedding);
            norm(normed.f(),weight(p+"attn_norm.weight"),g.embedding,g.rms_eps);
            mat(p+"attn_q.weight",normed.f(),q.f()); mat(p+"attn_k.weight",normed.f(),k.f());
            mat(p+"attn_v.weight",normed.f(),v.f()); mat(p+"attn_gate.weight",normed.f(),gate.f());
            norm(q.f(),weight(p+"attn_q_norm.weight"),g.head_dim,g.rms_eps,g.heads);
            norm(k.f(),weight(p+"attn_k_norm.weight"),g.head_dim,g.rms_eps,g.kv_heads);
            if (g.sliding[l]) {
                K::muse_rope(q.f(),g.heads,g.head_dim,pos,g.rope_base,stream);
                K::muse_rope(k.f(),g.kv_heads,g.head_dim,pos,g.rope_base,stream);
            }
            int capacity=g.sliding[l] ? std::min(g.window,context) : context;
            size_t offset=size_t(pos%capacity)*g.kv_heads*g.head_dim;
            copy(keys[l].f()+offset,k.f(),g.kv_heads*g.head_dim);
            copy(values[l].f()+offset,v.f(),g.kv_heads*g.head_dim);
            K::muse_attention(q.f(),keys[l].f(),values[l].f(),scores.f(),attention.f(),g.heads,g.kv_heads,g.head_dim,
                              std::min(pos+1,capacity),capacity,stream);
            norm(attention.f(),nullptr,g.head_dim,g.rms_eps,g.heads);
            K::muse_gate(attention.f(),gate.f(),g.heads*g.head_dim,stream);
            mat(p+"attn_output.weight",attention.f(),branch.f());
            norm(branch.f(),nullptr,g.embedding,g.post_eps);
            K::muse_residual(residual.f(),branch.f(),weight(p+"attn_resid_alpha.weight"),weight(p+"attn_resid_beta.weight"),g.embedding,stream);
            copy(normed.f(),residual.f(),g.embedding);
            norm(normed.f(),weight(p+"ffn_norm.weight"),g.embedding,g.rms_eps);
            mat(p+"ffn_latent_down.weight",normed.f(),latent.f());
            norm(latent.f(),weight(p+"ffn_pre_exp_norm.weight"),g.latent,g.rms_eps);
            mat(p+"ffn_gate_inp.weight",normed.f(),route.f());
            std::vector<float> probabilities(g.experts),bias(g.experts);
            cuda_check(cudaMemcpyAsync(probabilities.data(),route.p,g.experts*4,cudaMemcpyDeviceToHost,stream));
            cuda_check(cudaMemcpyAsync(bias.data(),weight(p+"exp_probs_b.bias"),g.experts*4,cudaMemcpyDeviceToHost,stream));
            cuda_check(cudaStreamSynchronize(stream));
            std::vector<int> order(g.experts); std::iota(order.begin(),order.end(),0);
            for (float& x:probabilities) x=1.f/(1.f+std::exp(-x));
            std::partial_sort(order.begin(),order.begin()+g.active,order.end(),[&](int a,int b) {
                float aa=probabilities[a]+bias[a],bb=probabilities[b]+bias[b];
                return aa==bb ? a<b : aa>bb;
            });
            float total=0; for (int j=0;j<g.active;++j) total+=probabilities[order[j]];
            total=std::max(total,6.103515625e-5f);
            cuda_check(cudaMemsetAsync(sum.p,0,sum.bytes,stream));
            for (int j=0;j<g.active;++j) {
                auto& ex=expert(l,order[j]); auto* base=static_cast<uint8_t*>(ex.gpu.p);
                mat(tensors.at(p+"ffn_gate_exps.weight"),base,latent.f(),expert_gate.f());
                mat(tensors.at(p+"ffn_up_exps.weight"),base+ex.up_offset,latent.f(),expert_up.f());
                K::muse_swiglu(expert_gate.f(),expert_up.f(),g.feed_forward,stream);
                mat(tensors.at(p+"ffn_down_exps.weight"),base+ex.down_offset,expert_gate.f(),expert_out.f());
                K::muse_mul(expert_out.f(),weight(p+"ffn_post_exp_norm.weight"),g.latent,stream);
                norm(expert_out.f(),nullptr,g.latent,g.post_eps);
                K::muse_accumulate(sum.f(),expert_out.f(),probabilities[order[j]]/total*g.expert_scale,g.latent,stream);
            }
            mat(p+"ffn_latent_up.weight",sum.f(),branch.f());
            K::muse_mul(branch.f(),weight(p+"post_ffw_norm.weight"),g.embedding,stream);
            norm(branch.f(),nullptr,g.embedding,g.post_eps);
            K::muse_residual(residual.f(),branch.f(),weight(p+"ffn_resid_alpha.weight"),weight(p+"ffn_resid_beta.weight"),g.embedding,stream);
        }
        norm(residual.f(),weight("output_norm.weight"),g.embedding,g.rms_eps);
        mat("output.weight",residual.f(),logits.f());
        K::muse_logits(logits.f(),g.vocab,g.logit_scale,g.logit_cap,stream);
        std::vector<float> result(g.vocab);
        cuda_check(cudaMemcpyAsync(result.data(),logits.p,result.size()*4,cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        next_position=pos+1;
        std::fprintf(stderr,"Muse pos=%d expert hits=%llu misses=%llu transferred=%.3f GiB\n",pos,
                     (unsigned long long)hits,(unsigned long long)misses,double(copied)/(1ULL<<30));
        return result;
    }
};
Model::Model(const std::string& path,int context,size_t cache_mib) : impl(std::make_unique<Impl>(path,context,cache_mib)) {}
Model::~Model()=default;
std::vector<float> Model::forward(int token,int pos) { return impl->forward(token,pos); }
int Model::vocabulary() const { return impl->g.vocab; }
}
