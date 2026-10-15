#include "strata/kernels/muse.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    float* p=nullptr; size_t n;
    explicit Device(size_t count):n(count) {ck(cudaMalloc(&p,n*4));}
    ~Device(){cudaFree(p);}
    void put(const std::vector<float>& x){ck(cudaMemcpy(p,x.data(),n*4,cudaMemcpyHostToDevice));}
    std::vector<float> get(){std::vector<float>x(n);ck(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
};
void compare(const char* name,const std::vector<float>& got,const std::vector<float>& want,float tolerance=2e-5f){
    float error=0;
    for(size_t i=0;i<got.size();++i){
        if(!std::isfinite(got[i])) throw std::runtime_error(std::string(name)+": non-finite result");
        error=std::max(error,std::abs(got[i]-want[i]));
    }
    std::printf("%s max_abs_error=%g\n",name,error);
    if(error>tolerance) throw std::runtime_error(std::string(name)+": mismatch");
}
std::vector<float> wave(int n,float offset){
    std::vector<float>x(n);for(int i=0;i<n;++i)x[i]=std::sin(i*.13f+offset);return x;
}
}
int main(){
    try{
        cudaStream_t s;ck(cudaStreamCreate(&s));
        namespace K=strata::kernels;
        const int n=8192;
        auto x=wave(n,.1f),y=wave(n,.7f),a=wave(n,1.3f),b=wave(n,2.f),want=x;
        Device dx(n),dy(n),da(n),db(n);dx.put(x);dy.put(y);da.put(a);db.put(b);
        K::muse_residual(dx.p,dy.p,da.p,db.p,n,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;++i)want[i]=x[i]*a[i]+y[i]*b[i];compare("residual",dx.get(),want);
        dx.put(x);K::muse_gate(dx.p,dy.p,n,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;++i)want[i]=x[i]/(1+std::exp(-y[i]));compare("attention gate",dx.get(),want);
        dx.put(x);K::muse_swiglu(dx.p,dy.p,n,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;++i)want[i]=x[i]/(1+std::exp(-x[i]))*y[i];compare("swiglu",dx.get(),want);
        dx.put(x);K::muse_mul(dx.p,dy.p,n,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;++i)want[i]=x[i]*y[i];compare("multiply before norm",dx.get(),want);
        dx.put(x);K::muse_accumulate(dx.p,dy.p,.3f,n,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;++i)want[i]=x[i]+.3f*y[i];compare("expert accumulation",dx.get(),want);
        dx.put(x);K::muse_logits(dx.p,n,.1767766923f,20.f,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;++i)want[i]=20.f*std::tanh(x[i]*.1767766923f/20.f);compare("logit softcap",dx.get(),want);
        dx.put(x);K::muse_rope(dx.p,128,64,137,500000.f,s);ck(cudaStreamSynchronize(s));
        for(int i=0;i<n;i+=2){
            double theta=137*std::pow(500000.,-double(i%64)/64);
            want[i]=x[i]*std::cos(theta)-x[i+1]*std::sin(theta);
            want[i+1]=x[i]*std::sin(theta)+x[i+1]*std::cos(theta);
        }compare("paired RoPE",dx.get(),want,4e-5f);
        for(int count:{1,3,4,9,257}){
            const int heads=8,kv=2,width=64;
            auto q=wave(heads*width,.3f),k=wave(count*kv*width,.6f),v=wave(count*kv*width,.9f);
            Device dq(q.size()),dk(k.size()),dv(v.size()),ds(heads*count),out(heads*width);
            dq.put(q);dk.put(k);dv.put(v);
            K::muse_attention(dq.p,dk.p,dv.p,ds.p,out.p,heads,kv,width,count,count,s);ck(cudaStreamSynchronize(s));
            std::vector<float> reference(heads*width);
            for(int h=0;h<heads;++h){
                std::vector<double> probabilities(count);
                for(int t=0;t<count;++t){
                    double dot=0;for(int d=0;d<width;++d)dot+=double(q[h*width+d])*k[(t*kv+h/(heads/kv))*width+d];
                    probabilities[t]=dot/std::sqrt(double(width));
                }
                double mx=*std::max_element(probabilities.begin(),probabilities.end()),sum=0;
                for(auto& p:probabilities){p=std::exp(p-mx);sum+=p;}
                for(int d=0;d<width;++d){
                    double value=0;for(int t=0;t<count;++t)value+=probabilities[t]/sum*v[(t*kv+h/(heads/kv))*width+d];
                    reference[h*width+d]=value;
                }
            }
            compare("grouped-query attention",out.get(),reference);
        }
        ck(cudaStreamDestroy(s));return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
