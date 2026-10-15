#include "model.hpp"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    try {
        std::string path,out,prompt="The capital of France is",csv;
        bool reference=false;
        int context=4096,generate=0;
        size_t cache=0;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            auto value=[&]() -> std::string {
                if(++i==argc) throw std::runtime_error("missing value for "+a);
                return argv[i];
            };
            if(a=="--model") path=value();
            else if(a=="--out") out=value();
            else if(a=="--prompt") prompt=value();
            else if(a=="--tokens") csv=value();
            else if(a=="--ctx") context=std::stoi(value());
            else if(a=="--cache-mib") cache=std::stoull(value());
            else if(a=="--generate") generate=std::stoi(value());
            else if(a=="--reference") reference=true;
            else throw std::runtime_error("unknown argument "+a);
        }
        if(path.empty() || out.empty() || context<1 || generate<0)
            throw std::runtime_error("usage: muse-check --model SHARD1 --out LOGITS.f32 [--reference] [--prompt TEXT | --tokens CSV] [--generate N] [--ctx N] [--cache-mib N]");
        llama_backend_init();
        auto mp=llama_model_default_params();
        mp.vocab_only=!reference;
        mp.n_gpu_layers=reference ? 999 : 0;
        using ModelPtr=std::unique_ptr<llama_model,decltype(&llama_model_free)>;
        ModelPtr model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
        if(!model) throw std::runtime_error("llama model load failed");
        const auto* vocab=llama_model_get_vocab(model.get());
        std::vector<llama_token> tokens;
        if(!csv.empty()) {
            std::istringstream ss(csv); std::string s;
            while(std::getline(ss,s,',')) tokens.push_back(std::stoi(s));
        } else {
            int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
            tokens.resize(n);
            n=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),true,true);
            if(n<0) throw std::runtime_error("tokenization failed");
            tokens.resize(n);
        }
        if(tokens.empty() || tokens.size()+generate>size_t(context)) throw std::runtime_error("invalid token/context count");
        const size_t prompt_count=tokens.size();
        for(int token:tokens) if(token<0 || token>=llama_vocab_n_tokens(vocab)) throw std::runtime_error("invalid token id");
        std::fprintf(stderr,"tokens:"); for(int t:tokens) std::fprintf(stderr," %d",t); std::fprintf(stderr,"\n");
        std::unique_ptr<strata::muse::Model> native;
        using ContextPtr=std::unique_ptr<llama_context,decltype(&llama_free)>;
        ContextPtr ctx(nullptr,llama_free);
        if(reference) {
            auto cp=llama_context_default_params();
            cp.n_ctx=context; cp.n_batch=1; cp.n_ubatch=1;
            cp.type_k=GGML_TYPE_F32; cp.type_v=GGML_TYPE_F32;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
            ctx.reset(llama_init_from_model(model.get(),cp));
            if(!ctx) throw std::runtime_error("reference context allocation failed");
        } else native=std::make_unique<strata::muse::Model>(path,context,cache);
        std::ofstream dump(out,std::ios::binary);
        std::ofstream tokenfile(out+".tokens");
        if(!dump || !tokenfile) throw std::runtime_error("cannot create output files");
        auto started=std::chrono::steady_clock::now();
        for(size_t i=0;i<tokens.size();++i) {
            std::vector<float> logits;
            if(reference) {
                auto batch=llama_batch_get_one(&tokens[i],1);
                if(llama_decode(ctx.get(),batch)) throw std::runtime_error("reference decode failed");
                const float* p=llama_get_logits_ith(ctx.get(),-1);
                logits.assign(p,p+llama_vocab_n_tokens(vocab));
            } else logits=native->forward(tokens[i],i);
            dump.write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));
            tokenfile<<tokens[i]<<'\n';
            if(!dump || !tokenfile) throw std::runtime_error("logit write failed");
            int best=std::max_element(logits.begin(),logits.end())-logits.begin();
            std::fprintf(stderr,"position=%zu top1=%d logit=%.8g\n",i,best,logits[best]);
            if(i+1>=prompt_count && tokens.size()<prompt_count+generate && !llama_vocab_is_eog(vocab,best)) {
                tokens.push_back(best);
                char piece[256];
                int n=llama_token_to_piece(vocab,best,piece,sizeof(piece),0,false);
                if(n>0) { std::fwrite(piece,1,n,stdout); std::fflush(stdout); }
            }
        }
        auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::fprintf(stderr,"processed=%zu seconds=%.3f tokens_per_second=%.3f\n",tokens.size(),seconds,tokens.size()/seconds);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"muse-check: %s\n",e.what()); return 1; }
}
