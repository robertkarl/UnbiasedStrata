#include "model.hpp"
#include "llama.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <map>
#include <memory>
#include <numeric>
#include <poll.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
class Input {
    std::string buffer;
public:
    bool closed=false;
    std::deque<std::string> lines;
    Input() {
        int flags=fcntl(STDIN_FILENO,F_GETFL);
        if(flags<0 || fcntl(STDIN_FILENO,F_SETFL,flags|O_NONBLOCK)<0) throw std::runtime_error("cannot configure stdin");
    }
    void read(bool wait) {
        pollfd p{STDIN_FILENO,POLLIN,0};
        if(poll(&p,1,wait ? -1 : 0)<0) {
            if(errno==EINTR) return;
            throw std::runtime_error("stdin poll failed");
        }
        char chunk[65536];
        while(true) {
            ssize_t n=::read(STDIN_FILENO,chunk,sizeof(chunk));
            if(n==0) { closed=true; break; }
            if(n<0) {
                if(errno==EAGAIN || errno==EWOULDBLOCK) break;
                if(errno==EINTR) continue;
                throw std::runtime_error("stdin read failed");
            }
            buffer.append(chunk,n);
            if(buffer.size()>64*1024*1024) throw std::runtime_error("engine request too large");
            size_t end;
            while((end=buffer.find('\n'))!=std::string::npos) {
                lines.push_back(buffer.substr(0,end)); buffer.erase(0,end+1);
            }
        }
    }
    bool stopped() {
        read(false);
        bool stop=closed;
        for(auto it=lines.begin();it!=lines.end();) {
            if(*it=="STOP") { stop=true; it=lines.erase(it); }
            else if(*it=="QUIT" || *it=="EXIT") { closed=true; stop=true; it=lines.erase(it); }
            else ++it;
        }
        return stop;
    }
};
double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
struct Request {
    int maximum=0;
    double temperature=0,top_p=1,min_p=0,presence=0,frequency=0,repeat=1;
    int top_k=0,last_n=64;
    uint64_t seed=0;
    std::vector<int> tokens;
    explicit Request(const std::string& line) {
        std::istringstream ss(line); std::string verb,part;
        if(!(ss>>verb>>maximum) || verb!="GEN" || maximum<1) throw std::runtime_error("invalid GEN header");
        while(ss>>part) {
            auto split=part.find('=');
            if(split!=std::string::npos) {
                auto name=part.substr(0,split),value=part.substr(split+1);
                if(name=="seed") seed=std::stoull(value);
                else {
                    double x=std::stod(value);
                    if(!std::isfinite(x)) throw std::runtime_error("non-finite sampling parameter");
                    if(name=="temperature") temperature=x;
                    else if(name=="top_p") top_p=x;
                    else if(name=="min_p") min_p=x;
                    else if(name=="top_k") top_k=x;
                    else if(name=="penalty_present") presence=x;
                    else if(name=="penalty_freq") frequency=x;
                    else if(name=="penalty_repeat") repeat=x;
                    else if(name=="penalty_last_n") last_n=x;
                    else throw std::runtime_error("unknown sampling parameter "+name);
                }
            } else {
                std::istringstream ids(part); std::string id;
                while(std::getline(ids,id,',')) tokens.push_back(std::stoi(id));
            }
        }
        if(tokens.empty() || temperature<0 || top_p<=0 || top_p>1 || min_p<0 || min_p>1 || repeat<=0 || last_n<0 || top_k<0)
            throw std::runtime_error("invalid tokens or sampling parameters");
    }
    int sample(std::vector<float> logits,std::mt19937_64& rng) const {
        std::map<int,int> counts;
        size_t begin=tokens.size()>size_t(last_n) ? tokens.size()-last_n : 0;
        for(size_t i=begin;i<tokens.size();++i) ++counts[tokens[i]];
        for(auto [id,count]:counts) {
            logits.at(id)=logits[id]<=0 ? logits[id]*repeat : logits[id]/repeat;
            logits[id]-=presence+frequency*count;
        }
        if(temperature==0) return std::max_element(logits.begin(),logits.end())-logits.begin();
        std::vector<int> ids(logits.size()); std::iota(ids.begin(),ids.end(),0);
        size_t n=top_k>0 ? std::min(size_t(top_k),ids.size()) : ids.size();
        std::partial_sort(ids.begin(),ids.begin()+n,ids.end(),[&](int a,int b){return logits[a]>logits[b];});
        ids.resize(n);
        std::vector<double> weights(n);
        double total=0,maximum=logits[ids[0]];
        for(size_t i=0;i<n;++i) { weights[i]=std::exp((logits[ids[i]]-maximum)/temperature); total+=weights[i]; }
        double cumulative=0;
        for(size_t i=0;i<n;++i) {
            cumulative+=weights[i]/total;
            if((i>0 && weights[i]<min_p*weights[0]) || cumulative>=top_p) {
                size_t keep=(i>0 && weights[i]<min_p*weights[0]) ? i : i+1;
                weights.resize(keep); break;
            }
        }
        return ids[std::discrete_distribution<size_t>(weights.begin(),weights.end())(rng)];
    }
};
}

int strata_engine_main(int argc,char** argv) {
    try {
        std::string path; int context=4096; size_t cache=0;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            if(i+1==argc) throw std::runtime_error("missing value for "+a);
            std::string value=argv[++i];
            if(a=="--native") path=value;
            else if(a=="--max-context") context=std::stoi(value);
            else if(a=="--cache-mib") cache=std::stoull(value);
            else throw std::runtime_error("unknown Muse engine option "+a);
        }
        if(path.empty()) throw std::runtime_error("--native is required");
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.vocab_only=true; mp.n_gpu_layers=0;
        using LlamaModel=std::unique_ptr<llama_model,decltype(&llama_model_free)>;
        LlamaModel tokenizer(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
        if(!tokenizer) throw std::runtime_error("cannot load Muse tokenizer");
        const auto* vocab=llama_model_get_vocab(tokenizer.get());
        strata::muse::Model model(path,context,cache);
        Input input;
        std::printf("READY %d stop\n",context); std::fflush(stdout);
        while(!input.closed) {
            if(input.lines.empty()) { input.read(true); continue; }
            auto line=std::move(input.lines.front()); input.lines.pop_front();
            if(line=="QUIT" || line=="EXIT") break;
            if(line=="STOP") continue;
            if(line=="STATS") { std::puts("STATS {\"backend\":\"native-muse\",\"expert_storage\":\"VRAM LRU and mmap\"}"); std::fflush(stdout); continue; }
            // Parse errors do not change model state; runtime failures terminate the child.
            std::unique_ptr<Request> request;
            try {
                request=std::make_unique<Request>(line);
                if(request->tokens.size()+size_t(request->maximum)>size_t(context)) throw std::runtime_error("request exceeds context");
                for(int token:request->tokens) if(token<0 || token>=model.vocabulary()) throw std::runtime_error("invalid token id");
            } catch(const std::exception& e) {
                std::printf("ERR %s\n",e.what()); std::fflush(stdout); continue;
            }
            auto& r=*request;
            const size_t prompt=r.tokens.size();
            std::mt19937_64 rng(r.seed);
            auto started=std::chrono::steady_clock::now();
            std::vector<float> logits;
            bool stopped=false;
            for(size_t i=0;i<prompt;++i) {
                if(input.stopped()) { stopped=true; break; }
                logits=model.forward(r.tokens[i],i);
            }
            double prompt_ms=elapsed(started);
            started=std::chrono::steady_clock::now();
            int produced=0;
            for(int i=0;!stopped && i<r.maximum;++i) {
                if(input.stopped()) { stopped=true; break; }
                int token=r.sample(logits,rng);
                if(llama_vocab_is_eog(vocab,token)) { stopped=true; break; }
                std::printf("T %d\n",token); std::fflush(stdout); ++produced;
                r.tokens.push_back(token);
                if(i+1<r.maximum) logits=model.forward(token,r.tokens.size()-1);
            }
            std::printf("DONE %d %zu %.3f %.3f %s 0 0\n",produced,prompt,prompt_ms,elapsed(started),stopped?"stop":"length");
            std::fflush(stdout);
        }
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"Muse engine: %s\n",e.what());
        std::printf("ERR %s\n",e.what()); std::fflush(stdout); return 1;
    }
}
