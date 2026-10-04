// frontend/main.cpp - UnbiasedStrata's front end: tokenizer, chat template and an OpenAI-style HTTP endpoint in
// front of the unmodified Strata engine.
//
// One binary, two roles.  Started normally it is the front end: it starts a copy of itself with `--engine`, which
// runs the engine's own program (frontend/engine_entry.cpp) in serve mode, and talks to it over a pipe in the
// engine's line protocol (token ids in, token ids out).  That is the process model upstream's Python server uses,
// so the engine runs exactly as upstream tested it.
//
// The tokenizer and the chat template come from llama.cpp (the pinned submodule): the GGUF's own vocabulary and
// its own Jinja template.  Nothing here opens an outgoing connection.
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int strata_engine_main(int argc, char** argv);

namespace {

const char* const kModelName = "qwen3.8-flash-next-iq3_xxs";

struct Options {
    std::string model;                  // shard 1 of the GGUF
    std::string pack;                   // the prepared pack directory
    std::string mtp;                    // the prepared draft-head directory
    std::string host = "127.0.0.1";
    int port = 8080;
    long long ctx = 32768;
    int threads = 0;                    // CPU expert workers; 0 = the engine's default (every physical core)
    std::vector<std::string> extra;     // after `--`: passed to the engine as they are
};

void usage() {
    std::fprintf(stderr,
        "usage: unbiased-strata --model SHARD1.gguf --pack DIR --mtp DIR [options] [-- engine options]\n\n"
        "  --model PATH    the model's first GGUF shard (...-00001-of-00002.gguf); the second must be beside it\n"
        "  --pack DIR      the prepared pack for that model\n"
        "  --mtp DIR       the prepared draft head\n"
        "  --ctx N         context length in tokens (default 32768)\n"
        "  --threads N     CPU threads for the experts the GPU does not hold (default: every physical core;\n"
        "                  set it inside a container, where the host's core count is not yours)\n"
        "  --host ADDR     address to listen on (default 127.0.0.1)\n"
        "  --port N        port to listen on (default 8080)\n\n"
        "Serves POST /v1/chat/completions (OpenAI style, with streaming), GET /v1/models and GET /health.\n");
}

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

// ---------------------------------------------------------------------------------------------------- the engine

struct GenResult {
    bool ok = false;
    std::string error;
    std::string finish = "length";
    long long produced = 0, prompt_n = 0, draft_accepted = 0, draft_offered = 0;
    double prompt_ms = 0, decode_ms = 0;
};

class Engine {
public:
    bool start(const std::vector<std::string>& args, std::string& err) {
        int to_child[2], from_child[2];
        if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) { err = "pipe failed"; return false; }
        pid_ = ::fork();
        if (pid_ < 0) { err = "fork failed"; return false; }
        if (pid_ == 0) {
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            ::close(to_child[0]); ::close(to_child[1]); ::close(from_child[0]); ::close(from_child[1]);
            std::vector<char*> argv;
            for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execv("/proc/self/exe", argv.data());
            std::fprintf(stderr, "unbiased-strata: could not start the engine: %s\n", std::strerror(errno));
            ::_exit(127);
        }
        ::close(to_child[0]);
        ::close(from_child[1]);
        in_fd_ = to_child[1];
        out_ = ::fdopen(from_child[0], "r");
        return out_ != nullptr;
    }

    /// Reads the engine's start-up lines until it says READY.  Loading the model takes minutes.
    bool wait_ready(std::string& err) {
        std::string line;
        while (read_line(line)) {
            if (line.rfind("READY", 0) == 0) return true;
            if (line.rfind("ERR", 0) == 0) { err = line; return false; }
        }
        err = "the engine exited before it was ready (its messages are above)";
        return false;
    }

    /// One request.  `on_token` returns false when the client has gone: the engine is told to stop and the rest of
    /// its output for this request is read and dropped, so the next request starts clean.
    GenResult generate(const std::string& header, const std::vector<llama_token>& ids,
                       const std::function<bool(llama_token)>& on_token) {
        GenResult r;
        std::string req = header;
        req.reserve(header.size() + ids.size() * 7 + 2);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) req += ',';
            req += std::to_string(ids[i]);
        }
        req += '\n';
        if (!send(req)) { r.error = "the engine is not running"; return r; }
        bool stopped = false;
        std::string line;
        while (read_line(line)) {
            if (line.size() > 2 && line[0] == 'T' && line[1] == ' ') {
                if (!stopped && !on_token((llama_token) std::atoi(line.c_str() + 2))) {
                    stopped = true;
                    send("STOP\n");
                }
            } else if (line.rfind("DONE ", 0) == 0) {
                char finish[32] = {0};
                std::sscanf(line.c_str(), "DONE %lld %lld %lf %lf %31s %lld %lld", &r.produced, &r.prompt_n, &r.prompt_ms,
                            &r.decode_ms, finish, &r.draft_accepted, &r.draft_offered);
                r.finish = finish;
                r.ok = true;
                return r;
            } else if (line.rfind("ERR", 0) == 0) {
                r.error = line.size() > 4 ? line.substr(4) : "the engine reported an error";
                return r;
            }
            // PP (prompt progress), RESUME, REUSED and INFO lines need no action here
        }
        alive_ = false;
        r.error = "the engine exited during the request";
        return r;
    }

    void quit() {
        if (pid_ > 0) {
            send("QUIT\n");
            int status = 0;
            for (int i = 0; i < 100 && ::waitpid(pid_, &status, WNOHANG) == 0; ++i) ::usleep(100 * 1000);
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    bool alive() const { return alive_; }
    std::mutex mu;   // one request at a time: the engine holds one conversation

private:
    bool send(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = ::write(in_fd_, s.data() + off, s.size() - off);
            if (n <= 0) { alive_ = false; return false; }
            off += (size_t) n;
        }
        return true;
    }

    bool read_line(std::string& line) {
        char* buf = nullptr;
        size_t cap = 0;
        const ssize_t n = ::getline(&buf, &cap, out_);
        if (n < 0) { std::free(buf); alive_ = false; return false; }
        line.assign(buf, (size_t) n);
        std::free(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        return true;
    }

    pid_t pid_ = -1;
    int in_fd_ = -1;
    FILE* out_ = nullptr;
    std::atomic<bool> alive_{true};
};

// ------------------------------------------------------------------------------------------------ text handling

/// The length of the longest prefix of `s` that does not end inside a UTF-8 character.
size_t utf8_complete_prefix(const std::string& s) {
    size_t i = s.size();
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char) s[i - 1];
        ++back;
        if ((c & 0xC0) == 0x80) { --i; continue; }             // a continuation byte: keep looking for its lead
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? i - 1 : s.size();
    }
    return s.size();
}

/// Splits the model's output into reasoning (inside the thinking tags) and the answer, as it streams.
class ThinkSplitter {
public:
    ThinkSplitter(bool in_think, std::string end_tag) : in_think_(in_think), end_(std::move(end_tag)) {}

    /// Feeds text; calls `emit(text, is_reasoning)` for what is certain so far.
    void feed(const std::string& text, bool final, const std::function<void(const std::string&, bool)>& emit) {
        buf_ += text;
        for (;;) {
            if (!in_think_) {
                if (strip_ws_) {                                // the blank lines the template puts after the end tag
                    size_t k = 0;
                    while (k < buf_.size() && (buf_[k] == '\n' || buf_[k] == '\r')) ++k;
                    buf_.erase(0, k);
                    if (buf_.empty() && !final) return;
                    strip_ws_ = false;
                }
                if (!buf_.empty()) emit(buf_, false);
                buf_.clear();
                return;
            }
            const size_t at = end_.empty() ? std::string::npos : buf_.find(end_);
            if (at != std::string::npos) {
                if (at > 0) emit(buf_.substr(0, at), true);
                buf_.erase(0, at + end_.size());
                in_think_ = false;
                strip_ws_ = true;
                continue;
            }
            // keep back a tail that could be the start of the end tag
            size_t keep = 0;
            if (!final && !end_.empty()) {
                const size_t maxk = std::min(buf_.size(), end_.size() - 1);
                for (size_t k = maxk; k > 0; --k)
                    if (buf_.compare(buf_.size() - k, k, end_, 0, k) == 0) { keep = k; break; }
            }
            if (buf_.size() > keep) {
                emit(buf_.substr(0, buf_.size() - keep), true);
                buf_.erase(0, buf_.size() - keep);
            }
            return;
        }
    }

private:
    bool in_think_;
    bool strip_ws_ = false;
    std::string end_;
    std::string buf_;
};

std::string make_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    char b[40];
    std::snprintf(b, sizeof b, "chatcmpl-%016llx", (unsigned long long) rng());
    return b;
}

long long now_s() {
    return (long long) std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void json_error(httplib::Response& res, int status, const std::string& message, const char* type) {
    common_json e = common_json::object();
    e["error"] = common_json::object();
    e["error"]["message"] = message;
    e["error"]["type"] = type;
    res.status = status;
    res.set_content(e.dump_safe(), "application/json");
}

// ------------------------------------------------------------------------------------------------- the request

struct Prepared {
    std::vector<llama_token> ids;
    std::string header;        // "GEN <max_new> key=value ... " - the ids follow
    bool in_think = false;
    std::string think_end = "</think>";
    bool stream = false;
    long long max_new = 0;
};

struct Server {
    Options opt;
    Engine engine;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    common_chat_templates_ptr tmpls;

    /// Parses an OpenAI chat request, renders the model's chat template and tokenizes the result.
    bool prepare(const common_json& body, Prepared& p, std::string& err) {
        if (!body.is_object() || !body.contains("messages")) { err = "the request needs a \"messages\" array"; return false; }
        if (body.contains("tools") && body.at("tools").is_array() && body.at("tools").size() > 0) {
            err = "tool calling is not supported by this server yet";
            return false;
        }
        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(body.at("messages"));
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        in.enable_thinking = true;
        // thinking: OpenAI's reasoning_effort, or the llama.cpp / vLLM convention chat_template_kwargs.enable_thinking
        if (body.contains("reasoning_effort") && body.at("reasoning_effort").is_string()) {
            const std::string effort = body.at("reasoning_effort").get<std::string>();
            if (effort == "none") in.enable_thinking = false;
            else in.chat_template_kwargs["reasoning_effort"] = common_json(effort).dump();
        }
        if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
            const common_json& kw = body.at("chat_template_kwargs");
            if (kw.contains("enable_thinking") && kw.at("enable_thinking").is_boolean())
                in.enable_thinking = kw.at("enable_thinking").get<bool>();
            if (kw.contains("reasoning_effort") && kw.at("reasoning_effort").is_string())
                in.chat_template_kwargs["reasoning_effort"] = kw.at("reasoning_effort").dump();
        }
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        p.ids = common_tokenize(vocab, cp.prompt, /*add_special=*/false, /*parse_special=*/true);
        if (p.ids.empty()) { err = "the prompt is empty"; return false; }

        const std::string start_tag = cp.thinking_start_tag.empty() ? "<think>" : cp.thinking_start_tag;
        if (!cp.thinking_end_tags.empty()) p.think_end = cp.thinking_end_tags.front();
        while (!p.think_end.empty() && (p.think_end.back() == '\n' || p.think_end.back() == ' ')) p.think_end.pop_back();
        const size_t s = cp.prompt.rfind(start_tag), e = cp.prompt.rfind(p.think_end);
        const size_t a = cp.prompt.rfind("<|im_start|>");
        p.in_think = s != std::string::npos && (a == std::string::npos || s > a) && (e == std::string::npos || e < s);

        const long long n = (long long) p.ids.size();
        if (n + 1 > opt.ctx) {
            err = "the prompt is " + std::to_string(n) + " tokens, which does not fit the context of " +
                  std::to_string(opt.ctx);
            return false;
        }
        long long max_new = opt.ctx - n;
        for (const char* key : {"max_completion_tokens", "max_tokens"})
            if (body.contains(key) && body.at(key).is_number()) {
                const long long want = body.at(key).get<long long>();
                if (want >= 1) max_new = std::min(max_new, want);
                break;
            }
        p.max_new = max_new;
        p.stream = body.value("stream", false);

        // sampling: greedy unless the request sets a temperature above zero
        std::string keys;
        const double temperature = body.contains("temperature") && body.at("temperature").is_number()
                                       ? body.at("temperature").get<double>() : 0.0;
        if (temperature > 0.0) {
            const double top_p = body.contains("top_p") && body.at("top_p").is_number() ? body.at("top_p").get<double>() : 0.95;
            long long top_k = body.contains("top_k") && body.at("top_k").is_number() ? body.at("top_k").get<long long>() : 20;
            top_k = std::max(1LL, std::min(64LL, top_k));       // the engine's sampler takes 1..64
            unsigned long long seed;
            if (body.contains("seed") && body.at("seed").is_number()) seed = (unsigned long long) body.at("seed").get<long long>();
            else seed = std::random_device{}();
            char b[160];
            std::snprintf(b, sizeof b, " temperature=%g top_p=%g top_k=%lld seed=%llu", temperature, top_p, top_k, seed);
            keys += b;
            if (body.contains("min_p") && body.at("min_p").is_number()) {
                std::snprintf(b, sizeof b, " min_p=%g", body.at("min_p").get<double>());
                keys += b;
            }
        }
        for (const auto& kv : {std::pair<const char*, const char*>{"presence_penalty", "penalty_present"},
                               std::pair<const char*, const char*>{"frequency_penalty", "penalty_freq"},
                               std::pair<const char*, const char*>{"repeat_penalty", "penalty_repeat"}})
            if (body.contains(kv.first) && body.at(kv.first).is_number()) {
                char b[64];
                std::snprintf(b, sizeof b, " %s=%g", kv.second, body.at(kv.first).get<double>());
                keys += b;
                if (keys.find("penalty_last_n") == std::string::npos) keys += " penalty_last_n=64";
            }
        p.header = "GEN " + std::to_string(max_new) + keys + " ";
        return true;
    }

    common_json usage_json(const GenResult& r) const {
        common_json u = common_json::object();
        u["prompt_tokens"] = r.prompt_n;
        u["completion_tokens"] = r.produced;
        u["total_tokens"] = r.prompt_n + r.produced;
        return u;
    }

    common_json timings_json(const GenResult& r) const {
        common_json t = common_json::object();
        t["prompt_n"] = r.prompt_n;
        t["prompt_ms"] = r.prompt_ms;
        t["prompt_per_second"] = r.prompt_ms > 0 ? 1000.0 * (double) r.prompt_n / r.prompt_ms : 0.0;
        t["predicted_n"] = r.produced;
        t["predicted_ms"] = r.decode_ms;
        t["predicted_per_second"] = r.decode_ms > 0 ? 1000.0 * (double) r.produced / r.decode_ms : 0.0;
        t["draft_n"] = r.draft_offered;
        t["draft_n_accepted"] = r.draft_accepted;
        return t;
    }

    /// Runs one request.  `emit(text, is_reasoning)` gets the text as it is produced and returns false when the
    /// client has gone.
    GenResult run(const Prepared& p, const std::function<bool(const std::string&, bool)>& emit) {
        std::lock_guard<std::mutex> lk(engine.mu);
        ThinkSplitter split(p.in_think, p.think_end);
        std::string pending;
        bool client_ok = true;
        auto out = [&](const std::string& text, bool reasoning) {
            if (client_ok && !text.empty()) client_ok = emit(text, reasoning);
        };
        GenResult r = engine.generate(p.header, p.ids, [&](llama_token id) {
            if (llama_vocab_is_eog(vocab, id)) return client_ok;
            pending += common_token_to_piece(vocab, id, /*special=*/true);
            const size_t n = utf8_complete_prefix(pending);
            if (n > 0) {
                split.feed(pending.substr(0, n), false, out);
                pending.erase(0, n);
            }
            return client_ok;
        });
        split.feed(pending, true, out);
        return r;
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        if (!engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        const common_json body = common_json::parse_no_throw(req.body);
        if (body.is_discarded()) return json_error(res, 400, "the request body is not valid JSON", "invalid_request_error");
        auto prep = std::make_shared<Prepared>();
        std::string err;
        try {
            if (!prepare(body, *prep, err)) return json_error(res, 400, err, "invalid_request_error");
        } catch (const std::exception& ex) {
            return json_error(res, 400, ex.what(), "invalid_request_error");
        }
        const std::string id = make_id();
        const long long created = now_s();

        if (!prep->stream) {
            std::string content, reasoning;
            const GenResult r = run(*prep, [&](const std::string& text, bool is_reasoning) {
                (is_reasoning ? reasoning : content) += text;
                return true;
            });
            if (!r.ok) return json_error(res, 500, r.error, "server_error");
            common_json msg = common_json::object();
            msg["role"] = "assistant";
            msg["content"] = content;
            if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
            common_json choice = common_json::object();
            choice["index"] = 0;
            choice["message"] = msg;
            choice["finish_reason"] = r.finish == "stop" ? "stop" : "length";
            common_json out = common_json::object();
            out["id"] = id;
            out["object"] = "chat.completion";
            out["created"] = created;
            out["model"] = kModelName;
            out["choices"] = common_json::array();
            out["choices"].push_back(choice);
            out["usage"] = usage_json(r);
            out["timings"] = timings_json(r);
            res.set_content(out.dump_safe(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [this, prep, id, created](size_t, httplib::DataSink& sink) {
            auto chunk = [&](const common_json& delta, const char* finish, const GenResult* done) {
                common_json choice = common_json::object();
                choice["index"] = 0;
                choice["delta"] = delta;
                if (finish) choice["finish_reason"] = finish; else choice["finish_reason"] = nullptr;
                common_json out = common_json::object();
                out["id"] = id;
                out["object"] = "chat.completion.chunk";
                out["created"] = created;
                out["model"] = kModelName;
                out["choices"] = common_json::array();
                out["choices"].push_back(choice);
                if (done) { out["usage"] = usage_json(*done); out["timings"] = timings_json(*done); }
                const std::string s = "data: " + out.dump_safe() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            common_json first = common_json::object();
            first["role"] = "assistant";
            first["content"] = "";
            bool ok = chunk(first, nullptr, nullptr);
            const GenResult r = run(*prep, [&](const std::string& text, bool is_reasoning) {
                common_json d = common_json::object();
                d[is_reasoning ? "reasoning_content" : "content"] = text;
                ok = ok && chunk(d, nullptr, nullptr);
                return ok;
            });
            if (ok) {
                if (r.ok) {
                    chunk(common_json::object(), r.finish == "stop" ? "stop" : "length", &r);
                } else {
                    common_json e = common_json::object();
                    e["error"] = common_json::object();
                    e["error"]["message"] = r.error;
                    e["error"]["type"] = "server_error";
                    const std::string s = "data: " + e.dump_safe() + "\n\n";
                    sink.write(s.data(), s.size());
                }
                static const std::string end = "data: [DONE]\n\n";
                sink.write(end.data(), end.size());
            }
            sink.done();
            return true;
        });
    }
};

Server* g_server = nullptr;
httplib::Server* g_http = nullptr;

void on_signal(int) {
    if (g_http) g_http->stop();
}

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "unbiased-strata: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") o.model = next("--model");
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--host") o.host = next("--host");
        else if (a == "--port") o.port = std::atoi(next("--port").c_str());
        else if (a == "--ctx") o.ctx = std::atoll(next("--ctx").c_str());
        else if (a == "--threads") o.threads = std::atoi(next("--threads").c_str());
        else if (a == "--") {
            for (++i; i < argc; ++i) {
                const std::string e = argv[i];
                // one GPU only: the engine's multi-GPU and remote-expert modes are not part of this fork
                for (const char* no : {"--gpus", "--layer-split", "--split-device", "--peer-device", "--peer-slots",
                                       "--remote-expert", "--vision", "--batch"})
                    if (e.rfind(no, 0) == 0) {
                        std::fprintf(stderr, "unbiased-strata: %s is not supported (single GPU, text only)\n", e.c_str());
                        std::exit(2);
                    }
                o.extra.push_back(e);
            }
        }
        else if (a == "-h" || a == "--help") return false;
        else { std::fprintf(stderr, "unbiased-strata: unknown option %s\n\n", a.c_str()); return false; }
    }
    return !o.model.empty() && !o.pack.empty() && !o.mtp.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--engine") == 0) return strata_engine_main(argc - 1, argv + 1);

    Server srv;
    Options& o = srv.opt;
    if (!parse_args(argc, argv, o)) { usage(); return 2; }

    // every file the engine needs must already be here: nothing is fetched
    std::string shard2 = o.model;
    const size_t at = shard2.rfind("-00001-of-00002");
    if (at == std::string::npos) {
        std::fprintf(stderr, "unbiased-strata: --model must be the first shard (...-00001-of-00002.gguf)\n");
        return 2;
    }
    shard2.replace(at, 15, "-00002-of-00002");
    const std::string profile = o.pack + "/expert-profile.bin";
    for (const std::string& f : {o.model, shard2, o.pack + "/index.txt", profile, o.mtp + "/experts.bin",
                                 o.mtp + "/draft_vocab.bin"})
        if (!file_exists(f)) { std::fprintf(stderr, "unbiased-strata: missing file: %s\n", f.c_str()); return 2; }
    if (o.ctx < 1024) { std::fprintf(stderr, "unbiased-strata: --ctx must be at least 1024\n"); return 2; }

    // the engine, started first: loading takes minutes, and it must be forked before any thread exists
    std::vector<std::string> eargs = {
        "unbiased-strata", "--engine", "--serve", "--pack", o.pack, "--native", o.model, "--ple-gguf", shard2,
        "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", o.mtp, "--max-context", std::to_string(o.ctx), "--ple-io", "ram"};
    if (o.ctx > 8192) { eargs.push_back("--kv"); eargs.push_back("int8"); }
    if (o.threads > 0) { eargs.push_back("--pool-workers"); eargs.push_back(std::to_string(o.threads)); }
    for (const std::string& e : o.extra) eargs.push_back(e);
    std::signal(SIGPIPE, SIG_IGN);
    std::string err;
    if (!srv.engine.start(eargs, err)) { std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str()); return 1; }

    // the tokenizer and the chat template, from the GGUF itself
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    srv.model = llama_model_load_from_file(o.model.c_str(), mp);
    if (srv.model == nullptr) {
        std::fprintf(stderr, "unbiased-strata: could not read the tokenizer from %s\n", o.model.c_str());
        srv.engine.quit();
        return 1;
    }
    srv.vocab = llama_model_get_vocab(srv.model);
    srv.tmpls = common_chat_templates_init(srv.model, "");

    std::fprintf(stderr, "unbiased-strata: loading the model (this takes a few minutes)...\n");
    if (!srv.engine.wait_ready(err)) {
        std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str());
        srv.engine.quit();
        return 1;
    }

    httplib::Server http;
    g_server = &srv;
    g_http = &http;
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = srv.engine.alive() ? 200 : 503;
        res.set_content(srv.engine.alive() ? "{\"status\":\"ok\"}" : "{\"status\":\"error\"}", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        common_json m = common_json::object();
        m["id"] = kModelName;
        m["object"] = "model";
        m["owned_by"] = "local";
        m["context_length"] = srv.opt.ctx;
        common_json out = common_json::object();
        out["object"] = "list";
        out["data"] = common_json::array();
        out["data"].push_back(m);
        res.set_content(out.dump_safe(), "application/json");
    });
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        srv.handle_chat(req, res);
    });
    http.set_read_timeout(600, 0);
    http.set_write_timeout(600, 0);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!http.bind_to_port(o.host, o.port)) {
        std::fprintf(stderr, "unbiased-strata: could not listen on %s:%d\n", o.host.c_str(), o.port);
        srv.engine.quit();
        return 1;
    }
    std::fprintf(stderr, "unbiased-strata: ready on http://%s:%d (context %lld tokens)\n", o.host.c_str(), o.port, o.ctx);
    http.listen_after_bind();

    std::fprintf(stderr, "unbiased-strata: stopping\n");
    srv.engine.quit();
    llama_model_free(srv.model);
    return 0;
}
