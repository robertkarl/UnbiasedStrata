#pragma once
#include "child_watch.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <csignal>
#include <unistd.h>

namespace strata_frontend {
struct GenResult {
    bool ok = false;
    std::string error;
    std::string finish = "length";
    long long produced = 0, prompt_n = 0, draft_accepted = 0, draft_offered = 0;
    double prompt_ms = 0, decode_ms = 0;
};

class Engine {
public:
    bool start(const std::vector<std::string>& args, std::string& err, const char* executable = "/proc/self/exe") {
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
            ::execv(executable, argv.data());
            std::fprintf(stderr, "unbiased-strata: could not start the engine: %s\n", std::strerror(errno));
            ::_exit(127);
        }
        ::close(to_child[0]);
        ::close(from_child[1]);
        in_fd_ = to_child[1];
        out_ = ::fdopen(from_child[0], "r");
        return out_ != nullptr;
    }

    /// Reads the engine's start-up lines until it says READY.  Loading the model takes minutes.  The READY line
    /// carries the context the engine actually allocated (`READY <max_context> stop`): its VRAM plan may have
    /// lowered the request, and the front end's request limit and /v1/models follow that figure, not --ctx.
    bool wait_ready(std::string& err, long long* context = nullptr) {
        std::string line;
        while (read_line(line)) {
            if (line.rfind("READY", 0) == 0) {
                if (context != nullptr) {
                    const long long n = std::atoll(line.c_str() + 5);
                    if (n > 0) *context = n;
                }
                return true;
            }
            if (line.rfind("ERR", 0) == 0) { err = line; return false; }
        }
        err = "the engine exited before it was ready (its messages are above)";
        return false;
    }

    /// One request.  `on_token` returns false when the client has gone: the engine is told to stop and the rest of
    /// its output for this request is read and dropped, so the next request starts clean.
    GenResult generate(const std::string& header, const std::vector<int32_t>& ids,
                       const std::function<bool(int32_t)>& on_token) {
        GenResult r;
        if (!alive_) { r.error = "the engine is not running"; return r; }
        std::string req = header;
        req.reserve(header.size() + ids.size() * 7 + 2);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) req += ',';
            req += std::to_string(ids[i]);
        }
        req += '\n';
        if (!send(req)) { r.error = "the engine is not running"; return r; }
        bool stopped = false;
        std::string callback_error;
        std::string line;
        while (read_line(line)) {
            if (line.size() > 2 && line[0] == 'T' && line[1] == ' ') {
                if (!stopped) {
                    bool keep_going = false;
                    try {
                        keep_going = on_token((int32_t) std::atoi(line.c_str() + 2));
                    } catch (const std::exception& ex) {
                        callback_error = ex.what();
                        std::fprintf(stderr, "unbiased-strata: response formatter failed; stopping and draining request: %s\n", ex.what());
                    }
                    if (!keep_going) {
                        stopped = true;
                        send("STOP\n");
                    }
                }
            } else if (line.rfind("DONE ", 0) == 0) {
                char finish[32] = {0};
                std::sscanf(line.c_str(), "DONE %lld %lld %lf %lf %31s %lld %lld", &r.produced, &r.prompt_n, &r.prompt_ms,
                            &r.decode_ms, finish, &r.draft_accepted, &r.draft_offered);
                r.finish = finish;
                if (r.prompt_n != (long long) ids.size()) {
                    r.error = "engine response prompt size mismatch: submitted " + std::to_string(ids.size()) +
                              ", returned " + std::to_string(r.prompt_n);
                    std::fprintf(stderr, "unbiased-strata: %s; refusing this response\n", r.error.c_str());
                    alive_ = false;
                    return r;
                }
                r.ok = callback_error.empty();
                if (!r.ok) r.error = "response formatter failed: " + callback_error;
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

    /// One command line between requests (the caller holds `mu`): the engine's one-line reply whose first word is
    /// `reply_prefix` lands in `reply`; an ERR line is the error.  `STATS` is the one in use (the server's /health).
    bool command(const std::string& line, const char* reply_prefix, std::string& reply, std::string& err) {
        if (!alive_) { err = "the engine is not running"; return false; }
        if (!send(line + "\n")) { err = "the engine is not running"; return false; }
        std::string got;
        while (read_line(got)) {
            if (got.rfind(reply_prefix, 0) == 0) { reply = got; return true; }
            if (got.rfind("ERR", 0) == 0) { err = got.size() > 4 ? got.substr(4) : "the engine reported an error"; return false; }
        }
        alive_ = false;
        err = "the engine exited";
        return false;
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

    bool reap_exit(std::string& why) {
        if (strata_frontend::child_running(pid_, why)) return false;
        alive_ = false;
        return true;
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

}
