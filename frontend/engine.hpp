// UnbiasedStrata: the engine as a child process.
//
// The engines (`strata`, `strata-poly`) speak lines over stdin/stdout: `GEN <max_new> key=value... <id,id,...>` in,
// `T <id>` per token, `PP`, `RESUME`, `DONE ...`, `ERR ...` out, `STOP` to cancel, `QUIT` to end (docs/BACKENDS.md,
// "How a backend works"). This class owns one such child. It never blocks longer than the caller allows: the server
// loop polls it between tasks so a cancel from a disconnected client is seen within the poll interval.
#pragma once

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace strata_frontend {

// Reap a dead child: kill(pid, 0) would report a zombie as alive. The pid slot is shared with quit() on another
// thread (the supervisor polls this while a SIGTERM handler shuts down), so it is read once and cleared by
// compare-exchange. Nothing here or in quit() may pass a pid that is not > 0 to kill(): kill(-1, SIGKILL) kills every
// process the user owns (it took out two whole login sessions on 2026-10-17).
inline bool child_running(std::atomic<pid_t> & pid_slot, std::string & why) {
    pid_t pid = pid_slot.load();
    if (pid <= 0) {
        return false;
    }
    int status = 0;
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == 0 || (result < 0 && errno == EINTR)) {
        return true;
    }
    if (result < 0) {
        why = std::string("waitpid: ") + std::strerror(errno);
    } else if (WIFSIGNALED(status)) {
        why = "engine killed by signal " + std::to_string(WTERMSIG(status));
    } else if (WIFEXITED(status)) {
        why = "engine exited with code " + std::to_string(WEXITSTATUS(status));
    } else {
        why = "engine exited unexpectedly";
    }
    pid_slot.compare_exchange_strong(pid, -1);
    return false;
}

// One line from the engine, already classified.
struct EngineLine {
    enum Kind { NONE, TOKEN, PP, RESUME, REUSED, DONE, ERR, INFO, OTHER, EOF_ } kind = NONE;
    std::string raw;
    int32_t token = -1;
    // PP <done> <total> [<ms> <tok/s>]
    long long pp_done = 0, pp_total = 0;
    // RESUME <n> / REUSED <n>
    long long n = 0;
    // DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> [<drafts_accepted> <drafts_offered> <reused> ...]
    long long produced = 0, prompt_n = 0, draft_accepted = 0, draft_offered = 0, reused = 0;
    double prompt_ms = 0, decode_ms = 0;
    std::string finish;   // stop | length | cancel | nonfinite
    std::string message;  // ERR text
};

class Engine {
public:
    Engine() = default;
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    ~Engine() { quit(); }

    // fork+exec `executable` with `args` (args[0] is argv[0]). Must be called before any thread exists.
    bool start(const std::string & executable, const std::vector<std::string> & args, std::string & err) {
        int to_child[2], from_child[2];
        if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) {
            err = std::string("pipe: ") + std::strerror(errno);
            return false;
        }
        const pid_t pid = ::fork();
        if (pid < 0) {
            err = std::string("fork: ") + std::strerror(errno);
            return false;
        }
        if (pid == 0) {
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            ::close(to_child[0]); ::close(to_child[1]); ::close(from_child[0]); ::close(from_child[1]);
            std::vector<char *> argv;
            for (const std::string & a : args) {
                argv.push_back(const_cast<char *>(a.c_str()));
            }
            argv.push_back(nullptr);
            ::execv(executable.c_str(), argv.data());
            std::fprintf(stderr, "unbiased-strata: could not start the engine %s: %s\n", executable.c_str(), std::strerror(errno));
            ::_exit(127);
        }
        pid_ = pid;
        ::close(to_child[0]);
        ::close(from_child[1]);
        in_fd_  = to_child[1];
        out_fd_ = from_child[0];
        alive_  = true;
        return true;
    }

    // Reads start-up lines until READY. Loading a model takes minutes; `timeout_s` bounds the wait (<= 0: forever).
    // `READY <max_context> [stop]`: the context the engine actually allocated (its VRAM plan may lower the request);
    // `stop` means it honours STOP mid-request. INFO lines land in `info`.
    bool wait_ready(std::string & err, int timeout_s = 0) {
        const int64_t deadline = timeout_s > 0 ? now_ms() + (int64_t) timeout_s * 1000 : 0;
        std::string line;
        while (true) {
            int wait_ms = 1000;
            if (deadline) {
                const int64_t left = deadline - now_ms();
                if (left <= 0) {
                    err = "the engine did not become ready within " + std::to_string(timeout_s) + " s";
                    return false;
                }
                wait_ms = (int) std::min<int64_t>(left, 1000);
            }
            const int rc = read_line(line, wait_ms);
            if (rc < 0) {
                err = "the engine exited before it was ready (its messages are above)";
                return false;
            }
            if (rc == 0) {
                continue;
            }
            if (line.rfind("READY", 0) == 0) {
                char stop[16] = {0};
                long long ctx = 0;
                std::sscanf(line.c_str(), "READY %lld %15s", &ctx, stop);
                max_context_ = ctx;
                can_stop_    = std::strcmp(stop, "stop") == 0;
                ready_       = true;
                return true;
            }
            if (line.rfind("INFO ", 0) == 0) {
                parse_info(line.substr(5));
                continue;
            }
            if (line.rfind("ERR", 0) == 0) {
                err = line.size() > 4 ? line.substr(4) : "the engine reported an error";
                return false;
            }
            if (line.rfind("FATAL", 0) == 0) {
                err = line;
                return false;
            }
            // anything else before READY is chatter on stdout; the engine's log is on stderr
        }
    }

    // Writes one request line. `header` is `GEN <max_new> key=value...` or `GENI <max_new> key=value... <file>`.
    bool send_request(const std::string & header, const std::vector<int32_t> & ids) {
        std::string req = header;
        req.reserve(header.size() + ids.size() * 7 + 2);
        req += ' ';
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) {
                req += ',';
            }
            req += std::to_string(ids[i]);
        }
        req += '\n';
        busy_ = true;
        return send(req);
    }

    // Asks the engine to end the current request. Its output continues until a DONE with finish=cancel.
    bool stop() {
        return send("STOP\n");
    }

    // One classified line, waiting at most `timeout_ms` (0: only what is buffered). Returns false on timeout.
    // kind == EOF_ when the engine closed its stdout (it exited).
    bool next_line(EngineLine & out, int timeout_ms) {
        std::string line;
        const int rc = read_line(line, timeout_ms);
        if (rc == 0) {
            return false;
        }
        out = EngineLine{};
        out.raw = line;
        if (rc < 0) {
            out.kind = EngineLine::EOF_;
            return true;
        }
        classify(line, out);
        if (out.kind == EngineLine::DONE || out.kind == EngineLine::ERR) {
            busy_ = false;
        }
        return true;
    }

    // One command between requests with a one-line reply whose first word is `reply_prefix` (`STATS`, `VRAM`).
    bool command(const std::string & line, const char * reply_prefix, std::string & reply, std::string & err, int timeout_ms = 5000) {
        if (!alive_) { err = "the engine is not running"; return false; }
        if (!send(line + "\n")) { err = "the engine is not running"; return false; }
        std::string got;
        const int64_t deadline = now_ms() + timeout_ms;
        while (true) {
            const int64_t left = deadline - now_ms();
            if (left <= 0) { err = "no reply from the engine"; return false; }
            const int rc = read_line(got, (int) left);
            if (rc < 0) { err = "the engine exited"; return false; }
            if (rc == 0) { continue; }
            if (got.rfind(reply_prefix, 0) == 0) { reply = got; return true; }
            if (got.rfind("SERR", 0) == 0) { err = got; return false; }
            if (got.rfind("ERR", 0) == 0) { err = got.size() > 4 ? got.substr(4) : "the engine reported an error"; return false; }
        }
    }

    // Stop the engine: QUIT, up to 10 s for it to leave, then SIGKILL. Safe against reap_exit() on another thread.
    void quit() {
        // Take the pid out of the shared slot first: from here only this call owns the child, and kill() below
        // targets this positive local copy, never the slot (which reads -1 once the supervisor has reaped it).
        const pid_t pid = pid_.exchange(-1);
        if (pid > 0) {
            send("QUIT\n");
            int status = 0;
            pid_t reaped = 0;
            for (int i = 0; i < 100 && (reaped = ::waitpid(pid, &status, WNOHANG)) == 0; ++i) {
                ::usleep(100 * 1000);
            }
            if (reaped == 0) {   // still running; -1 means another thread already reaped it
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &status, 0);
            }
        }
        if (in_fd_  >= 0) { ::close(in_fd_);  in_fd_  = -1; }
        if (out_fd_ >= 0) { ::close(out_fd_); out_fd_ = -1; }
        alive_ = false;
    }

    // Non-blocking: has the child exited? `why` names the signal or code.
    bool reap_exit(std::string & why) {
        if (child_running(pid_, why)) {
            return false;
        }
        alive_ = false;
        return true;
    }

    bool alive() const { return alive_; }
    bool busy()  const { return busy_; }
    bool ready() const { return ready_; }
    bool can_stop() const { return can_stop_; }
    long long max_context() const { return max_context_; }
    const std::map<std::string, std::string> & info() const { return info_; }
    pid_t pid() const { return pid_; }

    static int64_t now_ms() {
        struct timespec ts;
        ::clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    }

private:
    bool send(const std::string & s) {
        if (in_fd_ < 0) {
            return false;
        }
        size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = ::write(in_fd_, s.data() + off, s.size() - off);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                alive_ = false;
                return false;
            }
            off += (size_t) n;
        }
        return true;
    }

    // 1: a line; 0: timeout; -1: EOF/error
    int read_line(std::string & line, int timeout_ms) {
        const int64_t deadline = now_ms() + timeout_ms;
        while (true) {
            const size_t nl = buf_.find('\n');
            if (nl != std::string::npos) {
                line.assign(buf_, 0, nl);
                buf_.erase(0, nl + 1);
                while (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                return 1;
            }
            if (out_fd_ < 0) {
                return -1;
            }
            int wait = timeout_ms <= 0 ? 0 : (int) std::max<int64_t>(0, deadline - now_ms());
            struct pollfd pfd = { out_fd_, POLLIN, 0 };
            const int pr = ::poll(&pfd, 1, wait);
            if (pr < 0) {
                if (errno == EINTR) {
                    continue;
                }
                alive_ = false;
                return -1;
            }
            if (pr == 0) {
                return 0;
            }
            char chunk[65536];
            const ssize_t n = ::read(out_fd_, chunk, sizeof chunk);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                // EOF: the engine closed stdout. Hand out what is buffered without a newline, then -1.
                if (!buf_.empty()) {
                    line.swap(buf_);
                    buf_.clear();
                    return 1;
                }
                alive_ = false;
                return -1;
            }
            buf_.append(chunk, (size_t) n);
        }
    }

    void parse_info(const std::string & kv) {
        size_t i = 0;
        while (i < kv.size()) {
            while (i < kv.size() && kv[i] == ' ') ++i;
            const size_t j = kv.find(' ', i);
            const std::string tok = kv.substr(i, j == std::string::npos ? std::string::npos : j - i);
            const size_t eq = tok.find('=');
            if (eq != std::string::npos) {
                info_[tok.substr(0, eq)] = tok.substr(eq + 1);
            }
            if (j == std::string::npos) break;
            i = j + 1;
        }
    }

    static void classify(const std::string & line, EngineLine & out) {
        if (line.size() > 2 && line[0] == 'T' && line[1] == ' ') {
            out.kind  = EngineLine::TOKEN;
            out.token = (int32_t) std::atoi(line.c_str() + 2);
        } else if (line.rfind("DONE ", 0) == 0) {
            char finish[32] = {0};
            // DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused> ...
            const int got = std::sscanf(line.c_str(), "DONE %lld %lld %lf %lf %31s %lld %lld %lld", &out.produced, &out.prompt_n,
                                        &out.prompt_ms, &out.decode_ms, finish, &out.draft_accepted, &out.draft_offered, &out.reused);
            out.kind   = EngineLine::DONE;
            out.finish = got >= 5 ? finish : "stop";
        } else if (line.rfind("PP ", 0) == 0) {
            out.kind = EngineLine::PP;
            std::sscanf(line.c_str(), "PP %lld %lld", &out.pp_done, &out.pp_total);
        } else if (line.rfind("RESUME ", 0) == 0) {
            out.kind = EngineLine::RESUME;
            out.n = std::atoll(line.c_str() + 7);
        } else if (line.rfind("REUSED ", 0) == 0) {
            out.kind = EngineLine::REUSED;
            out.n = std::atoll(line.c_str() + 7);
        } else if (line.rfind("ERR", 0) == 0) {
            out.kind    = EngineLine::ERR;
            out.message = line.size() > 4 ? line.substr(4) : "the engine reported an error";
        } else if (line.rfind("FATAL", 0) == 0) {
            out.kind    = EngineLine::ERR;
            out.message = line;
        } else if (line.rfind("INFO ", 0) == 0 || line.rfind("WARN ", 0) == 0) {
            out.kind = EngineLine::INFO;
        } else {
            out.kind = EngineLine::OTHER;
        }
    }

    std::atomic<pid_t> pid_{-1};
    int in_fd_   = -1;
    int out_fd_  = -1;
    std::string buf_;
    std::atomic<bool> alive_{false};
    bool busy_      = false;
    bool ready_     = false;
    bool can_stop_  = false;
    long long max_context_ = 0;
    std::map<std::string, std::string> info_;
};

} // namespace strata_frontend
