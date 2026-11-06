// UnbiasedStrata: the engine adapter against the fake engine (frontend/tests/fake-engine.py). No GPU, no model.
//
//   c++ -std=c++20 -I frontend frontend/tests/engine_test.cpp -o engine_test && ./engine_test frontend/tests/fake-engine.py
//
// Cases, each the shape of a bug UnbiasedStrata 1 met:
//   1. READY parsed: context and the stop capability
//   2. a request: tokens come back, DONE carries the counts, the adapter is idle afterwards
//   3. cancel: STOP mid-request, the rest is drained to DONE finish=cancel, the next request is clean
//   4. backpressure: a 100k-token burst the caller abandons does not fill the pipe for ever (drain to DONE)
//   5. ERR: a refused request leaves the adapter usable
//   6. the engine dies mid-request: EOF is reported, reap_exit names the exit code
//   7. a DONE whose prompt size does not match is visible to the caller (the context refuses such a reply)
#include "engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// the child inherits this process's environment when it is created, as with fork+exec
static void set_env(const std::string & k, const std::string & v) { _putenv_s(k.c_str(), v.c_str()); }
static void unset_env(const std::string & k) { _putenv_s(k.c_str(), ""); }
#else
static void set_env(const std::string & k, const std::string & v) { ::setenv(k.c_str(), v.c_str(), 1); }
static void unset_env(const std::string & k) { ::unsetenv(k.c_str()); }
#endif

using strata_frontend::Engine;
using strata_frontend::EngineLine;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

struct Run {
    std::vector<int32_t> tokens;
    EngineLine done;
    bool got_done = false, got_err = false, got_eof = false;
    std::string err;
};

// read until DONE/ERR/EOF; `cancel_after` tokens -> send STOP; `abandon_after` tokens -> stop reading tokens into
// the vector (the caller is "gone") but keep draining
static Run run(Engine & e, int cancel_after = -1, int timeout_ms = 5000) {
    Run r;
    EngineLine line;
    const int64_t deadline = Engine::now_ms() + timeout_ms;
    while (Engine::now_ms() < deadline) {
        if (!e.next_line(line, 50)) continue;
        switch (line.kind) {
            case EngineLine::TOKEN:
                r.tokens.push_back(line.token);
                if (cancel_after >= 0 && (int) r.tokens.size() == cancel_after) e.stop();
                break;
            case EngineLine::DONE: r.done = line; r.got_done = true; return r;
            case EngineLine::ERR:  r.err = line.message; r.got_err = true; return r;
            case EngineLine::EOF_: r.got_eof = true; return r;
            default: break;
        }
    }
    std::fprintf(stderr, "FAIL: timed out waiting for DONE\n");
    ++failures;
    return r;
}

static bool start(Engine & e, const std::string & script, const std::vector<std::pair<std::string, std::string>> & env) {
    for (auto & kv : env) set_env(kv.first, kv.second);
    std::string err;
    const bool ok = e.start(script, {script}, err) && e.wait_ready(err, 10);
    for (auto & kv : env) unset_env(kv.first);
    if (!ok) std::fprintf(stderr, "start: %s\n", err.c_str());
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: engine_test <fake-engine.py>\n"); return 2; }
    const std::string script = argv[1];

    // 1 + 2: READY, one request echoing the prompt
    {
        Engine e;
        CHECK(start(e, script, {{"FAKE_CONTEXT", "2048"}}));
        CHECK(e.ready());
        CHECK(e.max_context() == 2048);
        CHECK(e.can_stop());
        CHECK(e.info().count("engine") && e.info().at("engine") == "fake");
        CHECK(e.send_request("GEN 10 temperature=0.6", {5, 6, 7}));
        CHECK(e.busy());
        Run r = run(e);
        CHECK(r.got_done);
        CHECK((r.tokens == std::vector<int32_t>{5, 6, 7}));
        CHECK(r.done.produced == 3 && r.done.prompt_n == 3 && r.done.finish == "stop");
        CHECK(!e.busy());
        // max_new caps it
        CHECK(e.send_request("GEN 2", {1, 2, 3, 4}));
        r = run(e);
        CHECK(r.got_done && r.tokens.size() == 2 && r.done.finish == "length");
        e.quit();
        CHECK(!e.alive());
    }

    // 3: cancel mid-request, then a clean next request
    {
        Engine e;
        CHECK(start(e, script, {{"FAKE_N_TOKENS", "1000"}, {"FAKE_DELAY_MS", "2"}}));
        CHECK(e.send_request("GEN 1000", {1}));
        Run r = run(e, /* cancel_after */ 5);
        CHECK(r.got_done);
        CHECK(r.done.finish == "cancel");
        CHECK(r.tokens.size() < 1000);
        CHECK(e.send_request("GEN 3", {9}));
        r = run(e);
        CHECK(r.got_done && r.done.finish == "length" && r.tokens.size() == 3);
        e.quit();
    }

    // 4: a burst the caller abandons; the pipe would hold ~64 KB, the burst is ~700 KB
    {
        Engine e;
        CHECK(start(e, script, {{"FAKE_N_TOKENS", "100000"}}));
        CHECK(e.send_request("GEN 100000", {1}));
        Run r = run(e, /* cancel_after */ 1, /* timeout */ 30000);
        CHECK(r.got_done);
        // FAKE_N_TOKENS still applies, so the reply is one token of id 100 cut by max_new; what matters is
        // that nothing of the abandoned burst leaks into it
        CHECK(e.send_request("GEN 1", {2}));
        r = run(e);
        CHECK(r.got_done && r.tokens.size() == 1 && r.done.finish == "length");
        e.quit();
    }

    // 5: ERR leaves the adapter usable
    {
        Engine e;
        CHECK(start(e, script, {{"FAKE_ERR", "prompt too long"}}));
        CHECK(e.send_request("GEN 1", {1}));
        Run r = run(e);
        CHECK(r.got_err && r.err == "prompt too long");
        CHECK(!e.busy());
        CHECK(e.alive());
        e.quit();
    }

    // 6: the engine dies mid-request
    {
        Engine e;
        CHECK(start(e, script, {{"FAKE_N_TOKENS", "50"}, {"FAKE_DIE_AFTER", "3"}}));
        CHECK(e.send_request("GEN 50", {1}));
        Run r = run(e);
        CHECK(r.got_eof);
        CHECK(r.tokens.size() == 3);
        std::string why;
        for (int i = 0; i < 50 && !e.reap_exit(why); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(why == "engine exited with code 3");
        CHECK(!e.alive());
    }

    // 7: prompt size mismatch is visible
    {
        Engine e;
        CHECK(start(e, script, {{"FAKE_PROMPT_N", "99"}}));
        CHECK(e.send_request("GEN 5", {1, 2}));
        Run r = run(e);
        CHECK(r.got_done && r.done.prompt_n == 99);
        e.quit();
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("engine_test: 7 cases passed\n");
    return 0;
}
