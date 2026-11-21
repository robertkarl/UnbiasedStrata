// UnbiasedStrata: the server context with a Strata engine behind it.
//
// This is the fork's replacement for llama.cpp's tools/server/server-context.cpp. Everything around it is llama.cpp's
// own, compiled unmodified from the submodule: the HTTP layer (server-http), the task queue and response reader
// (server-queue), the task and result types with their OpenAI/Anthropic/Responses JSON (server-task), request parsing,
// tokenization helpers and SSE formatting (server-common), the request schema (server-schema). What this file adds is
// the part that llama.cpp does with llama_context, llama_batch and common_sampler and that the engine does inside
// its own process: run one request at a time through `strata`/`strata-poly`, turn its `T <id>` lines into partial
// results, and its `DONE` line into the final one.
#pragma once

#include "server-http.h"

#include <memory>
#include <string>
#include <vector>

struct common_params;
struct strata_context_impl;

struct strata_context {
    std::unique_ptr<strata_context_impl> impl;

    strata_context();
    ~strata_context();

    // Registers the HTTP routes on `http`. Handlers run only once http.is_ready is set, after load().
    void init_routes(server_http_context & http);

    // fork+exec the engine. Call before any thread exists (before http.start()).
    bool start_engine(const common_params & params, const std::string & engine_exe, const std::vector<std::string> & engine_args, std::string & err);

    // Loads the vocabulary and chat template from the GGUF (vocab only, no tensors) and waits for the engine's READY.
    bool load(common_params & params, std::string & err);
    // --picture-cache: how many pictures' encoded rows are kept for the turns that resend them (0: none); before load()
    void set_picture_cache(size_t entries, size_t bytes);

    // Stop the task loop and wake pending HTTP readers before joining the HTTP thread.
    void start_loop();
    void stop_loop();
    void terminate();

    // Non-blocking: true once the engine process has exited; `why` names the signal or exit code.
    bool engine_exited(std::string & why);
    bool engine_alive() const;

    // the context the engine allocated (READY line)
    long long n_ctx() const;
};
