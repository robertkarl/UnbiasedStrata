// UnbiasedStrata: the server context with a Strata engine behind it. See strata_context.hpp.
//
// Where a function here mirrors one in llama.cpp's tools/server/server-context.cpp it says so; those are the places
// to look when the submodule pin moves.

#include "strata_context.hpp"
#include "engine.hpp"
#include "literals.hpp"
#include "picture_cache.hpp"

#include "server-common.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <set>
#include <chrono>
#include <thread>
#include <unistd.h>

using strata_frontend::Engine;
using strata_frontend::EngineLine;

namespace {

// how long update_slots() waits for an engine line before returning to the queue (which then sees cancels)
constexpr int ENGINE_POLL_MS = 20;

// the engine keeps this many context slots back; a prompt plus max_new may not reach the context (UnbiasedStrata 1,
// REGINA-112: every request without max_tokens got a 500 until the front end counted it)
constexpr int ENGINE_CONTEXT_RESERVE = 8;

// mirrors server_slot in server-context.cpp, for the one slot the engine is
struct Slot {
    std::unique_ptr<server_task> task;

    bool started    = false; // GEN written
    bool cancelled  = false; // the client went away: STOP sent, output is drained, no final result
    bool stop_sent  = false; // STOP sent for our own reason (stop string, limit); the final result is still sent

    std::string  generated_text;
    llama_tokens generated_tokens;
    size_t       n_sent_text = 0;

    server_slot_stats stats;
    stop_type   stop = STOP_TYPE_NONE;
    std::string stopping_word;
    bool        has_new_line = false;
    bool        truncated    = false;
    bool        has_next_token = true;

    int32_t n_predict = -1;

    std::string images_path; // the request's picture rows for GENI (step 8); unlinked when the slot is released
    std::vector<uint64_t> image_keys; // one per picture, for img_keys= on the GENI line (0: none)

    bool active() const { return task != nullptr; }

    void release() {
        if (!images_path.empty()) {
            ::unlink(images_path.c_str());
        }
        *this = Slot{};
    }

    bool has_budget() const {
        return n_predict < 0 || (int32_t) stats.n_gen < n_predict;
    }

    // mirrors server_slot::find_stopping_strings
    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) const {
        size_t stop_pos = std::string::npos;
        for (const std::string & word : task->params.antiprompt) {
            size_t pos;
            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            } else {
                pos = string_find_partial_stop(text, word);
            }
            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    const_cast<Slot *>(this)->stop          = STOP_TYPE_WORD;
                    const_cast<Slot *>(this)->stopping_word = word;
                    const_cast<Slot *>(this)->has_next_token = false;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }
};

// mirrors server_res_generator in server-context.cpp
struct res_generator : server_res_spipe {
    server_response_reader rd;
    res_generator(server_queue & queue_tasks, server_response & queue_results)
            : rd(queue_tasks, queue_results, /* polling_interval_seconds */ 1) {}
    void ok(const json & response_data) {
        status = 200;
        data   = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data   = safe_json_to_str({{ "error", error_data }});
    }
};

// mirrors ex_wrapper in server.cpp: a handler never throws; an exception becomes an error response
server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error   = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error   = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error   = ERROR_TYPE_SERVER;
            message = "unknown error";
        }
        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data   = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

std::string file_stem(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = name.rfind(".gguf");
    if (dot != std::string::npos) {
        name.erase(dot);
    }
    return name;
}

} // namespace

struct strata_context_impl {
    common_params params;

    llama_model *       model = nullptr;
    const llama_vocab * vocab = nullptr;

    // images (PLAN.md step 8): the model's vision encoder and projector (--mmproj) run in this process on the CPU
    // through llama.cpp's mtmd, as upstream's strata-vision did in a child; the rows go to the engine in a file named
    // by a GENI request. Without --mmproj mctx stays null and image_url parts are refused by
    // oaicompat_chat_params_parse with llama.cpp's own message.
    mtmd_context *        mctx = nullptr;
    mtmd_helper_init_opt  init_opt{};
    int32_t               vision_width = 0;  // the projector's output width, checked against the model's embedding
    int32_t               image_pad_id = -1; // the token whose runs in the ids the engine replaces with picture rows
    bool                  p_mmproj_gpu = false;
    strata_frontend::PictureCache pictures; // encoded rows per picture (--picture-cache N; 64 pictures, 1 GiB)

    // the texts of the vocabulary's control tokens and marker-like added tokens: inside a message they are text
    // (literals.hpp, USTRAT-14)
    strata_frontend::Literals literals;

    server_chat_params           chat_params;
    std::map<std::string, bool>  chat_caps;
    std::vector<llama_logit_bias> logit_bias_eog; // empty: the engine applies logit bias itself

    std::string model_name;
    std::string model_path;

    server_queue    queue_tasks;
    server_response queue_results;

    Engine                   engine;
    std::string              engine_exe;
    std::vector<std::string> engine_args;
    bool                     engine_is_poly = false;
    long long                n_ctx = 0;
    std::atomic<bool>        engine_dead{false};

    Slot slot;

    // ---- queue callbacks -------------------------------------------------------------------------------------------

    void send_error(int id_task, const std::string & error, enum error_type type, int32_t n_prompt_tokens = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());
        auto res = std::make_unique<server_task_result_error>();
        res->id       = id_task;
        res->err_type = type;
        res->err_msg  = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = (int32_t) n_ctx;
        queue_results.send(std::move(res));
    }

    void send_error(const server_task & task, const std::string & error, enum error_type type) {
        send_error(task.id, error, type, task.n_tokens());
    }

    // mirrors process_single_task in server-context.cpp, for the task types this front end has
    bool process_new_task(server_task && task, bool is_yielding) {
        (void) is_yielding; // never yields: the engine is not driven from this process
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
                if (slot.active()) {
                    queue_tasks.defer(std::move(task));
                    return true;
                }
                launch(std::move(task));
                return true;

            case SERVER_TASK_TYPE_CANCEL:
                if (slot.active() && slot.task->id == task.id_target) {
                    cancel_current();
                }
                return true;

            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                return true;

            case SERVER_TASK_TYPE_METRICS: {
                auto res = std::make_unique<server_task_result_metrics>();
                res->id = task.id;
                res->n_processing_slots = slot.active() ? 1 : 0;
                res->n_tasks_deferred   = (int) queue_tasks.queue_tasks_deferred_size();
                res->metrics            = metrics;
                queue_results.send(std::move(res));
                return true;
            }

            default:
                send_error(task, "this request type is not supported by this server", ERROR_TYPE_NOT_SUPPORTED);
                return true;
        }
    }

    // mirrors update_slots in server-context.cpp: called by the queue after each round of tasks
    void update_slots() {
        if (!slot.active()) {
            return;
        }
        // keep the loop coming back while a request runs
        {
            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // read what the engine has, for a bounded time, so a cancel posted meanwhile is seen within ENGINE_POLL_MS
        const int64_t until = Engine::now_ms() + ENGINE_POLL_MS;
        EngineLine line;
        while (slot.active()) {
            const int left = (int) (until - Engine::now_ms());
            if (!engine.next_line(line, std::max(left, 0))) {
                break; // nothing more right now
            }
            handle_line(line);
            if (left <= 0) {
                break;
            }
        }
    }

    // ---- one request --------------------------------------------------------------------------------------------

    // mirrors launch_slot_with_task
    void launch(server_task && task) {
        slot.release();
        slot.task = std::make_unique<server_task>(std::move(task));
        const server_task & t = *slot.task;

        if (!engine.alive()) {
            send_error(t, "the engine is not running", ERROR_TYPE_UNAVAILABLE);
            slot.release();
            return;
        }

        const int32_t n_prompt = t.n_tokens();
        if (n_prompt == 0) {
            send_error(t, "the prompt is empty", ERROR_TYPE_INVALID_REQUEST);
            slot.release();
            return;
        }
        std::vector<int32_t> ids;
        if (t.tokens.has_mtmd) {
            std::string ierr;
            if (!encode_pictures(t, ids, ierr)) {
                send_error(t, ierr, ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
        } else {
            ids = t.tokens.get_text_tokens();
        }
        const long long room = n_ctx - n_prompt - ENGINE_CONTEXT_RESERVE;
        if (room <= 0) {
            send_error(t, string_format("the request exceeds the available context size: %d prompt tokens, context %lld (%d kept back)",
                                        n_prompt, n_ctx, ENGINE_CONTEXT_RESERVE), ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            slot.release();
            return;
        }
        // n_predict: -1 and anything past the context become what is left, as llama-server does
        slot.n_predict = t.params.n_predict;
        if (slot.n_predict < 0 || slot.n_predict > room) {
            if (slot.n_predict > room) {
                SRV_WRN("n_predict %d exceeds the room left in the context, lowered to %lld\n", slot.n_predict, room);
            }
            slot.n_predict = (int32_t) room;
        }
        if (!t.params.sampling.grammar.grammar.empty()) {
            // the engine samples inside its own process and exposes no logits: a grammar (tools, json_schema,
            // tool_choice=required) cannot constrain it. Same as upstream PolyStrata. The parser still runs on the output.
            SRV_WRN("task %d: a grammar was requested (%zu chars); this engine does not constrain sampling, parsing only\n",
                    t.id, t.params.sampling.grammar.grammar.size());
        }

        const std::string header = gen_header(t, slot.n_predict, slot.images_path, slot.image_keys);
        if (!engine.send_request(header, ids)) {
            send_error(t, "the engine is not running", ERROR_TYPE_UNAVAILABLE);
            slot.release();
            return;
        }
        slot.started = true;
        slot.stats.update_prompt_start();
        SRV_INF("task %d: %s prompt=%d n_predict=%d\n", t.id, header.c_str(), n_prompt, slot.n_predict);

        if (t.params.stream) {
            // the HTTP side sends the 200 and the stream headers on this, before the first token
            send_partial_response(completion_token_output{}, /* is_progress */ false, /* is_begin */ true);
        }
    }

    // `GEN <max_new> key=value...` from the request's sampling parameters. Absent keys mean greedy to the engine
    // (generate.cpp: "Absent = greedy"), so temperature <= 0 sends none of them.
    std::string gen_header(const server_task & t, int32_t max_new, const std::string & images_path = std::string(),
                           const std::vector<uint64_t> & image_keys = {}) const {
        const auto & s = t.params.sampling;
        std::string h = (images_path.empty() ? "GEN " : "GENI ") + std::to_string(max_new);
        char b[128];
        if (s.temp > 0.0f) {
            std::snprintf(b, sizeof b, " temperature=%g", (double) s.temp); h += b;
            if (s.top_p > 0.0f && s.top_p < 1.0f) { std::snprintf(b, sizeof b, " top_p=%g", (double) s.top_p); h += b; }
            // strata's sampled path takes top_k in 1..64; 0 means "whole vocabulary" to llama.cpp
            int32_t top_k = s.top_k <= 0 ? 64 : std::min<int32_t>(s.top_k, 64);
            std::snprintf(b, sizeof b, " top_k=%d", top_k); h += b;
            if (s.min_p > 0.0f) { std::snprintf(b, sizeof b, " min_p=%g", (double) s.min_p); h += b; }
            if (s.seed != LLAMA_DEFAULT_SEED) { std::snprintf(b, sizeof b, " seed=%u", s.seed); h += b; }
        }
        const bool penalties = s.penalty_repeat != 1.0f || s.penalty_freq != 0.0f || s.penalty_present != 0.0f;
        if (penalties && s.penalty_last_n != 0) {
            std::snprintf(b, sizeof b, " penalty_last_n=%d penalty_repeat=%g penalty_freq=%g penalty_present=%g",
                          s.penalty_last_n < 0 ? (int) n_ctx : s.penalty_last_n,
                          (double) s.penalty_repeat, (double) s.penalty_freq, (double) s.penalty_present);
            h += b;
        }
        if (!s.logit_bias.empty() && !engine_is_poly) {
            std::string lb = " logit_bias=";
            for (size_t i = 0; i < s.logit_bias.size(); ++i) {
                if (i) lb += ',';
                std::snprintf(b, sizeof b, "%d:%g", s.logit_bias[i].token, (double) s.logit_bias[i].bias);
                lb += b;
            }
            h += lb;
        }
        if (!images_path.empty()) {
            // GENI <max_new> [key=value ...] image_pad=<id> <file> <ids>: the engine reads strata-vision records from the
            // file and places them at the runs of image_pad (src/poly/main.cpp take_pictures, generate.cpp for strata)
            // img_keys= (#2228): the conversation cache matches a picture by its key and grid, not its rows
            const std::string keys = strata_frontend::PictureCache::hex_list(image_keys);
            if (!keys.empty()) {
                h += " img_keys=" + keys;
            }
            h += " image_pad=" + std::to_string(image_pad_id) + " " + images_path;
        }
        return h;
    }

    // ---- pictures (step 8) ------------------------------------------------------------------------------------------

    // The projector file and the vision encoder, on the CPU: the GPU is the engine's. Fails when the projector's
    // output width is not the model's embedding width (the wrong mmproj for this model) or the vocabulary has no
    // <|image_pad|> token (a model family whose picture placeholders this front end does not know yet).
    bool init_vision(const common_params & p, std::string & err) {
        int32_t proj_width = 0;
        {
            gguf_init_params gp{ /* no_alloc */ true, /* ctx */ nullptr };
            gguf_context * gg = gguf_init_from_file(p.mmproj.path.c_str(), gp);
            if (gg == nullptr) {
                err = "cannot read the projector file " + p.mmproj.path;
                return false;
            }
            const int64_t k = gguf_find_key(gg, "clip.vision.projection_dim");
            if (k >= 0) {
                proj_width = (int32_t) gguf_get_val_u32(gg, k);
            }
            gguf_free(gg);
        }
        if (proj_width <= 0) {
            err = "the projector file has no clip.vision.projection_dim: " + p.mmproj.path;
            return false;
        }
        // the vocab-only model has no hparams; the embedding width comes from the GGUF's metadata
        char arch[64] = {0}, val[32] = {0};
        int32_t want = 0;
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof arch) > 0 &&
            llama_model_meta_val_str(model, (std::string(arch) + ".embedding_length").c_str(), val, sizeof val) > 0) {
            want = std::atoi(val);
        }
        if (want <= 0) {
            err = "cannot read the model's embedding width from " + p.model.path;
            return false;
        }
        if (proj_width != want) {
            err = string_format("the projector writes rows of %d values but the model's embedding width is %d: %s is not this model's mmproj",
                                proj_width, want, p.mmproj.path.c_str());
            return false;
        }
        const llama_tokens pad = common_tokenize(vocab, "<|image_pad|>", false, true);
        if (pad.size() != 1) {
            err = "the model has no <|image_pad|> token; pictures are supported for the Qwen family only so far";
            return false;
        }
        if (p.mmproj_use_gpu && ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
            // ggml in this build has no GPU backend (the engines bring their own CUDA; llama.cpp's ggml is built
            // without it), so mtmd would silently run on the CPU
            err = "--mmproj-gpu: this build's ggml has no GPU backend; the encoder can only run on the CPU";
            return false;
        }
        // mtmd's default logger writes every level to stderr, its debug line add_text() among them: the whole
        // rendered prompt, once per request, into the journal (USTRAT-14). Through the common log instead, where
        // debug lines need the server's verbosity raised.
        mtmd_log_set([](ggml_log_level level, const char * text, void *) {
            if (level == GGML_LOG_LEVEL_DEBUG && common_log_get_verbosity_thold() < LOG_DEFAULT_DEBUG) return;
            common_log_add(common_log_main(), level, "%s", text);
        }, nullptr);
        mtmd_context_params cp = mtmd_context_params_default();
        // the request parser (server-common) replaces each image_url part by a marker that is random per process
        // (get_media_marker); mtmd must look for that same marker, as server-context.cpp sets it
        cp.media_marker  = get_media_marker();
        cp.use_gpu       = p.mmproj_use_gpu; // CPU unless --mmproj-gpu (main.cpp): the card is the engine's
        cp.print_timings = false;
        cp.warmup        = p.mmproj_use_gpu; // on the card, allocate the work buffers now, not at the first picture
        cp.n_threads     = p.cpuparams.n_threads > 0 ? p.cpuparams.n_threads
                                                     : std::max(1u, std::thread::hardware_concurrency() / 2);
        // the picture's row budget: upstream's installer gave the CPU encoder 300 tokens and the GPU one 1024
        // (setup.py VISION); the encoder's attention is quadratic in it (a 1920x1080 screenshot uncapped is 2040
        // rows: 146 s on a Ryzen 7 1700). --image-max-tokens overrides.
        cp.image_max_tokens = p.image_max_tokens > 0 ? p.image_max_tokens : (p.mmproj_use_gpu ? 1024 : 300);
        if (p.image_min_tokens > 0) {
            cp.image_min_tokens = p.image_min_tokens;
        }
        mctx = mtmd_init_from_file(p.mmproj.path.c_str(), model, cp);
        if (mctx == nullptr || !mtmd_support_vision(mctx)) {
            if (mctx) { mtmd_free(mctx); mctx = nullptr; }
            err = "cannot load the vision encoder " + p.mmproj.path;
            return false;
        }
        vision_width = proj_width;
        image_pad_id = pad[0];
        p_mmproj_gpu = p.mmproj_use_gpu;
        LOG_INF("vision: %s, rows of %d, encoder on the %s (%d CPU threads), at most %d rows a picture, image_pad=%d\n",
                p.mmproj.path.c_str(), vision_width, cp.use_gpu ? "GPU" : "CPU", cp.n_threads, cp.image_max_tokens, image_pad_id);
        LOG_INF("vision: the rows of the last %zu pictures (%zu MiB at most) are kept for the turns that resend them\n",
                pictures.max_entries(), pictures.max_bytes() >> 20);
        return true;
    }

    // Encodes the request's pictures (the media chunks of its tokens) into one file of strata-vision records and
    // writes the ids the engine reads: the text tokens as they are, a run of image_pad per picture, one per row.
    bool encode_pictures(const server_task & t, std::vector<int32_t> & ids, std::string & err) {
        if (mctx == nullptr) {
            err = "images need the server started with --mmproj";
            return false;
        }
        std::FILE * f = open_pictures_file(err);
        if (f == nullptr) {
            return false;
        }
        const bool filled = fill_pictures(t, f, ids, err);
        const bool closed = std::fclose(f) == 0;
        if (!filled) {
            return false;
        }
        if (!closed) {
            err = "cannot write the pictures file";
            return false;
        }
        if (!ids.empty() && ids.back() == image_pad_id) {
            err = "a prompt cannot end in a picture";
            return false;
        }
        return true;
    }

    // the request's pictures file (slot.images_path, unlinked when the slot is released)
    std::FILE * open_pictures_file(std::string & err) {
        char name[] = "/tmp/unbiased-strata-pictures-XXXXXX";
        const int fd = ::mkstemp(name);
        if (fd < 0) {
            err = "cannot create the pictures file";
            return nullptr;
        }
        slot.images_path = name;
        std::FILE * f = ::fdopen(fd, "wb");
        if (f == nullptr) {
            ::close(fd);
            err = "cannot open the pictures file";
        }
        return f;
    }

    // the ids with every picture's rows written to `f` and image_pad in their place; slot.image_keys for the GENI line
    bool fill_pictures(const server_task & t, std::FILE * f, std::vector<int32_t> & ids, std::string & err) {
        const size_t n = t.tokens.size();
        ids.reserve(n);
        slot.image_keys.clear();
        for (size_t i = 0; i < n;) {
            if (t.tokens[i] != LLAMA_TOKEN_NULL) {
                ids.push_back(t.tokens[i]);
                ++i;
                continue;
            }
            const mtmd_input_chunk * ch = image_chunk(t, i);
            if (ch == nullptr) {
                err = "only pictures are supported, not audio or video";
                return false;
            }
            const auto rows = picture_rows(ch, t.id, (int) slot.image_keys.size() + 1, err);
            if (rows == nullptr) {
                return false;
            }
            if (!write_rows(f, *rows)) {
                err = "short write to the pictures file";
                return false;
            }
            ids.insert(ids.end(), (size_t) rows->n_tok, image_pad_id);
            slot.image_keys.push_back(strata_frontend::PictureCache::engine_key(chunk_id(ch)));
            i += (size_t) rows->n_tok;
        }
        return true;
    }

    // the image chunk at position i of the task's tokens, nullptr for anything else
    static const mtmd_input_chunk * image_chunk(const server_task & t, size_t i) {
        const mtmd_input_chunk * ch = nullptr;
        try {
            ch = t.tokens.find_chunk(i).get();
        } catch (const std::exception &) {
            return nullptr;
        }
        if (ch == nullptr || mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return nullptr;
        }
        return ch;
    }

    // mtmd names a bitmap by the SHA-256 of its bytes; the chunk carries it (empty for a placeholder)
    static std::string chunk_id(const mtmd_input_chunk * ch) {
        const char * id = mtmd_input_chunk_get_id(ch);
        return id == nullptr ? std::string() : std::string(id);
    }

    // one picture's rows: the cache's copy when the picture was seen before, else encoded now and remembered
    strata_frontend::PictureCache::Rows picture_rows(const mtmd_input_chunk * ch, int task_id, int index, std::string & err) {
        const std::string key = chunk_id(ch);
        if (auto hit = pictures.find(key)) {
            SRV_INF("task %d: picture %d: %d rows (%dx%d) from the cache\n", task_id, index, hit->n_tok, hit->nx, hit->ny);
            return hit;
        }
        const auto t0 = std::chrono::steady_clock::now();
        strata_frontend::PictureRows rows;
        if (!encode_rows(ch, index, rows, err)) {
            return nullptr;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        SRV_INF("task %d: picture %d: %d rows (%dx%d) encoded in %.0f ms on the %s\n", task_id, index, rows.n_tok, rows.nx, rows.ny, ms,
                p_mmproj_gpu ? "GPU" : "CPU");
        return pictures.put(key, std::move(rows));
    }

    // the projector's output for one chunk
    bool encode_rows(const mtmd_input_chunk * ch, int index, strata_frontend::PictureRows & rows, std::string & err) {
        const int32_t n_tok = (int32_t) mtmd_input_chunk_get_n_tokens(ch);
        if (mtmd_encode_chunk(mctx, ch) != 0) {
            err = "the vision encoder failed";
            return false;
        }
        const float * embd = mtmd_get_output_embd(mctx);
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);
        if (embd == nullptr || it == nullptr || n_tok < 1) {
            err = "the vision encoder returned nothing";
            return false;
        }
        if (!picture_grid(it, n_tok, rows.nx, rows.ny)) {
            err = string_format("picture %d: %d rows are not a grid (tiled encoders are not supported)", index, n_tok);
            return false;
        }
        rows.n_tok = n_tok;
        rows.width = vision_width;
        rows.embd.assign(embd, embd + (size_t) n_tok * (size_t) vision_width);
        return true;
    }

    // the grid from the decoder positions of the last row (strata-vision does the same)
    static bool picture_grid(const mtmd_image_tokens * it, int32_t n_tok, int32_t & nx, int32_t & ny) {
        const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n_tok - 1);
        nx = (int32_t) last.x + 1;
        ny = (int32_t) last.y + 1;
        if (n_tok > 1 && nx == n_tok && ny == n_tok) {
            ny = 1;
        }
        return (int64_t) nx * ny == n_tok;
    }

    // one strata-vision record: the header, then the rows
    static bool write_rows(std::FILE * f, const strata_frontend::PictureRows & r) {
        const int32_t hdr[5] = {0x31455653, r.n_tok, r.nx, r.ny, r.width};
        if (std::fwrite(hdr, sizeof hdr, 1, f) != 1) {
            return false;
        }
        return std::fwrite(r.embd.data(), sizeof(float), r.embd.size(), f) == r.embd.size();
    }

    void cancel_current() {
        if (!slot.active() || slot.cancelled) {
            return;
        }
        SRV_INF("task %d: cancelled by the client, stopping the engine's request\n", slot.task->id);
        slot.cancelled = true;
        if (!engine.stop()) {
            // the engine is gone; the supervisor will report it
            slot.release();
        }
    }

    void handle_line(const EngineLine & line) {
        switch (line.kind) {
            case EngineLine::TOKEN:
                if (!slot.cancelled && slot.has_next_token) {
                    on_token(line.token);
                }
                break;

            case EngineLine::RESUME:
                slot.stats.n_prompt_cached = (uint64_t) line.n;
                break;

            case EngineLine::PP:
                if (!slot.cancelled && slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(completion_token_output{}, /* is_progress */ true, false, line.pp_done);
                }
                break;

            case EngineLine::DONE:
                on_done(line);
                break;

            case EngineLine::ERR:
                SRV_ERR("task %d: engine: %s\n", slot.task->id, line.message.c_str());
                if (!slot.cancelled) {
                    send_error(*slot.task, "engine: " + line.message, ERROR_TYPE_SERVER);
                }
                finish_slot();
                break;

            case EngineLine::EOF_:
                SRV_ERR("%s", "the engine closed its output during a request\n");
                if (!slot.cancelled) {
                    send_error(*slot.task, "the engine exited during the request", ERROR_TYPE_UNAVAILABLE);
                }
                finish_slot();
                engine_dead = true;
                queue_tasks.terminate();
                break;

            case EngineLine::INFO:
                LOG_INF("engine: %s\n", line.raw.c_str());
                break;

            case EngineLine::REUSED:
            case EngineLine::OTHER:
            case EngineLine::NONE:
                break;
        }
    }

    bool accept_special_token(llama_token token) const {
        return params.special ||
            slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
    }

    // mirrors post_decode + process_token in server-context.cpp
    void on_token(llama_token tok) {
        if (slot.stats.n_gen == 0) {
            slot.stats.update_prompt_last();
        }
        slot.stats.n_gen++;
        slot.stats.update_gen_last();

        completion_token_output result;
        result.tok          = tok;
        result.prob         = 1.0f;
        result.text_to_send = common_token_to_piece(vocab, tok, accept_special_token(tok));

        const std::string token_str = result.text_to_send;
        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(tok);
        }
        slot.has_next_token = true;

        const bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());
            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(slot.generated_text.begin() + pos + stop_pos, slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, tok)) {
                stop_pos  = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            if (send_text) {
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
            } else {
                result.text_to_send = "";
            }
            if (slot.task->params.stream) {
                send_partial_response(result, false, false);
            }
        }

        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;
        }
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;
        }
        if (llama_vocab_is_eog(vocab, tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;
        }

        // a stop the engine does not know about (stop string, our limit): tell it, keep draining to its DONE
        if (!slot.has_next_token && slot.stop != STOP_TYPE_EOS && !slot.stop_sent) {
            slot.stop_sent = true;
            engine.stop();
        }
    }

    void on_done(const EngineLine & d) {
        if (!slot.active()) {
            return;
        }
        const server_task & t = *slot.task;
        // the engine's own clock: prompt and decode times as it measured them
        SRV_INF("task %d: DONE produced=%lld prompt=%lld prompt_ms=%.1f decode_ms=%.1f finish=%s drafts=%lld/%lld reused=%lld\n",
                t.id, d.produced, d.prompt_n, d.prompt_ms, d.decode_ms, d.finish.c_str(), d.draft_accepted, d.draft_offered, d.reused);

        if (d.prompt_n != (long long) t.n_tokens()) {
            // the engine answered for another prompt: never pass that off as this request's reply
            SRV_ERR("task %d: engine prompt size mismatch: sent %d, engine reports %lld; refusing the response\n", t.id, t.n_tokens(), d.prompt_n);
            if (!slot.cancelled) {
                send_error(t, "engine response prompt size mismatch", ERROR_TYPE_SERVER);
            }
            finish_slot();
            return;
        }

        slot.stats.n_prompt_cached    = (uint64_t) std::max<long long>(d.reused, (long long) slot.stats.n_prompt_cached);
        slot.stats.n_prompt_processed = (uint64_t) std::max<long long>(0, d.prompt_n - (long long) slot.stats.n_prompt_cached);
        slot.stats.n_draft_tokens     = (uint64_t) d.draft_offered;
        slot.stats.n_draft_accepted   = (uint64_t) d.draft_accepted;
        if (slot.stats.t_prompt_last == 0) {
            slot.stats.update_prompt_last();
        }
        if (slot.stats.t_gen_last == 0 && slot.stats.n_gen > 0) {
            slot.stats.update_gen_last();
        }

        if (slot.cancelled) {
            finish_slot();
            return;
        }
        if (d.finish == "nonfinite") {
            send_error(t, "the engine produced non-finite logits", ERROR_TYPE_SERVER);
            finish_slot();
            return;
        }
        if (slot.stop == STOP_TYPE_NONE) {
            if (d.finish == "length") {
                slot.stop      = STOP_TYPE_LIMIT;
                slot.truncated = false;
            } else {
                slot.stop = STOP_TYPE_EOS;
            }
        }
        send_final_response();
        finish_slot();
    }

    void finish_slot() {
        slot.release();
        queue_tasks.pop_deferred_task(-1);
    }

    // mirrors send_partial_response in server-context.cpp
    void send_partial_response(const completion_token_output & tkn, bool is_progress, bool is_begin, long long pp_done = 0) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();
        const server_task & t = *slot.task;

        res->id    = t.id;
        res->index = t.index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = t.n_tokens();
            res->progress.cache     = (int32_t) slot.stats.n_prompt_cached;
            res->progress.processed = (int32_t) pp_done;
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = (int32_t) slot.stats.n_gen;
        res->n_prompt_tokens       = t.n_tokens();
        res->n_prompt_tokens_cache = (int32_t) slot.stats.n_prompt_cached;
        res->post_sampling_probs   = t.params.post_sampling_probs;

        res->verbose           = t.params.verbose;
        res->res_type          = t.params.res_type;
        res->oaicompat_model   = t.params.oaicompat_model;
        res->oaicompat_cmpl_id = t.params.oaicompat_cmpl_id;

        if (slot.stop != STOP_TYPE_NONE || t.params.timings_per_token) {
            res->stats = slot.stats;
        }
        queue_results.send(std::move(res));
    }

    // mirrors send_final_response in server-context.cpp
    void send_final_response() {
        auto res = std::make_unique<server_task_result_cmpl_final>();
        server_task & t = *slot.task;

        res->id      = t.id;
        res->id_slot = 0;
        res->index   = t.index;

        if (t.params.stream) {
            res->content = "";
            res->tokens  = llama_tokens{};
        } else {
            res->content = std::move(slot.generated_text);
            res->tokens  = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = tokens_to_str(vocab, t.tokens.get_text_tokens());
        res->response_fields = std::move(t.params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = (int32_t) slot.stats.n_gen;
        res->n_prompt_tokens       = t.n_tokens();
        res->n_prompt_tokens_cache = (int32_t) slot.stats.n_prompt_cached;
        res->n_tokens_cached       = t.n_tokens() + (int32_t) slot.stats.n_gen;
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = t.params.post_sampling_probs;

        res->verbose           = t.params.verbose;
        res->stream            = t.params.stream;
        // always report usage in the final streamed chunk: upstream PolyStrata's server does, and its clients
        // (tools/strata_bench.py reads prompt_tokens from it) do not send stream_options.include_usage
        res->include_usage     = true;
        res->res_type          = t.params.res_type;
        res->oaicompat_model   = t.params.oaicompat_model;
        res->oaicompat_cmpl_id = t.params.oaicompat_cmpl_id;

        res->generation_params = t.params;

        // cumulative server metrics
        metrics.add_prompt(slot.stats.n_prompt_processed, (uint64_t) std::max<double>(0, slot.stats.t_prompt_ms() * 1000));
        metrics.add_prompt_cached(slot.stats.n_prompt_cached);
        metrics.predict.add(slot.stats.n_gen, slot.stats.n_gen_steps(), (uint64_t) slot.stats.t_gen_us());
        metrics.n_draft_tokens   += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted += slot.stats.n_draft_accepted;

        queue_results.send(std::move(res));
    }

    server_metrics metrics;

    // ---- routes ----------------------------------------------------------------------------------------------------

    std::unique_ptr<res_generator> create_response() {
        return std::make_unique<res_generator>(queue_tasks, queue_results);
    }

    json model_info_json() const {
        json modalities = json::array({"text"});
        if (chat_params.allow_image) {
            modalities.push_back("image");
        }
        return json {
            {"id",       model_name},
            {"object",   "model"},
            {"created",  std::time(0)},
            {"owned_by", "unbiasedstrata"},
            {"context_length",   n_ctx},
            {"input_modalities", modalities},
            {"meta", {
                {"vocab_type",  llama_vocab_type(vocab)},
                {"n_vocab",     llama_vocab_n_tokens(vocab)},
                {"n_ctx",       n_ctx},
                {"engine",      engine_is_poly ? "strata-poly" : "strata"},
            }},
        };
    }

    // ---- the literals of a request (literals.hpp) ----------------------------------------------------------------

    // the control tokens' texts and the marker-like added tokens' texts, from the vocabulary
    void init_literals() {
        std::vector<std::string> texts;
        const int32_t n = llama_vocab_n_tokens(vocab);
        for (int32_t id = 0; id < n; ++id) {
            const llama_token_attr attr = llama_vocab_get_attr(vocab, id);
            const char * text = llama_vocab_get_text(vocab, id);
            if (text == nullptr || *text == '\0') continue;
            if (attr & LLAMA_TOKEN_ATTR_CONTROL) {
                texts.emplace_back(text);
            } else if ((attr & LLAMA_TOKEN_ATTR_USER_DEFINED) && strata_frontend::Literals::marker_like(text)) {
                texts.emplace_back(text);
            }
        }
        literals = strata_frontend::Literals::from_texts(std::move(texts));
        LOG_INF("literals: %zu control-token texts are text inside a message%s\n", literals.tags.size(),
                literals.think_open.empty() ? "" : " (an assistant message keeps a leading <think> block)");
    }

    // the first UTF-8 character boundary at or after the middle of `s`; 0 when `s` is one character
    static size_t utf8_half(const std::string & s) {
        size_t cut = s.size() / 2;
        while (cut < s.size() && ((unsigned char) s[cut] & 0xC0) == 0x80) ++cut;
        return cut < s.size() ? cut : 0;
    }

    // Text inside a plain span, tokenized as text. parse_special=false keeps the control tokens as text; an added
    // token the tokenizer matches regardless (USER_DEFINED: <think>, <tool_call> in Qwen's vocabulary) is split in
    // two and each half tokenized again, until no piece is one. A one-character added token is that character.
    void tokenize_plain(const std::string & text, llama_tokens & out) const {
        for (llama_token t : common_tokenize(vocab, text, false, false)) {
            const llama_token_attr attr = llama_vocab_get_attr(vocab, t);
            if ((attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) == 0) {
                out.push_back(t);
                continue;
            }
            const std::string piece = common_token_to_piece(vocab, t, true);
            const size_t cut = utf8_half(piece);
            if (cut == 0) { // one character: that character
                out.push_back(t);
                continue;
            }
            tokenize_plain(piece.substr(0, cut), out);
            tokenize_plain(piece.substr(cut), out);
        }
    }

    // the rendered prompt tokenized: the plain spans (literals.hpp) as text, the rest with specials parsed and the
    // media markers resolved, as process_mtmd_prompt / tokenize_input_prompts do for a prompt without plain spans
    server_tokens tokenize_prompt(const std::string & prompt, const std::vector<std::pair<size_t, size_t>> & plain,
                                  const std::vector<raw_buffer> & files) const {
        server_tokens out(llama_tokens{}, mctx != nullptr);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        if (add_bos) {
            out.push_back(llama_vocab_bos(vocab));
        }
        const std::string marker = mctx != nullptr ? std::string(get_media_marker()) : std::string();
        size_t next_file = 0;
        auto special_segment = [&](const std::string & seg) {
            if (seg.empty()) return;
            size_t n_media = 0;
            if (!marker.empty()) {
                for (size_t p = seg.find(marker); p != std::string::npos; p = seg.find(marker, p + marker.size())) ++n_media;
            }
            if (n_media == 0) {
                for (llama_token t : common_tokenize(vocab, seg, false, true)) out.push_back(t);
                return;
            }
            if (next_file + n_media > files.size()) {
                throw std::invalid_argument("the prompt has more media markers than files");
            }
            const std::vector<raw_buffer> sub(files.begin() + (long) next_file, files.begin() + (long) (next_file + n_media));
            next_file += n_media;
            server_tokens part = process_mtmd_prompt(mctx, seg, sub, init_opt);
            // mtmd adds the BOS this prompt already has
            size_t i = (add_bos && part.size() > 0 && part[0] == llama_vocab_bos(vocab)) ? 1 : 0;
            while (i < part.size()) {
                if (part[i] == LLAMA_TOKEN_NULL) {
                    const mtmd_input_chunk * chunk = part.find_chunk(i).get();
                    out.push_back(chunk);
                    i += mtmd_input_chunk_get_n_tokens(chunk);
                } else {
                    out.push_back(part[i]);
                    ++i;
                }
            }
        };
        size_t pos = 0;
        llama_tokens plain_toks;
        for (const auto & span : plain) {
            special_segment(prompt.substr(pos, span.first - pos));
            plain_toks.clear();
            tokenize_plain(prompt.substr(span.first, span.second - span.first), plain_toks);
            for (llama_token t : plain_toks) out.push_back(t);
            pos = span.second;
        }
        special_segment(prompt.substr(pos));
        if (next_file != files.size()) {
            throw std::invalid_argument("the prompt has fewer media markers than files");
        }
        return out;
    }

    // mirrors handle_completions_impl in server-context.cpp without the parts the engine has no use for
    // (n_cmpl > 1, prompt logging, message spans for checkpoints). `plain`: the prompt's byte spans that are text
    // (a control token's text inside a message, literals.hpp)
    std::unique_ptr<res_generator> handle_completions(const server_http_req & req, const json & data,
                                                      const std::vector<raw_buffer> & files, task_response_type res_type,
                                                      const std::vector<std::pair<size_t, size_t>> & plain = {}) {
        auto res = create_response();
        auto completion_id = gen_chatcmplid();
        auto & rd = res->rd;

        res->set_req(&req);

        int32_t sse_ping_interval = params.sse_ping_interval;

        try {
            const auto & prompt = data.at("prompt");

            std::vector<server_tokens> inputs;
            if (!plain.empty() && prompt.is_string()) {
                inputs.push_back(tokenize_prompt(prompt.get<std::string>(), plain, files));
            } else if (res_type != TASK_RESPONSE_TYPE_NONE && mctx != nullptr) {
                inputs.push_back(process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt));
            } else {
                inputs = tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt);
            }
            if (inputs.size() != 1) {
                throw std::invalid_argument("this server takes one prompt per request");
            }

            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id     = rd.get_new_id();
            task.tokens = std::move(inputs[0]);
            task.params = server_schema::eval_llama_cmpl_schema(vocab, params, logit_bias_eog, data);
            if (task.params.n_cmpl > 1) {
                throw std::invalid_argument("n > 1 is not supported by this server");
            }
            sse_ping_interval = task.params.sse_ping_interval;

            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = model_name;

            std::vector<server_task> tasks;
            tasks.push_back(std::move(task));
            rd.post_tasks(std::move(tasks));
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const bool stream = json_value(data, "stream", false);

        if (!stream) {
            auto all_results = rd.wait_for_all(req.should_stop);
            if (all_results.is_terminated) {
                return res; // connection closed
            } else if (all_results.error) {
                res->error(all_results.error->to_json());
                return res;
            } else {
                GGML_ASSERT(all_results.results.size() == 1);
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(all_results.results[0].get()) != nullptr);
                res->ok(all_results.results[0]->to_json());
            }
            return res;
        }

        // streaming: the first error is a plain (non-stream) error response, as OpenAI does
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }
        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // headers and status only
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status       = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({{"event", "error"}, {"data", res_json}});
                }
                return format_oai_sse(json {{ "error", res_json }});
            };
            auto effective_should_stop = [&res_this]() { return res_this->should_stop(); };

            try {
                if (effective_should_stop()) {
                    return false;
                }
                if (!res_this->data.empty()) {
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }
                server_response_reader & rd = res_this->rd;
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;
                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    return false;
                }
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true;
                    }
                    if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t) sse_ping_interval * 1000) {
                        timeout = true;
                        return true;
                    }
                    return false;
                });
                if (timeout) {
                    // a client with an idle timeout (pi: 5 minutes) stays attached; a client that is gone fails this write
                    output = ":\n\n";
                    return true;
                }
                if (result == nullptr) {
                    GGML_ASSERT(effective_should_stop());
                    return false;
                }
                if (result->is_error()) {
                    output = format_error(res_type, result->to_json());
                    return false;
                }
                GGML_ASSERT(
                    dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr ||
                    dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                );
                json res_json = result->to_json();
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    output = format_anthropic_sse(res_json);
                } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                    output = format_oai_resp_sse(res_json);
                } else {
                    output = format_oai_sse(res_json);
                }
                return true;
            } catch (const std::exception & e) {
                output = format_error(res_type, format_error_response(e.what(), ERROR_TYPE_SERVER));
                return false;
            }
        });
        return res;
    }
};

// ---- strata_context -------------------------------------------------------------------------------------------------

strata_context::strata_context() : impl(std::make_unique<strata_context_impl>()) {}
strata_context::~strata_context() = default;

bool strata_context::start_engine(const common_params & params, const std::string & engine_exe,
                                  const std::vector<std::string> & engine_args, std::string & err) {
    impl->params      = params;
    impl->engine_exe  = engine_exe;
    impl->engine_args = engine_args;
    impl->engine_is_poly = engine_exe.find("strata-poly") != std::string::npos;

    std::vector<std::string> argv;
    argv.push_back(engine_exe);
    argv.push_back("--serve");
    for (const auto & a : engine_args) {
        argv.push_back(a);
    }
    return impl->engine.start(engine_exe, argv, err);
}

void strata_context::set_picture_cache(size_t entries, size_t bytes) {
    impl->pictures.configure(entries, bytes);
}

bool strata_context::load(common_params & params, std::string & err) {
    auto & I = *impl;
    I.params = params;

    // the vocabulary and the chat template, from the GGUF's metadata; no tensor is read
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    I.model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (I.model == nullptr) {
        // An architecture this llama.cpp does not know (GLM 5.3 Flash's file says "glm5-next", from neurall's fork)
        // still has an ordinary vocabulary and chat template; the engine, not llama.cpp, runs the tensors. A
        // vocabulary-only load reads nothing architecture-specific, so the name is overridden to one llama.cpp knows.
        LOG_WRN("vocabulary load failed; retrying with general.architecture overridden to \"llama\" (vocabulary only)\n");
        llama_model_kv_override ov[2] = {};
        snprintf(ov[0].key, sizeof(ov[0].key), "%s", "general.architecture");
        ov[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        snprintf(ov[0].val_str, sizeof(ov[0].val_str), "%s", "llama");
        ov[1].key[0] = 0;
        mparams.kv_overrides = ov;
        I.model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    }
    if (I.model == nullptr) {
        err = "could not read the vocabulary from " + params.model.path;
        return false;
    }
    I.vocab      = llama_model_get_vocab(I.model);
    I.init_literals();
    I.model_path = params.model.path;
    I.model_name = params.model_alias.empty() ? file_stem(params.model.path) : *params.model_alias.begin();

    if (!params.mmproj.path.empty()) {
        if (!I.init_vision(params, err)) {
            return false;
        }
    }

    // mirrors the chat template setup in server_context_impl::load_model
    {
        common_chat_templates_ptr chat_templates;
        bool enable_thinking = false;
        try {
            chat_templates = common_chat_templates_init(I.model, params.chat_template);
            LOG_INF("chat template, example_format: '%s'\n",
                    common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
            const bool template_supports_thinking = params.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            enable_thinking = params.enable_reasoning != 0 && template_supports_thinking;
            LOG_INF("chat template, thinking = %d\n", enable_thinking);
        } catch (const std::exception & e) {
            err = std::string("chat template parsing error: ") + e.what();
            return false;
        }
        I.chat_params = {
            /* use_jinja             */ params.use_jinja,
            /* prefill_assistant     */ params.prefill_assistant,
            /* reasoning_format      */ params.reasoning_format,
            /* chat_template_kwargs  */ params.default_template_kwargs,
            /* tmpls                 */ std::move(chat_templates),
            /* allow_image           */ I.mctx != nullptr,
            /* allow_audio           */ false,
            /* allow_video           */ false,
            /* enable_thinking       */ enable_thinking,
            /* reasoning_budget      */ -1, // needs a sampler the engine does not expose
            /* reasoning_budget_msg  */ std::string(),
            /* media_path            */ params.media_path,
            /* force_pure_content    */ params.force_pure_content_parser
        };
        I.chat_caps = common_chat_templates_get_caps(I.chat_params.tmpls.get());
    }

    // the engine: READY carries the context it allocated
    LOG_INF("waiting for the engine (%s) to load\n", I.engine_exe.c_str());
    std::string eerr;
    if (!I.engine.wait_ready(eerr, /* timeout_s */ 3600)) {
        err = "engine: " + eerr;
        return false;
    }
    I.n_ctx = I.engine.max_context() > 0 ? I.engine.max_context() : params.n_ctx;
    if (I.n_ctx <= 0) {
        err = "the engine did not report its context size and -c was not given";
        return false;
    }
    if (!I.engine.can_stop()) {
        LOG_WRN("%s", "the engine does not honour STOP: a cancelled request runs to its end\n");
    }
    for (const auto & kv : I.engine.info()) {
        LOG_INF("engine info: %s=%s\n", kv.first.c_str(), kv.second.c_str());
    }
    LOG_INF("engine ready: context %lld tokens, model %s\n", I.n_ctx, I.model_name.c_str());

    I.queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
        return impl->process_new_task(std::move(task), is_yielding);
    });
    I.queue_tasks.on_update_slots([this]() {
        impl->update_slots();
    });
    I.metrics.init();
    return true;
}

void strata_context::start_loop() {
    impl->queue_tasks.start_loop();
}

void strata_context::terminate() {
    impl->queue_tasks.terminate();
    impl->queue_results.terminate();
    impl->engine.quit();
    if (impl->model) {
        llama_model_free(impl->model);
        impl->model = nullptr;
    }
}

bool strata_context::engine_exited(std::string & why) {
    if (impl->engine_dead) {
        why = "the engine closed its output";
        return true;
    }
    return impl->engine.reap_exit(why);
}

bool strata_context::engine_alive() const {
    return impl->engine.alive() && !impl->engine_dead;
}

long long strata_context::n_ctx() const {
    return impl->n_ctx;
}

void strata_context::init_routes(server_http_context & http) {
    strata_context_impl * I = impl.get();

    auto get_health = [I](const server_http_req &) {
        auto res = I->create_response();
        if (!I->engine.alive() || I->engine_dead) {
            res->error(format_error_response("the engine is not running", ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        // llama.cpp's shape ("status") plus the fields upstream PolyStrata clients (tools/strata_bench.py,
        // its is_strata / wait_ready) look for: "service" and "loaded". The engine only serves once loaded.
        res->ok({{"status", "ok"}, {"busy", I->slot.active()},
                 {"service", "strata"}, {"loaded", true},
                 {"model", I->model_name}, {"max_context", I->n_ctx}});
        return res;
    };

    auto get_models = [I](const server_http_req &) {
        auto res = I->create_response();
        res->ok(json {
            {"object", "list"},
            {"data", json::array({ I->model_info_json() })},
        });
        return res;
    };

    auto get_props = [I](const server_http_req &) {
        auto res = I->create_response();
        json caps = json::object();
        for (const auto & kv : I->chat_caps) {
            caps[kv.first] = kv.second;
        }
        res->ok(json {
            {"default_generation_settings", {{"n_ctx", I->n_ctx}}},
            {"total_slots", 1},
            {"model_path", I->model_path},
            {"model_alias", I->model_name},
            {"modalities", {{"vision", I->chat_params.allow_image}, {"audio", false}}},
            {"chat_template", common_chat_templates_source(I->chat_params.tmpls.get())},
            {"chat_template_caps", caps},
            {"build_info", std::string(llama_build_info())},
            {"engine", I->engine_is_poly ? "strata-poly" : "strata"},
        });
        return res;
    };

    // a control token's text inside a message stays text (literals.hpp): marked before the template renders,
    // unmarked after, and the spans it covered are tokenized as text
    auto parse_chat = [I](json & body, std::vector<raw_buffer> & files, std::vector<std::pair<size_t, size_t>> & plain) {
        bool marked = false;
        if (body.contains("messages")) {
            json * tools = body.contains("tools") ? &body.at("tools") : nullptr;
            marked = I->literals.mark_request(body.at("messages"), tools);
        }
        json data = oaicompat_chat_params_parse(body, I->chat_params, files);
        if (marked) {
            std::string prompt = data.at("prompt").get<std::string>();
            plain = I->literals.unmark(prompt);
            data["prompt"] = std::move(prompt);
            SRV_INF("%zu control-token texts inside the messages are sent as text\n", plain.size());
        }
        return data;
    };

    auto post_chat_completions = [I, parse_chat](const server_http_req & req) {
        std::vector<raw_buffer> files;
        std::vector<std::pair<size_t, size_t>> plain;
        json body = json::parse(req.body);
        json body_parsed = parse_chat(body, files, plain);
        return I->handle_completions(req, body_parsed, files, TASK_RESPONSE_TYPE_OAI_CHAT, plain);
    };

    auto post_completions_oai = [I](const server_http_req & req) {
        std::vector<raw_buffer> files;
        const json body = json::parse(req.body);
        return I->handle_completions(req, body, files, TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    auto post_completions = [I](const server_http_req & req) {
        std::vector<raw_buffer> files;
        const json body = json::parse(req.body);
        return I->handle_completions(req, body, files, TASK_RESPONSE_TYPE_NONE);
    };

    auto post_apply_template = [I, parse_chat](const server_http_req & req) {
        auto res = I->create_response();
        std::vector<raw_buffer> files;
        std::vector<std::pair<size_t, size_t>> plain;
        json body = json::parse(req.body);
        json data = parse_chat(body, files, plain);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    auto post_tokenize = [I](const server_http_req & req) {
        auto res = I->create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special   = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces   = json_value(body, "with_pieces", false);
            llama_tokens tokens = tokenize_mixed(I->vocab, body.at("content"), add_special, parse_special);
            if (with_pieces) {
                for (const auto & token : tokens) {
                    std::string piece = common_token_to_piece(I->vocab, token);
                    json piece_json;
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }
                    tokens_response.push_back({{"id", token}, {"piece", piece_json}});
                }
            } else {
                tokens_response = tokens;
            }
        }
        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    auto post_detokenize = [I](const server_http_req & req) {
        auto res = I->create_response();
        const json body = json::parse(req.body);
        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(I->vocab, tokens);
        }
        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    // Disk sessions: llama-server's slot 0 save/restore surface, forwarded to the engine's SAVE/RESTORE
    // commands (both engines implement them). The directory comes from --slot-save-path (common_params).
    auto get_slots = [I](const server_http_req &) {
        auto res = I->create_response();
        const long long n_prompt = I->slot.active() ? (long long) I->slot.task->n_tokens() : 0;
        res->ok(json::array({ json{
            {"id", 0},
            {"id_task", -1},
            {"n_ctx", I->n_ctx},
            {"n_prompt_tokens", n_prompt},
            {"is_processing", I->slot.active()},
        } }));
        return res;
    };

    auto post_slots = [I](const server_http_req & req) {
        auto res = I->create_response();
        const std::string action = req.get_param("action");
        const std::string id     = req.get_param("id_slot");
        if (id != "0") {
            res->error(format_error_response("only slot 0 exists", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (action != "save" && action != "restore") {
            res->error(format_error_response("action must be save or restore", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (I->params.slot_save_path.empty()) {
            res->error(format_error_response("start the server with --slot-save-path DIR", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (I->slot.active()) {
            res->error(format_error_response("the engine is busy; save or restore between requests", ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        if (body.count("filename") == 0 || !body.at("filename").is_string()) {
            res->error(format_error_response("filename is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string filename = body.at("filename").get<std::string>();
        bool bad = filename.empty() || filename.size() > 255 || filename == "." || filename == ".." ||
                   filename.find_first_of("/\\:") != std::string::npos || filename.front() == '.' ||
                   filename.back() == '.' || filename.back() == ' ';
        for (unsigned char c : filename) {
            if (c < 0x20 || c == 0x7f) bad = true;
        }
        if (bad) {
            res->error(format_error_response("filename must be a plain file name inside --slot-save-path",
                                             ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const bool save = action == "save";
        const std::string path = I->params.slot_save_path + filename;
        std::string reply, err;
        if (!I->engine.command((save ? "SAVE " : "RESTORE ") + path, save ? "SAVED" : "RESTORED", reply, err, 600000)) {
            enum error_type type = ERROR_TYPE_SERVER;
            if (err.rfind("SERR invalid", 0) == 0) {
                type = ERROR_TYPE_INVALID_REQUEST;
            } else if (err.rfind("SERR memory", 0) == 0) {
                type = ERROR_TYPE_UNAVAILABLE;
            }
            res->error(format_error_response(err, type));
            return res;
        }
        long long tokens = 0, bytes = 0;
        double ms = 0;
        if (save) {
            std::sscanf(reply.c_str(), "SAVED %lld %lld %lf", &tokens, &bytes, &ms);
        } else {
            std::sscanf(reply.c_str(), "RESTORED %lld %lld %lf", &tokens, &bytes, &ms);
        }
        json out = {{"id_slot", 0}, {"filename", filename}};
        if (save) {
            out["n_saved"]   = tokens;
            out["n_written"] = bytes;
            out["timings"]   = {{"save_ms", ms}};
        } else {
            out["n_restored"] = tokens;
            out["n_read"]     = bytes;
            out["timings"]    = {{"restore_ms", ms}};
        }
        res->ok(out);
        return res;
    };

    http.get ("/health",              ex_wrapper(get_health));
    http.get ("/v1/health",           ex_wrapper(get_health));
    http.get ("/slots",               ex_wrapper(get_slots));
    http.post("/slots/:id_slot",      ex_wrapper(post_slots));
    http.get ("/props",               ex_wrapper(get_props));
    http.get ("/models",              ex_wrapper(get_models));
    http.get ("/v1/models",           ex_wrapper(get_models));
    http.post("/completion",          ex_wrapper(post_completions));
    http.post("/completions",         ex_wrapper(post_completions));
    http.post("/v1/completions",      ex_wrapper(post_completions_oai));
    http.post("/chat/completions",    ex_wrapper(post_chat_completions));
    http.post("/v1/chat/completions", ex_wrapper(post_chat_completions));
    http.post("/tokenize",            ex_wrapper(post_tokenize));
    http.post("/detokenize",          ex_wrapper(post_detokenize));
    http.post("/apply-template",      ex_wrapper(post_apply_template));
}
