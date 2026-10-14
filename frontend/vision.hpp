#pragma once
#include "common.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace strata_frontend {
// Private per-request file, retained through streaming completion/cancellation.
struct ImageFile {
    std::string path;
    ImageFile() {
        char name[] = "/tmp/unbiased-strata-image-XXXXXX";
        int fd = mkstemp(name);
        if (fd < 0) throw std::runtime_error("cannot create image embedding file");
        close(fd); path = name;
    }
    ~ImageFile() { if (!path.empty()) unlink(path.c_str()); }
    ImageFile(const ImageFile&) = delete;
    ImageFile& operator=(const ImageFile&) = delete;
};

inline std::vector<unsigned char> image_data(const std::string& url) {
    const size_t comma = url.find(',');
    if (url.rfind("data:image/", 0) != 0 || comma == std::string::npos ||
        comma < 7 || url.substr(comma - 7, 7) != ";base64")
        throw std::invalid_argument("image_url must contain a base64 data:image URL (PNG, JPEG, BMP or GIF)");
    const std::string data = url.substr(comma + 1);
    if (data.empty() || data.size() > 32 * 1024 * 1024 || data.size() % 4)
        throw std::invalid_argument("empty, oversized or malformed base64 image");
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::array<int, 256> table; table.fill(-1);
    for (size_t i = 0; i < alphabet.size(); ++i) table[(unsigned char) alphabet[i]] = (int) i;
    std::vector<unsigned char> out; out.reserve(data.size() / 4 * 3);
    for (size_t i = 0; i < data.size(); i += 4) {
        int a = table[(unsigned char)data[i]], b = table[(unsigned char)data[i+1]];
        bool p2 = data[i+2] == '=', p3 = data[i+3] == '=';
        int c = p2 ? 0 : table[(unsigned char)data[i+2]], d = p3 ? 0 : table[(unsigned char)data[i+3]];
        if (a < 0 || b < 0 || c < 0 || d < 0 || (p2 && !p3) ||
            ((p2 || p3) && i + 4 != data.size()) || (p2 && (b & 15)) || (!p2 && p3 && (c & 3)))
            throw std::invalid_argument("malformed base64 image");
        out.push_back((a << 2) | (b >> 4));
        if (!p2) out.push_back((b << 4) | (c >> 2));
        if (!p3) out.push_back((c << 6) | d);
    }
    return out;
}
struct EncodedImage {
    int32_t tokens, nx, ny, width;
    std::vector<float> values;
};

// CPU encoder leaves the engine's full VRAM expert cache available. Its callers
// take engine.mu, serializing image encoding and generation just like upstream.
class Vision {
    mtmd_context* ctx_ = nullptr;
    int width_ = 0;
    int max_tokens_ = 1024;
    size_t cache_bytes_ = 0;
    std::map<std::string, std::shared_ptr<EncodedImage>> cache_;
public:
    ~Vision() { if (ctx_) mtmd_free(ctx_); }
    bool enabled() const { return ctx_ != nullptr; }
    void init(const std::string& path, const llama_model* model, int threads, int max_tokens) {
        gguf_init_params gp{true, nullptr};
        auto gg = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_from_file(path.c_str(), gp), gguf_free);
        auto key = gg ? gguf_find_key(gg.get(), "clip.vision.projection_dim") : -1;
        if (key < 0) throw std::runtime_error("vision encoder has no projection_dim");
        width_ = (int) gguf_get_val_u32(gg.get(), key);
        // the projector must produce rows of the model's own width (2560 for Flash-Next, 2048 for Qwen3.6-35B-A3B);
        // the tokenizer-only model the front end holds has no hparams, so the width is read from its metadata
        char arch[64] = {0}, val[32] = {0};
        int want = 0;
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof arch) > 0 &&
            llama_model_meta_val_str(model, (std::string(arch) + ".embedding_length").c_str(), val, sizeof val) > 0)
            want = std::atoi(val);
        if (want <= 0) throw std::runtime_error("cannot read the model's embedding width");
        if (width_ != want)
            throw std::runtime_error("vision projector width " + std::to_string(width_) +
                                     " does not match the model's embedding width " + std::to_string(want));
        max_tokens_ = max_tokens;
        auto cp = mtmd_context_params_default();
        cp.use_gpu = false; cp.print_timings = false; cp.warmup = false;
        cp.n_threads = threads > 0 ? threads : 8;
        cp.image_max_tokens = max_tokens;
        ctx_ = mtmd_init_from_file(path.c_str(), model, cp);
        if (!ctx_ || !mtmd_support_vision(ctx_)) throw std::runtime_error("cannot load vision encoder: " + path);
    }
    std::shared_ptr<EncodedImage> encode(const std::string& url) {
        if (!ctx_) throw std::invalid_argument("image input requires starting UnbiasedStrata with --mmproj PATH");
        auto bytes = image_data(url);
        auto t0 = std::chrono::steady_clock::now();
        auto bw = mtmd_helper_bitmap_init_from_buf(ctx_, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(bw.bitmap, mtmd_bitmap_free);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
        if (!bitmap || mtmd_bitmap_is_audio(bitmap.get())) throw std::invalid_argument("cannot decode image; use PNG, JPEG, BMP or GIF");
        const char* hash = mtmd_bitmap_get_id(bitmap.get());
        if (!hash || !*hash) throw std::runtime_error("image decoder did not supply a content hash");
        std::string id(hash);
        auto cached = cache_.find(id);
        if (cached != cache_.end()) return cached->second;
        auto chunks = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bm[] = {bitmap.get()};
        if (mtmd_tokenize(ctx_, chunks.get(), &txt, bm, 1)) throw std::invalid_argument("image preprocessing failed");
        const mtmd_input_chunk* image = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto ch = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (image) throw std::invalid_argument("multiple image chunks are unsupported for this model");
                image = ch;
            }
        }
        if (!image) throw std::invalid_argument("no image tokens produced");
        size_t n = mtmd_input_chunk_get_n_tokens(image);
        if (!n || n > (size_t)max_tokens_) throw std::invalid_argument("image token count exceeds configured limit");
        if (mtmd_encode_chunk(ctx_, image)) throw std::runtime_error("vision encoding failed");
        auto last = mtmd_image_tokens_get_decoder_pos(mtmd_input_chunk_get_tokens_image(image), 0, n-1);
        int nx = (int)last.x + 1, ny = (int)last.y + 1;
        if ((size_t)nx * ny != n) throw std::runtime_error("image grid is not rectangular");
        auto result = std::make_shared<EncodedImage>();
        result->tokens = (int32_t)n; result->nx = nx; result->ny = ny; result->width = width_;
        auto values = mtmd_get_output_embd(ctx_);
        if (!values) throw std::runtime_error("encoder returned no image embeddings");
        result->values.assign(values, values+n*width_);
        size_t size = result->values.size()*sizeof(float);
        if (cache_bytes_ + size > 128*1024*1024) { cache_.clear(); cache_bytes_ = 0; }
        cache_[id] = result; cache_bytes_ += size;
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();
        std::fprintf(stderr, "unbiased-strata: encoded image %.0f ms, %d tokens (%dx%d), CPU\n", ms, result->tokens, nx, ny);
        return result;
    }
};
inline void write_images(const ImageFile& file, const std::vector<std::shared_ptr<EncodedImage>>& images) {
    auto f = std::unique_ptr<FILE, decltype(&std::fclose)>(std::fopen(file.path.c_str(), "wb"), std::fclose);
    if (!f) throw std::runtime_error("cannot open image embedding file");
    for (auto& im : images) {
        int32_t hdr[] = {0x31455653, im->tokens, im->nx, im->ny, im->width};
        if (std::fwrite(hdr, sizeof(hdr), 1, f.get()) != 1 ||
            std::fwrite(im->values.data(), sizeof(float), im->values.size(), f.get()) != im->values.size())
            throw std::runtime_error("short write to image embedding file");
    }
    if (std::fflush(f.get())) throw std::runtime_error("cannot flush image embedding file");
}
} // namespace strata_frontend
