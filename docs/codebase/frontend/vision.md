[Codebase docs](../README.md) › [Front end](README.md) › Vision

# Vision: `--mmproj` image input

## Purpose

Image input for the OpenAI `image_url` content part. The front end encodes each image on the CPU with llama.cpp's
`mtmd` library and the model's own vision projector (mmproj GGUF), splices a run of `<|image_pad|>` tokens into the
prompt, writes the embeddings to a temporary file, and sends a `GENI` request so the engine substitutes those rows
for the pad tokens and gives them M-RoPE grid positions. Running the encoder on the CPU leaves the engine's whole
VRAM expert cache in place (`frontend/vision.hpp:60-61`).

Files: `frontend/vision.hpp` (all of it), the image part of `Server::prepare` (`frontend/main.cpp:211-228`,
`262-293`), option checks (`main.cpp:672-674`, `710-715`), and the engine's `GENI` handling in
`src/program/generate.cpp:6100-6105`, `6680-6768`.

## Main types and functions (`frontend/vision.hpp`)

| Item | Lines | What it does |
| --- | --- | --- |
| `ImageFile` | 17-28 | per-request temporary file `/tmp/unbiased-strata-image-XXXXXX` (`mkstemp`), unlinked in the destructor; non-copyable. Held by `shared_ptr` in `Prepared`, which the streaming lambda captures, so the file outlives streaming and cancellation |
| `image_data(url)` | 30-54 | accepts only `data:image/...;base64,<data>`; data non-empty, at most 32 MiB of base64 text, length a multiple of 4; strict decoder (alphabet, padding only at the end, zero trailing bits) |
| `EncodedImage` | 55-58 | `tokens`, grid `nx` x `ny`, `width` (embedding width), `values` (`tokens * width` floats) |
| `Vision::init(path, model, threads, max_tokens)` | 71-95 | reads `clip.vision.projection_dim` from the mmproj GGUF and checks it equals the text model's `<arch>.embedding_length` (read from metadata, because the front end's vocab-only model has no hparams); creates the `mtmd` context with `use_gpu = false`, `warmup = false`, `n_threads = threads > 0 ? threads : 8`, `image_max_tokens = max_tokens`; requires `mtmd_support_vision` |
| `Vision::encode(url)` | 96-140 | decode bytes -> `mtmd_helper_bitmap_init_from_buf` (rejects audio and undecodable data) -> cache lookup by the bitmap's content hash (`mtmd_bitmap_get_id`) -> `mtmd_tokenize` with the default marker -> exactly one image chunk -> token count 1..`max_tokens` -> `mtmd_encode_chunk` -> grid from the last token's decoder position (`nx * ny` must equal the token count) -> copy `mtmd_get_output_embd` rows; logs `encoded image <ms> ms, <n> tokens (<nx>x<ny>), CPU` |
| image cache | 64-67, 107-108, 135-136 | `std::map<content hash, shared_ptr<EncodedImage>>`; when adding an entry would pass 128 MiB of floats, the whole cache is cleared first |
| `write_images(file, images)` | 142-152 | for each image in prompt order: int32 header `{0x31455653, tokens, nx, ny, width}` then `tokens * width` float32 values; flushed before the request is sent |

`0x31455653` is the bytes `S V E 1` in little-endian order; the engine calls these "strata-vision records".

## Data flow

```
image_url part (data:image/png;base64,...)
  -> Server::prepare (holding Engine::mu)
       Vision::encode -> EncodedImage (cached by content hash)
       part replaced by {"type":"media_marker","text":"UNBIASED_IMAGE_chatcmpl-<hex>_<i>"}
  -> chat template renders the marker text where the image goes
  -> prompt split at each marker (each must appear exactly once):
       text ids | <|vision_start|> | <|image_pad|> x tokens | <|vision_end|> | text ids ...
  -> ImageFile + write_images (SVE1 records)
  -> header "GENI <max_new> [keys] /tmp/unbiased-strata-image-XXXXXX " + ids
  -> engine: reads the file, maps each run of image-pad ids (248056) to one record's rows,
     M-RoPE positions t = p, h = p + y, w = p + x; text after the image continues at p + max(nx, ny)
```

Details in `Server::prepare`:

- Images are found in any message whose `content` is an array, parts of `type: "image_url"` with
  `image_url.url` (`main.cpp:216-227`). At most 32 per request.
- The encode happens under `Engine::mu`, which serializes encoding with generation (and protects the cache).
- `<|image_pad|>`, `<|vision_start|>` and `<|vision_end|>` must each tokenize (with special parsing) to one token,
  else "model does not have the expected image special tokens" (`main.cpp:265-270`).
- Text segments are tokenized with special parsing, but any `<|image_pad|>` token appearing in them is replaced by
  the tokens of the literal text `<|image_pad|>` parsed without special tokens (`main.cpp:271-276`), so only the
  front end creates image cells.
- A `rendered` string with the pad runs written out is kept for the optional prompt capture
  (`SUGMEOFF_CAPTURE_FIRST_PROMPT`).

## Engine side (`src/program/generate.cpp`)

- Started with `--vision` (the front end adds it when `--mmproj` is given). A `GENI` to an engine without it gets
  `ERR this engine was started without --vision` (`generate.cpp:6679`).
- `GENI <max_new> [key=value ...] <file> <ids>`: the file path is the first token without `=` after the keys
  (`generate.cpp:6665-6670`).
- File reading and validation (`generate.cpp:6693-6745`): magic, positive counts, `nx * ny == n`, width equal to the model's
  `n_embd`; errors such as "bad embeddings file", "short embeddings file", "the prompt has more images than the
  embeddings file", "image k has n rows but fewer <|image_pad|> tokens", "the embeddings file has more images than
  the prompt", "the prompt cannot end in an image".
- Each image's grid and rows are hashed (`fnv1a`) into the conversation-cache identity, so a reused checkpoint must
  match the same picture (`generate.cpp:6720-6725`).
- On any error the M-RoPE table is reset to the identity so the next text request is untouched, and the request gets
  `ERR <reason>` (`generate.cpp:6758-6765`), which the front end returns as 500 (or a stream error event).

## Configuration

| Knob | Default | Effect |
| --- | --- | --- |
| `--mmproj PATH` | off | load the encoder; the file must exist; adds `--vision` to the engine; `/v1/models` lists `image` |
| `--vision-tokens N` | 1024 | `image_max_tokens` for mtmd and the per-image refusal limit; 16..4096 |
| `--threads N` | | also the encoder's thread count (8 when 0 and no cgroup quota) |

`--vision` given directly after `--` is refused by the front end ("use --mmproj for image input").

## Model-specific behaviour

- The projector width must match the text model: 2560 for Flash-Next, 2048 for Qwen3.6-35B-A3B
  (`vision.hpp:77-78`). A mismatched mmproj fails start-up with exit 1.
- Encoders named in the repository README: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (0.9 GB) for Flash-Next and
  `mmproj-F16.gguf` from the Qwen3.6 -MTP repository (899 MB). The README reports about 25 s for a 1-megapixel
  screenshot on 8 threads (Flash-Next) and 24 s / 70 s for a full-screen image on an i9-9900K / Ryzen 7 1700
  (Qwen3.6); repeats come from the cache.
- The engine's `kImagePad = 248056` ("qwen4exp.ple.image_token_id", `generate.cpp:6104`) is shared by both models'
  vocabularies as far as the code shows; the front end looks up `<|image_pad|>` from the GGUF instead of assuming it.

## Tests

No unit test covers the vision path; `chat_parser_test.cpp` uses text-only requests. The step-5 serving checks in
`docs/qwen36-port/step5-serve.md` include an image request through the HTTP front end.

## Gotchas

- Only base64 `data:` URLs; `http(s)://` URLs are refused (the front end opens no outgoing connection).
- The cache is cleared wholesale at 128 MiB, not evicted per entry.
- Encoding happens before the request reaches the engine but under the same lock, so a long image encode delays
  every other request, and a request with images waits for a running generation before encoding starts.
- Image errors raised as `std::runtime_error` (e.g. "vision encoding failed") are also reported as 400, because
  `handle_chat` maps every exception from `prepare` to 400 (`main.cpp:472-476`).

## See also

- [chat-and-parsing.md](chat-and-parsing.md) (prompt tokenization), [http-api.md](http-api.md), [engine-process.md](engine-process.md)
- [../program/README.md](../program/README.md) - the engine's `GENI` path and M-RoPE table.
- [README.md (repository)](../../../README.md), "Run" (images) and the Qwen3.6 section.
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "Images (vision)" - upstream's description (its server, not this one).
