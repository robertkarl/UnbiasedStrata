[Codebase docs](../README.md) › [Front end](README.md) › Chat template and parsing

# Chat template, tokenization and output parsing

## Purpose

How an OpenAI message list becomes prompt token ids, and how generated token ids become reasoning text, answer
text and tool calls, both for a whole response and for a stream. Everything here uses llama.cpp's `common` library
from the pinned submodule (`chat.h`, `common.h`, `llama.h`): the GGUF's own vocabulary and its own Jinja template.
The engine never sees text.

Files: `frontend/main.cpp` (`Server::prepare`, `StreamParse`, `Server::run`, `utf8_complete_prefix`),
`patches/llama.cpp-qwen3-coder-any-argument-order.patch`, test `frontend/chat_parser_test.cpp`.

## Loading

`main` loads the model file with `llama_model_params::vocab_only = true` (`frontend/main.cpp:698-708`): no weights,
only the vocabulary and metadata. `common_chat_templates_init(model, "")` (`main.cpp:709`) builds the templates
from the GGUF's `tokenizer.chat_template` (the empty string means "no override"). Both live in `Server`
(`main.cpp:203-205`).

## Rendering the prompt (`Server::prepare`, `frontend/main.cpp:208-299`)

1. Image parts are replaced by `media_marker` parts first ([vision.md](vision.md)).
2. `common_chat_msgs_parse_oaicompat(messages)` -> `common_chat_templates_inputs::messages`.
3. Tools: if `tools` is a non-empty array, `common_chat_tools_parse_oaicompat`; `tool_choice` if it is a string
   (`common_chat_tool_choice_parse_oaicompat`); `parallel_tool_calls` from the request or, when absent, from the
   template's `supports_parallel_tool_calls` capability (`main.cpp:230-236`).
4. Fixed inputs: `add_generation_prompt = true`, `use_jinja = true`,
   `reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK` (reasoning goes to `reasoning_content`),
   `enable_thinking = true` (`main.cpp:237-240`).
5. Thinking controls (`main.cpp:241-259`), applied in this order:
   - top-level `reasoning_effort`: `"none"` sets `enable_thinking = false`; any other string is stored as
     `chat_template_kwargs["reasoning_effort"]` (JSON-encoded);
   - `chat_template_kwargs.enable_thinking` (bool) overrides `enable_thinking`;
   - `chat_template_kwargs.reasoning_effort` (string) overrides the template kwarg;
   - `chat_template_kwargs.preserve_thinking` (bool) is passed through. Qwen3.6's template drops the reasoning of
     every reply before the last user message unless it is set; with it, and the client sending
     `reasoning_content` back, the history renders as generated, so the engine can reuse its checkpoint at the end
     of the previous reply rather than only the one at its header (measured in
     `docs/qwen36-port/step5-serve.md`, section (d)).
   No other `chat_template_kwargs` key is forwarded.
6. `common_chat_templates_apply(tmpls, in)` returns `common_chat_params cp`: the rendered `cp.prompt` and the
   grammar/parser description for this request's output (`cp.parser`).

## Tokenizing (`main.cpp:261-294`)

- Text-only: `common_tokenize(vocab, rendered, /*add_special*/ false, /*parse_special*/ true)`. The template already
  writes the special tokens as text (`<|im_start|>` etc.), so `parse_special` turns them into control tokens; note
  that this also applies to any special-token text inside message content.
- With images, the prompt is split at the image markers and each image becomes `<|vision_start|>`, `n` x
  `<|image_pad|>`, `<|vision_end|>`; any literal `<|image_pad|>` written in the text is re-tokenized as plain text
  so text cannot forge image cells ([vision.md](vision.md)).
- An empty token list is a 400. The resulting ids go to the engine as a comma-separated list; the engine checks
  each id is inside the vocabulary (`src/program/generate.cpp:6775-6777`).

## Parser parameters (`main.cpp:296-299`)

`p.parser = common_chat_parser_params(cp)` with `reasoning_format = DEEPSEEK` and `parse_tool_calls = true`; if the
template produced a PEG parser description (`cp.parser`), it is loaded with `p.parser.parser.load(cp.parser)`. These
parameters are per request, because the parser depends on the tools offered and on thinking being on or off.

## Detokenizing (`Server::run`, `frontend/main.cpp:397-459`)

For every `T <id>` from the engine (`main.cpp:415-432`):

- end-of-generation tokens (`llama_vocab_is_eog`) are dropped; the engine sends its EOS token and then `DONE`;
- otherwise `common_token_to_piece(vocab, id, /*special*/ true)` appends the piece to `pending`, so special tokens
  such as `</think>` and `<tool_call>` appear as text for the parser;
- `utf8_complete_prefix(pending)` (`main.cpp:100-112`) returns the length of the longest prefix that does not end
  inside a UTF-8 character; only that part moves to `text`, the rest waits for the next token;
- for a streaming request the parse is updated after every token (`update(true)`); a non-streaming request is
  parsed once, at the end.

After `DONE` the leftover `pending` bytes are appended and a final, non-partial parse runs (`update(false)`). If that
throws, the request fails with "response formatter failed: ..." and the raw text is logged between
`formatter-input-begin` / `-end` markers. If the text contains `<tool_call>` but the final message has no tool call,
the raw output is logged (`raw-output-begin` / `-end`) because "the parser accepts a message with no tool call
even when the text holds one it could not match", and the client sees `finish_reason: stop`
(`main.cpp:443-450`).

## `StreamParse`: what is published, and when (`frontend/main.cpp:149-197`)

`StreamParse` holds the message already published (`msg`) and the tool-call ids assigned so far (`call_ids`).
`step(text, partial, parser)`:

1. `common_chat_parse(text, partial, parser)`. If it throws on a partial parse, nothing is published (wait for more
   text). If it throws on the final parse, the whole text becomes the `content` of an assistant message.
2. `content_parts` is cleared.
3. Partial parse:
   - tool calls are cleared: while a call is being written its arguments are partial JSON that the parser closes
     and normalizes on every token, so earlier output could change, and an OpenAI stream cannot take back what it
     sent;
   - if the new `content` does not extend the published `content` (the parser re-read the tail as the start of a
     tool call), the published content is kept, i.e. nothing new is sent for content until it extends again; the
     same rule for `reasoning_content`.
4. Final parse: for `content` and `reasoning_content`, text already sent stays sent. If the complete parse no longer
   holds it (the parser trims white space before a tool call that the partial parses still carried), the published
   text is kept; if the two genuinely differ, a warning is logged.
5. `set_tool_call_ids` gives new calls ids `call_<16 hex>`, then
   `common_chat_msg_diff::compute_diffs(msg, now)` produces the deltas and `msg = now`.

Net effect: reasoning and answer text stream token by token; each tool call is published once, complete (name, id
and full arguments in one delta), when the message ends; text before a call streams. "This changes delivery, never
generated tokens." This is the REGINA-110 fix (c3b41cc), which replaced an earlier rule that cleared content with
tool calls on every partial parse (`docs/qwen36-port/step5-serve.md`, "A5 on the 4090").

## The Qwen3-Coder argument-order patch

The tool calls these models write, as exercised by `chat_parser_test.cpp` and parsed by llama.cpp's Qwen3-Coder
parser, look like
(`<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>`). Upstream
llama.cpp's grammar (`common/parsers/qwen3-coder.cpp`) accepted required arguments in any order but optional ones
only after them; Flash-Next often writes optional ones first (`limit offset path`), and because the tool-call group
is optional and the grammar has no end anchor, such a call was dropped silently (client sees `finish_reason: stop`
and an empty answer). `patches/llama.cpp-qwen3-coder-any-argument-order.patch` accepts all arguments in any order.
It is applied by `make build` (`Makefile:28-29`) and also at CMake configure time
(`CMakeLists.txt:1150-1166`, REGINA-104) when ggml comes from the in-tree submodule.

## Configuration

Request-level only: `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning_effort`, `chat_template_kwargs`
(`enable_thinking`, `reasoning_effort`, `preserve_thinking`), `stream`. No CLI flag or environment variable changes
the template; it always comes from the GGUF.

## Model-specific behaviour

The front end is model-agnostic; differences come from each GGUF's template. Noted in code: Qwen3.6's template and
`preserve_thinking` (`main.cpp:253-256`); Flash-Next's habit of writing optional arguments first (the patch). The
test asserts that the template advertises `supports_parallel_tool_calls` (`chat_parser_test.cpp:22`); it was run on
the Qwen3.6 IQ3_XXS and Q4_K_XL files (step5-serve.md).

## Tests

`frontend/chat_parser_test.cpp` (`make test-chat-parser MODEL=<shard 1>`), see [tests.md](tests.md): parallel tool
calls default and explicit, arguments in any order, and seven streaming cases fed one UTF-8 character at a time.

## Gotchas

- `parse_special = true` on the rendered prompt means special-token text in user content becomes control tokens
  (only `<|image_pad|>` is guarded, and only on the image path).
- A tool call is invisible to a streaming client until the message ends; a long argument is silent except for the
  15 s keepalive comment.
- A `<tool_call>` the parser cannot match ends as plain `content` with `finish_reason: stop`; the log has the raw
  text.
- Only string `tool_choice` values are honoured; an object form (`{"type":"function",...}`) is ignored.

## See also

- [http-api.md](http-api.md) (response shapes), [vision.md](vision.md) (image tokens), [engine-process.md](engine-process.md)
- [docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md) - streaming, tool round trips, `preserve_thinking` reuse.
- [docs/QWEN36-35B-A3B.md](../../QWEN36-35B-A3B.md), section 3.11 (tokenizer, chat template, vision).
