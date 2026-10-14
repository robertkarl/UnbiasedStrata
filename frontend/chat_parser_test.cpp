// Exercise request preparation and final tool parsing with the actual GGUF.
// No inference engine or CUDA context is created.
#define main unused_serving_main
#include "main.cpp"
#undef main
#include <cassert>
#include <sstream>

int strata_engine_main(int, char**) { return 99; }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    Server server;
    server.opt.ctx = 196608;
    auto params = llama_model_default_params(); params.vocab_only = true;
    server.model = llama_model_load_from_file(argv[1], params);
    assert(server.model);
    server.vocab = llama_model_get_vocab(server.model);
    server.tmpls = common_chat_templates_init(server.model, "");
    assert(common_chat_templates_get_caps(server.tmpls.get()).at("supports_parallel_tool_calls"));
    auto request = common_json::parse_no_throw(R"({
        "messages":[{"role":"user","content":"Check the files and the date."}],
        "tools":[{"type":"function","function":{"name":"bash","parameters":{
            "type":"object","properties":{"command":{"type":"string"}},"required":["command"]
        }}}]
    })");
    const std::string output = "Reasoning.\n</think>\n\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\ndate\n</parameter>\n</function>\n</tool_call>";
    Prepared default_request, explicit_request;
    std::string error;
    assert(server.prepare(request, default_request, error));
    auto message = common_chat_parse(output, false, default_request.parser);
    assert(message.tool_calls.size() == 2);
    assert(message.tool_calls[0].arguments == "{\"command\":\"ls\"}");
    assert(message.tool_calls[1].arguments == "{\"command\":\"date\"}");
    request["parallel_tool_calls"] = true;
    assert(server.prepare(request, explicit_request, error));
    assert(explicit_request.ids == default_request.ids);
    assert(common_chat_parse(output, false, explicit_request.parser).tool_calls.size() == 2);
    request["parallel_tool_calls"] = false;
    assert(server.prepare(request, explicit_request, error));
    assert(explicit_request.ids == default_request.ids);
    assert(common_chat_parse(output, false, explicit_request.parser).tool_calls.size() == 1);
    // Arguments in any order: the model writes optional parameters before required ones, and upstream's
    // required-first rule dropped such calls without an error (patches/llama.cpp-qwen3-coder-any-argument-order.patch).
    auto read_request = common_json::parse_no_throw(R"({
        "messages":[{"role":"user","content":"Read the file."}],
        "tools":[{"type":"function","function":{"name":"read","parameters":{
            "type":"object","required":["path"],"properties":{"path":{"type":"string"},
            "offset":{"type":"number"},"limit":{"type":"number"}}
        }}}]
    })");
    Prepared read_prepared;
    assert(server.prepare(read_request, read_prepared, error));
    for (const char* order : {"limit offset path", "offset path limit", "path limit offset", "path"}) {
        std::string text = "Reasoning.\n</think>\n\n<tool_call>\n<function=read>\n";
        std::istringstream names(order);
        for (std::string name; names >> name;)
            text += "<parameter=" + name + ">\n" + (name == "path" ? "/tmp/a.txt" : "40") + "\n</parameter>\n";
        text += "</function>\n</tool_call>";
        auto parsed = common_chat_parse(text, false, read_prepared.parser);
        assert(parsed.tool_calls.size() == 1);
        assert(parsed.tool_calls[0].name == "read");
        assert(parsed.tool_calls[0].arguments.find("\"path\":\"/tmp/a.txt\"") != std::string::npos);
    }
    // Streaming (REGINA-110): the response's text grows a UTF-8 character at a time, as it does token by token
    // in run(); the answer must be published as it grows, not in one piece at the end, and the pieces must add
    // up to the parse of the whole message.  Tool calls are published once, complete.
    struct StreamCase { const char* name; bool thinking; bool tools; std::string output; size_t calls; };
    const std::string call_ls = "<tool_call>\n<function=bash>\n<parameter=command>\nls -l \"a b\" | head -n 3\n"
                                "</parameter>\n</function>\n</tool_call>";
    const std::string answer = "A lighthouse stands on the cape: caf\xC3\xA9 lights, 2 < 3 and <b>bold</b>.\n\n"
                               "```json\n{\"a\": [1, 2], \"s\": \"x\\ny\"}\n```\nIt writes <tool and <function= as text.";
    const StreamCase cases[] = {
        {"reasoning then answer", true, false, "The user asks.\nTwo steps.\n</think>\n\n" + answer, 0},
        {"answer, thinking off", false, false, answer, 0},
        {"answer, tools declared", true, true, "No tool is needed.\n</think>\n\n" + answer, 0},
        {"answer, tools declared, thinking off", false, true, answer, 0},
        {"text then a tool call", true, true, "List first.\n</think>\n\nI will list the files.\n\n" + call_ls, 1},
        {"a tool call alone", false, true, call_ls, 1},
        {"two tool calls", true, true, "Both.\n</think>\n\n" + call_ls + "\n" + call_ls, 2},
    };
    for (const StreamCase& c : cases) {
        auto body = common_json::parse_no_throw(R"({"messages":[{"role":"user","content":"Describe a lighthouse."}],
            "stream":true})");
        if (c.tools) body["tools"] = request.at("tools");
        body["chat_template_kwargs"] = common_json::object();
        body["chat_template_kwargs"]["enable_thinking"] = c.thinking;
        Prepared prepared;
        assert(server.prepare(body, prepared, error));
        const common_chat_msg whole = common_chat_parse(c.output, false, prepared.parser);
        StreamParse parse;
        std::string content, reasoning;
        size_t content_deltas = 0, reasoning_deltas = 0, call_deltas = 0, steps = 0;
        bool calls_only_at_end = true;
        auto take = [&](const std::vector<common_chat_msg_diff>& diffs, bool partial) {
            for (const auto& d : diffs) {
                if (!d.content_delta.empty()) { content += d.content_delta; ++content_deltas; }
                if (!d.reasoning_content_delta.empty()) { reasoning += d.reasoning_content_delta; ++reasoning_deltas; }
                if (d.tool_call_index != std::string::npos) { ++call_deltas; if (partial) calls_only_at_end = false; }
            }
        };
        for (size_t n = 1; n < c.output.size(); ++n) {
            if (utf8_complete_prefix(c.output.substr(0, n)) != n) continue;
            take(parse.step(c.output.substr(0, n), true, prepared.parser), true);
            ++steps;
        }
        take(parse.step(c.output, false, prepared.parser), false);
        std::printf("stream case '%s': %zu steps, %zu reasoning deltas, %zu content deltas, %zu tool-call deltas\n",
                    c.name, steps, reasoning_deltas, content_deltas, call_deltas);
        if (std::getenv("STREAM_TEST_VERBOSE"))
            std::printf("  whole content [%s]\n  streamed      [%s]\n", whole.content.c_str(), content.c_str());
        assert(reasoning == whole.reasoning_content);
        // the streamed answer is the whole one; before a tool call it may keep the white space the parser trims
        assert(content.compare(0, whole.content.size(), whole.content) == 0);
        assert(content.find_first_not_of(" \n", whole.content.size()) == std::string::npos);
        assert(whole.content.size() < 20 || content_deltas * 2 >= whole.content.size());
        assert(whole.reasoning_content.size() < 10 || reasoning_deltas * 2 >= whole.reasoning_content.size());
        assert(whole.tool_calls.size() == c.calls && parse.msg.tool_calls.size() == c.calls);
        assert(calls_only_at_end && call_deltas == c.calls);
        for (size_t i = 0; i < c.calls; ++i) {
            assert(parse.msg.tool_calls[i].arguments == "{\"command\":\"ls -l \\\"a b\\\" | head -n 3\"}");
            assert(!parse.msg.tool_calls[i].id.empty());
        }
    }
    llama_model_free(server.model);
    std::puts("PASS: omitted parallel_tool_calls uses the model capability; explicit false remains honored; "
              "tool arguments parse in any order; the answer streams as it grows and tool calls arrive whole");
}
