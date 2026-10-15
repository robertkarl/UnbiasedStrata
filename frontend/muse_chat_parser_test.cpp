// Exercise the serving adapter with Muse's real tokenizer/template, without inference.
#define main unused_serving_main
#include "main.cpp"
#undef main

int strata_engine_main(int, char**) { return 99; }

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: muse-chat-test MODEL.gguf");
        llama_log_set(quiet_log, nullptr);
        llama_backend_init();
        Server server;
        server.opt.ctx = 4096;
        auto params = llama_model_default_params(); params.vocab_only = true;
        server.model = llama_model_load_from_file(argv[1], params);
        require(server.model != nullptr, "tokenizer load failed");
        server.vocab = llama_model_get_vocab(server.model);
        server.tmpls = common_chat_templates_init(server.model, "");
        auto request = common_json::parse_no_throw(R"({
          "messages":[{"role":"user","content":"Check the weather in Paris."}],
          "reasoning_effort":"low","temperature":0,
          "tools":[{"type":"function","function":{"name":"weather","parameters":{
            "type":"object","properties":{"city":{"type":"string"}},"required":["city"]
          }}}]
        })");
        Prepared prepared;
        std::string error;
        require(server.prepare(request, prepared, error), error);
        const std::string reasoning = "I should check the weather.";
        const std::string prefix = " to=self<|message|>" + reasoning + "<|eom|><|start|>assistant";
        const std::string call = " to=weather<|message|><atem:function_calls>\n"
          "<atem:invoke name=\"weather\"><atem:parameter name=\"city\">Paris</atem:parameter>"
          "</atem:invoke>\n</atem:function_calls>";
        for (bool tool : {false, true}) {
            const std::string output = prefix + (tool ? call : " to=user<|message|>It is sunny.");
            const auto whole = common_chat_parse(common_chat_input(output), false, prepared.parser);
            require(whole.reasoning_content == reasoning, "reasoning was not extracted");
            require(whole.tool_calls.size() == size_t(tool), "incorrect tool count");
            if (tool) {
                require(whole.tool_calls[0].name == "weather", "incorrect tool name");
                require(whole.tool_calls[0].arguments == "{\"city\":\"Paris\"}", "incorrect tool arguments");
            } else require(whole.content == "It is sunny.", "incorrect answer content");
            StreamParse stream;
            std::string content, analysis;
            size_t calls = 0;
            for (size_t n = 1; n <= output.size(); ++n) {
                bool partial = n != output.size();
                for (const auto& d : stream.step(output.substr(0, n), partial, prepared.parser)) {
                    content += d.content_delta;
                    analysis += d.reasoning_content_delta;
                    if (d.tool_call_index != std::string::npos) {
                        require(!partial, "partial tool arguments were published");
                        ++calls;
                    }
                }
            }
            require(content == whole.content && analysis == whole.reasoning_content, "streaming differs from final parse");
            require(calls == size_t(tool), "streamed tool count differs");
        }
        request["reasoning_effort"] = "none";
        bool rejected = false;
        try { rejected = !server.prepare(request, prepared, error); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported reasoning accepted");
        request["reasoning_effort"] = "low";
        request["messages"].push_back(common_json::parse_no_throw(R"({"role":"assistant","content":null,
          "tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]})"));
        request["messages"].push_back(common_json::parse_no_throw(R"({"role":"tool","tool_call_id":"call_1","content":"Sunny, 20 C."})"));
        require(server.prepare(request, prepared, error), "tool result round trip: " + error);
        std::puts("PASS: Muse reasoning, answer, tool arguments, streaming, tool-result history, and invalid effort");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "muse-chat-test: %s\n", e.what()); return 1;
    }
}
