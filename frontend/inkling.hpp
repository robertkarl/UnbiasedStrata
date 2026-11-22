// The Inkling 2 chat template emits one typed block per content part, including
// a separate block for each tool invocation. The same PEG handles partial SSE output.
#pragma once

#include "chat-peg-parser.h"

namespace strata_frontend {

// The pinned Jinja lacks tojson(sort_keys=true). The template constructs its
// outer descriptors in order. Sort nested tool schemas once before rendering.
inline void inkling_sort_keys(common_json & value) {
    if (value.is_object()) {
        std::map<std::string, common_json> sorted;
        for (auto item : value.items()) {
            inkling_sort_keys(item.value());
            sorted.emplace(item.key(), item.value());
        }
        value.clear();
        for (auto & [key, item] : sorted) value[key] = std::move(item);
    } else if (value.is_array()) {
        for (auto & item : value) inkling_sort_keys(item);
    }
}

inline bool prepare_inkling_chat(common_json & body, const std::map<std::string, std::string> & defaults) {
    if (body.contains("tools")) inkling_sort_keys(body.at("tools"));
    const auto kwargs = body.value("chat_template_kwargs", common_json::object());
    auto setting = [&](const std::string & name) {
        if (kwargs.contains(name)) return kwargs.at(name);
        auto it = defaults.find(name);
        return it == defaults.end() ? common_json(nullptr) : common_json::parse(it->second);
    };
    const auto effort = body.value("reasoning_effort", setting("reasoning_effort"));
    if (effort != "none" && effort != 0 && setting("enable_thinking") != false) return false;
    body["chat_template_kwargs"]["reasoning_effort"] = "none";
    body.erase("reasoning_effort");
    return true;
}

inline void inkling_chat_parser(common_json & data, const common_json & tools, bool extract_reasoning) {
    const std::string model = "<|message_model|>";
    const std::string text = "<|content_text|>";
    const std::string tool_error = "<|content_tool_error|>";
    const std::string tool_text = "<|content_invoke_tool_text|>";
    const std::string think = "<|content_thinking|>";
    const std::string invoke = "<|content_invoke_tool_json|>";
    const std::string end = "<|end_message|>";
    const std::string eos = "<|content_model_end_sampling|>";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto body = p.until_one_of({end, eos});
        auto close = p.optional(p.literal(end));
        auto text_kind = p.choice({p.literal(text), p.literal(tool_error), p.literal(tool_text)});
        auto answer = text_kind + p.content(body) + close;
        auto reasoning = extract_reasoning
            ? p.literal(think) + p.reasoning(body) + close
            : p.content(p.literal(think) + body + close);
        std::vector<common_peg_parser> kinds{answer, reasoning};
        if (!tools.empty()) {
            auto call = p.standard_json_tools(invoke, end, tools, false, true, "name", "args");
            kinds.push_back(p.until_one_of({invoke, end, eos}) + call);
        }
        auto block = p.optional(p.literal(model)) + p.choice(kinds);
        return p.zero_or_more(block) + p.optional(p.literal(eos)) + p.end();
    });
    data["chat_format"] = static_cast<int>(COMMON_CHAT_FORMAT_PEG_NATIVE);
    data["chat_parser"] = parser.save();
    data["preserved_tokens"] = std::vector<std::string>{model, text, tool_error, tool_text, think, invoke, end, eos};
}

} // namespace strata_frontend
