# FAKE MODEL for tests and CI, it is NOT a language model
# It follows simple keyword rules so the graph, tools and memory can be tested without an API key
# Every answer starts with [FAKE MODEL] so it can never be confused with a real model run

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

LANGUAGES = ["Kazakh", "Russian", "English"]


class FakeHelpdeskModel(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-helpdesk-model"

    # The real model needs bind_tools; the fake one already knows the tool names
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        reply = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _tool_call(self, name: str, args: dict) -> AIMessage:
        return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"fake-{self.calls}"}])

    def _decide(self, messages) -> AIMessage:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        last = messages[-1]

        # After tool results: answer with a summary of the results
        if isinstance(last, ToolMessage):
            results = [m.content for m in messages[-3:] if isinstance(m, ToolMessage)]
            return AIMessage("[FAKE MODEL] Tool results: " + " | ".join(results))

        text = last.content.lower()
        language = next((lang for lang in LANGUAGES if lang.lower() in text), None)

        # Rule 1: a request to remember a language
        if "remember" in text and "prefer" in text and language:
            return self._tool_call("save_language_preference", {"language": language})
        # Rule 2: a status check
        if "status" in text or "down" in text:
            service = "account_recovery" if ("password" in text or "recovery" in text) else "wifi"
            return self._tool_call("check_service_status", {"service": service})
        # Rule 3: instructions
        if "steps" in text and "which" not in text or "cannot connect" in text:
            topic = "password" if "password" in text else "wifi"
            return self._tool_call("find_instructions", {"topic": topic})

        # Rule 4: answer from memory only (conversation messages + user memory section)
        earlier = [m.content for m in messages[:-1] if isinstance(m, HumanMessage)]
        saved = re.search(r"preferred reply language = ([^.]+)\.", system)
        parts = ["[FAKE MODEL]"]
        parts.append(f"Saved language: {saved.group(1)}." if saved else "No saved language preference.")
        parts.append("Earlier messages in this conversation: " + " / ".join(earlier)
                     if earlier else "No previous problem is recorded in this conversation.")
        return AIMessage(" ".join(parts))