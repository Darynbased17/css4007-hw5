# Part 2 loop tests with a scripted FAKE model (no API key)

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_graph import MAX_TOOL_ROUNDS, build_agent_graph, run_turn
from conftest import make_config


# A fake model that returns prepared messages in order
class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, call_id):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


# Catches: tools not executed, results not returned to the model, or tool chaining broken
def test_tool_request_is_executed_and_result_returned():
    model = ScriptedModel(messages=iter([
        call("check_service_status", {"service": "wifi"}, "c1"),
        call("prepare_ticket", {"category": "wifi", "summary": "No connection",
                                "steps_tried": ["Enabled Wi-Fi"], "service_status": "operational"}, "c2"),
        AIMessage("Draft saved."),
    ]))
    graph = build_agent_graph(model, checkpointer=InMemorySaver())
    turn = run_turn(graph, "Check Wi-Fi and prepare a ticket", make_config())
    assert turn["nodes_visited"] == ["agent", "tools", "agent", "tools", "agent"]
    first_result = turn["steps"][1]["tool_results"][0]
    assert first_result["tool"] == "check_service_status"
    assert first_result["result"]["status"] == "operational"
    assert len(turn["ticket_files"]) == 1
    assert turn["final_reply"] == "Draft saved." and not turn["loop_limit_reached"]


# Catches: the status snapshot not reaching the tool through config
def test_outage_profile_reaches_tool():
    model = ScriptedModel(messages=iter([call("check_service_status", {"service": "wifi"}, "c1"),
                                         AIMessage("Wi-Fi is down.")]))
    graph = build_agent_graph(model, checkpointer=InMemorySaver())
    turn = run_turn(graph, "Is Wi-Fi down?", make_config(profile="wifi_outage"))
    assert turn["steps"][1]["tool_results"][0]["result"]["status"] == "outage"
    assert turn["ticket_files"] == []


# Catches: a missing or broken loop limit (the run would never stop)
def test_loop_limit_stops_endless_tool_requests():
    endless = iter(call("find_instructions", {"topic": "wifi"}, f"c{i}") for i in range(100))
    graph = build_agent_graph(ScriptedModel(messages=endless), checkpointer=InMemorySaver())
    turn = run_turn(graph, "loop forever", make_config())
    assert turn["loop_limit_reached"] is True
    assert turn["nodes_visited"].count("tools") == MAX_TOOL_ROUNDS
    assert turn["nodes_visited"][-1] == "stop_at_limit"


# Catches: a plain answer that still goes to the tools node
def test_plain_answer_ends_without_tools():
    graph = build_agent_graph(ScriptedModel(messages=iter([AIMessage("Hello")])))
    turn = run_turn(graph, "hi", {"configurable": {}})
    assert turn["nodes_visited"] == ["agent"] and turn["final_reply"] == "Hello"
    