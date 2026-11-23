# Part 2: a model that chooses tools inside a LangGraph loop
#   START -> agent --(model asked for tools)--> tools -> back to agent
#            agent --(plain answer)-----------> END
#            agent --(too many tool rounds)---> stop_at_limit -> END

import json
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from helpdesk_tools import TOOLS

MAX_TOOL_ROUNDS = 5  # loop limit: at most 5 trips to the tools node per user message

SYSTEM_PROMPT = """You are a university IT helpdesk assistant for a PRACTICE exercise.
All data is fictional. You only help with Wi-Fi and password (account recovery) problems.

Rules:
- Use find_instructions to get Wi-Fi or password steps. Do not write steps from your own knowledge.
- Use check_service_status before saying whether a service is up or down.
  Wi-Fi -> service 'wifi'. Password / account recovery -> service 'account_recovery'.
- Status 'unknown' means you could NOT check it: it is not operating and not a confirmed outage.
- Follow the user's conditions exactly. Example: "if it is operating, prepare a ticket" means:
  operational -> call prepare_ticket; outage -> do NOT prepare a ticket; unknown -> do NOT prepare it,
  explain the status could not be checked and ask if they still want a ticket draft.
- An outage only blocks a ticket when the user's own condition says so. If the user set no such
  condition, you may offer to prepare a ticket draft describing the error.
- prepare_ticket needs the problem and the steps the user already tried. Use only what the user said.
  If something is missing, ask the user. Never invent steps, repair times or outages.
- A ticket is a LOCAL DRAFT. Never say it was sent. Give the ticket file name.
- Never ask for or store passwords or recovery codes.
- Answer briefly and clearly."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # whole conversation, new messages are appended
    tool_rounds: int                         # trips to the tools node for the current message
    limit_reached: bool                      # True if the run stopped at MAX_TOOL_ROUNDS


# Real model from .env; OpenRouter works through the OpenAI-compatible API
def make_model():
    from langchain_openai import ChatOpenAI
    load_dotenv()
    return ChatOpenAI(
        model=os.environ["MODEL_NAME"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
    )


def build_agent_graph(model, checkpointer=None, system_prompt=SYSTEM_PROMPT):
    model_with_tools = model.bind_tools(TOOLS)   # tells the model which tools exist
    tools_by_name = {t.name: t for t in TOOLS}

    # Node 1: the model reads the conversation and either requests tools or answers
    def agent(state: AgentState):
        reply = model_with_tools.invoke([SystemMessage(system_prompt)] + state["messages"])
        return {"messages": [reply]}

    # Node 2: the program executes the tools requested by the model
    def tools(state: AgentState, config):
        results = []
        for call in state["messages"][-1].tool_calls:
            tool_fn = tools_by_name.get(call["name"])
            try:
                output = tool_fn.invoke(call["args"], config=config) if tool_fn \
                    else {"error": f"Unknown tool {call['name']}"}
            except Exception as error:  # invalid arguments from the model are reported back to it
                output = {"error": str(error)}
            results.append(ToolMessage(content=json.dumps(output, ensure_ascii=False),
                                       tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1}

    # Node 3: safety stop when the loop becomes too long
    def stop_at_limit(state: AgentState):
        # every requested tool call must get a ToolMessage, otherwise the next API call fails
        unanswered = [ToolMessage(content='{"error": "Not run: tool limit reached."}',
                                  tool_call_id=c["id"], name=c["name"])
                      for c in state["messages"][-1].tool_calls]
        final = AIMessage(f"I stopped after {MAX_TOOL_ROUNDS} tool rounds (loop limit). "
                          "Please rephrase or give more details.")
        return {"messages": unanswered + [final], "limit_reached": True}

    # Conditional edge after the agent node
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "end"                         # plain answer: finish
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return "stop_at_limit"               # the loop limit is enforced here
        return "tools"                           # tool requested: run it

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_node("stop_at_limit", stop_at_limit)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue,
                                  {"tools": "tools", "stop_at_limit": "stop_at_limit", "end": END})
    builder.add_edge("tools", "agent")           # after the tools, always back to the model
    builder.add_edge("stop_at_limit", END)
    return builder.compile(checkpointer=checkpointer)


# Sends one user message and records a readable trace of the run
def run_turn(graph, message: str, config: dict) -> dict:
    steps, tickets = [], []
    new_input = {"messages": [HumanMessage(message)], "tool_rounds": 0, "limit_reached": False}
    for chunk in graph.stream(new_input, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            step = {"node": node}
            for msg in (update or {}).get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    step.setdefault("model_tool_requests", []).extend(
                        {"tool": c["name"], "args": c["args"]} for c in msg.tool_calls)
                elif isinstance(msg, AIMessage):
                    step["model_answer"] = msg.content
                elif isinstance(msg, ToolMessage):
                    result = json.loads(msg.content)
                    step.setdefault("tool_results", []).append({"tool": msg.name, "result": result})
                    if isinstance(result, dict) and "ticket_file" in result:
                        tickets.append(result["ticket_file"])
            if "tool_rounds" in (update or {}):
                step["state_change"] = {"tool_rounds": update["tool_rounds"]}
            steps.append(step)
    final = graph.get_state(config).values if config.get("configurable", {}).get("thread_id") else {}
    last_answer = next((s["model_answer"] for s in reversed(steps) if "model_answer" in s), None)
    return {
        "user_message": message,
        "nodes_visited": [s["node"] for s in steps],
        "steps": steps,
        "final_reply": last_answer,
        "ticket_files": tickets,
        "loop_limit_reached": bool(final.get("limit_reached", False)),
        "messages_in_thread": len(final.get("messages", [])),
    }