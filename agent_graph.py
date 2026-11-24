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
from memory import UserMemory, save_language_preference

MAX_TOOL_ROUNDS = 5

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
  condition, report the outage and OFFER to prepare a local ticket draft describing the error.
  Never say that you cannot prepare a ticket.
- When a service has an outage, also call find_instructions for that topic and follow its steps
  (for example, GUIDE-PASSWORD says to prepare a ticket if recovery is unavailable).
- prepare_ticket needs the problem and the steps the user already tried. Use only what the user said.
  If something is missing, ask the user. Never invent steps, repair times or outages.
- A ticket is a LOCAL DRAFT. Never say it was sent. Give the ticket file name.
- Never ask for or store passwords or recovery codes.
- Answer briefly and clearly."""

MEMORY_RULES = """
Memory rules:
- Earlier messages of THIS conversation are your conversation memory. Use them to understand
  references such as "that" or "the steps I tried".
- If the user asks you to remember a reply language, call save_language_preference.
- The "User memory" section below is loaded by the program for the current user only.
  If a preferred language is saved there, reply in that language.
- You do not know anything from this user's other conversations except the User memory section.
  If nothing is saved, say that no preference or previous problem is recorded."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # whole conversation, new messages are appended
    tool_rounds: int                         # trips to the tools node for the current message
    limit_reached: bool                      # True if the run stopped at MAX_TOOL_ROUNDS
    user_memory: dict                        # Part 3: what was loaded from user memory


# Real model from .env; OpenRouter works through the OpenAI-compatible API
# MODEL_NAME=fake gives the rule-based fake model used by the tests (no API key)
def make_model():
    load_dotenv()
    if os.getenv("MODEL_NAME") == "fake":
        from fake_model import FakeHelpdeskModel
        return FakeHelpdeskModel()
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.environ["MODEL_NAME"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
    )


# Text added to the system prompt with the user memory of the current user
def user_memory_text(memory: dict) -> str:
    if memory.get("language"):
        return f"User memory: preferred reply language = {memory['language']}. Reply in {memory['language']}."
    return "User memory: nothing is saved for this user."


def build_agent_graph(model, checkpointer=None, system_prompt=SYSTEM_PROMPT, use_user_memory=False):
    tool_list = TOOLS + ([save_language_preference] if use_user_memory else [])
    model_with_tools = model.bind_tools(tool_list)   # tells the model which tools exist
    tools_by_name = {t.name: t for t in tool_list}

    # Part 3 node: reads the saved preference of the user given in config by the program
    def load_user_memory(state: AgentState, config):
        user_id = config["configurable"]["user_id"]
        return {"user_memory": UserMemory().get(user_id)}

    # Node 1: the model reads the conversation and either requests tools or answers
    def agent(state: AgentState):
        prompt = system_prompt
        if use_user_memory:
            prompt += "\n" + MEMORY_RULES + "\n\n" + user_memory_text(state.get("user_memory", {}))
        reply = model_with_tools.invoke([SystemMessage(prompt)] + state["messages"])
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
        update = {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1}
        # after a preference is saved, reload it so the next model call uses it
        if any(c["name"] == "save_language_preference" for c in state["messages"][-1].tool_calls):
            update["user_memory"] = UserMemory().get(config["configurable"]["user_id"])
        return update

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
    if use_user_memory:
        builder.add_node("load_user_memory", load_user_memory)
        builder.add_edge(START, "load_user_memory")
        builder.add_edge("load_user_memory", "agent")
    else:
        builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue,
                                  {"tools": "tools", "stop_at_limit": "stop_at_limit", "end": END})
    builder.add_edge("tools", "agent")           # after the tools, always back to the model
    builder.add_edge("stop_at_limit", END)
    return builder.compile(checkpointer=checkpointer)


# Sends one user message and records a readable trace of the run
def run_turn(graph, message: str, config: dict) -> dict:
    steps, tickets = [], []
    has_thread = bool(config.get("configurable", {}).get("thread_id"))
    # what the checkpointer already holds for this thread before the new message
    before = graph.get_state(config).values if has_thread else {}
    earlier_user_messages = [m.content for m in before.get("messages", []) if isinstance(m, HumanMessage)]
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
            if "user_memory" in (update or {}):
                step["user_memory_loaded"] = update["user_memory"]
            steps.append(step)
    final = graph.get_state(config).values if has_thread else {}
    last_answer = next((s["model_answer"] for s in reversed(steps) if "model_answer" in s), None)
    used_tools = {r["tool"] for s in steps for r in s.get("tool_results", [])}
    return {
        "user_message": message,
        "memory_read": {
            "conversation_memory": {"thread_id": config.get("configurable", {}).get("thread_id"),
                                    "messages_before_this_turn": len(before.get("messages", [])),
                                    "earlier_user_messages": earlier_user_messages},
            "user_memory": next((s["user_memory_loaded"] for s in steps
                                 if s["node"] == "load_user_memory"), "not used"),
            "reference_file": "data/helpdesk.json via find_instructions"
                              if "find_instructions" in used_tools else "not read",
        },
        "nodes_visited": [s["node"] for s in steps],
        "steps": steps,
        "final_reply": last_answer,
        "ticket_files": tickets,
        "loop_limit_reached": bool(final.get("limit_reached", False)),
        "messages_in_thread": len(final.get("messages", [])),
    }