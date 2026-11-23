"""Part 1: a helpdesk graph with NO language model.

Two versions of the graph:
  - "original":            read_request -> wifi_help / password_help / other_help
  - "with_clarification":  same, plus clarify_problem when both topics appear
"""

import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

DATA_DIR = Path(__file__).parent / "data"
HELPDESK = json.loads((DATA_DIR / "helpdesk.json").read_text(encoding="utf-8"))


#1. The state: what the graph carries between nodes 
class HelpdeskState(TypedDict, total=False):
    message: str
    category: str
    reply: str

#2. Helper functions
def mentions(text: str, topic: str) -> bool:
    """True if the text contains any alias of the topic."""
    text = text.lower()
    return any(alias in text for alias in HELPDESK["topic_aliases"][topic])


def format_instructions(topic: str) -> str:
    guide = HELPDESK["instructions"][topic]
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(guide["steps"], 1))
    return f"[{guide['instruction_id']}] {guide['title']}\n{steps}"


#3. Nodes: each reads the state and returns ONLY what it changes
def make_read_request(detect_both: bool):
    """Build read_request. The changed graph also detects 'both topics'."""
    def read_request(state: HelpdeskState) -> dict:
        msg = state["message"]
        wifi, password = mentions(msg, "wifi"), mentions(msg, "password")
        if detect_both and wifi and password:
            category = "both"
        elif wifi:
            category = "wifi"
        elif password:
            category = "password"
        else:
            category = "other"
        return {"category": category}

    return read_request


def wifi_help(state: HelpdeskState) -> dict:
    return {"reply": format_instructions("wifi")}


def password_help(state: HelpdeskState) -> dict:
    return {"reply": format_instructions("password")}


def other_help(state: HelpdeskState) -> dict:
    return {"reply": "I can help with Wi-Fi or password problems. "
                     "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state: HelpdeskState) -> dict:
    return {"reply": "Your message mentions both Wi-Fi and a password. "
                     "Is the problem connecting to Wi-Fi, or recovering your account password?"}


#4. The conditional edge: chooses the NEXT node, changes nothing
def choose_next_node(state: HelpdeskState) -> str:
    return {
        "wifi": "wifi_help",
        "password": "password_help",
        "both": "clarify_problem",
        "other": "other_help",
    }[state["category"]]


#5. Build and compile the graph
def build_graph(variant: str = "original"):
    with_clarification = variant == "with_clarification"
    builder = StateGraph(HelpdeskState)

    builder.add_node("read_request", make_read_request(detect_both=with_clarification))
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]
    if with_clarification:
        builder.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")

    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", choose_next_node, targets)
    for node in targets:
        builder.add_edge(node, END)

    return builder.compile()


#6. Run once and record the trace 
def run_with_trace(message: str, variant: str = "original") -> dict:
    """Run the graph and record each node with the full state after it."""
    graph = build_graph(variant)
    state = {"message": message}
    steps = []
    # stream_mode="updates" yields {node_name: update} after every node
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node_name, update in chunk.items():
            state = {**state, **update}   # merge, exactly like LangGraph does
            steps.append({"node": node_name, "update": update, "state_after": dict(state)})
    return {
        "variant": variant,
        "message": message,
        "route": ["START"] + [s["node"] for s in steps] + ["END"],
        "steps": steps,
        "final_state": state,
    }


if __name__ == "__main__":
    import sys
    variant = "with_clarification" if "--clarify" in sys.argv else "original"
    text = " ".join(a for a in sys.argv[1:] if a != "--clarify") or "I cannot connect to Wi-Fi."
    result = run_with_trace(text, variant)
    print(" -> ".join(result["route"]))
    print(result["final_state"]["reply"])