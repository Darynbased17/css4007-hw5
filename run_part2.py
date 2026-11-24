import json
import os
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from agent_graph import MAX_TOOL_ROUNDS, build_agent_graph, make_model, run_turn
from helpdesk_tools import check_service_status, find_instructions, prepare_ticket

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SCENARIOS = {s["id"]: s for s in json.loads((ROOT / "data/scenarios.json").read_text(encoding="utf-8"))["scenarios"]}
PART2_IDS = ["S06", "S07", "S08", "S09", "S14"]


# Calls each tool on its own, without the agent
def direct_tool_calls():
    cfg = {"configurable": {"user_id": "demo-user-a", "thread_id": "direct-demo"}}
    calls = [
        ("find_instructions", {"topic": "wifi"}, None),
        ("find_instructions", {"topic": "printer"}, None),
        ("check_service_status", {"service": "wifi"}, "normal"),
        ("check_service_status", {"service": "wifi"}, "wifi_outage"),
        ("check_service_status", {"service": "wifi"}, "status_unavailable"),
        ("check_service_status", {"service": "account_recovery"}, "recovery_outage"),
        ("prepare_ticket", {"category": "wifi", "summary": "Campus-Demo will not connect.",
                            "steps_tried": ["Enabled Wi-Fi", "Forgot the saved network"],
                            "service_status": "operational"}, "normal"),
        ("prepare_ticket", {"category": "wifi", "summary": "", "steps_tried": [],
                            "service_status": "not checked"}, "normal"),
    ]
    tools = {"find_instructions": find_instructions, "check_service_status": check_service_status,
             "prepare_ticket": prepare_ticket}
    log = []
    for name, args, profile in calls:
        config = {"configurable": {**cfg["configurable"], "status_profile": profile or "normal"}}
        result = tools[name].invoke(args, config=config)
        log.append({"tool": name, "args": args, "status_profile": profile, "result": result})
        print(f"{name}({args}) [{profile}]\n  -> {json.dumps(result, ensure_ascii=False)[:160]}")
    (OUT / "part2_direct_tool_calls.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved outputs/part2_direct_tool_calls.json")


# Runs every event of one scenario in the same graph and saves the log
def run_scenario(scenario_id: str, graph):
    sc = SCENARIOS[scenario_id]
    turns = []
    print(f"\n===== {scenario_id}: {sc['title']}  [status_profile={sc['status_profile']}]")
    for event in sc["events"]:
        config = {"configurable": {"thread_id": event["thread_id"], "user_id": event["user_id"],
                                   "status_profile": sc["status_profile"]},
                  "recursion_limit": 25}
        print(f"\nUSER: {event['message']}")
        turn = run_turn(graph, event["message"], config)
        for step in turn["steps"]:
            for req in step.get("model_tool_requests", []):
                print(f"  model requests -> {req['tool']}({req['args']})")
            for res in step.get("tool_results", []):
                print(f"  tool result    <- {res['tool']}: {json.dumps(res['result'], ensure_ascii=False)[:150]}")
        print(f"HELPDESK: {turn['final_reply']}")
        turns.append(turn)
    log = {
        "scenario": scenario_id,
        "title": sc["title"],
        "command": f"python run_part2.py {scenario_id}",
        "provider": "OpenRouter (OpenAI-compatible API)",
        "model": os.getenv("MODEL_NAME"),
        "status_profile": sc["status_profile"],
        "user_id": sc["events"][0]["user_id"],
        "thread_id": sc["events"][0]["thread_id"],
        "loop_limit": MAX_TOOL_ROUNDS,
        "data_sources": "data/helpdesk.json (find_instructions), data/service_status.json "
                        "(check_service_status), conversation state (earlier messages in this thread)",
        "turns": turns,
    }
    (OUT / f"{scenario_id}_part2.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> saved outputs/{scenario_id}_part2.json")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "direct":
        direct_tool_calls()
    else:
        model = make_model()
        for sid in (PART2_IDS if what == "all" else [what.upper()]):
            # a fresh in-memory checkpointer per scenario implements reset_before: true
            run_scenario(sid, build_agent_graph(model, checkpointer=InMemorySaver()))