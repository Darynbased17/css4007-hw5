# Part 3 runner: memory scenarios S10-S13 and the demo D01 ("I already tried that")
#   python run_part3.py all    -> runs D01, S10, S11, S12, S13
#   python run_part3.py S11    -> runs one scenario
# A restart_process event really ends the Python process: every part of a scenario between
# restarts runs in a NEW process (python run_part3.py S11 --segment 1) against the same storage

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_graph import MAX_TOOL_ROUNDS, build_agent_graph, make_model, run_turn
from memory import conversation_db_path, open_checkpointer, reset_storage, storage_dir, user_memory_path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Kazakh letters in any terminal

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
SCENARIOS = {s["id"]: s for s in
             json.loads((ROOT / "data/scenarios.json").read_text(encoding="utf-8"))["scenarios"]}

# Extra demo required by Part 3: the user refers to an earlier message with "that"
SCENARIOS["D01"] = {
    "id": "D01", "part": 3, "title": "Refer to an earlier message ('I already tried that')",
    "status_profile": "normal", "reset_before": True,
    "events": [
        {"action": "message", "user_id": "demo-user-a", "thread_id": "that-demo",
         "message": "I cannot connect to Wi-Fi."},
        {"action": "message", "user_id": "demo-user-a", "thread_id": "that-demo",
         "message": "I already tried that. What should I do next?"},
    ],
}
PART3_IDS = ["D01", "S10", "S11", "S12", "S13"]


# Splits the events of a scenario into parts separated by restart_process
def split_into_processes(scenario: dict) -> list[list[dict]]:
    parts = [[]]
    for event in scenario["events"]:
        if event["action"] == "restart_process":
            parts.append([])
        else:
            parts[-1].append(event)
    return parts


# Path relative to the project folder when possible, for readable logs
def shown(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def segment_file(scenario_id: str, index: int) -> Path:
    return storage_dir() / "runs" / f"{scenario_id}_process{index + 1}.json"


# Runs inside a child process: opens the saved storage, sends the messages, saves the trace
def run_segment(scenario_id: str, index: int):
    scenario = SCENARIOS[scenario_id]
    events = split_into_processes(scenario)[index]
    graph = build_agent_graph(make_model(), checkpointer=open_checkpointer(), use_user_memory=True)
    turns = []
    for event in events:
        config = {"configurable": {"thread_id": event["thread_id"], "user_id": event["user_id"],
                                   "status_profile": scenario["status_profile"]},
                  "recursion_limit": 25}
        print(f"\n[{event['user_id']} | thread={event['thread_id']}] USER: {event['message']}")
        turn = run_turn(graph, event["message"], config)
        mem = turn["memory_read"]
        print(f"  conversation memory: {mem['conversation_memory']['messages_before_this_turn']} "
              f"earlier messages in thread '{event['thread_id']}'")
        print(f"  user memory loaded:  {mem['user_memory']}")
        for step in turn["steps"]:
            for req in step.get("model_tool_requests", []):
                print(f"  model requests -> {req['tool']}({req['args']})")
            for res in step.get("tool_results", []):
                print(f"  tool result    <- {res['tool']}: {json.dumps(res['result'], ensure_ascii=False)[:120]}")
        print(f"HELPDESK: {turn['final_reply']}")
        turns.append({"user_id": event["user_id"], "thread_id": event["thread_id"], **turn})
    path = segment_file(scenario_id, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"process_id": os.getpid(), "turns": turns}, indent=2,
                               ensure_ascii=False), encoding="utf-8")


# Runs in the parent process: resets storage if needed and starts one child process per part
def run_scenario(scenario_id: str):
    scenario = SCENARIOS[scenario_id]
    print(f"\n===== {scenario_id}: {scenario['title']}  [status_profile={scenario['status_profile']}]")
    if scenario["reset_before"]:
        reset_storage()
        print("(storage reset: conversation and user memory deleted)")
    processes = []
    parts = split_into_processes(scenario)
    for index in range(len(parts)):
        if index > 0:
            print("\n----- restart_process: the previous Python process has exited -----")
        subprocess.run([sys.executable, __file__, scenario_id, "--segment", str(index)], check=True)
        processes.append(json.loads(segment_file(scenario_id, index).read_text(encoding="utf-8")))
    log = {
        "scenario": scenario_id,
        "title": scenario["title"],
        "command": f"python run_part3.py {scenario_id}",
        "model": os.getenv("MODEL_NAME"),
        "status_profile": scenario["status_profile"],
        "reset_before": scenario["reset_before"],
        "restarts": len(parts) - 1,
        "storage": {"conversation_memory": shown(conversation_db_path()),
                    "user_memory": shown(user_memory_path())},
        "loop_limit": MAX_TOOL_ROUNDS,
        "processes": processes,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{scenario_id}_part3.json").write_text(json.dumps(log, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(f"-> saved outputs/{scenario_id}_part3.json")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    args = sys.argv[1:] or ["all"]
    if "--segment" in args:
        run_segment(args[0].upper(), int(args[args.index("--segment") + 1]))
    else:
        for sid in (PART3_IDS if args[0] == "all" else [args[0].upper()]):
            run_scenario(sid)