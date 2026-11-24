# The finished helpdesk as a terminal program, with tools and memory
#   python chat.py --user demo-user-a --thread t1 --profile normal          (interactive, /exit to quit)
#   python chat.py --user demo-user-a --thread t1 --message "text"          (one message, then exit)
#   python chat.py --reset                                                  (delete all saved memory)
# Every turn is also appended to local-storage/logs/<user>__<thread>.jsonl

import argparse
import json
import sys

from dotenv import load_dotenv

from agent_graph import build_agent_graph, make_model, run_turn
from memory import open_checkpointer, reset_storage, storage_dir

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Kazakh letters in any terminal


def print_turn(turn: dict):
    mem = turn["memory_read"]
    print(f"  [conversation memory: {mem['conversation_memory']['messages_before_this_turn']} earlier messages"
          f" | user memory: {mem['user_memory']}]")
    for step in turn["steps"]:
        for req in step.get("model_tool_requests", []):
            print(f"  [model requests -> {req['tool']}({req['args']})]")
        for res in step.get("tool_results", []):
            print(f"  [tool result <- {res['tool']}: {json.dumps(res['result'], ensure_ascii=False)[:120]}]")
    print(f"HELPDESK: {turn['final_reply']}")
    if turn["ticket_files"]:
        print(f"  [local ticket draft: {', '.join(turn['ticket_files'])}]")
    if turn["loop_limit_reached"]:
        print("  [loop limit reached]")


def main():
    parser = argparse.ArgumentParser(description="University IT helpdesk (practice)")
    parser.add_argument("--user", default="demo-user-a", help="invented user id, e.g. demo-user-a")
    parser.add_argument("--thread", default="default", help="conversation id")
    parser.add_argument("--profile", default="normal",
                        choices=["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
    parser.add_argument("--message", action="append", help="send this message and exit (can repeat)")
    parser.add_argument("--reset", action="store_true", help="delete saved conversations and preferences")
    args = parser.parse_args()

    if args.reset:
        reset_storage()
        print(f"Deleted {storage_dir()}")
        return

    load_dotenv()
    graph = build_agent_graph(make_model(), checkpointer=open_checkpointer(), use_user_memory=True)
    config = {"configurable": {"thread_id": args.thread, "user_id": args.user,
                               "status_profile": args.profile}, "recursion_limit": 25}
    log_file = storage_dir() / "logs" / f"{args.user}__{args.thread}.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"user={args.user}  thread={args.thread}  status_profile={args.profile}")

    messages = args.message or []
    interactive = not messages
    while True:
        if interactive:
            text = input("\nYOU: ").strip()
            if text in ("/exit", "/quit"):
                break
            if not text:
                continue
        elif messages:
            text = messages.pop(0)
            print(f"\nYOU: {text}")
        else:
            break
        turn = run_turn(graph, text, config)
        print_turn(turn)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"user_id": args.user, "thread_id": args.thread,
                                "status_profile": args.profile, **turn}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()