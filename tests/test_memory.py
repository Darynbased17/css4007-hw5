# Part 3 memory tests with the rule-based FAKE model (MODEL_NAME=fake, no API key)

import os
import subprocess
import sys

from agent_graph import build_agent_graph, make_model, run_turn
from conftest import ROOT, make_config
from memory import UserMemory, open_checkpointer


def new_app():
    # a new graph with a new connection to the same storage, as after a restart
    return build_agent_graph(make_model(), checkpointer=open_checkpointer(), use_user_memory=True)


def chat_process(*args) -> str:
    # runs chat.py as a separate Python process and returns what it printed
    result = subprocess.run([sys.executable, str(ROOT / "chat.py"), *args], capture_output=True,
                            text=True, encoding="utf-8", env=os.environ.copy(), cwd=ROOT, check=True)
    return result.stdout


# Catches: conversation state not saved by the checkpointer or not reloaded for the same thread
def test_conversation_survives_new_graph_instance():
    run_turn(new_app(), "My Wi-Fi fails. I forgot the saved network.", make_config(thread_id="s10"))
    turn = run_turn(new_app(), "Which steps did I already try?", make_config(thread_id="s10"))
    assert turn["memory_read"]["conversation_memory"]["messages_before_this_turn"] > 0
    assert "forgot the saved network" in turn["final_reply"]


# Catches: memory that lives only inside one Python process (real restart through chat.py)
def test_memory_survives_process_restart():
    chat_process("--user", "demo-user-a", "--thread", "restart-a", "--message",
                 "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi.")
    output = chat_process("--user", "demo-user-a", "--thread", "restart-a", "--message",
                          "Which problem and steps did we discuss?")
    assert "Saved language: Kazakh" in output
    assert "My issue is Wi-Fi" in output


# Catches: a new thread that reuses old messages, or a preference that is not shared across threads
def test_new_thread_has_preference_but_not_old_messages():
    run_turn(new_app(), "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.",
             make_config(thread_id="preference-a1"))
    turn = run_turn(new_app(), "Which language did I ask you to use?",
                    make_config(thread_id="preference-a2"))
    assert turn["memory_read"]["conversation_memory"]["messages_before_this_turn"] == 0
    assert turn["memory_read"]["user_memory"] == {"language": "Kazakh"}
    assert "Wi-Fi problem" not in turn["final_reply"]


# Catches: one user's preference or conversation leaking to another user
def test_users_are_separated():
    run_turn(new_app(), "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.",
             make_config(user_id="demo-user-a", thread_id="isolation-a"))
    turn = run_turn(new_app(), "Have I told you a preferred language or a previous problem?",
                    make_config(user_id="demo-user-b", thread_id="isolation-b"))
    assert turn["memory_read"]["user_memory"] == {}
    assert "Kazakh" not in turn["final_reply"]
    assert UserMemory().get("demo-user-a") == {"language": "Kazakh"}
    assert UserMemory().get("demo-user-b") == {}