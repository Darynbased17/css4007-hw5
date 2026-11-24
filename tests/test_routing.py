# Part 1 routing tests (no model)

import pytest

from part1_graph import run_with_trace


# Catches: a broken keyword rule, a broken lower() call, or a wrong conditional edge mapping
@pytest.mark.parametrize("message, expected_node", [
    ("I cannot connect to Wi-Fi.", "wifi_help"),
    ("wifi is slow", "wifi_help"),
    ("I forgot my PASSWORD.", "password_help"),
    ("My laptop screen is broken.", "other_help"),
])
def test_original_graph_routes(message, expected_node):
    result = run_with_trace(message, "original")
    assert result["route"] == ["START", "read_request", expected_node, "END"]


# Catches: rule order changed by accident (S04 must still go to wifi_help in the original graph)
def test_both_topics_original_graph_prefers_wifi():
    result = run_with_trace("My Wi-Fi password does not work.", "original")
    assert result["route"][2] == "wifi_help"
    assert result["final_state"]["category"] == "wifi"


# Catches: the clarify_problem node missing or not connected (S05)
def test_both_topics_changed_graph_clarifies():
    result = run_with_trace("My Wi-Fi password does not work.", "with_clarification")
    assert result["route"] == ["START", "read_request", "clarify_problem", "END"]
    assert "Wi-Fi" in result["final_state"]["reply"] and "password" in result["final_state"]["reply"]


# Catches: help nodes that stop using data/helpdesk.json, or a node that overwrites message
def test_reply_comes_from_reference_file_and_message_is_kept():
    result = run_with_trace("I forgot my password", "original")
    state = result["final_state"]
    assert state["reply"].startswith("[GUIDE-PASSWORD]")
    assert state["message"] == "I forgot my password"
    assert [s["node"] for s in result["steps"]] == ["read_request", "password_help"]