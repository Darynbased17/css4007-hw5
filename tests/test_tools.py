# Part 2 tool tests: each tool is called directly, without the agent

import json
from pathlib import Path

import pytest

from conftest import make_config
from helpdesk_tools import check_service_status, find_instructions, prepare_ticket


# Catches: wrong topic aliases or a changed data file format
def test_find_instructions_known_and_unknown_topic():
    wifi = find_instructions.invoke({"topic": "Wi-Fi"})
    assert wifi["instruction_id"] == "GUIDE-WIFI" and len(wifi["steps"]) == 4
    assert find_instructions.invoke({"topic": "password"})["instruction_id"] == "GUIDE-PASSWORD"
    assert "error" in find_instructions.invoke({"topic": "printer"})


# Catches: ignoring the selected snapshot (always returning "operational")
@pytest.mark.parametrize("profile, service, expected", [
    ("normal", "wifi", "operational"),
    ("wifi_outage", "wifi", "outage"),
    ("status_unavailable", "wifi", "unknown"),
    ("recovery_outage", "account_recovery", "outage"),
    ("recovery_outage", "password", "outage"),  # password topic maps to account_recovery
])
def test_check_service_status_uses_selected_profile(profile, service, expected):
    result = check_service_status.invoke({"service": service}, config=make_config(profile=profile))
    assert result["status"] == expected
    assert result["status_profile"] == profile


# Catches: a ticket that is not saved, has wrong fields, or invents steps
def test_prepare_ticket_saves_local_draft(isolated_storage):
    args = {"category": "wifi", "summary": "Campus-Demo will not connect.",
            "steps_tried": ["Enabled Wi-Fi", "Forgot the saved network"],
            "service_status": "operational"}
    result = prepare_ticket.invoke(args, config=make_config(thread_id="s06"))
    path = Path(result["ticket_file"])
    assert path.exists() and path.parent == isolated_storage / "tickets"
    ticket = json.loads(path.read_text(encoding="utf-8"))
    assert ticket["status"] == "draft"
    assert ticket["user_id"] == "demo-user-a" and ticket["thread_id"] == "s06"
    assert ticket["steps_tried"] == ["Enabled Wi-Fi", "Forgot the saved network"]
    assert ticket["reference_ids"] == ["GUIDE-WIFI"]
    assert ticket["ticket_id"].startswith("DRAFT-")


# Catches: a ticket created from missing details (S09 rule: ask, do not invent)
def test_prepare_ticket_refuses_missing_details(isolated_storage):
    result = prepare_ticket.invoke({"category": "wifi", "summary": "", "steps_tried": [],
                                    "service_status": "not checked"}, config=make_config())
    assert "error" in result
    assert set(result["missing"]) == {"problem description", "steps already tried"}
    assert not (isolated_storage / "tickets").exists()


# Catches: the model being able to see or choose the profile, user or thread
def test_model_cannot_see_program_settings():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    for tool_fn in (check_service_status, prepare_ticket):
        params = convert_to_openai_tool(tool_fn)["function"]["parameters"]["properties"]
        assert not {"config", "user_id", "thread_id", "status_profile"} & set(params)