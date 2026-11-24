# Shared test setup
# Every test gets its own empty storage and ticket folder, and uses the FAKE model (no API key)

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "fake")
    monkeypatch.setenv("HELPDESK_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("HELPDESK_TICKET_DIR", str(tmp_path / "tickets"))
    return tmp_path


# Config that the program would build for one user and one conversation
def make_config(user_id="demo-user-a", thread_id="t1", profile="normal"):
    return {"configurable": {"user_id": user_id, "thread_id": thread_id, "status_profile": profile},
            "recursion_limit": 25}