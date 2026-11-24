# Part 3: the two kinds of memory and where they are stored

import json
import os
import shutil
import sqlite3
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(__file__).parent


def storage_dir() -> Path:
    folder = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))
    return folder if folder.is_absolute() else ROOT / folder


def conversation_db_path() -> Path:
    return storage_dir() / "conversations.sqlite"


def user_memory_path() -> Path:
    return storage_dir() / "user_memory.json"


# Deletes all saved conversations and user preferences
def reset_storage():
    shutil.rmtree(storage_dir(), ignore_errors=True)


# Conversation memory: a LangGraph checkpointer that saves the state of every thread in SQLite
def open_checkpointer() -> SqliteSaver:
    storage_dir().mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(conversation_db_path(), check_same_thread=False)
    return SqliteSaver(connection)


# User memory: a small JSON file {user_id: {"language": ...}}
class UserMemory:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else user_memory_path()

    def _read_all(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    # Returns only the data of this user, other users are never read
    def get(self, user_id: str) -> dict:
        return self._read_all().get(user_id, {})

    def set(self, user_id: str, key: str, value):
        data = self._read_all()
        data.setdefault(user_id, {})[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# Tool for the model: saves the reply language for the CURRENT user
# The user_id comes from config set by the program, the model cannot choose another user
@tool(description="Save the user's preferred reply language (for example 'Kazakh') so it is "
                  "remembered in later conversations. Call it only when the user asks you to "
                  "remember a language preference.")
def save_language_preference(language: str, config: RunnableConfig) -> dict:
    user_id = ((config or {}).get("configurable") or {}).get("user_id")
    if not user_id:
        return {"error": "No user_id in config; preference not saved."}
    UserMemory().set(user_id, "language", language.strip())
    return {"saved": True, "user_id": user_id, "language": language.strip(),
            "storage": "user memory (local-storage/user_memory.json)"}