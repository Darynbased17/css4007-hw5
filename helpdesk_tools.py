import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


HELPDESK = load_json("helpdesk.json")
STATUS_PROFILES = load_json("service_status.json")["profiles"]
TICKET_TEMPLATE = load_json("ticket_template.json")["ticket"]


# Folder for ticket drafts, can be changed with HELPDESK_TICKET_DIR
def ticket_dir() -> Path:
    return Path(os.getenv("HELPDESK_TICKET_DIR", ROOT / "outputs" / "tickets"))


# Reads a value that the program put into config["configurable"]
def setting(config, key: str, default=None):
    return ((config or {}).get("configurable") or {}).get(key, default)


# Normalises a topic name: Wi-Fi -> wifi, PASSWORD -> password, anything else -> None
def to_topic(word: str):
    word = word.lower().strip()
    for topic, aliases in HELPDESK["topic_aliases"].items():
        if word == topic or word in aliases:
            return topic
    return None


# Tool 1: instructions from data/helpdesk.json
@tool(description="Look up the local helpdesk instructions. topic must be 'wifi' or 'password'.")
def find_instructions(topic: str) -> dict:
    key = to_topic(topic)
    if key is None:
        return {"error": f"No instructions for '{topic}'.",
                "available_topics": list(HELPDESK["instructions"])}
    guide = HELPDESK["instructions"][key]
    return {"topic": key, "instruction_id": guide["instruction_id"],
            "title": guide["title"], "steps": guide["steps"]}


# Tool 2: service status from the snapshot selected by the program
@tool(description="Check whether a service is operating. service must be 'wifi' or "
                  "'account_recovery' (password problems use 'account_recovery').")
def check_service_status(service: str, config: RunnableConfig) -> dict:
    profile = setting(config, "status_profile", "normal")
    topic = to_topic(service)
    name = HELPDESK["topic_services"][topic] if topic else service.lower().strip()
    services = STATUS_PROFILES[profile]
    if name not in services:
        return {"error": f"Unknown service '{service}'.", "available_services": list(services)}
    entry = services[name]
    return {"service": name, "status": entry["status"],
            "message": entry["message"], "status_profile": profile}


# Tool 3: a local ticket draft saved as JSON, nothing is sent
@tool(description="Save a LOCAL ticket draft (nothing is sent). "
                  "category: 'wifi' or 'password'. summary: the problem in the user's words. "
                  "steps_tried: only steps the user said they tried. "
                  "service_status: 'operational', 'outage', 'unknown' or 'not checked'.")
def prepare_ticket(category: str, summary: str, steps_tried: list[str],
                   service_status: str, config: RunnableConfig) -> dict:
    topic = to_topic(category)
    missing = []
    if topic is None:
        missing.append("category (wifi or password)")
    if not summary.strip():
        missing.append("problem description")
    if not [s for s in steps_tried if s.strip()]:
        missing.append("steps already tried")
    if missing:
        # Missing details are not invented: the model is told to ask the user
        return {"error": "Ticket not prepared: missing information.", "missing": missing,
                "next": "Ask the user for the missing information."}

    ticket = dict(TICKET_TEMPLATE)
    ticket.update({
        "ticket_id": f"DRAFT-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}",
        "user_id": setting(config, "user_id"),
        "thread_id": setting(config, "thread_id"),
        "category": topic,
        "summary": summary.strip(),
        "steps_tried": [s.strip() for s in steps_tried if s.strip()],
        "service_status": service_status,
        "status_profile": setting(config, "status_profile", "normal"),
        "reference_ids": [HELPDESK["instructions"][topic]["instruction_id"]],
    })
    folder = ticket_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
    shown_path = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {"ticket_file": shown_path, "ticket": ticket,
            "note": "Local draft only. Nothing was sent."}


TOOLS = [find_instructions, check_service_status, prepare_ticket]