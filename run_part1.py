# Runs the Part 1 scenarios (S01-S05) and saves one log per scenario in outputs/
# Usage: python run_part1.py

import json
from pathlib import Path

from part1_graph import build_graph, run_with_trace

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# Predictions written before running, copied into SUBMISSION.md
PREDICTIONS = {
    "S01": ["START", "read_request", "wifi_help", "END"],
    "S02": ["START", "read_request", "password_help", "END"],
    "S03": ["START", "read_request", "other_help", "END"],
    "S04": ["START", "read_request", "wifi_help", "END"],
    "S05": ["START", "read_request", "clarify_problem", "END"],
}

# Scenario id, message, graph variant
SCENARIOS = [
    ("S01", "I cannot connect to Wi-Fi.", "original"),
    ("S02", "I forgot my PASSWORD.", "original"),
    ("S03", "My laptop screen is broken.", "original"),
    ("S04", "My Wi-Fi password does not work.", "original"),
    ("S05", "My Wi-Fi password does not work.", "with_clarification"),
]

for scenario_id, message, variant in SCENARIOS:
    result = run_with_trace(message, variant)
    log = {
        "scenario": scenario_id,
        "command": "python run_part1.py",
        "status_profile": "normal (not used in Part 1)",
        "user_id": "demo-user-a",
        "thread_id": scenario_id.lower(),
        "predicted_route": PREDICTIONS[scenario_id],
        "prediction_correct": PREDICTIONS[scenario_id] == result["route"],
        "data_source": "data/helpdesk.json (reference file)",
        **result,
    }
    path = OUT / f"{scenario_id}_part1.json"
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = "OK " if log["prediction_correct"] else "DIFF"
    print(f"{ok} {scenario_id} [{variant}] {message!r}")
    print("     route:", " -> ".join(result["route"]))

# Save both graph drawings as Mermaid text
for variant in ["original", "with_clarification"]:
    mermaid = build_graph(variant).get_graph().draw_mermaid()
    (OUT / f"part1_graph_{variant}.mmd").write_text(mermaid, encoding="utf-8")
print("\nLogs and graph drawings saved in outputs/")