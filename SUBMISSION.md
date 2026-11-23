# HW5 — My LangGraph helpdesk

## Part 1 — How the graph works

Code: `part1_graph.py` (graph), `run_part1.py` (runs S01–S05 and saves logs).
Command: `python run_part1.py`

### My graph and state

```text
START → read_request → choose_next_node (conditional edge)
                         ├─ category == "wifi"     → wifi_help     → END
                         ├─ category == "password" → password_help → END
                         └─ category == "other"    → other_help    → END
```

State (`HelpdeskState`):

| Field | What it holds | Who sets it |
| --- | --- | --- |
| `message` | the user's original message | the caller, at START |
| `category` | `wifi`, `password`, `other` (later also `both`) | `read_request` |
| `reply` | the answer text | one help node |

| Node | Reads | Changes |
| --- | --- | --- |
| `read_request` | `message` | `category` |
| `choose_next_node` (edge, not a node) | `category` | nothing — only returns the next node's name |
| `wifi_help` / `password_help` | nothing from the state; reads `data/helpdesk.json` | `reply` |
| `other_help` | nothing | `reply` (asks: Wi-Fi or password?) |

**1. What does each node read and change?** See the table above. Every node returns only the fields it changes, e.g. `read_request` returns `{"category": "wifi"}`.

**2. Node update vs conditional edge.** A node *changes the state*: it returns a dictionary that LangGraph merges into the state. A conditional edge *does not change the state*: `choose_next_node` only reads `category` and returns the name of the next node. Deciding "what is true" (node) is separate from deciding "where to go" (edge).

**3. Does the next node get only the previous update or the whole state?** The whole current state. LangGraph merges each update into the shared state, so `wifi_help` receives `{message, category}`, not only `{category}`. The `state_after` field in every log shows this.

**4. START, END, compile.** `START` is the virtual entry point: the input `{"message": ...}` becomes the first state and goes to `read_request`. `END` is the virtual exit: when a node leads to `END`, the run stops and the final state is returned. `compile()` checks the builder (all edges point to existing nodes, there is an entry point) and turns it into a runnable object with `invoke()` / `stream()`.

### Predictions before running

| Message | Predicted route | Actual route | What I learned |
| --- | --- | --- | --- |
| I cannot connect to Wi-Fi. | read_request → wifi_help | read_request → wifi_help ([S01](outputs/S01_part1.json)) | Matching works with the hyphen `wi-fi` alias. |
| I forgot my PASSWORD. | read_request → password_help | read_request → password_help ([S02](outputs/S02_part1.json)) | Lowercasing the message makes the rule case-insensitive. |
| My laptop screen is broken. | read_request → other_help | read_request → other_help ([S03](outputs/S03_part1.json)) | Anything without a keyword falls to the fallback node, which asks a question. |
| My Wi-Fi password does not work. | read_request → wifi_help | read_request → wifi_help ([S04](outputs/S04_part1.json)) | Both keywords match, but the Wi-Fi rule is checked first, so the password part is silently ignored. |

**Message with both topics (S04).** In `read_request` the `if wifi:` check comes before `elif password:`. The first true condition wins, so the order of the rules decides the result: the user gets Wi-Fi instructions even though the problem may be the password.

### My change to the graph

```text
START → read_request → choose_next_node
                         ├─ "both"     → clarify_problem → END   (new)
                         ├─ "wifi"     → wifi_help       → END
                         ├─ "password" → password_help   → END
                         └─ "other"    → other_help      → END
```

Change: `read_request` first checks `wifi and password` → `category = "both"`, and the conditional edge maps `"both"` to the new node `clarify_problem`. Mermaid drawings: [original](outputs/part1_graph_original.mmd), [with clarification](outputs/part1_graph_with_clarification.mmd).

New trace for `My Wi-Fi password does not work.` ([S05](outputs/S05_part1.json)):

| Step | Node | Update | State after |
| --- | --- | --- | --- |
| 1 | read_request | `category = "both"` | message, category=both |
| 2 | clarify_problem | `reply = "…Is the problem connecting to Wi-Fi, or recovering your account password?"` | message, category, reply |

This is a different route (a different node runs), not just a different sentence from the same node.

**What LangGraph controls:** the order of execution (START → node → edge → node → END), passing and merging the state between nodes, calling my edge function and following the name it returns, and stopping at END.

**What my Python functions decide:** everything about meaning — which keywords count (`mentions`), the rule order (`read_request`), which category goes to which node (`choose_next_node`) and the reply text (help nodes). No model makes any decision in Part 1; the same input always gives the same route.

## Part 2 — Model and tools

TODO: draw the model/tool loop. Answer the four questions in the brief. Record the model and provider, direct calls to each tool, and one actual run that uses multiple tools. Explain the loop limit.

## Part 3 — Memory

TODO: explain conversation state, thread_id, user identifier, user preference storage, and retrieval from helpdesk.json.

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | | | |
| Close, restart and resume | | | |
| Same user, new conversation | | | |
| Different user | | | |

## Part 4 — Running and checking my work

TODO: point to the README setup steps, package versions, memory-reset instructions, tests and GitHub Actions workflow. Explain which mistake each test catches and which runs use a fake model.

## Choices and limitations

TODO: explain the storage and libraries I chose, why they fit this task, and what still does not work reliably.

## Supplied scenarios — my actual results

| Scenario | Command or manual steps | Log file | What happened / what I changed |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | | | |
| S02 — Case-insensitive password route | | | |
| S03 — Unsupported request | | | |
| S04 — Two topics before the graph change | | | |
| S05 — Two topics after adding clarification | | | |
| S06 — Instructions then local ticket | | | |
| S07 — The same request during a Wi-Fi outage | | | |
| S08 — Status source unavailable | | | |
| S09 — Missing ticket details | | | |
| S10 — Use the current conversation | | | |
| S11 — Persist and recall after restart | | | |
| S12 — Preference across threads without replay | | | |
| S13 — Do not share memory between users | | | |
| S14 — Password recovery outage | | | |