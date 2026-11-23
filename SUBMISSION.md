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

Code: `helpdesk_tools.py` (3 tools), `agent_graph.py` (graph + loop), `run_part2.py` (runner).
Provider: **OpenRouter** (OpenAI-compatible API through `langchain-openai`). Model: **`openai/gpt-4o-mini`**, temperature 0.

### The model/tool loop

```text
START → agent ──should_continue──┬─ model asked for tools  → tools ──→ agent (loop)
                                 ├─ plain text answer       → END
                                 └─ tool_rounds ≥ 5         → stop_at_limit → END
```

1. `agent`: the model gets the system rules + the whole conversation and returns either a text answer or a request for tools (name + arguments).
2. `tools`: my program finds each requested tool by name, runs it and adds the result as a `ToolMessage`.
3. Back to `agent`: the model reads the result and decides again.

Drawing generated by LangGraph: [outputs/part2_graph.mmd](outputs/part2_graph.mmd).

State (`AgentState`): `messages` (whole conversation; new messages are appended by `add_messages`), `tool_rounds` (trips to `tools` for the current user message, reset to 0 for each new message), `limit_reached`.

**Loop limit.** `MAX_TOOL_ROUNDS = 5`, enforced in the conditional edge `should_continue`: if the model asks for a tool again after 5 tool rounds, the graph goes to `stop_at_limit` instead of `tools`. That node answers "I stopped after 5 tool rounds" and sets `limit_reached = True`. LangGraph's `recursion_limit = 25` is a second safety net.

### The four questions

**1. How does the graph decide between running a tool and ending with an answer?**
After the `agent` node, the conditional edge `should_continue` looks at the model's last message. If the message contains tool calls (and the limit is not reached), the graph goes to `tools`; if it is a plain text answer, the graph goes to `END`. The model decides by what it outputs; my edge function reads that output and picks the next node.

**2. Which component requests a tool, and which component actually executes it?**
The model only *requests* a tool: it returns a message like "call `check_service_status` with `{"service": "wifi"}`". It cannot run code. My `tools` node *executes* the Python function and returns the result. The status profile, user_id and thread_id are passed by my program through `config`; the model cannot see or choose them (its tool schemas only contain `topic`, `service`, and the ticket fields).

**3. What changes in the state after a tool result returns?**
A `ToolMessage` with the result (JSON) is appended to `messages`, linked to the request by `tool_call_id`, and `tool_rounds` increases by 1. On the next `agent` call the model sees the result and decides what to do next. `prepare_ticket` also writes a file in `outputs/tickets/` (a side effect outside the state).

**4. What makes this part different from the fixed routing rules in Part 1?**
In Part 1 my `if` rules fixed the route in advance: keywords only, same input → same route. Here the graph shape is fixed (agent ↔ tools), but the path through it is chosen at run time by the model, from the meaning of the message and from tool results. The same request leads to a ticket when Wi-Fi is operational and to no ticket during an outage. My code still controls which tools exist, how they run, what data they read, and the loop limit.

### Direct tool calls (no agent)

Command: `python run_part2.py direct` → [outputs/part2_direct_tool_calls.json](outputs/part2_direct_tool_calls.json)

| Call | Result |
| --- | --- |
| `find_instructions(topic="wifi")` | GUIDE-WIFI with 4 steps |
| `find_instructions(topic="printer")` | error + available topics |
| `check_service_status(service="wifi")` with normal / wifi_outage / status_unavailable | operational / outage / unknown |
| `check_service_status(service="account_recovery")` with recovery_outage | outage |
| `prepare_ticket(...)` with problem + steps | draft saved in `outputs/tickets/` |
| `prepare_ticket(...)` with empty problem and steps | refused: "missing information, ask the user" |

### Runs with the real model

Command: `python run_part2.py all` (or one: `python run_part2.py S06`).

| Scenario | Status profile | Tool requests by the model (in order) | Outcome | Log |
| --- | --- | --- | --- | --- |
| S06, turn 1 | normal | `find_instructions(topic="wifi")` | GUIDE-WIFI steps returned verbatim from the reference file | [S06](outputs/S06_part2.json) |
| S06, turn 2 | normal | `check_service_status(service="wifi")` → `operational`; `prepare_ticket(category="wifi", summary="Cannot connect to Campus-Demo network.", steps_tried=["Enabled Wi-Fi", "Forgot the saved network"], service_status="operational")` | Draft `outputs/tickets/DRAFT-20261123-235823-9d12.json` created; the reply states that nothing was sent | [S06](outputs/S06_part2.json) |
| S07 | wifi_outage | `check_service_status(service="wifi")` → `outage` | No ticket requested; the condition "if it is operating" was not met; no repair time was stated | [S07](outputs/S07_part2.json) |
| S08 | status_unavailable | `check_service_status(service="wifi")` → `unknown` | No ticket requested; the model reported that the status could not be determined and asked whether a draft is still wanted | [S08](outputs/S08_part2.json) |
| S09, turn 1 | normal | none | The model asked for the problem description and the steps already tried instead of inventing them | [S09](outputs/S09_part2.json) |
| S09, turn 2 | normal | `check_service_status(service="wifi")` → `operational`; `prepare_ticket(category="wifi", steps_tried=["Enabled Wi-Fi", "Tried reconnecting"], service_status="operational")` | Draft `outputs/tickets/DRAFT-20261123-235835-2c4c.json` created from the user's own details | [S09](outputs/S09_part2.json) |
| S14 | recovery_outage | `check_service_status(service="account_recovery")` → `outage` | See the note below | [S14](outputs/S14_part2.json) |

**Multi-tool request (S06, turn 2).** A single user message produced two tool rounds: the model first requested `check_service_status`, read the result `operational`, and only then requested `prepare_ticket`. The second request depended on the first result: in S07 and S08 the identical condition stopped after the status check.

**S14 correction.** In the first run ([outputs/S14_part2_before_fix.json](outputs/S14_part2_before_fix.json)) the model correctly selected `account_recovery` and reported the outage without a repair time, but it also stated "I cannot prepare a ticket for this issue". This is incorrect: the user set no condition, and GUIDE-PASSWORD says that a ticket should be prepared when recovery is unavailable. The model over-generalised the outage rule from the system prompt. I changed `SYSTEM_PROMPT` in two steps and re-ran S14 after each:

1. Added "An outage only blocks a ticket when the user's own condition says so; otherwise you may offer a ticket draft". The re-run produced the same refusal, so the permissive wording ("may offer") was not strong enough to override the earlier outage rule.
2. Replaced it with "report the outage and OFFER to prepare a local ticket draft describing the error. Never say that you cannot prepare a ticket", plus a rule to consult `find_instructions` during an outage. After this change the model requested `check_service_status(service="account_recovery")` → `outage` and replied: *"The password recovery service is currently experiencing an outage and is unavailable. Unfortunately, there is no information on a repair time. Would you like me to prepare a ticket draft describing this issue?"* ([outputs/S14_part2.json](outputs/S14_part2.json)).

The final behaviour is correct: the outage is reported from the snapshot, no repair time is invented, and a draft is offered rather than refused. The model did not follow the additional rule to call `find_instructions`, which shows that prompt rules influence but do not guarantee the model's tool choices. A guarantee would require enforcing the step in code (for example, a graph node that always attaches the guide when a status is `outage`).

**Observed limitation.** In S09 the final reply did not repeat that the ticket is a local draft; this information is present only in the tool result (`"note": "Local draft only. Nothing was sent."`).

The same conditional request ("check whether Wi-Fi is down; if it is operating, prepare a ticket") gives three different results in S06/S07/S08 because the model's next step depends on the result of `check_service_status`.

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
| S06 — Instructions then local ticket | `python run_part2.py S06` | [S06_part2.json](outputs/S06_part2.json) | Instructions retrieved, then status `operational` led to a local ticket draft |
| S07 — The same request during a Wi-Fi outage | `python run_part2.py S07` | [S07_part2.json](outputs/S07_part2.json) | Status `outage`; no ticket prepared, as required by the user's condition |
| S08 — Status source unavailable | `python run_part2.py S08` | [S08_part2.json](outputs/S08_part2.json) | Status `unknown`; no ticket prepared; the model asked whether a draft is still wanted |
| S09 — Missing ticket details | `python run_part2.py S09` | [S09_part2.json](outputs/S09_part2.json) | Turn 1: the model asked for details; turn 2: draft created from the user's details |
| S10 — Use the current conversation | | | |
| S11 — Persist and recall after restart | | | |
| S12 — Preference across threads without replay | | | |
| S13 — Do not share memory between users | | | |
| S14 — Password recovery outage | `python run_part2.py S14` | [S14_part2.json](outputs/S14_part2.json) | Outage reported without a repair time and a ticket draft offered; the first two runs wrongly refused a ticket, so the system prompt was corrected twice (see Part 2) |