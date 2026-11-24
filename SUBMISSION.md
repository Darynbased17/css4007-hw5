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

Code: `memory.py` (storage and the `save_language_preference` tool), `agent_graph.py` (`use_user_memory=True`), `run_part3.py` (scenario runner with real process restarts), `chat.py` (interactive terminal application).

### Graph with memory

```text
START → load_user_memory → agent ──should_continue──┬─ tools → agent (loop)
                                                    ├─ END
                                                    └─ stop_at_limit → END
        (compiled with a SqliteSaver checkpointer)
```

Drawing generated by LangGraph: [outputs/part3_graph.mmd](outputs/part3_graph.mmd).

### Two kinds of memory

| | Conversation memory | User memory |
| --- | --- | --- |
| What it holds | All messages of one conversation: the current problem, the steps the user said they tried, tool requests and tool results | The user's saved preference (reply language) |
| Key | `thread_id` | `user_id` |
| Storage | `local-storage/conversations.sqlite`, written by LangGraph's `SqliteSaver` checkpointer after every node | `local-storage/user_memory.json`, `{user_id: {"language": ...}}` |
| Written by | LangGraph automatically, because the graph is compiled with the checkpointer | The tool `save_language_preference`, requested by the model when the user asks to remember a language |
| Read by | LangGraph automatically: on `invoke`/`stream` with a `thread_id` it loads the saved state of that thread before the new message is appended | The node `load_user_memory` at the start of every turn; the result is inserted into the system prompt |
| Scope | One conversation | All conversations of one user |

**`thread_id` versus user identifier.** A `thread_id` identifies one conversation: the checkpointer stores a separate state history for each `thread_id`, so a new `thread_id` starts with an empty message list. A `user_id` identifies a person, who may have many threads. User memory is keyed by `user_id`, so a preference saved in thread `preference-a1` is available in thread `preference-a2` of the same user, while the messages of `preference-a1` are not. Both identifiers are supplied by the program through `config["configurable"]`; the model cannot choose them, so it cannot load another user's data. `save_language_preference` reads `user_id` from `config`, not from a model argument.

**Retrieval versus memory.** Looking up Wi-Fi instructions is *retrieval*: `find_instructions` reads fixed reference data from `data/helpdesk.json`, which is the same for every user and is not changed by conversations. Recalling the user's language is *memory*: the information did not exist before the user stated it, it was learned from a message, saved, and is read back later for that user only.

**Restart.** `run_part3.py` executes every `restart_process` event as a real process restart: each part of a scenario between restarts runs as a separate child process (`python run_part3.py S11 --segment 1`). The new process opens the same SQLite file and JSON file. Each log records the `process_id` of every part to show that the processes were different.

**Trace of memory reads.** Each turn in the logs contains `memory_read` with three entries: `conversation_memory` (how many messages of this thread were loaded from the checkpointer and the earlier user messages), `user_memory` (what `load_user_memory` loaded for this `user_id`), and `reference_file` (whether `data/helpdesk.json` was read through `find_instructions`).

### Demonstrations

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | `python run_part3.py D01` (also S10) | [D01_part3.json](outputs/D01_part3.json), [S10_part3.json](outputs/S10_part3.json) | D01: turn 1 retrieved GUIDE-WIFI (reference file) and checked the status. In turn 2, "I already tried that" was resolved through conversation memory (6 earlier messages in thread `that-demo`): the model listed the three GUIDE-WIFI steps as `steps_tried` and prepared a local draft. S10: with 6 earlier messages loaded, the model answered that the user had enabled Wi-Fi and forgotten the saved network, and that reconnecting failed. |
| Close, restart and resume | `python run_part3.py S11` | [S11_part3.json](outputs/S11_part3.json) | Process 1 saved `language = Kazakh` (user memory) and answered in Kazakh. After the process exited, process 2 loaded 7 earlier messages of `restart-a` from SQLite and `{'language': 'Kazakh'}` from user memory, and answered in Kazakh that the problem was Wi-Fi and that the user had enabled it and tried reconnecting. The old transcript was not passed as input. |
| Same user, new conversation | `python run_part3.py S12` | [S12_part3.json](outputs/S12_part3.json) | After a restart, the new thread `preference-a2` started with 0 earlier messages, while user memory supplied `{'language': 'Kazakh'}`. The model answered in Kazakh that Kazakh had been requested and listed what it can help with; it did not mention the Wi-Fi problem from `preference-a1`. |
| Different user | `python run_part3.py S13` | [S13_part3.json](outputs/S13_part3.json) | demo-user-a saved Kazakh. For demo-user-b the trace shows 0 earlier messages and empty user memory `{}`; the reply was "No preference or previous problem is recorded for you." and it was in English. |

**Observations.** In D01 the model prepared a ticket although the user only asked "What should I do next?"; this is consistent with GUIDE-WIFI step 4 (the status had already been checked as operational), but a stricter design would ask before creating a draft. It also counted all three guide steps as tried, which is the most natural reading of "that" but is still an inference. In S11 the recall answer repeated one sentence. These are wording issues of the model; the memory mechanisms (loaded message counts, user memory contents and process ids) behaved as designed in every run.

Manual equivalent of S11 with the interactive program (two separate launches):

```text
python chat.py --user demo-user-a --thread restart-a --message "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."
python chat.py --user demo-user-a --thread restart-a --message "Which problem and steps did we discuss?"
```

## Part 4 — Running and checking my work

- **Setup and run commands:** [README.md](README.md) gives the exact installation steps for Windows and macOS/Linux, the `.env` configuration for OpenRouter, every run command and the interactive `chat.py` options. The original brief was moved to [ASSIGNMENT.md](ASSIGNMENT.md).
- **Python and package versions:** Python 3.13 (tested with 3.13.18). [requirements.txt](requirements.txt) pins `langgraph==1.2.24`, `langgraph-checkpoint==4.3.2`, `langgraph-checkpoint-sqlite==3.1.3`, `langchain-core==1.8.4`, `langchain-openai==1.8.5`, `python-dotenv==1.2.3`, `pytest==9.1.1`.
- **Memory location and reset:** conversations in `local-storage/conversations.sqlite`, preferences in `local-storage/user_memory.json` (folder set by `HELPDESK_STORAGE_DIR`, ignored by git). `python chat.py --reset` deletes both.
- **Continuous integration:** [.github/workflows/tests.yml](.github/workflows/tests.yml) installs `requirements.txt` on Python 3.13 and runs `python -m pytest -v` on every push and pull request. No secret is used.

### Tests and what each one would catch

Command: `python -m pytest -v` (24 tests). `tests/conftest.py` sets `MODEL_NAME=fake` and gives every test a separate temporary storage and ticket folder, so tests never touch real memory and never call a paid model.

| Test | What would break if it failed |
| --- | --- |
| `test_original_graph_routes` (4 cases) | A keyword rule, the case-insensitive comparison, or the conditional edge mapping |
| `test_both_topics_original_graph_prefers_wifi` | The Part 1 rule order (S04 behaviour) changed by accident |
| `test_both_topics_changed_graph_clarifies` | `clarify_problem` is missing or not connected (S05) |
| `test_reply_comes_from_reference_file_and_message_is_kept` | Help nodes stop using `helpdesk.json`, or a node overwrites `message` |
| `test_find_instructions_known_and_unknown_topic` | Topic aliases or the data format of `helpdesk.json` |
| `test_check_service_status_uses_selected_profile` (5 cases) | The tool ignores the selected snapshot or the password → `account_recovery` mapping |
| `test_prepare_ticket_saves_local_draft` | The draft is not saved, has wrong fields, loses user/thread ids or changes the user's steps |
| `test_prepare_ticket_refuses_missing_details` | A ticket is created from missing information (S09 rule) |
| `test_model_cannot_see_program_settings` | The profile, user or thread becomes a model-visible argument |
| `test_tool_request_is_executed_and_result_returned` | Tools are not executed, results do not return to the model, or chaining stops |
| `test_outage_profile_reaches_tool` | The snapshot is not passed from the program to the tool through `config` |
| `test_loop_limit_stops_endless_tool_requests` | The loop limit is missing, so a model that keeps requesting tools never stops |
| `test_plain_answer_ends_without_tools` | A text answer is routed to `tools` instead of `END` |
| `test_conversation_survives_new_graph_instance` | The checkpointer does not save or reload a thread |
| `test_memory_survives_process_restart` | Memory exists only in RAM; this test runs `chat.py` twice as separate processes |
| `test_new_thread_has_preference_but_not_old_messages` | A new thread reuses old messages, or the preference is not shared across threads |
| `test_users_are_separated` | One user's preference or conversation leaks to another user |

**Fake models.** Two clearly labelled fakes are used, only in tests: `ScriptedModel` in `test_agent_loop.py` returns a prepared list of tool requests to test the loop mechanics exactly, and `FakeHelpdeskModel` in `fake_model.py` follows keyword rules and prefixes every reply with `[FAKE MODEL]`, so the memory tests can run end to end. Every run reported in Parts 2 and 3 used the real model `openai/gpt-4o-mini`.

## Choices and limitations

**Libraries.** LangGraph builds the graphs and provides checkpointing. `langchain-openai` is used as the model adapter because OpenRouter exposes an OpenAI-compatible API, so only `OPENAI_BASE_URL` had to change. `python-dotenv` keeps the key in an ignored `.env` file. `pytest` and GitHub Actions run the tests.

**Model.** `openai/gpt-4o-mini` through OpenRouter, temperature 0: it supports tool calling, is inexpensive for repeated scenario runs, and writes Kazakh adequately for the preference demonstrations.

**Conversation storage.** `SqliteSaver` from `langgraph-checkpoint-sqlite` is LangGraph's own checkpointer, so saving and reloading a thread requires no custom code. SQLite is a single local file with no server, which survives restarts and is easy to delete. `InMemorySaver` is used only in `run_part2.py`, where persistence is not part of the task.

**User memory storage.** A JSON file keyed by `user_id`. The only preference in this task is the reply language, so a readable file is easier to inspect and to reset than a second database. It is written only by `save_language_preference` and read only by `load_user_memory`, both of which take `user_id` from the program's config.

**Separate runners.** Each part has its own runner so that the Part 1 graph stays model-free and the Part 2 logs are not affected by memory. `run_part3.py` starts a new Python process for every part of a scenario after `restart_process`, which makes the restart real rather than a second call in the same process.

**What does not work reliably.**

- Model behaviour depends on the prompt and is not guaranteed by code. S14 needed two prompt corrections, and the model ignored the rule to call `find_instructions` during an outage. In D01 it prepared a ticket without being asked.
- Wording varies between runs even at temperature 0, so the logs show one observed run, not a fixed result. The tests check mechanisms with fake models instead of real model text.
- The whole conversation is sent to the model on every turn. Long threads would increase cost and could exceed the context window; a production version would need trimming or summarisation.
- The user memory JSON file is rewritten on every save and has no locking, so two processes saving at the same moment could overwrite each other. This is acceptable for a single-user demo but not for a shared service.
- Language preference detection relies on the model calling `save_language_preference`; the preference is not extracted by code.

## AI-use declaration

I used Claude (Anthropic) as an assistant to explain LangGraph concepts, to draft the code structure, tests and documentation, and to review my scenario outputs. I ran every scenario myself with my own OpenRouter key, checked the logs, and identified and corrected the S14 behaviour from my own runs.

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
| S10 — Use the current conversation | `python run_part3.py S10` | [S10_part3.json](outputs/S10_part3.json) | The second message was answered from conversation memory: the model listed enabling Wi-Fi and forgetting the saved network |
| S11 — Persist and recall after restart | `python run_part3.py S11` | [S11_part3.json](outputs/S11_part3.json) | Real process restart (two different process ids); problem, steps and Kazakh preference recalled from SQLite and user memory |
| S12 — Preference across threads without replay | `python run_part3.py S12` | [S12_part3.json](outputs/S12_part3.json) | New thread had no earlier messages; the Kazakh preference was loaded from user memory |
| S13 — Do not share memory between users | `python run_part3.py S13` | [S13_part3.json](outputs/S13_part3.json) | demo-user-b received no preference and no previous problem from demo-user-a |
| S14 — Password recovery outage | `python run_part2.py S14` | [S14_part2.json](outputs/S14_part2.json) | Outage reported without a repair time and a ticket draft offered; the first two runs wrongly refused a ticket, so the system prompt was corrected twice (see Part 2) |