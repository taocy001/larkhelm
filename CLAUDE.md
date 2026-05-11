# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working inside this repository.

> User-facing docs (install / config fields / command table / voice setup) live in `README.md`. **This file is not a duplicate** — it captures internal architecture, conventions, and gotchas that an AI dev agent or new maintainer needs to ship a correct change.

---

## Module layout

```
larkhelm/
├── bridge.py            WebSocket event registration + main entry
├── config.py            runtime config + memory limits + dataclass mirror
├── memory_watchdog.py   per-process RSS watchdog (gc soft / SIGTERM hard, 60s debounce)
├── handlers/            Feishu event dispatch
│   ├── _message.py      message routing, /command dispatch, audio branch
│   ├── _query.py        AI query lifecycle: streaming cards, soft/hard timeout, cancel
│   └── _card_action.py  button-callback dispatch
├── commands.py          /run, /cd, /status, /model, /voice etc
├── cmd_doc.py           /doc + /doc wiki families
├── cmd_plan.py          /plan multi-step planner
├── chat_state.py        per-chat persistent state (cwd / model / voice_lang / crons)
├── concurrency.py       per-chat locks + cancel events (LRU 500 / 1000)
├── dedup.py             event/message dedup (LRU 500)
├── log.py               .md + all.jsonl conversation logs + DEBUG_LOG rotation
├── token_stats.py       token usage accounting (LRU 5000)
├── lark_client.py       Feishu API wrappers; native-table descendant API
├── card_builder.py      JSON 2.0 card construction + markdown split
├── ai_runner.py         thin shim re-exporting runners
├── runner_base.py       BaseProcessRunner: semaphore / watch / retry template
├── runner_claude.py     Claude CLI subprocess (--print stream-json)
├── runner_gemini.py     Gemini CLI subprocess
├── runner_kimi.py       Kimi CLI subprocess
├── runner_deepseek.py   DeepSeek HTTP (requests, openai-compat shape)
├── backend_registry.py  BackendSpec + capability scoring + health tracking
├── backend_api.py       anthropic / google_genai / openai_compat wrappers
├── backend_cli.py       dispatch wrapper around runner_*
├── health_signals.py    AUTH / QUOTA / TRANSIENT classifier
├── model_probe.py       startup health probes (CLI + API)
├── voice/               M3.2 STT subpackage
│   ├── transcribe.py    engine dispatcher; faster-whisper singleton
│   ├── _engine_dashscope.py  DashScope Paraformer opt-in adapter
│   ├── merge.py         30s window merge buffer (Timer + OrderedDict)
│   └── system_probe.py  `larkhelm voice probe` CLI implementation
├── crew/                multi-agent orchestration
│   ├── _commands.py     /crew planner + manager prompt (dynamic backend menu)
│   ├── _runner.py       agent execution + DAG scheduling
│   ├── _pipeline.py     /dev fixed pipeline (pm → architect → impl → qa → reviewer)
│   ├── _checkpoint.py   checkpoint persistence + resume
│   ├── _scheduler.py    crew cron scheduler
│   └── _state.py        global crew state
├── crew_types.py        AgentSpec / AgentState / CrewState / CrewPhase
├── crew_card.py         crew card rendering + heartbeat
├── agent_hub/           Phase 5 intent router + agent dispatch (gated off)
│   ├── intent_router.py     resolve_intent: explicit cmd → L1 keyword → L2 LLM JSON
│   ├── agent_dispatcher.py  ACL + audit + transparent card
│   ├── agent_base.py        AgentExecutor (ABC) + AGENT_REGISTRY singleton
│   ├── model_selector.py    BackendRegistry.rank_for_task() integration
│   ├── plugin_loader.py     entry-points('larkhelm.agents') + config['agent_plugins']
│   └── builtin/             ChatAgent / DevAgent / CrewAgent / PlanAgent / DocAgent
├── perm.py              permission approval socket + YOLO grants
├── perm_hook.py         permission hook for Claude subprocess
├── mcp_server.py        MCP stdio server (`larkhelm mcp-server`)
└── __main__.py          CLI entry: start / doc / mcp-server / voice
```

---

## Architecture invariants

### Event flow

```
WebSocket → handle_message → dedup → /command branch  OR  threading.Thread(_do_query)
                                                              ↓
                                                       orchestration.py:
                                                         DELEGATE <backend> | AGENT <type>
                                                              ↓
                                                       backend_cli.run_*  /  backend_api.run_*
                                                              ↓
                                                       stream events → on_text / on_tool callbacks
                                                              ↓
                                                       lark_client streams card updates
```

* **Per-chat lock** serializes queries; **cancel event** is per-chat too — `/cancel` triggers it
* **Card schema** is **JSON 2.0 unconditional** (post-commit `4b7c68e`). Buttons go directly into `body.elements[]` (no `action` container); multi-button rows wrap in `column_set width:"auto"` columns. `_card_action.py:27` parses `CallBackAction.value` schema-agnostically.
* **Streaming**: cards update in-place; split via `_split_md()` when content exceeds `max_card_len`.

### AI runner contract

Every backend (CLI or HTTP) goes through `backend_cli.run_*` or `backend_api.run_*` which:
1. Acquires `MAX_AI_PROCS=3` semaphore (`runner_base.py`)
2. Spawns subprocess / makes HTTP call
3. Streams `tool_use` / `text` / `result` events to callbacks
4. Calls `BACKEND_REGISTRY.record_call_success/failure(spec_id, ...)` on completion (real-traffic health feedback)
5. Releases semaphore in `finally`

`record_call_failure` classifies the error via `health_signals.classify_error()` and either flips `healthy=False` instantly (AUTH/QUOTA/MODEL_NOT_FOUND) or appends to a sliding window (TRANSIENT, threshold 3 within 600s).

### State persistence (under `DATA_DIR`)

```
.feishu_sessions/{chat_id}.sid         Claude session IDs
.feishu_sessions/gemini_{chat_id}.sid  Gemini session IDs
.feishu_state.json                     per-chat {cwd, model, voice_lang, crons, …}
.feishu_logs/{chat_id}/{YYYY-MM-DD}.md markdown conversation log
.feishu_logs/all.jsonl                 global event log (auto-rotated at _MAX_JSONL_BYTES)
```

`chat_state.py` is the canonical writer; `state.py` is now a thin back-compat re-export shim.

### Phase 5 agent_hub (gated off by default)

`intent_router_enabled=false` keeps `_message.py` from even importing `agent_hub`. When opened, dispatch becomes Intent → Agent → Backend (three layers):

* `intent_router.resolve_intent()` — explicit `/cmd` → L1 keyword → L2 cheap-LLM JSON
* `AgentDispatcher.dispatch()` — ACL check (`agent_acl` config) + audit (JSONL 0600) + transparent decision card
* `model_selector.resolve_backend_for_task()` — calls `BackendRegistry.rank_for_task(TaskProfile)` for capability scoring

Plugin agents register via entry-points `larkhelm.agents` or config `agent_plugins`. Detailed design lives in `.crew_workspace/design.md` (R1+R2+R3 APPROVED).

> **`intent_feedback_path` / `intent_audit_path`**: keep under `DATA_DIR` (e.g. `DATA_DIR/audit/intent.jsonl`). Files are 0600 but landing them outside `DATA_DIR` breaks backup hygiene and leaks raw user queries via misconfigured share scripts. Module falls back to `tempfile.gettempdir()/intent_*.jsonl` when `DATA_DIR` is unset (early bootstrap).

---

## Conventions enforced across new code

### Import discipline (which module owns what)

| Need | Import from |
|---|---|
| Per-chat state | `larkhelm.chat_state` (`_get_chat_state` / `_set_chat_field`) |
| Per-chat lock / cancel event | `larkhelm.concurrency` (`_get_chat_lock` / `_get_cancel_event`) |
| Conversation logging | `larkhelm.log` (`log_entry`) |
| Debug diagnostics | `larkhelm.log` (`_debug_log` / `safe_log` / `lazy_debug_log`) |
| Token accounting | `larkhelm.token_stats` |
| Message dedup | `larkhelm.dedup` |
| Crew dataclasses | `larkhelm.crew_types` |

Don't reach into module privates across this boundary; if a helper is missing, add it to the owning module.

### Exception handling — three buckets

| Severity | Pattern | Example |
|---|---|---|
| **High** (silent business failure) | `_debug_log` + user-visible ⚠️ card | `/reset` API history clear failed |
| **Mid** (auxiliary op) | `_debug_log`, do not interrupt main flow | token stat update, callback errors, memory load |
| **Low** (acceptable silent) | `except Exception: pass` allowed | `proc.kill()`, stderr drain, scratch I/O |

Don't introduce a new `except Exception: pass` without classifying it. Format: `[HH:MM:SS] [{Module}] {operation} failed: {exception}` written to `_cfg.DEBUG_LOG`.

### Log prefix — PascalCase `[Module]`

New code: first arg to `_debug_log` / `safe_log` / `lazy_debug_log` / `info` / `warn` / `error` starts with `[Module]`. Module name PascalCase (matches Python class style), no spaces, no underscores. Sub-components space-separated (e.g. `[Crew] Manager: …`).

Exceptions kept lower-case (third-party process names): `[claude]` / `[gemini]` / `[kimi]` / `[upgrade]`.

Some historical lower-case prefixes (`[memory]`, `[router]`, `[token_stats]`, `[lark_client]`, `[agent_audit]`) remain — rename on touch; don't batch-rewrite.

### Helper choice

| Helper | Level | When |
|---|---|---|
| `_debug_log(msg)` | DEBUG | Main-path diagnostics; caller has `larkhelm.log` loaded |
| `safe_log(msg)` | DEBUG | Exception cleanup / never-throw path (`_debug_log` in try) |
| `lazy_debug_log(msg)` | DEBUG | Bootstrap / circular-import edge; module may not be fully imported |
| `info(msg)` | INFO | Phase progression, state change ("backend X registered") |
| `warn(msg)` | WARN | Degraded behavior, credential fetch failure, operator-visible |
| `error(msg)` | ERROR | User-visible task interrupted (helps ticket correlation) |

Don't keep local copies of these — `safe_log` superseded 4 local `_safe_log` copies in `agent_hub/`; `lazy_debug_log` replaced the double-try-import pattern in `config.py` / `agent_base.abort()` / `intent_router`.

### Log gate

`LARKHELM_LOG_LEVEL=DEBUG|INFO|WARN|ERROR` env (default DEBUG) is parsed **once at import time** — runtime changes don't take effect (deliberate, prevents mid-flight visibility flip). Output:
* DEBUG keeps `[HH:MM:SS] [Module] msg` format (grep-compatible)
* INFO/WARN/ERROR add explicit level: `[HH:MM:SS] <LEVEL> [Module] msg`
* Unknown values fall back to DEBUG with a stderr warning (never silent-mute)

### Documented "acceptable silent" exceptions

These existing `except: pass` sites are *intentional* and have been audited:

| Site | Reason |
|---|---|
| `runner_base.py` `proc.kill()` (×3) | OS-level, process already exited is expected |
| `runner_base.py` `_drain_stderr` thread | Display-only, no side effects on failure |
| `runner_base.py` `_cleanup_tmp` unlink | Temp file already gone is expected |
| `log.py` `_debug_log` inner two `pass` | Logging infra — recursive failure is meaningless |
| `log.py:105` JSONL row parse skip | Design intent: tolerate corrupted lines |
| `memory.py` `_global_memory_file` chat_state access | Explicit fallback returns None |
| `crew/_runner.py` git diff | Non-git repo is expected |
| `mcp_server.py` config inner parse | Line-level fault tolerance |

---

## Adding a feature (recipes)

### New `/command`

1. **Implement** `_cmd_x(chat_id, args="", msg_id=None)` in `commands.py` — call `send_card_reply(...)` for output, never raise.
2. **Import** in `handlers/_message.py` (around the existing imports near line 38).
3. **Route** in the sync command block in `_message.py` (before the `else: return`) — `if tl.startswith("/x"): _cmd_x(...); return`.
4. **State R/W**: use `from larkhelm.chat_state import _get_chat_state, _set_chat_field`. Lock with `_get_chat_lock(chat_id)` if mutating shared state.
5. **Help text**: update `_cmd_help()` in `commands.py`.
6. (Optional) **Button callback**: add a case in `_card_action.py` or the `_dispatch_button_cmd` chain in `commands.py`.

### New extension point

| Extension | File | Notes |
|---|---|---|
| Register a Feishu event | `bridge.py:86-91` | `.register_p2_xxx_yyy(handler)` |
| New Feishu API call | `lark_client.py` | Follow existing builder pattern; use the descendant API for nested blocks (tables) |
| Card layout tweak | `card_builder.py` | Modify `_make_card_dict` — always JSON 2.0 |
| New per-chat state field | `chat_state.py` | Add to `_chat_state_store` schema; bump field default in `_init_runtime` |
| New permission rule | `perm.py` | Add check before `grant_yolo` / regular approval flow |
| New backend (CLI or API) | `backend_registry.py` + `runner_*.py` | Register a `BackendSpec`; implement `run_*` that respects the `MAX_AI_PROCS` semaphore + records call outcome |

### New background task that should refresh memory

`memory.maybe_auto_update(chat_id)` has **two trigger paths**:

| Trigger | When | Frequency |
|---|---|---|
| Normal cadence | After `_do_query` completes | every `AUTO_UPDATE_EVERY=10` turns |
| Milestone cadence | `/dev` / `/crew` / `/plan` finish | each completion + 60s debounce |

If you add a long-running background task that resembles `/dev` / `/crew`, **call `record_milestone(chat_id, kind, summary)` in its `finally`** — without it, the task's outcome never reaches session memory.

`record_milestone` writes a `role="milestone"` log entry, then calls `maybe_auto_update(force=True)` guarded by `_get_update_lock(chat_id)`. The auto-update filter accepts `role in {user, assistant, milestone}` and skips `model in {crew, shell}`, so the milestone surfaces but the sub-task chatter stays muted.

---

## Feishu document writes (Claude Code workflow)

When asked to write content to a Feishu doc **inside this repo**, use the `larkhelm doc` CLI — don't write a standalone Python script.

```bash
cat report.md | larkhelm doc create "Title"          # → prints new doc URL
cat more.md   | larkhelm doc append "<doc-url>"
cat fresh.md  | larkhelm doc write  "<doc-url>"       # overwrites
```

Owner defaults to `~/.config/larkhelm/config.json:default_owner_open_id`. **Don't** create `upload_to_feishu.py` or similar — the CLI replaces all such helpers and supports native Feishu tables via `lark_client._md_to_descendants()`.

---

## lark-oapi SDK cheat-sheet

Path: `~/.local/lib/python3.13/site-packages/lark_oapi/`.

| Namespace | Version | Resources |
|---|---|---|
| `client.docx` | v1 | `document`, `document_block`, `document_block_children`, `document_block_descendant` |
| `client.drive` | v1, v2 | `file`, `file_comment`, `media`, `meta`, `permission_member`, `permission_public`, `export_task`, `import_task` |
| `client.sheets` | v3 | `spreadsheet`, `spreadsheet_sheet`, `spreadsheet_sheet_filter`, `spreadsheet_sheet_filter_view` |
| `client.wiki` | v1, v2 | v1: `node`. v2: `space`, `space_node`, `space_member`, `task` |
| `client.docs` | v1 | `content` (legacy doc text) |
| `client.document_ai` | v1 | OCR / doc intelligence |
| `client.im` | v1 | `message`, `message_resource`, `chat`, `chat_members`, `file` |

Builder pattern, e.g.:

```python
resp = client.docx.v1.document.get(
    GetDocumentRequest.builder().document_id("doc_token_xxx").build()
)
```

For nested block writes (native tables, etc.) use `POST /docx/v1/documents/:doc_id/blocks/:block_id/descendant` (singular). See `lark_client._md_to_descendants()` + `_append_descendants_http()`.
