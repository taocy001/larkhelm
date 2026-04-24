# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LarkHelm** is a Python integration layer that bridges Feishu (Lark) messenger with Claude and Gemini AI CLIs. It maintains a WebSocket connection to Feishu and dispatches user messages to AI processes, streaming responses back as interactive Feishu cards.

## Running the Bridge

```bash
# 推荐通过 install.sh 安装（依赖通过 pyproject.toml 统一管理）
# 参见 README.md 快速开始

# 开发模式手动安装
pipx install -e .
# 或不使用 pipx：
pip install -e .

# Run
python3 -m larkhelm start
python3 -m larkhelm start --config /path/to/config.json --data-dir /path/to/data
```

## Testing

No formal test framework. Manual testing via:

```bash
# 检查包能否正常导入
python3 -c "import larkhelm.bridge; print('OK')"

# 直接以模块方式启动（--help 查看参数）
python3 -m larkhelm --help
```

## Configuration

Config file is auto-detected in this priority order:

```
CLI --config > LARKHELM_CONFIG env > /etc/larkhelm/config.json > ~/.config/larkhelm/config.json
```

Data directory priority:

```
CLI --data-dir > LARKHELM_DATA_DIR env > /var/lib/larkhelm > ~/.local/share/larkhelm
```

### Config fields

| Field | Purpose |
|---|---|
| `APP_ID`, `APP_SECRET` | Feishu app credentials (required) |
| `claude_command`, `gemini_command` | CLI binary paths (default: `"claude"`, `"gemini"`) |
| `default_model` | `"claude"` or `"gemini"` |
| `default_cwd` | Initial working directory for AI subprocess |
| `skip_permissions` | Auto-confirm Claude permission prompts |
| `response_timeout` | Soft timeout seconds per query (default: 300) |
| `hard_timeout` | Hard timeout seconds per query (default: 21600，即 6 小时) |
| `max_card_len` | Feishu card char limit (default: 3000) |
| `allowed_chat_ids` | Whitelist of chat IDs (empty = all allowed) |
| `gemini_idle_ttl` | Gemini process idle TTL in seconds (default: 1800) |
| `timezone` | Cron task timezone (e.g. `"Asia/Shanghai"`) |

> **超时层级说明**：
> - `response_timeout`（软超时）：AI 响应无更新超过此时长，释放主锁但后台继续运行，默认 300s
> - `hard_timeout`（硬超时）：强制终止子进程，默认 21600s
> - Shell 命令（`/run`）：固定 30s 硬超时，不受上述配置影响

## Architecture

Project is structured as the `larkhelm/` package:

```
larkhelm/
├── bridge.py           (147 行)  - 主程序入口、WebSocket 事件监听与注册
├── config.py           (242 行)  - 运行时配置加载、路径初始化
├── handlers/                    - 飞书事件处理器子包
│   ├── __init__.py     (24 行)   - re-export
│   ├── _message.py     (393 行)  - 消息接收、路由、表情处理
│   ├── _query.py       (461 行)  - AI 查询流程（流式卡片更新、超时、取消）
│   └── _card_action.py (186 行)  - 卡片按钮回调分发
├── commands.py         (893 行)  - 命令实现（/run, /cd, /ls 等核心命令）
├── cmd_doc.py          (441 行)  - /doc 和 /doc wiki 子命令族
├── chat_state.py       (165 行)  - Per-chat 状态持久化（cwd/model/crons）
├── concurrency.py      (135 行)  - 并发原语（per-chat 锁、取消事件、信号量）
├── dedup.py            (34 行)   - 消息去重（OrderedDict 缓存）
├── log.py              (89 行)   - 对话日志读写（.md + all.jsonl）
├── token_stats.py      (146 行)  - Token 使用量统计与持久化
├── lark_client.py      (1021 行) - 飞书 API 调用封装、卡片操作、权限引导卡片
├── card_builder.py     (166 行)  - 卡片 JSON 构建、Markdown 分割
├── ai_runner.py        (900+ 行) - Claude/Gemini/Kimi CLI 子进程运行
├── crew/                        - 多 Agent 协作子包
│   ├── __init__.py     (118 行)  - re-export
│   ├── _commands.py    (686 行)  - /crew 和 /dev 入口命令
│   ├── _runner.py      (911 行)  - Agent 执行与 DAG 调度
│   ├── _state.py       (121 行)  - 全局 crew 状态变量
│   ├── _checkpoint.py  (260 行)  - Checkpoint 持久化与恢复
│   ├── _pipeline.py    (194 行)  - /dev 固定流水线定义
│   └── _scheduler.py   (137 行)  - Cron 调度器
├── crew_types.py       (120+ 行) - Crew 数据类型（AgentSpec/AgentState/CrewState/CrewPhase 等）
├── crew_card.py        (267 行)  - Crew 飞书卡片构建与心跳推送
├── perm.py             (298 行)  - 权限审批系统
├── perm_hook.py        (74 行)   - 权限审批 hook
└── __main__.py         (42 行)   - 命令行入口
```

### 1. Event Handler (`handlers/`)

`lark_oapi` SDK holds a persistent WebSocket to Feishu. `handle_message()` deduplicates events (via `OrderedDict`) and routes to commands or `_do_query()`.

Two dispatch points, both using if/elif chains:
- **Main message routing**: `handlers/_message.py` (inside `handle_message`)
- **Card button callbacks**: `handlers/_card_action.py` + `commands.py` (`_dispatch_button_cmd`)

### 2. AI Query Engine (`ai_runner.py`)

- **Claude**: Spawns `claude --print --output-format stream-json --verbose` as a subprocess per query. Session IDs are passed via `--resume` for conversation continuity. On crash, clears session and retries once.
- **Gemini**: Spawns `gemini -y --output-format stream-json` per query, session via `--resume`.
- **Kimi**: Spawns `kimi --print --output-format stream-json --input-format stream-json` per query, session via `--session`.

All three support cancellation via threading events and stream structured JSON events (`tool_use`, `text`, `result`) back to callbacks. A global semaphore (`MAX_AI_PROCS=3`) limits concurrent subprocess count.

### 3. State & Session Persistence (`chat_state.py` / `concurrency.py`)

> `state.py` 现为向后兼容的 re-export 层，实际逻辑已拆分到以下子模块：
> - `chat_state.py`：Per-chat 状态字典（cwd/model/crons）的读写与持久化
> - `concurrency.py`：Per-chat 锁（`_get_chat_lock`）与取消事件管理
> - `dedup.py`：消息事件去重缓存
> - `log.py`：对话日志读写
> - `token_stats.py`：Token 用量累计与查询

Persisted state structure per chat:
```python
{
    "cwd":   "/home/user/code",
    "model": "claude",
    "name":  "项目名称",       # optional label
    "crons": [{"id": "uuid", "expr": "0 9 * * *", "query": "...", "model": "claude"}]
}
```

Data files (under `DATA_DIR`):
- `.feishu_sessions/{chat_id}.sid` — Claude session IDs per chat
- `.feishu_sessions/gemini_{chat_id}.sid` — Gemini session IDs per chat
- `.feishu_state.json` — Per-chat working directory and model preference
- `.feishu_logs/{chat_id}/{YYYY-MM-DD}.md` — Markdown conversation logs
- `.feishu_logs/all.jsonl` — Global event log

### 4. Concurrency Model

- Per-chat lock serializes queries (no overlapping requests per chat)
- Per-chat cancellation event allows `/cancel` to interrupt in-flight queries
- Separate locks for: global state, logging, event deduplication, Gemini process pool

### 5. Card Rendering (`lark_client.py` + `card_builder.py`)

Responses are sent as Feishu interactive cards (Markdown). Cards are updated in-place during streaming. When content exceeds `max_card_len`, it is split across multiple cards via `_split_md()`.

Card format selection: has buttons → JSON 1.0 (supports action tags); no buttons → JSON 2.0 (richer rich text).

## 写入飞书文档（Claude Code CLI 集成）

在此项目工作时，若需将内容写入飞书文档，**直接使用 `larkhelm doc` CLI**，无需编写任何脚本：

```bash
# 创建新文档（stdin 传入内容），输出文档 URL
cat report.md | larkhelm doc create "文档标题"

# 追加内容到已有文档
cat more.md | larkhelm doc append "https://feishu.cn/docx/xxxx"

# 覆盖写入
cat updated.md | larkhelm doc write "https://feishu.cn/docx/xxxx"
```

- Owner 由 `~/.config/larkhelm/config.json` 中的 `default_owner_open_id` 控制，创建后自动转移
- 无需初始化 larkhelm 服务，CLI 独立运行
- **不要**为此目的编写独立的 Python 脚本或使用 `upload_to_feishu.py`

## User-Facing Commands

所有命令均以 `/` 开头。

| Command | Function | File | Action |
|---|---|---|---|
| `/c`, `/claude <prompt>` | `_cmd_cli_native()` | commands.py:517 | Force Claude |
| `/g`, `/gemini <prompt>` | `_cmd_cli_native()` | commands.py:517 | Force Gemini |
| `/model claude\|gemini` | `_cmd_model()` | commands.py:490 | Switch default model for this chat |
| `/reset [claude\|gemini\|perm]` | `_cmd_reset()` | commands.py:61 | Clear session(s) / permissions |
| `/status` | `_cmd_status()` | commands.py:78 | Show versions, session IDs, runtime info |
| `/help` | `_cmd_help()` | commands.py:173 | Show help |
| `/cancel` | inline | handlers.py | Interrupt current query |
| `/run <cmd>` | `_cmd_run()` | commands.py:471 | Execute shell command (30s timeout) |
| `/cd <path>` | `_cmd_cd()` | commands.py:432 | Change working directory |
| `/pwd` | `_cmd_pwd()` | commands.py:447 | Show current working directory |
| `/ls [path]` | `_cmd_ls()` | commands.py:451 | List files (max 60 entries) |
| `/pickup` | `_cmd_pickup()` | commands.py:221 | Print commands to resume sessions in terminal |
| `/history [all]` | `_cmd_history()` | commands.py:238 | Last 10 conversation summaries |
| `/stats` | `_cmd_stats()` | commands.py:298 | Token usage statistics |
| `/upgrade` | `_cmd_upgrade()` | commands.py | 更新 larkhelm 到最新版本 |
| `/cron` | `_cmd_cron()` | commands.py:350 | Manage scheduled tasks |
| `/btw <prompt>` | `_cmd_btw()` | commands.py:608 | Quick side question (bypasses main lock) |
| `/crew <task>` | `cmd_crew()` | crew.py | Multi-agent collaborative planning |
| `/dev <task>` | `cmd_dev()` | crew.py | Software engineering pipeline |
| `/doc <sub>` | `_cmd_doc_*()` | cmd_doc.py | 读写飞书文档（read/write/append/list 等子命令） |
| `/doc wiki <sub>` | `_cmd_doc_wiki_*()` | cmd_doc.py | 飞书 Wiki 操作（read/create/list 等） |
| `/rename <名称>` | inline | handlers.py | 给当前会话命名 |

## Key Features

### 飞书文档自动注入（Auto Doc Context Injection）

当用户消息中包含飞书文档/Wiki URL 时，`handlers.py` 中的 `_inject_doc_context()` 会**自动读取文档内容**并追加到发送给 AI 的上下文中。用户无需任何额外操作——这是 LarkHelm 与原生飞书 AI 集成的核心差异化能力。

支持的 URL 类型：
- `https://xxx.feishu.cn/docx/...` — 新版文档
- `https://xxx.feishu.cn/wiki/...` — Wiki 页面
- `https://xxx.feishu.cn/sheets/...` — 电子表格

### Crew 断点机制（Human-in-the-Loop）

`/crew` 任务支持在执行过程中插入人工确认节点（`_breakpoint_events`）。当 Agent 到达断点时，飞书卡片上会出现「继续」/「取消」按钮，支持人工审核后再决定是否继续执行。

### Dev 模式 Git 快照（Auto Git Commit）

`/dev` 流水线在每个关键阶段完成后，会通过 `_git_auto_commit()` 自动提交变更作为快照，便于查看每步的 diff 和在出错后回滚。

## Adding a New Command

**Step 1** — define in `commands.py`:
```python
def _cmd_new_cmd(chat_id: str, args: str = ""):
    send_card(chat_id, "Title", "Body", color="blue")
```

**Step 2** — add import in `handlers.py:38-42`:
```python
from larkhelm.commands import _cmd_new_cmd
```

**Step 3** — add route in `handlers.py:671-740`:
```python
if tl.startswith("/new_cmd"):
    _cmd_new_cmd(chat_id, text[9:].strip())
    return
```

**Step 4** (optional) — add button route in `commands.py:681-716`

**Step 5** (optional) — update `_cmd_help()` help text

> **状态读写提示**：若需读写 Per-chat 状态（cwd、model 等），
> 请使用 `from larkhelm.chat_state import _get_chat_state, _set_chat_field`。
> 并发锁请使用 `from larkhelm.concurrency import _get_chat_lock`。

## Other Extension Points

| Extension point | File | Notes |
|---|---|---|
| Register new Feishu events | bridge.py:86-91 | `.register_p2_xxx_yyy(handler)` |
| New Feishu API calls | lark_client.py | Follow existing patterns |
| Card layout | card_builder.py | Modify `_make_card()` |
| State fields | chat_state.py | Add to `_chat_state_store` data structure |
| Permission rules | perm.py | Add new permission check logic |

### 6. 状态模块导入指南

各专属模块分工明确，直接按需导入：

| 需求 | 导入来源 |
|---|---|
| Per-chat 状态（cwd/model/crons） | `larkhelm.chat_state` |
| 并发锁、取消事件 | `larkhelm.concurrency` |
| 日志读写、调试输出 | `larkhelm.log` |
| Token 统计 | `larkhelm.token_stats` |
| 消息去重 | `larkhelm.dedup` |
| Crew 数据类型 | `larkhelm.crew_types` |

## lark-oapi SDK — Available API Namespaces

SDK install path: `~/.local/lib/python3.13/site-packages/lark_oapi/`

### Docs & Drive

| Namespace | Version | Resources |
|---|---|---|
| `client.docx` | v1 | `document`, `document_block`, `document_block_children`, `document_block_descendant`, `chat_announcement` |
| `client.drive` | v1, v2 | `file`, `file_comment`, `file_version`, `export_task`, `import_task`, `media`, `meta`, `permission_member`, `permission_public` |
| `client.sheets` | v3 | `spreadsheet`, `spreadsheet_sheet`, `spreadsheet_sheet_filter`, `spreadsheet_sheet_filter_view` |
| `client.wiki` | v1, v2 | `node` (v1) / `space`, `space_node`, `space_member`, `task` (v2) |
| `client.docs` | v1 | `content` — fetch legacy doc content |
| `client.document_ai` | v1 | Document intelligence / OCR |

### Usage examples

```python
# Get document content
resp = client.docx.v1.document.get(
    GetDocumentRequest.builder().document_id("doc_token_xxx").build()
)

# Upload file to Drive
resp = client.drive.v1.media.upload_all(
    UploadAllMediaRequest.builder()
    .request_body(
        UploadAllMediaRequestBody.builder()
        .file_name("report.pdf")
        .parent_type("explorer")
        .parent_node("folder_token_xxx")
        .size(file_size)
        .file(open("report.pdf", "rb"))
        .build()
    ).build()
)

# List Drive files
resp = client.drive.v1.file.list(
    ListFileRequest.builder().folder_token("folder_token_xxx").build()
)

# Get Wiki node
resp = client.wiki.v2.space_node.get(
    GetSpaceNodeRequest.builder()
    .space_id("space_id_xxx")
    .node_token("node_token_xxx")
    .build()
)

# List spreadsheet sheets
resp = client.sheets.v3.spreadsheet_sheet.query(
    QuerySpreadsheetSheetRequest.builder()
    .spreadsheet_token("sheet_token_xxx")
    .build()
)
