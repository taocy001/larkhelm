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
| `max_ai_procs` | 并发 AI 子进程上限：正整数 / `"auto"` / 缺省（默认 `"auto"`，根据 cgroup MemoryMax 或物理 RAM 探测，公式见 `runner_base._compute_max_procs`） |
| `timezone` | Cron task timezone (e.g. `"Asia/Shanghai"`) |
| `voice_enabled` | M3.2 语音转文字总开关（默认 `false`；关闭时 bridge 行为完全不变） |
| `voice_model_size` | faster-whisper 模型规格 `tiny`/`base`/`small`/`medium`/`large-v3`（默认 `"small"`） |
| `voice_compute_type` | faster-whisper compute_type，例如 `int8` / `float16`（默认 `"int8"`） |
| `voice_max_duration_ms` | 单条音频上限毫秒，floor `1000`（默认 `180000`） |
| `voice_default_lang` | 全局默认转录语种 `zh` / `en` / `auto`（默认 `"zh"`） |
| `voice_merge_window_sec` | 多条语音消息合并窗口秒数，floor `0`（`0` = 禁用合并；默认 `0`） |
| `voice_max_merge` | 单次最多合并几条语音，floor `1`（默认 `5`） |
| `voice_keep_audio` | 转录后是否保留原音频文件（默认 `false`，即转录完即删） |
| `metrics_text_legacy` | P2 REQ-01：强制 `/metrics` 走 P1 手写文本路径，即使 prometheus-client 已装。默认 `false`；翻 `true` 用于 bisect 指标回归 |
| `memory_extract_buffer_window_sec` | P2 REQ-06：session→cascade buffer 合并窗口（秒）；默认 `0` = 禁用 buffer，每次 update 立即 cascade（P1 byte-compat），>0 合并 |
| `memory_session_smart_compress` | P2 REQ-07：session-layer 走句子级评分 + top-K（确定性，无 LLM），默认 `false` = P1 尾截断 |
| `memory_global_profile_slot_enabled` | P2 REQ-05.1：global memory 按 style/format/domain/expertise 4 槽位写入（每槽 ≤200 chars），默认 `false` = 整段文本 |
| `memory_project_section_enabled` | P2 REQ-05.2：project memory 按 TechStack/Conventions/Architecture/Constraints 4 段写入，默认 `false` = 整段文本 |
| `query_session_v2_traffic` | P3 REQ-02：v2 路径灰度比 0.0–1.0；默认 `0.0` = legacy。`query_session_v2_enabled=true` 时强制走 v2（traffic 视为 1.0）|
| `intent_embedding_top_k_threshold` | P3 REQ-03：embedding L2 分类器最低 cosine 置信，默认 `0.30`；低于则退回 LLM JSON 路径 |
| `llm_router_circuit_failures` | P3 REQ-04：cheap 后端连续失败阈值，超过则开断路，默认 `5` |
| `llm_router_circuit_cooldown_sec` | P3 REQ-04：断路 cool-down 秒数，默认 `30.0`；超过后允许 1 次半开探测 |
| `cascade_backoff_max_attempts` | P3 REQ-05：memory cascade / extract buffer 的 ExponentialBackoff 最大尝试次数，默认 `3` 即 sleep 序列 `[1.0s, 2.0s]`；要 `[1.0s, 2.0s, 4.0s]` 共 3 次 sleep 需显式设为 `4`（单次 sleep cap 30s）|
| `plan_retry_strategy` | P3 REQ-06：`/plan` step 失败时的重试策略，`now`/`manual`/`off`，默认 `off` = 保持 P0-P2 行为 |
| `plugin_report_card_enabled` | P3 REQ-07：boot 后将 plugin 加载失败汇总成飞书橙色卡片推送给 admin，默认 `false` |
| `admin_chat_id` | P3 REQ-07：失败卡片目标 chat_id；为空时退回 `default_owner_open_id` 私聊；都空则只 log |
| `memory_gc_interval_hours` | P3 REQ-08：MemoryGC daemon tick 周期（小时），默认 `6.0`；`0` = 走 P2 的 boot-only 一次性扫描 |
| `crew_checkpoint_ttl_days` | P3 REQ-09：`.crew_workspace/*/crew_checkpoint.json` 孤儿清理 TTL，默认 `7.0` 天 |
| `dev_stage_timeouts` | P3 REQ-10：`/dev` 单 stage 超时覆盖（秒），形如 `{"pm": 600, "implementer": 7200}`；未列出的 stage 走默认公式 |
| `recent_turns_cache_enabled` | P1 REQ-01：`_get_recent_turns` 走 LRU 缓存（key = chat_id + max_turns + max_chars + dedup_prefix_hash + all.jsonl mtime_ns + size），默认 `true`；flip `false` 直走原 tail-read 路径用于 bisect |
| `memory_legacy_cache_enabled` | P1 REQ-02：memory `_layer_global / _layer_project / _layer_session` 走 LRU 缓存（key = layer + path + mtime_ns），默认 `true`；包含 `global_slots` / `project_sections` 两个独立 layer，单层 LRU 容量 128 |
| `doc_inject_cache_enabled` | P1 REQ-03：`_inject_doc_context` 走 TTL 缓存（key = chat_id + doc_type + token + max_chars），默认 `true`；`DocPermissionError` 与 `DocError` 不入缓存（用户可现场授权重试）|
| `doc_inject_cache_ttl_sec` | P1 REQ-03 / P4 REQ-04：TTL 秒数，floor `1`（`0` 视为关闭，回退到默认值）。默认 `600`（P3 之前为 `300`）；命中时 `_inject_doc_context` 在文档头部加「（缓存版本，N 分钟前读取，如内容已变请提示刷新）」age hint（P4 REQ-05），且 metric outcome 是 `hit_with_age_hint` 而非 `hit`（P4 REQ-06）|
| `workspace_hint_keyword_gate` | P3 REQ-02：bool，默认 `false`。`true` 时 `.crew_workspace/` 文件清单仅当用户消息正则匹配 `(workspace\|计划\|任务\|设计\|prd\|design\|tasks\|review\|qa\|crew)`（大小写不敏感）才注入；未命中则整段跳过并 bump `larkhelm_workspace_hint_total{outcome="skipped_by_gate"}`。即使 `true` + 命中关键词，文案仍是 P3 REQ-01 的被动条件式（「如果与本次问题相关，再读取；否则忽略」）|
| `stats_agent_type_breakdown_enabled` | P5 REQ-09：bool，默认 `true`。`/stats` 的 Crew Agents 块按 agent_type 桶（planner/engineer/qa/reviewer/chat/dev/crew/plan/doc/其它）降序输出每桶 `agents/合计 tokens/费用`。`false` 时退回 P2 单行汇总（`**🤖 Crew Agents（本进程）**` 不带「按类型」），用于桶数过多导致卡片溢出 `max_card_len` 的极端情况 |
| `cli_skip_recent_turns_when_sid` | P1 REQ-04：CLI（claude/kimi/gemini）`sid` 非空时跳过 recent_turns 注入；`deepseek_api` 在 `load_history` 非空时同步跳过。默认 `true`；flip `false` 恢复每次注入（多花 ~500 input tokens / call）|
| `anthropic_extended_cache_enabled` | bool 默认 `true`。Anthropic API 适配器请求时携带 `anthropic-beta: extended-cache-ttl-2025-04-11` header 并将 `cache_control.ttl` 升级为 `1h`；若该 beta 在该账号未开通而被拒，进程内自动回退到 5min ephemeral 并写一次 `[anthropic_api]` 调试日志，整进程不再重试。设为 `false` 强制保留 5min |
| `claude_session_auto_reset_enabled` | P0 缓存出血面收敛：当 `claude --resume` 累积 prefix 越过下面两个阈值任一时，自动调 `_clear_sid("claude", chat_id)` + 清零累计器 + 写 milestone + 触发 `larkhelm_session_auto_reset_total{reason}`。默认 `true`；翻 `false` 关掉自动 reset（仍统计累计，仅不触发） |
| `claude_session_reset_cache_tokens` | P0：单 session 累计 `usage.cache_read` 触发自动 reset 的阈值（tokens），默认 `5000000`（5M）。`reason="cache_tokens"` |
| `claude_session_reset_turns` | P0：单 session 累计 `record_token_usage(model="claude")` 调用次数触发自动 reset 的阈值，默认 `50`。`reason="turns"` |
| `chat_agent_cheap_routing_enabled` | P1 缓存出血面收敛：`ChatAgent.execute` 调 `resolve_backend_for_task(profile=chat, cost_ceiling=0.10)` 把 chat 流量导向 DeepSeek/Kimi 等 cheap backend；rank 无健康候选时回落 `_get_chat_model` 并写 `[ChatAgent] fell back to chat model`。默认 `true`；翻 `false` 直接走用户偏好 model |
| `recent_crew_sticky_ttl_sec` | P2：sticky crew context（`get_recent_crew_context` / `consume_recent_crew_context`）生存秒数，floor 60s。默认 `1800`（30 min；之前硬编码 7200/2h）。过期时 `_recent_crew_by_chat` 懒删除并 bump `larkhelm_sticky_context_evicted_total{reason="ttl"}` |
| `recent_crew_sticky_max_injections` | P2：同一 sticky entry 经 `consume_recent_crew_context` 注入到主路径 prompt 多少次后强制淘汰，默认 `5`。`0` = 禁用 per-count 淘汰（仅 TTL）。淘汰时 bump `larkhelm_sticky_context_evicted_total{reason="max_injections"}` |

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
├── doc_handlers.py     (441 行)  - 飞书文档/Wiki 读写 dispatcher（natural-language 入口，被 DocAgent 调用；用户面 /doc 已下线）
├── chat_state.py       (165 行)  - Per-chat 状态持久化（cwd/model/crons）
├── concurrency.py      (135 行)  - 并发原语（per-chat 锁、取消事件、信号量）
├── dedup.py            (34 行)   - 消息去重（OrderedDict 缓存）
├── _message_pure.py    (200 行)  - P2 REQ-03：5 个纯函数（dedup/ACL/doc-url/分类/路由），无飞书 SDK 依赖，可单测
├── metrics.py          (350 行)  - P2 REQ-01：Prometheus 注册中心（4 核心 Gauge + 5 Counter + 1 Histogram），可选 prometheus-client
├── memory_global_slots.py     (~200 行) - P2 REQ-05.1：global memory 4 槽位读写（style/format/domain/expertise）
├── memory_project_sections.py (~170 行) - P2 REQ-05.2：project memory 4 段读写（TechStack/Conventions/Architecture/Constraints）
├── memory_session_compress.py (~250 行) - P2 REQ-07：句子级 score + top-K 压缩，确定性、无 LLM
├── memory_extract_buffer.py   (~230 行) - P2 REQ-06：进程内 session-cascade buffer，timer/capacity/shutdown 三触发
├── backend_api_streaming.py   (~290 行) - P2 REQ-09：StreamingAPIAdapter Protocol + 3 实现 + 通用模板
├── log.py              (89 行)   - 对话日志读写（.md + all.jsonl）
├── token_stats.py      (146 行)  - Token 使用量统计与持久化
├── lark_client.py      (1021 行) - 飞书 API 调用封装、卡片操作、权限引导卡片
├── card_builder.py     (166 行)  - 卡片 JSON 构建、Markdown 分割
├── ai_runner.py        (112 行)  - thin shim，re-export runner_base/runner_claude 等公共接口
├── runner_base.py      (358 行)  - BaseProcessRunner 抽象基类（信号量、watch、retry 模板方法）
├── runner_claude.py    (225 行)  - ClaudeRunner 子类（MCP config、设置文件临时文件管理）
├── runner_kimi.py      (155 行)  - KimiRunner 子类
├── runner_gemini.py    (120 行)  - GeminiRunner 子类
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

Card schema: **JSON 2.0 unconditionally** (post-commit `4b7c68e`). Buttons land
as native `{"tag":"button", ...}` elements in `body.elements[]`; multi-button
rows wrap in `column_set` with `width:"auto"` columns. Callback payload sits
in `behaviors:[{"type":"callback","value":{"cmd":...}}]` — `_card_action.py:27`
parses ``CallBackAction.value`` schema-agnostically. The legacy `_make_card_json10_dict`
path (which used `{"tag":"div","text":{"tag":"lark_md",...}}` for body and
`{"tag":"action","actions":[...]}` for buttons) was deleted because Feishu's
`lark_md` element rendered body markdown at a different default font size
than the JSON 2.0 `markdown` element AND silently dropped bullet lists,
fenced code blocks, and block quotes.

### 6. Phase 5 智能编排层 (`larkhelm/agent_hub/`)

Phase 5 引入意图识别 + Agent 分发层，与现有显式命令并存（不替换）。

**包结构**：

```
larkhelm/agent_hub/
├── intent_types.py    - IntentResult / TaskProfile / AgentDispatch / AgentContext / AgentResult
├── agent_base.py      - AgentExecutor (ABC) + AgentRegistry 单例 AGENT_REGISTRY
├── intent_router.py   - resolve_intent(): 显式命令 → L1 关键词 → L2 cheap LLM JSON
├── model_selector.py  - resolve_backend_for_task() 调用 BackendRegistry.rank_for_task
├── agent_dispatcher.py - AgentDispatcher.dispatch() 透明化卡片 + ACL + 审计
├── intent_feedback.py - record_feedback / register_pending / resolve_pending（JSONL 0600）
├── agent_audit.py     - write_audit / aggregate_daily（JSONL 0600）
├── plugin_loader.py   - importlib.metadata.entry_points('larkhelm.agents') + config['agent_plugins']
└── builtin/           - ChatAgent / DevAgent / CrewAgent / PlanAgent / DocAgent（薄壳调用现有命令）
```

**灰度开关**（`config.json`）：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `intent_router_enabled` | `false` | 总开关，关闭时 `_message.py` 完全不导入 `agent_hub` |
| `intent_router_traffic` | `0.0` | 0.0–1.0 灰度比例，按 `chat_id` 哈希一致性分流 |
| `intent_layer2_strategy` | `"llm"` | `llm` 或 `embedding`（P2 预留） |
| `agent_plugins` | `[]` | 第三方 plugin 入口点字符串，如 `mypkg.module:my_agent` |
| `agent_acl` | `{}` | `{agent_type: ["chat_id_glob", ...]}` |
| `intent_feedback_path` / `intent_audit_path` | 空 | 默认 `DATA_DIR/intent_*.jsonl` |
| `intent_feedback_extended_signals` | `true` | 总开关，关闭时只保留 `signal_type="force_chat"` 一种行为（向后兼容 Phase-D 之前）|
| `intent_feedback_cancel_window_sec` | `60.0` | `/cancel` 在 dispatch 后多少秒内仍算作 `cancel_after_dispatch` 信号 |
| `intent_feedback_signal_text_max` | `800` | 观察类信号（`l1_gray_zone` / `l2_dispatched` 等）的 `text` 截断阈值；force_chat 不截断以保持旧记录字节兼容 |
| `intent_feedback_l1_gray_band` | `0.10` | L1 灰区宽度——置信度落在 `[promotion_threshold, threshold + band)` 时记为 `l1_gray_zone` 难例 |

**扩展信号采集**（`intent_feedback_extended_signals=true` 时启用，默认 ON）：

每条 `intent_feedback.jsonl` 都会带一个新的 `signal_type` 字段（向后兼容：旧
record 没有该字段视为 `force_chat`）。当前采集的 6 种信号：

| signal_type | 触发位置 | corrected_intent | 用途 |
|---|---|---|---|
| `force_chat` | `_card_action.force_chat` 按钮 | `"chat"` | 用户显式纠错（已存在）|
| `cancel_after_dispatch` | `/cancel` 在 `intent_feedback_cancel_window_sec` 内 | `"chat"` | 用户立即取消，强信号 |
| `agent_reswitch` | `AgentDispatcher.dispatch` 或 `_message.py` backend override 在 120s 内 | 新 agent_type | 用户切换 agent / backend |
| `dispatch_failed` | `AgentDispatcher._fallback_to_chat` | `"chat"` | Agent 抛错落回 chat |
| `l1_gray_zone` | `intent_router._maybe_record_l1_gray_zone` | `""` | L1 置信处于灰区的难例（观察） |
| `l2_dispatched` | `intent_router._maybe_record_l2_dispatched` | `""` | L1 弃权后由 L2 接管的样本（观察）|

`corrected_intent=""` 的观察类信号被
`scripts/train_intent_classifier.py:_load_feedback` 自动跳过（其要求非空 label），
只用于 L1 关键词调优脚本的难例挖掘，不会被当作监督样本污染训练。

每次写入会 `larkhelm_intent_feedback_total{signal_type}` +1，Grafana 可监控各
信号的实际产生率（>0 即说明扩展信号正常工作）。

> **路径安全**：当显式设置 `intent_feedback_path` / `intent_audit_path` 时，
> 强烈建议路径**位于 `DATA_DIR` 之内**（例如 `DATA_DIR/audit/intent.jsonl`）。
> 这两份 JSONL 以 0600 权限写入，落到 `DATA_DIR` 之外可能：
> 1. 不参与 `DATA_DIR` 的备份策略，丢失审计；
> 2. 与运维的备份/分享脚本冲突，泄露用户原始 query。
> 若 `DATA_DIR` 未设置（早期 bootstrap / 单文件调试），模块会回退到
> `tempfile.gettempdir() / intent_*.jsonl`，避免文件意外落入当前工作目录。

**第三方 Agent 接入**：

```python
# 子类化 AgentExecutor，实现 execute(intent, ctx) -> AgentResult
from larkhelm.agent_hub import AgentExecutor, AgentResult, AgentContext, IntentResult

class MyAgent(AgentExecutor):
    agent_type = "translate"
    description = "中英互译 Agent"

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        ...
```

通过 entry-point group `larkhelm.agents` 暴露：

```toml
# pyproject.toml
[project.entry-points."larkhelm.agents"]
translate = "mypkg.translate:MyAgent"
```

或在 `config.json` 中追加 `"agent_plugins": ["mypkg.translate:MyAgent"]`。

**详细设计文档**：`.crew_workspace/design.md`（v1.0，2026-05-09）。

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
| `/rename <名称>` | inline | handlers.py | 给当前会话命名 |
| `/voice [status\|lang <zh\|en\|auto>]` | `_cmd_voice()` | commands.py | 查看 / 切换语音转写设置（卡片显示当前 engine + 状态）|
| `/memory export` | `_cmd_memory()` | commands.py | 导出当前 chat 的所有持久化数据为 zip 文件，机器人回复文件消息 |
| `/memory import [file_key]` | `_cmd_memory()` | commands.py | 从 zip 导入记忆数据；无参时等待用户回复 zip 文件 |
| `/memory status` | `_cmd_memory()` | commands.py | 查看持久化层摘要（chat 数、日志大小、记忆文件数等）|

外部 CLI（不是飞书命令）：

| CLI | 作用 |
|---|---|
| `larkhelm voice probe [--no-benchmark] [--no-write]` | 安装时一次性 probe：检 ffmpeg + CPU flags + RAM + 实测 RTF；自动写回 `config.json` 的 `voice_enabled` / `voice_engine` / `voice_model_size` |
| `larkhelm memory export [output.zip] [--chat-ids ID…] [--data-dir DIR] [--include-debug-log]` | 导出持久化数据到 zip 文件（无需 bridge 在线） |
| `larkhelm memory import <archive.zip> [--replace] [--dry-run] [--data-dir DIR]` | 从 zip 恢复数据；默认合并（state.json merge + JSONL 去重）；`--replace` 覆盖写入 |

## Key Features

### 飞书文档自动注入（Auto Doc Context Injection）

当用户消息中包含飞书文档/Wiki URL 时，`handlers.py` 中的 `_inject_doc_context()` 会**自动读取文档内容**并追加到发送给 AI 的上下文中。用户无需任何额外操作——这是 LarkHelm 与原生飞书 AI 集成的核心差异化能力。

支持的 URL 类型：
- `https://xxx.feishu.cn/docx/...` — 新版文档
- `https://xxx.feishu.cn/wiki/...` — Wiki 页面
- `https://xxx.feishu.cn/sheets/...` — 电子表格

### Crew 断点机制（Human-in-the-Loop）

`/crew` 任务支持在执行过程中插入人工确认节点（`_breakpoint_events`）。当 Agent 到达断点时，飞书卡片上会出现「继续」/「取消」按钮，支持人工审核后再决定是否继续执行。

**Phase C 超时**：等待时长由 `crew_breakpoint_timeout_sec`（默认 1800s）控制；
超时后自动 `state.cancel_ev.set()` 并通过 `_failure_card.emit_breakpoint_timeout`
推送橙色提示卡片，已完成阶段保留 checkpoint，用户可重启续跑。

### 为 Crew 新增 Agent 时如何挑 task_profile

`AgentSpec.task_profile` 是 Phase C 引入的字段；它**取代**了之前 `model="claude"`
硬编码的 dispatch 决策，让 backend 选择由 `crew/_backend_resolver.py` 根据
`BACKEND_REGISTRY.rank_for_task` 动态决定。规则：

| 选 profile | 何时使用 | 实际权重（design.md §3.3） |
|---|---|---|
| `planner` | PRD / 需求分解 / 架构设计 / 长链推理 | reasoning=1.0, long_context=0.6 |
| `engineer` | 代码实现 / 修复 / 重构（必须能调用 Write/Edit/Bash 工具） | coding=1.0, tools=0.8, require_tools=True |
| `qa` | 测试编写 / 静态检查 / 验收（需要工具，但偏中等复杂度） | coding=0.6, reasoning=0.8, tools=0.8, require_tools=True |
| `reviewer` | 代码审查 / 8 项 checklist / 长上下文阅读 | reasoning=1.0, long_context=0.5 |
| `chat` | 单轮闲聊 / 简单问答 / 摘要类 fast 任务 | chat=1.0, latency_pref="fast" |

**写入新 AgentSpec 时**：

```python
AgentSpec(
    id="my_new_agent",
    role="...",
    model="",                  # ← 留空，让 resolver 走 task_profile 路径
    task_profile="engineer",   # ← 选上方 5 个之一
    system="...",
    prompt="...",
    depends_on=[...],
    timeout=...,
)
```

**留空 / 兼容性路径**：`task_profile=""` 时 resolver 退到 `model` 字符串路径
（`gemini` / `kimi` / `deepseek` / `hermes_*` 直接 dispatch；其他值或空字符串
退到 `BACKEND_REGISTRY.get_orchestrator()`）。这条路径专门为旧 checkpoint 与
第三方 plugin 保留，**新 agent 不要走这条路径**。

**没有 backend 可用时**：resolver 抛 `NoBackendAvailableError`，
`_run_agent_wrapper` 捕获后调 `_failure_card.emit_agent_failure(stage="backend_select")`
推送 ⚠️ 卡片，提示用户检查 `/status`。**不会**重试 — 这是 config / 健康问题，
不是瞬时失败。

### Phase D · Phase 2 召回栈

Phase 2 引入 **embedding + hybrid 召回**、**stale slice 软删除**、**审计 v2 schema**
与 `/memory diagnose` 飞书命令；默认全关，灰度默认 0%（与 Phase 1 完全
byte-compatible，AC-01）。`memory_retriever_enabled` + `memory_retriever_traffic`
是 **Stage A**（Phase 1 / Phase 2 共享）；`embedding_enabled` + `embedding_traffic`
是 **Stage B**（仅 Phase 2 引入）。两段独立 hash 分桶——同一 chat 必须同时命中
Stage A 与 Stage B 才会真正走到 hybrid。

**Hybrid 调用顺序**：
`KeywordRetriever(top_k × multiplier)` → cosine rerank → `α·cos_sim + (1-α)·BM25_norm`
（默认 α=0.6）→ stale × `memory_stale_decay`（默认 0.5）→ top_k。

**核心新模块**：
- `larkhelm/memory_embedding.py` — `EmbeddingBackend` 三实现（Local ONNX / HTTP /
  Stub）+ `EmbeddingCache(maxsize=2048)` + circuit breaker。**numpy / onnxruntime 是
  optional**，缺装时 `get_embedding_backend()` 自动返回 `None`，retriever 安全退到
  Keyword 路径。
- `larkhelm/memory_lifecycle.py` — `mark_stale_slices()` / `inject_stale_marks()` /
  `unstale_slice_id()`。每个 `.md` 配一个 0600 的 `.meta.json` sidecar，
  保存 `stale_slice_ids` 列表与 `last_gc_at`。

**飞书命令 / CLI**：
- `/memory diagnose [N]` — 取该 chat 最近 24h 的 N 条召回审计，渲染 mode /
  elapsed / 命中 slice 标题（**不展示 body**，避免泄漏）。N 默认 3，上限 10。
- `/memory status` — 在原状态卡上新增 `Stale slice` 与 `上次 GC` 行。
- `larkhelm memory audit-summary [--since 1h] [--chat-id X] [--mode hybrid] [--json]`
  — 跨 rotation 聚合审计 JSONL，输出 mode 分布 / p95 / fail_open_rate / 每 agent
  细分。运维健康面板首选。
- `larkhelm memory unstale --slice-id <12-hex>` — 从所有 `.meta.json` sidecar 中
  剔除某个 id（重新激活被降权的 slice）。

**审计 v2 schema**（`schema_version="2"`）新增字段：`mode` / `declared_mode` /
`hybrid_alpha` / `query_token_count` / `top_k_returned` / `stale_hit_count`。
Phase 1 reader 兼容——多余字段 `json.loads` 忽略即可（已 pinned 于
`tests/test_memory_audit_summary.py::test_legacy_fixture_still_parses`）。

**stale 概念**：连续 `memory_stale_window_days`（默认 90 天）未在审计日志中
被命中的 slice 标记为 stale；检索时 `relevance *= memory_stale_decay`（默认 0.5），
reason 串后缀 `stale`。**不删 `.md` 文件**——`/memory unstale` 可单条恢复。
boot 时 `bridge._start_memory_boot_warmup()` 启动一次 daemon 跑全量重计算 +
embedding 模型预热；每次常规 `MemoryGC` tick 也会触发 audit JSONL rotate +
新一轮 stale 标记（一日一次 + 32MiB 翻滚 + 30 天 unlink）。

**当 embedding 失败时**（onnxruntime 缺失 / HTTP 5xx / circuit 打开）：
`HybridRetriever.retrieve` 把已计算好的 KeywordRetriever pool 截到 top_k 返回；
`_build_with_retriever` 顶层另有一层 `try/except`，捕获任何下游异常后 fail-open 到
KeywordRetriever 并写 `audit.fail_open=True`。`_build_with_retriever` 自己异常时
仍按 Phase 1 行为 fall back 到 `_build_legacy_v2`（外层 `build()` 的 try/except）。

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

## 监控集成（Prometheus，P2 REQ-01）

`health_endpoint_port > 0` 时，`larkhelm.health_server` 暴露三个端点：
`/health`、`/ready`、`/metrics`。`/metrics` 默认走 prometheus-client
渲染（需安装 `pip install -e ".[metrics]"`），缺装或 `metrics_text_legacy=true`
时自动回退到 P1 手写文本路径（byte-compat）。

核心指标列表（前缀 `larkhelm_`）：

| 名称 | 类型 | label | 含义 |
|---|---|---|---|
| `larkhelm_backend_healthy` | Gauge | `name` | 1=健康 / 0=失败 |
| `larkhelm_active_queries` | Gauge | — | 当前 `_do_query` 并发数 |
| `larkhelm_memory_rss_bytes` | Gauge | — | 进程 RSS（bytes） |
| `larkhelm_cascade_active` | Gauge | — | 在飞 cascade extract 数 |
| `larkhelm_cascade_extract_total` | Counter | `kind`,`outcome` | kind∈{project,global}, outcome∈{success,unchanged,rejected,cancelled,error} |
| `larkhelm_cascade_dropped_total` | Counter | — | 因 sem 满或 cancel 被丢的 cascade |
| `larkhelm_cascade_midflight_cancelled_total` | Counter | — | mid-LLM 取消的 cascade |
| `larkhelm_query_duration_seconds` | Histogram | — | query 端到端延时（buckets: 0.5/1/2/5/10/30/60/120/300/600） |
| `larkhelm_extract_buffer_flushes_total` | Counter | `trigger` | trigger∈{timer,capacity,manual,shutdown} |
| `larkhelm_llm_router_circuit_state` | Gauge | `backend` | P3 REQ-04：memory_llm_router 断路器状态，0=closed / 1=half_open / 2=open |
| `larkhelm_tokens_total` | Counter | `backend`,`kind` | 每次 `record_token_usage` 触发一次 4-bucket inc；kind ∈ {input, output, cache_read, cache_create}；backend 取调用方传入的 `model` 标识（CLI 是 `claude`/`gemini`/`kimi`/`deepseek`，API 流式是 `spec.model or spec.id`）|
| `larkhelm_session_auto_reset_total` | Counter | `reason` | P0 缓存出血面收敛：`claude_session_guard.maybe_auto_reset_session` 触发一次自动 reset 时 +1；`reason` ∈ {`cache_tokens`, `turns`}（先 check cache_read 阈值，再 check turn 阈值）|
| `larkhelm_sticky_context_evicted_total` | Counter | `reason` | P2 缓存出血面收敛：sticky crew context entry 被淘汰一次 +1；`reason` ∈ {`ttl`（超过 `recent_crew_sticky_ttl_sec`）, `max_injections`（达到 `recent_crew_sticky_max_injections` 次注入）}|
| `larkhelm_workspace_hint_total` | Counter | `outcome` | P3 REQ-03：`handle_message` 每条消息进入工作区注入段时 +1（恰好一次）；`outcome` ∈ {`injected_passive`（注入被动文案）, `injected_active_legacy`（保留位，REQ-01 改文案后已不再发，留给未来回滚）, `skipped_by_gate`（`workspace_hint_keyword_gate=true` 且关键词未命中）, `skipped_empty`（`.crew_workspace/` 不存在或无 `.md`/`.json` 文件）}|
| `larkhelm_doc_inject_cache_total` | Counter | `outcome` | P1 REQ-03 / P4 REQ-06：`_inject_doc_context` 每次走 doc cache 时 +1；`outcome` ∈ {`hit`（旧 `cached_doc_read` 调用路径）, `hit_with_age_hint`（新 `cached_doc_read_with_meta` 路径，age hint 已注入）, `miss`, `bypass`, `evict`, `invalidate`}；Grafana 总命中率应 query `outcome=~"hit\|hit_with_age_hint"`|
| `larkhelm_intent_feedback_total` | Counter | `signal_type` | Phase D 跟进：每条 `intent_feedback.jsonl` 写入 +1；`signal_type` ∈ {`force_chat`（按钮）, `cancel_after_dispatch`（`/cancel` 落在 dispatch 后 ≤ `intent_feedback_cancel_window_sec` 秒内）, `agent_reswitch`（同 chat 在 120s 内切换 agent / backend）, `dispatch_failed`（agent 抛错回退 chat）, `l1_gray_zone`（L1 置信落在灰区）, `l2_dispatched`（非-chat L2 命中）}；`intent_feedback_extended_signals=false` 时只剩 `force_chat` 一种|

Prometheus scrape 配置示例：

```yaml
scrape_configs:
  - job_name: larkhelm
    static_configs:
      - targets: ['127.0.0.1:9300']
    scrape_interval: 15s
```

**⚠️ 安全提示**：`health_bind_addr` 默认 `127.0.0.1`；不要直接 bind `0.0.0.0`，
经身份验证的反向代理（nginx/Caddy）才暴露给 scraper。

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

### 7. 记忆系统的两条触发路径

`memory.maybe_auto_update(chat_id)` 是后台 LLM 摘要器，把最近对话浓缩成
session memory，再级联抽取 project / global memory。它有**两条触发路径**：

| 触发 | 何时 | 谁触发 | 频率 |
|---|---|---|---|
| 普通节奏 | 普通 `/chat` 查询完成后 | `handlers/_query.py:742` | 每 `AUTO_UPDATE_EVERY=10` 轮一次 |
| 里程碑节奏 | `/dev` / `/crew` / `/plan` 完成时 | `record_milestone(chat_id, kind, summary)`（`memory.py`）| 每次完成 + 60s 防抖 |

里程碑节奏修复了之前"用 `/dev` 干完一整天但 memory 一字未变"的问题：

- `record_milestone` 写一条 `role="milestone"`、`model="milestone"` 的日志条目
- `maybe_auto_update` 的过滤器接受 `role in {user, assistant, milestone}`、
  排除 `model in {crew, shell}`，所以 milestone 条目能被 LLM 摘要看到，
  普通 crew 子任务的喧嚣仍被屏蔽
- 然后 `record_milestone` 强制调 `maybe_auto_update(force=True)`，由
  `_get_update_lock(chat_id)` 防止并发风暴；`_MILESTONE_DEBOUNCE_SEC=60`
  防止短时间多次开销

新写后台任务（类似 `/dev` / `/crew`）务必在 finally 加 `record_milestone`，
否则任务结果不会进入 memory。

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

## 异常处理规范

**三类分类标准**（禁止未经分类就引入新的 `except Exception: pass`）：

| 分类 | 处理方式 | 典型场景 |
|---|---|---|
| 高危—业务静默失败 | `_debug_log` + 用户 ⚠️ 卡片 | `/reset` API history 清除失败 |
| 中危—辅助操作失败 | `_debug_log` 记录，不打断主流程 | token 统计、回调、所有权转移、memory 加载 |
| 低危/零危—可接受静默 | 保持 `except Exception: pass` | `proc.kill()`、stderr drain、调试 I/O |

**日志格式**（写入 `_cfg.DEBUG_LOG`）：

```
[HH:MM:SS] [{Module}] {operation} failed: {exception}
```

### 日志前缀规范

**新代码**约定：`_debug_log` / `safe_log` / `lazy_debug_log` 的第一个参数必须以
`[Module]` 开头，**模块名采用 PascalCase**（与 Python 类名一致），多词不加空格、
不加下划线。子组件用空格分隔（例：`[Crew] Manager: ...`）。

下表是当前已完成的小写 → PascalCase 迁移清单（其它模块下次顺手就改，不强制
批量重写历史日志）：

| ✅ 推荐 | ❌ 已迁移 | 说明 |
|---|---|---|
| `[Crew]` / `[Crew] Manager: …` | `[crew]` / `[Crew/Manager]` | 模块统一大写，子组件空格分隔 |
| `[Checkpoint]` | `[checkpoint]` | |
| `[Perm]` | `[perm]` | |
| `[IntentRouter]` | `[intent_router]` | |
| `[Plan]` | `[plan]` | |
| `[Dev]` | `[dev]` | |
| `[BackendRegistry]` | `[recover_thread]` | 用模块名而非线程名 |

**例外**：第三方进程协议 / 外部 CLI 二进制名保留小写（与命令名对齐）：
`[claude]` / `[gemini]` / `[kimi]` / `[upgrade]` 等。

**未迁移的历史小写前缀**（如 `[memory]` / `[router]` / `[token_stats]` /
`[lark_client]` / agent_hub 内部 `[agent_audit]` 等）属于知情遗留：新写日志
请用 PascalCase，遇到时顺手重命名即可，不必专门起 PR 批量改写。

**helper 选择**：

| Helper | 级别 | 何时用 | 实现位置 |
|---|---|---|---|
| `_debug_log(msg)` | DEBUG | 主路径诊断；调用方已确保 `larkhelm.log` 已加载 | `log.py:_debug_log` |
| `safe_log(msg)` | DEBUG | 异常清理 / 永不抛路径；底层等价于 `_debug_log` 套 try-except | `log.py:safe_log` |
| `lazy_debug_log(msg)` | DEBUG | bootstrap / 循环 import 边缘；模块本身可能尚未完成 import | `log.py:lazy_debug_log` |
| `info(msg)` | INFO | 阶段性进展、状态变更（"backend X 已注册"） | `log.py:info` |
| `warn(msg)` | WARN | 降级行为、credential 拉取失败、操作员需要看到的告警 | `log.py:warn` |
| `error(msg)` | ERROR | 用户可见任务被打断的失败（便于与工单关联） | `log.py:error` |

> `safe_log` 取代 R3 之前 `agent_hub/` 4 处本地 `_safe_log` 副本；
> `lazy_debug_log` 取代 `config.py`、`agent_hub/agent_base.py.abort()`、
> `agent_hub/intent_router.py` 中的"双层 try-import"模式。新代码避免再
> 引入这两种模式的本地拷贝。

### `LARKHELM_LOG_LEVEL` 环境变量过滤

支持通过 `LARKHELM_LOG_LEVEL=DEBUG|INFO|WARN|ERROR` 环境变量在启动时
过滤诊断写入。默认 `DEBUG`（保留原始 verbose 行为，所有 250+ 现存
`_debug_log` 调用照旧落盘）。生产环境想降噪：

```bash
export LARKHELM_LOG_LEVEL=WARN
python3 -m larkhelm start
```

实施细节：

- 读取时机：模块 import 时一次性解析（运行中改环境变量**不**生效，
  避免日志过滤中途翻转造成的诊断盲区）。
- 输出格式：DEBUG 级别保持 `[HH:MM:SS] [Module] msg` **不变**（grep
  兼容）；INFO/WARN/ERROR 加显式标签 `[HH:MM:SS] <LEVEL> [Module] msg`。
- 未知值（如 `VERBOSE`）回退到 DEBUG 并向 stderr 写一行警告，避免拼写
  错误意外静音整个 bridge。
- `safe_log` / `lazy_debug_log` 也走 gate（与 `_debug_log` 一致）。

**保留静默清单**（已确认无需改动）：

| 位置 | 原因 |
|---|---|
| `runner_base.py` `proc.kill()`（3处：watch/stdin/FileNotFoundError）+ `runner_claude/kimi/gemini` 各 0 | OS 级，进程已退出为预期 |
| `runner_base.py` `_drain_stderr` 线程 | display-only，失败无副作用 |
| `runner_base.py` `_cleanup_tmp` unlink | 临时文件已不存在为预期 |
| `log.py` `_debug_log` 内部两处 pass | 调试基础设施，递归报错无意义 |
| `log.py:105` JSONL 行解析跳过 | 设计意图：容错读取损坏行 |
| `memory.py` `_global_memory_file` chat_state 访问 | 已有明确 fallback（返回 None） |
| `crew/_runner.py` git diff | 非 git 仓库为预期行为 |
| `mcp_server.py` config inner parse | MCP config 行级容错，解析失败继续下一行 |
| `crew/_failure_card.py` `emit_agent_failure` / `emit_terminal_failure` / `emit_breakpoint_timeout` 顶层 try | 错误上报路径「永不抛」契约——这三个 emit 入口本身就是其它路径的失败兜底，再抛只会复合污染。docstring 显式说明；三者均由 `test_crew_failure_card.py::test_*_never_raises_on_lark_error` pinned |

## P3 变更摘要（2026-05-18）

P3 引入 10 项 prod-ready 收尾改进，全部默认关闭 / 状态不变。详见 `.crew_workspace/prd.md` + `.crew_workspace/design.md`：

- **REQ-01 Voice duration gate**：`voice/transcribe.py` 强制 `VOICE_MAX_DURATION_MS`；超长音频返回 `error="duration_exceeded"` 而不走推理（ffprobe / wave 探测元数据，无解码成本）
- **REQ-02 Query session v2 traffic**：`query_session_v2_traffic` 灰度比，复用 `_gating.hash_bucket_allows`；`query_session_v2_enabled=true` 仍是强制全开
- **REQ-03 Intent embedding L2**：`intent_layer2_strategy="embedding"` 时走 cosine top-1 + `intent_embedding_top_k_threshold`，失败静默退 LLM
- **REQ-04 LLM router circuit breaker**：`memory_circuit.CircuitBreaker` wrap cheap-backend，metric `larkhelm_llm_router_circuit_state`
- **REQ-05 Cascade exponential backoff**：`memory.cascade_extract` + `memory_extract_buffer.flush` 加 `ExponentialBackoff(max_attempts=cascade_backoff_max_attempts)`，默认 `3` 即 sleep 序列 `[1.0s, 2.0s]`；要 `[1.0s, 2.0s, 4.0s]`（共 4 次尝试）需显式 `cascade_backoff_max_attempts=4`，单次 sleep cap 30s
- **REQ-06 Plan retry engine**：`plan_retry.PlanRetryEngine`，三策略 `now`/`manual`/`off`，默认 `off` 保持现状
- **REQ-07 Plugin failure card**：`plugin_loader.load_plugins` 返回 `PluginLoadReport`；`plugin_report_card_enabled=true` 时 bridge 启动后推送橙色卡片给 `admin_chat_id`
- **REQ-08 MemoryGC daemon**：`memory_gc.MemoryGC` 加 `interval_hours` + `start/stop`；tick 内 audit rotate + stale 重算
- **REQ-09 Checkpoint TTL GC**：`crew/_checkpoint_gc.CheckpointGC` 由 MemoryGC tick 调度，7d 孤儿 checkpoint 清理
- **REQ-10 Dev stage timeouts**：`dev_stage_timeouts: {stage_id: seconds}` 覆盖 `_make_dev_pipeline` 的公式

## P3 / P4 / P5 变更摘要（2026-05-23）

第二批 prefix 出血面收尾 — 减少自动 Read 触发与提高 doc cache 命中率。详见 `.crew_workspace/prd.md` + `.crew_workspace/changes.md`：

- **REQ-01 工作区提示改被动**：`handlers/_message.py::_build_workspace_hint` 把 *「请读取这些文件」* 改成 *「如果与本次问题相关，再读取；否则忽略」*，不再主动诱导模型 Read
- **REQ-02 关键词门控**：`workspace_hint_keyword_gate=true` 时 `.crew_workspace/` 文件清单仅当用户消息正则匹配 `(workspace|计划|任务|设计|prd|design|tasks|review|qa|crew)` 才注入；门关闭（默认）时维持被动文案直接注入
- **REQ-03 工作区注入仪表**：`larkhelm_workspace_hint_total{outcome}` Counter 注册，outcome ∈ {`injected_passive`, `injected_active_legacy`（保留位）, `skipped_by_gate`, `skipped_empty`}
- **REQ-04 doc cache TTL 默认值升级**：`doc_inject_cache_ttl_sec` 从 300 提到 600（10 min），与 Anthropic 1h extended cache 形成 6× 衰减档；5 处定义点同步（`config.py` 默认值 + dataclass 默认值 + setdefault 校验 + 2 处 fallback）
- **REQ-05 doc cache age hint**：`_context_cache.cached_doc_read_with_meta` 返回 `DocReadResult(payload, from_cache, age_sec)`；`_inject_doc_context` 命中缓存时在文档头部追加「（缓存版本，N 分钟前读取，如内容已变请提示刷新）」，`< 60s` 渲染「不到 1 分钟前」。首次注入不加 hint
- **REQ-06 doc cache 命中类型可观测**：`larkhelm_doc_inject_cache_total{outcome}` 新增 `hit_with_age_hint` outcome 与原 `hit` 并列；Grafana 总命中率应 `outcome=~"hit|hit_with_age_hint"`
- **REQ-07/08 /stats 按 agent_type 分桶**：`token_stats.summarize_crew_agent_tokens_by_type` 把 crew agent token 按 `_AGENT_TYPE_MAP` 归到 planner/engineer/qa/reviewer/chat/dev/crew/plan/doc 桶，`commands._render_crew_agent_breakdown` 渲染降序行
- **REQ-09 退路开关**：`stats_agent_type_breakdown_enabled=false` 让 `_cmd_stats` 退回 P2 单行汇总（用于桶数过多导致卡片溢出的极端情况）
- **REQ-10 开关独立性**：本批 4 个新 flag（`workspace_hint_keyword_gate` / `stats_agent_type_breakdown_enabled` / 复用既有 `doc_inject_cache_ttl_sec` / `doc_inject_cache_enabled`）默认值在 `config.py` 与本表保持一致、彼此独立可翻，无隐式依赖
