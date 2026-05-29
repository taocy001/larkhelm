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

仓库自带 pytest 套件（`tests/` 下 ~170 个测试文件），统一入口在 `Makefile` / `scripts/check.sh`：

```bash
make test                       # pytest 全套（含 pytest-timeout 守护）
make lint                       # ruff bug-detector 子集
make type                       # mypy 严格白名单
make all                        # 上面三个

make test ARGS="-k pure -x"     # 转发额外参数
./scripts/check.sh test         # 没装 make 时直接调脚本

python3 -c "import larkhelm.bridge; print('OK')"   # 仅检查包能否 import
python3 -m larkhelm --help                          # 命令行入口
```

> 跑测试 / lint / type check 需要 `pip install -e ".[dev]"`（带 pytest-timeout / mypy / ruff）。

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

> **本表只列 agent 在改代码时高频会碰到 / 必须尊重的字段**——凭证、超时、
> 行为开关、灰度总开关。完整字段清单（含 doc/voice/health/embedding/P1·P2·P3
> 灰度旋钮共 80+ 项）见：
>
> - `larkhelm_config.example.json` — 带 `_comment_*` 注释的运行示例与默认值
> - `README.md → 配置文件` / `### Phase D 召回灰度开关` / `### 启用语音功能` — 用户面字段说明
> - `larkhelm/config.py` `setdefault(...)` 调用列表 — **运行时真源**
> - `.crew_workspace/config_diff.md` — 四方比对矩阵
>
> 新增字段时同步：`config.py` setdefault + `larkhelm_config.example.json` +
> （对用户可见时）`README.md` 配置表。

**凭证与超时**

| Field | Purpose |
|---|---|
| `APP_ID`, `APP_SECRET` | Feishu app credentials (required) |
| `claude_command`, `gemini_command`, `kimi_command` | CLI binary paths (default: 命令名) |
| `default_model` | `"claude"` / `"gemini"` / `"kimi"` / `"deepseek"` |
| `default_cwd` | Initial working directory for AI subprocess |
| `skip_permissions` | Auto-confirm Claude permission prompts |
| `response_timeout` | 软超时（秒），AI 无更新即释放主锁，后台继续；默认 `300` |
| `hard_timeout` | 硬超时（秒），强制 kill 子进程；默认 `21600`（6h） |
| `shell_timeout_sec` | `/run` 命令硬超时；默认 `30`，floor 1s（不走 `response_timeout` / `hard_timeout`） |
| `max_card_len` | Feishu card char limit；默认 `3000` |
| `max_ai_procs` | 并发 AI 子进程上限：`正整数` / `"auto"` / 缺省（默认 `"auto"`，由 `runner_base._compute_max_procs` 按 cgroup MemoryMax 或物理 RAM 探测） |
| `allowed_chat_ids` | 白名单 chat ID 列表，空 = 不限制 |
| `gemini_idle_ttl` / `timezone` | 见 README |

**关键行为开关**（改记忆 / cache / runner 代码必看）

| Field | Purpose |
|---|---|
| `anthropic_extended_cache_enabled` | 默认 `true`。Anthropic API 适配器带 `anthropic-beta: extended-cache-ttl-2025-04-11` header 并将 `cache_control.ttl` 升级为 `1h`；该 beta 未开通时进程内自动回退 5min ephemeral，整进程不再重试。设 `false` 强制保留 5min |
| `claude_session_auto_reset_enabled` | 默认 `true`。`claude --resume` 累积 prefix 越过下面任一阈值时自动 `_clear_sid + reset` 并写 milestone；翻 `false` 仅统计不触发 |
| `claude_session_reset_cache_tokens` | 累计 `usage.cache_read` 阈值，默认 `5_000_000`（5M）；`reason="cache_tokens"` |
| `claude_session_reset_turns` | 累计 `record_token_usage(model="claude")` 次数阈值，默认 `50`；`reason="turns"` |
| `chat_agent_cheap_routing_enabled` | 默认 `true`。`ChatAgent.execute` 调 `resolve_backend_for_task(profile=chat, cost_ceiling=0.10)` 走 DeepSeek/Kimi 等 cheap backend；无健康候选时回落 `_get_chat_model` |
| `backend_aware_budget_enabled` | 默认 `false`。按 backend context window tier 动态缩放记忆注入预算：Gemini/Kimi（≥256K）+20%，mid-tier（≥64K）不变，小窗口（<64K）−30%；联动字段：`context_window_<id>`（9 个，值 0 = 用内置默认，见 `token_budget.DEFAULT_CONTEXT_WINDOWS`）|
| `cli_skip_recent_turns_when_sid` | 默认 `true`。`sid` 非空时跳过 recent_turns 注入（多省 ~500 input tokens / call）；flip `false` 强制每次注入 |
| `memory_extract_buffer_window_sec` | 默认 `0` = 禁 buffer，每次 update 立即 cascade（P1 byte-compat）；>0 合并 |
| `memory_session_smart_compress` | 默认 `true` = 句子级评分 + top-K（确定性，无 LLM）；flip `false` 退回尾截断 |
| `embedding_enabled` | 默认 `true`。embedding 功能总开关；`false` 时强制 keyword 路径，不受 `embedding_traffic` 影响 |
| `embedding_traffic` | 默认 `0.1`。embedding / hybrid 召回流量比例（0.0–1.0）；需同时开启 `embedding_enabled=true` 且 `embedding_backend!="none"` |
| `memory_global_profile_slot_enabled` | **默认 `true`（P5-OPT6）** = 4 槽位（style/format/domain/expertise，≤200 chars 每槽）；局部编辑不会让整段 body 漂移、Anthropic prompt-cache prefix 命中率更稳。翻 `false` 退回整段文本 |
| `memory_project_section_enabled` | **默认 `true`（P5-OPT6）** = 4 段（TechStack/Conventions/Architecture/Constraints）；翻 `false` 退回整段文本 |
| `recent_turns_cache_enabled` | 默认 `true`，`_get_recent_turns` 走 LRU；flip `false` 直走 tail-read（bisect 用） |
| `memory_legacy_cache_enabled` | 默认 `true`，三层 memory 走 LRU（key = layer + path + mtime_ns），单层容量 128 |
| `doc_inject_cache_enabled` / `doc_inject_cache_ttl_sec` | 默认 `true` / `600`s。命中时 `_inject_doc_context` 加 age hint，metric outcome = `hit_with_age_hint`；`DocPermissionError` 与 `DocError` 不入缓存 |
| `workspace_hint_keyword_gate` | 默认 `false`。`true` 时 `.crew_workspace/` 文件清单仅匹配 `(workspace|计划|任务|设计|prd|design|tasks|review|qa|crew)` 时注入 |
| `recent_crew_sticky_ttl_sec` | sticky crew context 生存秒数，floor 60s，默认 `1800`（30 min） |
| `recent_crew_sticky_max_injections` | sticky entry 注入到 prompt 多少次后强制淘汰，默认 `5`；`0` = 仅 TTL 淘汰 |

**灰度总开关 / 失败上报**

| Field | Purpose |
|---|---|
| `intent_router_enabled` / `intent_router_traffic` | Phase 5 智能编排总开关 + 0.0–1.0 灰度。默认开启 10% 灰度（`enabled=true` + `traffic=0.1`）；详细分层开关（L1/microlearn/feedback）见 `larkhelm_config.example.json`，升档/回滚流程见 `.crew_workspace/intent_router_rollout.md` |
| `query_session_v2_traffic` | `_do_query` v2 灰度比 0.0–1.0；默认 `0.0`。`query_session_v2_enabled=true` 时强制 v2 |
| `metrics_text_legacy` | 默认 `false`；`true` 强制 `/metrics` 走 P1 手写文本（bisect 指标回归用） |
| `plan_retry_strategy` | `/plan` step 失败重试策略：`now` / `manual` / `off`，默认 `off`（保持 P0-P2 行为） |
| `cascade_backoff_max_attempts` | memory cascade / extract buffer 的 ExponentialBackoff 最大尝试次数，默认 `3`（即 sleep `[1.0s, 2.0s]`，单次 cap 30s） |
| `llm_router_circuit_failures` / `llm_router_circuit_cooldown_sec` | cheap 后端连续失败阈值（默认 `5`）+ cool-down 秒（默认 `30.0`）；改 `memory_circuit.py` 必看 |
| `failure_report_card_enabled` / `plugin_report_card_enabled` | 默认 `false`。失败 / 插件加载诊断卡片总开关，目标 chat 由 `admin_chat_id`（默认 `""`） 决定，为空时退回 `default_owner_open_id` 私聊 |
| `crew_checkpoint_ttl_days` / `dev_stage_timeouts` | crew 孤儿 checkpoint TTL（默认 `7.0`）+ `/dev` 单 stage 超时覆盖（默认 `{}`，未列 stage 走默认公式） |
| `memory_gc_interval_hours` | MemoryGC daemon tick 周期（小时），默认 `6.0`；`0` = boot-only 一次性扫描 |
| `stats_agent_type_breakdown_enabled` | `/stats` Crew Agents 按 agent_type 分桶输出，默认 `true`；`false` 退回单行汇总 |
| `voice_enabled` | 语音转写总开关，默认 `false`；其余 `voice_*` 见 README §启用语音功能 |

> **超时层级说明**：
> - `response_timeout`（软超时）：AI 响应无更新超过此时长，释放主锁但后台继续运行，默认 300s
> - `hard_timeout`（硬超时）：强制终止子进程，默认 21600s
> - Shell 命令（`/run`）：默认 30s 硬超时，由 `shell_timeout_sec` 调整（floor 1s），不走 `response_timeout` / `hard_timeout`

## Architecture

> **文件清单不再写行号**——行号一过几次重构就全错。下面只列符号入口（函数 / 类 / 模块用途）。需要看具体行号请 `grep -n <symbol>` 或读源码。

Project is structured as the `larkhelm/` package. 核心模块按角色分组：

### 入口与配置
- `__main__.py` — CLI 入口（`larkhelm start` / `larkhelm voice probe` / `larkhelm memory ...` / `larkhelm doc ...`）
- `bridge.py` — `main()` 主程序、`_start_memory_boot_warmup()` 启动 daemon、`.register_p2_*` 注册飞书事件
- `config.py` — 运行时配置加载、~91 个 `setdefault`、自动发现 backend（`_auto_discover_cli` / `_auto_discover_http`）
- `command_registry.py` — `CommandSpec` + `COMMAND_REGISTRY` 集中式 slash 命令注册表（替代旧的 600 行 if/elif 链）

### 飞书事件层 (`handlers/`)
- `_message.py` — `handle_message()` 主消息路由（ACL / dedup / 工作区注入 / intent router / `/cancel` `/rename` `/btw` 直接处理）；其余命令交给 `COMMAND_REGISTRY.dispatch`
- `_query.py` — `_do_query()` AI 查询主流程；`_inject_doc_context()` doc URL 注入、流式卡片更新、超时、取消
- `_query_session.py` — `QuerySession` 类化重构（v2 灰度由 `query_session_v2_traffic` 控制）
- `_query_card_state.py` / `_query_pure.py` — query 子组件（卡片状态机 / 纯函数）
- `_card_action.py` — 卡片按钮回调分发，调用 `commands._dispatch_button_cmd`

### 命令实现
- `commands.py` — 所有 `_cmd_*` 函数（`/run` `/cd` `/ls` `/help` `/reset` `/status` `/history` `/stats` `/memory` `/voice` `/cron` 等）+ `_dispatch_button_cmd()`
- `cmd_plan.py` — `/plan` 多阶段串行流水线入口
- `doc_handlers.py` — 飞书文档/Wiki 读写 dispatcher；**无用户面 slash 命令**，由 `larkhelm doc` CLI 与 DocAgent（natural-language 入口）共同调用
- `file_handler.py` — 飞书文件消息（M4.1）处理
- `orchestration.py` — `_do_query_with_delegation` 的 DELEGATE / FORWARD 协议
- `router.py` — backend 路由分发

### 状态与并发
- `chat_state.py` — Per-chat 状态字典（cwd / model / crons / sender_open_id），磁盘 `.feishu_state.json`
- `concurrency.py` — `_get_chat_lock`（per-chat 串行锁）+ `is_shutting_down` 取消事件 + Gemini 进程池信号量
- `dedup.py` — 消息事件去重（OrderedDict 缓存）
- `log.py` — `_debug_log` / `safe_log` / `lazy_debug_log` / `info/warn/error` + 对话日志（`.md` + `all.jsonl`）+ rotate
- `token_stats.py` — Token 用量累计、`summarize_crew_agent_tokens_by_type` 按 agent_type 分桶

### AI Runner
- `ai_runner.py` — thin shim，re-export `runner_base` / `runner_claude` 等公共接口
- `runner_base.py` — `BaseProcessRunner` 抽象基类（信号量、watch、retry、`_compute_max_procs` 公式）
- `runner_claude.py` — ClaudeRunner（`--print --output-format stream-json --verbose`，`--resume <sid>`，MCP config 临时文件）
- `runner_gemini.py` — GeminiRunner（`-y --output-format stream-json`，`--resume <sid>`）
- `runner_kimi.py` — KimiRunner（`--print --output-format stream-json --input-format stream-json`，`--session <sid>`）
- `runner_deepseek.py` — DeepSeek HTTP backend（无官方 CLI，纯 API 透传）
- `backend_api_streaming.py` — `StreamingAPIAdapter` Protocol + 3 实现 + 通用模板（P2 REQ-09）
- `backend_registry.py` — 11 个 backend 注册 + `rank_for_task` 选型 + 健康探测

### 飞书 SDK 封装
- `lark_client.py` — 飞书 API 调用封装、卡片增删改、权限引导卡片、`BOT_OPEN_ID`
- `card_builder.py` — 卡片 JSON 构建（JSON 2.0 unconditionally）、`_split_md` Markdown 分割

### 记忆系统
- `memory.py` — 三层记忆（global / project / session）+ `maybe_auto_update` LLM 摘要 + `record_milestone`
- `memory_global_slots.py` — global memory 4 槽位（style / format / domain / expertise）— P2 REQ-05.1
- `memory_project_sections.py` — project memory 4 段（TechStack / Conventions / Architecture / Constraints）— P2 REQ-05.2
- `memory_session_compress.py` — 句子级 score + top-K 压缩（确定性，无 LLM）— P2 REQ-07
- `memory_extract_buffer.py` — session-cascade buffer（timer / capacity / shutdown 三触发）— P2 REQ-06
- `memory_embedding.py` — `EmbeddingBackend` 三实现（Local ONNX / HTTP / Stub）+ `EmbeddingCache` + circuit breaker — Phase D
- `memory_lifecycle.py` — `mark_stale_slices` / `inject_stale_marks` / `unstale_slice_id` — Phase D
- `memory_circuit.py` — `CircuitBreaker`（cheap-backend 断路保护）
- `memory_gc.py` — `MemoryGC` daemon（audit rotate + stale 重算）
- `_context_cache.py` — doc cache（`cached_doc_read_with_meta` 返回 `DocReadResult`）

### Crew 多 Agent (`crew/`)
- `_commands.py` — `cmd_crew()` / `cmd_dev()` 入口
- `_runner.py` — Agent 执行 + DAG 调度 + checkpoint 恢复
- `_state.py` — 全局 crew 状态变量
- `_checkpoint.py` — Checkpoint 持久化与恢复
- `_checkpoint_gc.py` — 孤儿 checkpoint 清理（TTL = `crew_checkpoint_ttl_days`，默认 7d）
- `_pipeline.py` — `/dev` 固定流水线定义（PM → 架构 → 工程 → QA → Review）
- `_backend_resolver.py` — `task_profile` → backend 选型
- `_scheduler.py` — Cron 调度器
- `_failure_card.py` — `emit_agent_failure` / `emit_terminal_failure` / `emit_breakpoint_timeout`（永不抛契约）
- `_hermes_orchestrator.py` — Hermes / 多 Agent orchestrator 后端

### Crew 共享
- `crew_types.py` — `AgentSpec` / `AgentState` / `CrewState` / `CrewPhase` 等数据类型
- `crew_card.py` — Crew 飞书卡片构建与心跳推送

### Crew 输出协议防护层（F1-F6 + LOW2/LOW3）

Crew agent 的 `output_file` 在**写盘前**和**验证时**都过 sentinel scan，防止非-tool backend（如 DeepSeek）把内部 tool-call token 当 markdown 流出污染下游 stage。所有 helper 都集中在 `crew/_runner.py`。

关键 helper（grep anchor 见函数名）：

| 名称 | 作用 |
|---|---|
| `_OUTPUT_SENTINELS` | 6 个**严格** token 串，覆盖 DeepSeek DSML（`<｜｜DSML｜｜tool_call`）/ OpenAI（`<tool_calls>`）两套风格，bare substring match 即拒。Anthropic XML（`<function_calls>` / `<invoke name=`）已移出本列表 — 见下方 `_ANTHROPIC_LOOSE_SENTINEL_RE` |
| `_ANTHROPIC_LOOSE_SENTINEL_RE` | SEC-v2-MED-1：Anthropic XML 走**结构正则**而非裸字符串。匹配 `<function_calls>` + `<invoke name=` + `</function_calls>` 三段在 4 KiB 窗口内同时出现的完整 shape — 这种 shape 只有非-tool backend 真泄漏才会出现。narrative prose 单独提及任一 tag 不命中（修了 LOW3 把 Claude API 文档 / 项目 README / CLAUDE.md 自身误标的问题）|
| `_anthropic_loose_check` | 调用 `_ANTHROPIC_LOOSE_SENTINEL_RE`，命中即视为 sentinel-class 违约；按 `crew_sentinel_anthropic_loose_enabled` gate enforce/observe，metric `larkhelm_crew_validate_anthropic_loose_total{outcome}` 始终 emit |
| `_strip_code_evidence` | 剥 fenced (```` ``` ````) / inline-backtick (`` ` ``) / blockquote (`>`) 三种合法引用形式，scrub 后再做 sentinel scan，避免误伤合法 review / 文档 |
| `_validate_output_artifact` | 写盘后扫描；命中后调 `_quarantine_invalid_output` 把文件改名为 `<output>.invalid` 留证 |
| `_persist_result_to_output_file_if_missing` | 写盘前 pre-scan（LOW2）；命中即 skip persist 防止 corrupt 文件出现在 disk 上 |
| `_sanitize_quarantined_content` | synth 阶段从 `.invalid` 读 sanitized 摘要喂给 final reviewer（F5） |
| `_banner_throttle_should_send` | F6 红 banner 去重：throttle key `(crew_id, agent_id)`，单 crew 同 agent 只推一次 |

用户面：

- 红色 banner 卡片由 `_failure_card.emit_agent_failure(stage=...)` 推送，`stage ∈ {validate, backend_select, oom, timeout}` 才会推；其他 stage（如普通 `run` 异常）走 `_debug_log` + 中性失败卡，不触发红 banner
- `.crew_workspace/<output>.invalid` 是被 quarantine 的可疑输出，可手动 inspect 后删除；synth 阶段已自动消费 sanitized 摘要

约束（写新 backend / agent 必看）：

- agent 的 prose output **不要直引** `<｜｜DSML｜｜tool_call` / `<tool_calls>` 等严格 sentinel；必须引用时套 ```` ``` ```` fenced block 或 `` ` `` inline backticks，让 `_strip_code_evidence` 能正确 scrub。Anthropic XML（`<function_calls>` / `<invoke name=`）由结构正则把关——单独提及任一 tag 现在**允许直引**，只有 opening + invoke + closing 三段同时出现的完整 shape 才被拒
- 非-tool-capable backend（`tags` 不含 `tools`）**不要 dispatch** 给有 `output_file` 的 agent；resolver Path 2（F3，`_backend_has_tools` gate）已硬拦，走 `BackendRegistry.rank_for_task` 路径也需要 `task_profile.require_tools=True`
- SEC-CRIT-4 layer-2 启发式（已落地，默认 observe-only）：`_validate_output_artifact` 末段调 `_layer2_check`，layer-1 miss 后按 `raw_hits` + `drop_ratio` 二段判定，默认 `crew_sentinel_layer2_enabled=false` + `traffic=0.0` 只走 observe（`larkhelm_crew_validate_layer2_total{outcome, mode}` 始终 emit），等运营校准阈值后再翻 `true` 强制；阈值 `crew_sentinel_layer2_raw_threshold=3` / `drop_ratio=0.30` / `paranoid_threshold=5`，全部走 `setdefault` 可覆盖。背景见 `.crew_workspace/review_security.md` § SEC-CRIT-4
- SEC-v2-MED-1 Anthropic XML 结构检查（已落地，**默认 enforced**）：`crew_sentinel_anthropic_loose_enabled=true` 时 `<function_calls>` + `<invoke name=` + `</function_calls>` 三段完整 shape 命中即拒，命中时 `larkhelm_crew_validate_anthropic_loose_total{outcome="hit_enforced"}` +1；翻 `false` 进 observe-only（`hit_observed`），保留 metric 但不 quarantine。结构正则的窗口上限 4 KiB，跨段提及任一 tag 不命中（修了 LOW3 bare-substring 误报）。背景见 `.crew_workspace/review_security_v2.md` § SEC-v2-MED-1
- SEC-v2-MED-2 backend 排除 TTL（已落地，**默认 60s**）：`AgentState.excluded_backends_until` 是 `dict[str, float]`（backend_id → 失效时间戳，原 `excluded_backend_ids: list[str]` 已重命名+换型）。`_run_agent_wrapper` validate-fail 时按 `time.time() + crew_backend_exclusion_cooldown_sec` 写入；resolver site 与 `_execute` 跨 round retry-target reset block 都只承认/保留 TTL 未过期的 entry，**不再 unconditional clear**。若同一 backend 被重复写入（已有 entry）→ 视为 swing，`larkhelm_crew_backend_swing_total{agent_id}` +1 并 warn 行尾标 `swing-repeat`。设 `crew_backend_exclusion_cooldown_sec=0` 退回 pre-fix 行为（每轮立即 clear）。背景见 `.crew_workspace/review_security_v2.md` § SEC-v2-MED-2

### 智能编排 (`agent_hub/`) · Phase 5
> 详细设计见 [`.crew_workspace/design.md`](.crew_workspace/design.md) §Phase 5
- `intent_types.py` / `agent_base.py` — `AgentExecutor` ABC + `AGENT_REGISTRY` 单例
- `intent_router.py` — `resolve_intent`: 显式命令 → L1 关键词 → L2 cheap LLM
- `intent_keywords.py` — L1 关键词分类器
- `intent_embedding.py` — embedding L2 分类器（P3 REQ-03）
- `intent_microlearn.py` — 微学习分类器（Phase D-D）
- `intent_feedback.py` — 6 种 signal_type 反馈（force_chat / cancel_after_dispatch / agent_reswitch / dispatch_failed / l1_gray_zone / l2_dispatched）
- `agent_audit.py` — write_audit / aggregate_daily（JSONL 0600）
- `agent_dispatcher.py` — `AgentDispatcher.dispatch` + ACL + 透明化卡片
- `model_selector.py` — `resolve_backend_for_task` 调 `BackendRegistry.rank_for_task`
- `plugin_loader.py` / `plugin_report.py` — entry-point 插件加载 + 失败汇总卡片
- `builtin/` — ChatAgent / DevAgent / CrewAgent / PlanAgent / DocAgent（薄壳调用现有命令）

### 语音 (`voice/`)
- `voice/transcribe.py` — faster-whisper / DashScope 引擎封装 + duration gate（P3 REQ-01）
- `voice/merge.py` — 多条语音合并窗口

### 监控与运维
- `metrics.py` — Prometheus 注册中心（核心 Gauge + Counter + Histogram），可选 prometheus-client — P2 REQ-01
- `health_server.py` — `/health` `/ready` `/metrics` HTTP 端点
- `perm.py` — 权限审批系统（飞书卡片 + socket 协议）
- `perm_hook.py` — Claude Code PreToolUse hook 入口（**注意**：editable 安装下，hook command 必须用 `python3 -m larkhelm.perm_hook` 而非绝对文件路径，因为 site-packages 里没有物理文件，靠 `.pth` 解析）
- `mcp_server.py` — MCP 工具服务器（larkhelm-as-MCP-server）
- `workspace_finalize.py` — `.crew_workspace/` 文件清单注入与 finalize
- `oauth_user.py` — 飞书 user-token OAuth 流程
- `_message_pure.py` — 5 个纯函数(dedup/ACL/doc-url/分类/路由)，无飞书 SDK 依赖 — P2 REQ-03

### 1. Event Handler (`handlers/`)

`lark_oapi` SDK holds a persistent WebSocket to Feishu. `handle_message()` deduplicates events (via `OrderedDict`) and routes to commands or `_do_query()`.

Two dispatch points, both using if/elif chains:
- **Main message routing**: `handlers/_message.py` (inside `handle_message`)
- **Card button callbacks**: `handlers/_card_action.py` + `commands.py` (`_dispatch_button_cmd`)

### 2. AI Query Engine (`ai_runner.py` + `runner_*.py` + `backend_registry.py`)

四个 backend，前三个是 CLI 子进程，最后一个是 HTTP API 透传：

- **Claude**: spawns `claude --print --output-format stream-json --verbose`, session via `--resume <sid>`. On crash, clears session and retries once.
- **Gemini**: spawns `gemini -y --output-format stream-json`, session via `--resume <sid>`.
- **Kimi**: spawns `kimi --print --output-format stream-json --input-format stream-json`, session via `--session <sid>`.
- **DeepSeek**: no CLI — `runner_deepseek.py` 走 HTTP API（`DEEPSEEK_BASE_URL/chat/completions`），session 历史保存在 `.feishu_sessions/deepseek_*.sid`。

All four backends:
- Stream structured JSON events (`tool_use` / `text` / `result`) back to callbacks
- Support cancellation via threading events
- 受 `BackendRegistry` 健康探测、`rank_for_task` 选型管理
- CLI backend 共享 `MAX_AI_PROCS` 信号量（`max_ai_procs="auto"` 时由 `_compute_max_procs` 公式决定）

复杂任务可由 `orchestration.py:_do_query_with_delegation` 走 DELEGATE / FORWARD 协议委托给 Crew / 子 agent。

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
in `behaviors:[{"type":"callback","value":{"cmd":...}}]` — `_card_action.py`
parses ``CallBackAction.value`` schema-agnostically. The legacy `_make_card_json10_dict`
path (which used `{"tag":"div","text":{"tag":"lark_md",...}}` for body and
`{"tag":"action","actions":[...]}` for buttons) was deleted because Feishu's
`lark_md` element rendered body markdown at a different default font size
than the JSON 2.0 `markdown` element AND silently dropped bullet lists,
fenced code blocks, and block quotes.

### 6. Phase 5 智能编排层 (`larkhelm/agent_hub/`)

Phase 5 引入意图识别 + Agent 分发层，与现有显式命令**并存**（不替换）。**默认 10% 灰度**（`intent_router_enabled=true` + `intent_router_traffic=0.1`），关闭时 `_message.py` 不 import `agent_hub`；升档 / 回滚 SOP 见 [`.crew_workspace/intent_router_rollout.md`](.crew_workspace/intent_router_rollout.md)。

- 包结构、灰度开关、扩展信号采集、第三方 Agent 接入：详见 [`.crew_workspace/design.md`](.crew_workspace/design.md) §Phase 5
- 关键审计 / 反馈 JSONL：`DATA_DIR/intent_*.jsonl`（0600 权限）
- 6 种 signal_type：`force_chat` / `cancel_after_dispatch` / `agent_reswitch` / `dispatch_failed` / `l1_gray_zone` / `l2_dispatched`
- 指标：`larkhelm_intent_feedback_total{signal_type}`
- 第三方 plugin 通过 entry-point group `larkhelm.agents` 或 `config["agent_plugins"]` 接入

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

所有命令均以 `/` 开头。详细命令清单与示例见 README.md「聊天命令」段；下表按入口函数 / 模块整理，便于代码导航。

| Command | Function | Module | Action |
|---|---|---|---|
| `/c`, `/claude <prompt>` | `_cmd_cli_native()` | `commands.py` | Force Claude |
| `/g`, `/gemini <prompt>` | `_cmd_cli_native()` | `commands.py` | Force Gemini |
| `/k`, `/kimi <prompt>` | `_cmd_cli_native()` | `commands.py` | Force Kimi |
| `/d`, `/deepseek <prompt>` | `_cmd_cli_native()` | `commands.py` | Force DeepSeek (HTTP API) |
| `/model claude\|gemini\|kimi\|deepseek` | `_cmd_model()` | `commands.py` | Switch default model（`/lock` 是别名）|
| `/lock [id\|off]` | `_cmd_lock()` | `commands.py` | List / lock / unlock backend |
| `/reset [claude\|gemini\|kimi\|deepseek\|perm\|memory]` | `_cmd_reset()` | `commands.py` | Clear session(s) / permissions / session memory |
| `/status` | `_cmd_status()` | `commands.py` | Show versions, session IDs, backend health |
| `/help` | `_cmd_help()` | `commands.py` | Show help |
| `/cancel` | inline | `handlers/_message.py` | Interrupt current query |
| `/rename <名称>` | inline | `handlers/_message.py` | 给当前会话命名 |
| `/btw <prompt>` | `_cmd_btw()` | `commands.py` | Quick side question (bypasses main lock) |
| `/run <cmd>` | `_cmd_run()` | `commands.py` | Execute shell command (30s timeout) |
| `/cd <path>` | `_cmd_cd()` | `commands.py` | Change working directory |
| `/pwd` | `_cmd_pwd()` | `commands.py` | Show current working directory |
| `/ls [path]` | `_cmd_ls()` | `commands.py` | List files (max 60 entries) |
| `/pickup` | `_cmd_pickup()` | `commands.py` | Print commands to resume sessions in terminal |
| `/history [all]` | `_cmd_history()` | `commands.py` | Conversation summaries |
| `/stats [intent]` | `_cmd_stats()` | `commands.py` | Token usage statistics（含 `intent` 子命令）|
| `/upgrade` | `_cmd_upgrade()` | `commands.py` | 更新 larkhelm 到最新版本 |
| `/cron add\|list\|del` | `_cmd_cron()` | `commands.py` | Manage scheduled tasks |
| `/voice [status\|lang <zh\|en\|auto>]` | `_cmd_voice()` | `commands.py` | 查看 / 切换语音转写设置 |
| `/memory [status\|update\|clear\|gc\|export\|import\|diagnose\|observe\|set\|list]` | `_cmd_memory()` | `commands.py` | 记忆系统操作（详见 `commands.py` 内子命令分发）|
| `/crew <task>` | `cmd_crew()` | `crew/_commands.py` | Multi-agent collaborative planning |
| `/dev <task> [--no-confirm]` | `cmd_dev()` | `crew/_commands.py` | Software engineering pipeline |
| `/plan <task>` | `cmd_plan()` | `cmd_plan.py` | 多阶段串行：`[dev]` `[review]` `[fix]` `[test]` |

> **文档操作**：没有 `/doc` 用户命令。读：消息含飞书 URL 时由 `_inject_doc_context` 自动注入内容；写：`larkhelm doc` CLI 或意图识别后的 DocAgent（两者共用 `doc_handlers.py` 的 dispatcher）。命令面入口由 `command_registry.py` 集中注册，硬编码命令保留在 `handlers/_message.py` 顶部（仅 `/cancel` / `/rename` / `/btw` / model shortcut 等需要触发 per-chat 锁 / cancel_event 的特例）。

外部 CLI（不是飞书命令）：

| CLI | 作用 |
|---|---|
| `larkhelm voice probe [--no-benchmark] [--no-write]` | 安装时一次性 probe：检 ffmpeg + CPU flags + RAM + 实测 RTF；自动写回 `config.json` 的 `voice_enabled` / `voice_engine` / `voice_model_size` |
| `larkhelm memory export [output.zip] [--chat-ids ID…] [--data-dir DIR] [--include-debug-log]` | 导出持久化数据到 zip 文件（无需 bridge 在线） |
| `larkhelm memory import <archive.zip> [--replace] [--dry-run] [--data-dir DIR]` | 从 zip 恢复数据；默认合并（state.json merge + JSONL 去重）；`--replace` 覆盖写入 |

## Key Features

### 飞书文档自动注入（Auto Doc Context Injection）

当用户消息中包含飞书文档/Wiki URL 时，`handlers/_query.py` 的 `_inject_doc_context()` 会**自动读取文档内容**并追加到发送给 AI 的上下文中。用户无需任何额外操作——这是 LarkHelm 与原生飞书 AI 集成的核心差异化能力。

支持的 URL 类型：
- `https://xxx.feishu.cn/docx/...` — 新版文档
- `https://xxx.feishu.cn/wiki/...` — Wiki 页面
- `https://xxx.feishu.cn/sheets/...` — 电子表格

> 缓存命中走 `_context_cache.cached_doc_read_with_meta`，注入时会标注「N 分钟前读取」age hint（P4 REQ-05），默认 TTL `doc_inject_cache_ttl_sec=600`（10 min）。

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

> 详细设计见 [`.crew_workspace/phase_d_recall.md`](.crew_workspace/phase_d_recall.md)

关键模块：`memory_embedding.py`（EmbeddingBackend + circuit breaker）、`memory_lifecycle.py`（stale slice + `.meta.json` sidecar）。默认全关（灰度 0%），与 Phase 1 完全 byte-compatible。Hybrid 路径：KeywordRetriever → cosine rerank → `α·BM25_norm`（α=0.6）→ stale decay → top_k。embedding 失败时 fail-open 到 Keyword 路径。

### Dev 模式 Git 快照（Auto Git Commit）

`/dev` 流水线在每个关键阶段完成后，会通过 `_git_auto_commit()` 自动提交变更作为快照，便于查看每步的 diff 和在出错后回滚。

## Adding a New Command

90% 的新命令应该走 `command_registry.py` 注册表，不要再往 `handlers/_message.py` 里加 if/elif 分支（那条老路径已被收敛，目前仅保留 `/cancel` / `/rename` / `/btw` / model shortcut 等需要 per-chat 锁 / cancel_event 的特例）。

**Step 1** — 在 `commands.py` 实现处理函数（同步入参 `DispatchContext`）：

```python
from larkhelm.command_registry import DispatchContext

def _cmd_new_cmd(ctx: DispatchContext) -> None:
    args = ctx.raw_args.strip()
    if not args:
        send_card(ctx.chat_id, "用法", "/new_cmd <arg>", color="orange")
        return
    send_card(ctx.chat_id, "标题", f"已处理：{args}", color="blue")
```

**Step 2** — 在 `command_registry.py` 的 `_default_registrations()` 里 `register()`：

```python
register(CommandSpec(
    name="/new_cmd",
    handler=_lazy_import("larkhelm.commands", "_cmd_new_cmd"),  # 懒 import，避免启动期连锁
    match_kind="prefix",          # "exact" 当命令不带参数；"prefix" 当带参数（自动剥离前缀拿 raw_args）
    aliases=("/nc",),             # 可选别名
    usage_card="/new_cmd <arg>",  # raw_args 为空时自动回的橙色用法卡
    run_async=False,              # IO 阻塞 > 1s 的处理器设为 True，会自动包 daemon thread + 错误卡
    thread_label="new_cmd",       # run_async=True 时显示在错误卡里的标签
    hidden=False,                 # /help 是否列出（M1 增 description/examples 字段后由 help 渲染器消费）
    description="单行命令描述（≤80 字）",          # P1-2a 元数据，仅供未来 help 渲染器与第三方插件消费
    examples=("/new_cmd foo", "/new_cmd bar"),  # 0–3 条可粘贴样例（每条 ≤120 字，不含占位符）
))
```

**Step 3**（可选）— 卡片按钮回调：在 `commands.py::_dispatch_button_cmd` 内加分支（button payload 的 `cmd` 字段触发）。

**Step 4**（可选）— 在 `_cmd_help` 的硬编码 help 字符串里加一行（**M1 已规划自动生成**，届时这步将消失）。

**Step 5**（可选）— 写测试：`tests/test_command_registry.py` 与 `tests/test_pure_*.py` 是 pin 现有契约的位置；新加 spec 至少 cover「`/new_cmd` 触发」与「`/new_cmd ` 空参回 usage_card」两条用例。

> **状态读写**：`from larkhelm.chat_state import _get_chat_state, _set_chat_field`。并发锁：`from larkhelm.concurrency import _get_chat_lock`。日志：`from larkhelm.log import _debug_log, info, warn, error`（前缀按 PascalCase，如 `[NewCmd] …`）。

## Other Extension Points

| Extension point | File | Notes |
|---|---|---|
| Register new Feishu events | `bridge.py` | 在 `main()` builder 链上加 `.register_p2_xxx_yyy(handler)` |
| New Feishu API calls | `lark_client.py` | Follow existing patterns；`BOT_OPEN_ID` 缓存在模块 attribute |
| Card layout | `card_builder.py` | Modify `_make_card()` / `_split_md()`（JSON 2.0 unconditionally）|
| State fields | `chat_state.py` | Add to `_chat_state_store` data structure |
| Permission rules | `perm.py` | Add new permission check logic |
| New backend | `backend_registry.py` | `BACKEND_REGISTRY.register(BackendSpec(...))`，并在 `runner_*.py` 实现 streaming adapter |
| New crew agent type | `agent_hub/builtin/` 子类化 `AgentExecutor`，或通过 `pyproject.toml` `[project.entry-points."larkhelm.agents"]` 暴露 plugin | 详见 [`.crew_workspace/design.md`](.crew_workspace/design.md) §Phase 5 |
| New memory layer | `memory.py` + sidecar（如 `memory_*_slots.py`）+ retriever 接入 | 走 LRU + mtime_ns 缓存协议 |

## 监控集成（Prometheus，P2 REQ-01）

> 完整指标列表（23 条）见 [`.crew_workspace/metrics_reference.md`](.crew_workspace/metrics_reference.md)

`health_endpoint_port > 0` 时，`larkhelm.health_server` 暴露 `/health` `/ready` `/metrics`。`/metrics` 默认走 prometheus-client（需 `pip install -e ".[metrics]"`），缺装或 `metrics_text_legacy=true` 时回退 P1 手写文本。

新增指标时在 `metrics.py` 注册，同步更新 `.crew_workspace/metrics_reference.md`。`health_bind_addr` 默认 `127.0.0.1`，不要直接暴露 `0.0.0.0`。

## 状态模块导入指南

各专属模块分工明确，直接按需导入：

| 需求 | 导入来源 |
|---|---|
| Per-chat 状态（cwd/model/crons） | `larkhelm.chat_state` |
| 并发锁、取消事件 | `larkhelm.concurrency` |
| 日志读写、调试输出 | `larkhelm.log` |
| Token 统计 | `larkhelm.token_stats` |
| 消息去重 | `larkhelm.dedup` |
| Crew 数据类型 | `larkhelm.crew_types` |

## 记忆系统的两条触发路径

`memory.maybe_auto_update(chat_id)` 是后台 LLM 摘要器，把最近对话浓缩成
session memory，再级联抽取 project / global memory。它有**两条触发路径**：

| 触发 | 何时 | 谁触发 | 频率 |
|---|---|---|---|
| 普通节奏 | 普通 `/chat` 查询完成后 | `handlers/_query.py` 内 `maybe_auto_update(chat_id)` 调用点 | 每 `AUTO_UPDATE_EVERY=10` 轮一次 |
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

> Namespace 列表与用法示例见 [`.crew_workspace/lark_sdk_reference.md`](.crew_workspace/lark_sdk_reference.md)

SDK install path: `~/.local/lib/python3.13/site-packages/lark_oapi/`。主要命名空间：`client.docx.v1` / `client.drive.v1` / `client.sheets.v3` / `client.wiki.v2`。

## 异常处理规范

**三类分类标准**（禁止未经分类就引入新的 `except Exception: pass`）：

| 分类 | 处理方式 | 典型场景 |
|---|---|---|
| 高危—业务静默失败 | `_debug_log` + 用户 ⚠️ 卡片 | `/reset` API history 清除失败 |
| 中危—辅助操作失败 | `_debug_log` 记录，不打断主流程 | token 统计、回调、所有权转移、memory 加载 |
| 低危/零危—可接受静默 | 保持 `except Exception: pass` | `proc.kill()`、stderr drain、调试 I/O |
| **第四类**—红色 banner（throttled 强提醒） | `emit_agent_failure` 推红色 banner 卡 + `_debug_log`，**必须** throttle 防 alert fatigue | crew agent 的 `validate` / `backend_select` / `oom` / `timeout` 四个 stage（见 `crew/_failure_card.py:emit_agent_failure`） |

**第四类准入门槛**（新加红 banner 路径前必须满足）：

1. 操作员收到后有具体可执行动作（不是「试试看再跑一次」——那是中危）
2. 已 throttle（per `(subject_id, agent/stage)` 或 LRU），单一事件不会刷屏
3. 文案走 `_safe_error_repr` → `redact_error`，不带 token / 凭证 / 绝对路径

新增 stage 时同步更新 `_failure_card.emit_agent_failure` 顶部 stage 白名单注释，并在 `tests/test_crew_failure_card.py` 加 pin。

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
| `log.py` JSONL 行解析跳过 | 设计意图：容错读取损坏行 |
| `memory.py` `_global_memory_file` chat_state 访问 | 已有明确 fallback（返回 None） |
| `crew/_runner.py` git diff | 非 git 仓库为预期行为 |
| `mcp_server.py` config inner parse | MCP config 行级容错，解析失败继续下一行 |
| `crew/_failure_card.py` `emit_agent_failure` / `emit_terminal_failure` / `emit_breakpoint_timeout` 顶层 try | 错误上报路径「永不抛」契约——这三个 emit 入口本身就是其它路径的失败兜底，再抛只会复合污染。docstring 显式说明；三者均由 `test_crew_failure_card.py::test_*_never_raises_on_lark_error` pinned |

## 变更记录索引

每批 REQ 的简短摘要 + flag 与回滚开关：见 [`.crew_workspace/changes.md`](.crew_workspace/changes.md)
关键子系统的详细设计（Phase 5 / Phase D / Crew task_profile / 断点机制）：见 [`.crew_workspace/design.md`](.crew_workspace/design.md)
PRD / 任务清单占位：[`prd.md`](.crew_workspace/prd.md) / [`tasks.md`](.crew_workspace/tasks.md)

> 新增 REQ 或灰度 flag 时，请同步：`larkhelm_config.example.json` + `config.py` setdefault + `.crew_workspace/changes.md` + （对用户可见时）README.md。本文件 CLAUDE.md 只保留扩展点契约与架构符号入口，不再粘贴变更日志。
