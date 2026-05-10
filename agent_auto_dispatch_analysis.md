# LarkHelm 智能 Agent 自动分发系统：可行性分析与价值分析

> **分析对象**：LarkHelm（Feishu ↔ Claude/Gemini/Kimi 桥接平台）
> **核心命题**：从"手动命令驱动"（`/dev`、`/crew`、`/plan`）升级为"自然语言意图驱动"的自动 Agent 选择与模型匹配系统。

---

## 一、技术可行性分析

### 1.1 意图识别层（Intent Recognition）

#### 现状诊断

当前 `larkhelm/handlers/_message.py` 采用**硬编码命令前缀匹配**：

```python
if tl.startswith("/crew"): ...
elif tl == "/dev" or tl.startswith("/dev "): ...
elif tl.startswith("/plan"): ...
```

这是一种 "Command Dispatcher" 模式，优势是确定性高、零误判，但缺陷是：
- 用户必须记忆命令词汇表（`/dev`、`/crew`、`/plan`、`/doc` 等）
- 无法处理模糊需求（如"帮我看看这个代码有没有问题"应该走 `/dev` 还是普通对话？）
- 无层级意图结构（无法区分"写代码"vs"审查代码"vs"修复代码"）

#### 技术路径对比

| 方案 | 原理 | 精度 | 延迟 | 改造成本 | 推荐度 |
|------|------|------|------|----------|--------|
| **A. 规则引擎 + 关键词** | 基于正则/关键词匹配（如现有 `_detect_hermes_mode`） | 中 | 极低 | 低 | ⭐⭐⭐ 可作为 Fallback |
| **B. 轻量分类模型** | 在 cheap/fast 模型（如 Gemini Flash）上附加 System Prompt 做意图分类 | 中高 | 低（~500ms） | 中 | ⭐⭐⭐⭐ **推荐主方案** |
| **C. Embedding 语义检索** | 将用户 query 与预设意图描述做向量相似度匹配 | 中 | 中 | 中 | ⭐⭐⭐ 适合扩展阶段 |
| **D. 端到端 LLM 规划** | 直接让 Orchestrator 模型输出结构化执行计划（类似现有 `/crew` Manager） | 高 | 高（2-5s） | 低 | ⭐⭐⭐⭐⭐ **长期目标** |

#### 推荐架构：双层意图识别

```
┌─────────────────────────────────────────────┐
│  Layer 1: 极速意图预筛（规则引擎）            │  ← 0ms，本地执行
│  - 精确命令匹配（保留 /crew /dev 等向后兼容）  │
│  - 正则快速分类（含"bug"→修复，含"review"→审查）│
│  - 媒体类型触发（图片→vision，文档→analysis）  │
└──────────────────┬──────────────────────────┘
                   │ 未命中规则
┌──────────────────▼──────────────────────────┐
│  Layer 2: 智能意图解析（Cheap Model）         │  ← ~300-800ms
│  调用 gemini-3-flash-lite / cheap backend    │
│  输出结构化意图 JSON：                         │
│  {                                            │
│    "intent": "dev|crew|plan|doc|chat|search", │
│    "sub_intent": "review|fix|implement|test", │
│    "complexity": "simple|medium|complex",     │
│    "required_tags": ["vision","tools"],       │
│    "confidence": 0.92,                        │
│    "reasoning": "用户要求审查代码，属于dev子任务"│
│  }                                            │
└─────────────────────────────────────────────┘
```

**与现有代码的结合点**：

在 `handlers/_message.py` 的命令解析块之前插入 `IntentRecognizer`：

```python
# 现有代码位置（约 line 206-220）
# tl = text.lower().strip()
# 
# 插入意图识别层：
from larkhelm.intent_router import resolve_intent
intent = resolve_intent(text, images=_msg_images, has_doc_urls=bool(doc_urls))
# intent.agent_type → "dev" | "crew" | "plan" | "chat" | "doc" | ...
# intent.target_tags → ["vision", "tools"]
# intent.complexity → "simple" | "complex"
```

**可行依据**：
1. `backend_registry.py` 已定义 `cheap` role，恰好适合做轻量意图分类（成本低、速度快）。
2. `orchestration.py` 的 `build_orchestrator_system_prompt` 已证明：通过 System Prompt 引导模型输出特定格式（DELEGATE 协议）是可靠的。
3. `_detect_hermes_mode`（`crew/_commands.py:124-130`）已验证了"关键词启发式 → 模式选择"的工程可行性。

---

### 1.2 自动分发（Orchestration）

#### 现状诊断

当前系统有两层调度：

1. **Backend 路由**（`router.py`）：基于消息特征选择**单个模型后端**
   - 图片 → vision backend
   - 文档 URL → tools backend
   - 短消息 + `enable_cheap_routing` → cheap backend
   - 用户锁定 → locked backend

2. **委托协议**（`orchestration.py`）：Orchestrator 运行时自主决定**是否委派**给 Worker
   - Orchestrator 输出 `DELEGATE <backend_id>\n<query>\nEND_DELEGATE`
   - `_detect_delegation` 解析并路由到 specialist

这两层是**互补**的：Router 解决"选谁执行"，Orchestrator 解决"是否转发"。但当前二者**没有打通 Agent 语义层**。

#### 目标架构：三层分发模型

```
┌──────────────────────────────────────────────────────────────┐
│  L1: Intent Router（意图路由器）                              │
│  输入：用户自然语言 + 上下文                                   │
│  输出：AgentType + TaskProfile                                 │
│  映射：                                                       │
│    "实现登录功能"        → AgentType=dev,   profile=full_pipeline│
│    "对比两个方案的优劣"   → AgentType=crew,  profile=hermes_race │
│    "先实现再测试再修复"   → AgentType=plan,  profile=multi_stage │
│    "查一下最近的进展"     → AgentType=chat,  profile=quick_lookup│
│    "分析这张架构图"       → AgentType=chat,  profile=vision_deep │
└──────────────────────┬───────────────────────────────────────┘
                       │ TaskProfile
┌──────────────────────▼───────────────────────────────────────┐
│  L2: Agent Dispatcher（Agent 调度器）← 新建模块                │
│  根据 AgentType 实例化对应 Agent：                              │
│    - DevAgent: 执行 PM→Architect→Engineer→QA→Reviewer         │
│    - CrewAgent: 执行 Manager 规划 → 拓扑并行 → 合成             │
│    - PlanAgent: 执行步骤编排 + 人机确认节点                     │
│    - ChatAgent: 直接对话（复用现有 _do_query）                  │
│    - DocAgent: 飞书文档读写                                   │
│    - (PluginAgent): 第三方插件...                             │
└──────────────────────┬───────────────────────────────────────┘
                       │ ExecutionContext
┌──────────────────────▼───────────────────────────────────────┐
│  L3: Backend Selector（模型选择器）← 复用 router.py 并扩展     │
│  基于 TaskProfile.required_tags + complexity + health          │
│  从 BackendRegistry 匹配最优 backend：                          │
│    - complex + code → worker with "tools" + strong model      │
│    - simple + quick → cheap backend                           │
│    - vision + analysis → worker with "vision" tag             │
│    - thinking required → worker with "thinking" tag           │
└──────────────────────────────────────────────────────────────┘
```

#### 与现有代码的结合点

**复用 `orchestration.py`**：
- 当前 `build_orchestrator_system_prompt` 生成的是 "specialist 列表" 用于**模型自决策**。
- 升级为"结构化任务描述"，使 Orchestrator 输出更丰富的调度指令：

```python
# 扩展现有 DELEGATE 协议为完整的 AGENT 协议
AGENT <agent_type>
BACKEND <backend_id>
MODE <mode>
TASK <task_description>
END_AGENT
```

示例：
```
AGENT dev
BACKEND claude
MODE full_pipeline
TASK 实现一个用户登录模块，包含 JWT 认证和密码加密
END_AGENT
```

**新建 `agent_dispatcher.py`**：
核心职责是将 `IntentResult` 映射到 `AgentExecutor`：

```python
class AgentDispatcher:
    _registry: dict[str, AgentExecutor] = {}

    def register(self, agent_type: str, executor: AgentExecutor): ...
    def dispatch(self, intent: IntentResult, ctx: ChatContext) -> AgentResult: ...
```

---

### 1.3 模型自适应选择

#### 现状诊断

`backend_registry.py` 已经建立了非常完善的**能力标签体系**：

```python
@dataclass
class BackendSpec:
    role: str        # "orchestrator" | "worker" | "cheap"
    tags: list[str]  # ["vision", "tools", "cheap", "fast", ...]
    capabilities: str
```

`router.py` 已实现了基于 tags 的基础匹配：
- `get_by_tag(["vision"], prefer_role="orchestrator")`
- `get_by_tag(["cheap", "fast"])`

**但当前体系的问题是**：
1. 标签是**静态的**，无法表达模型能力梯度（如 `gemini-3-pro` 比 `gemini-flash-lite` 更适合复杂推理，但二者都带 `tools`）
2. 没有**复杂度评估**机制（用户说"1+1=？"和"设计一个分布式系统"都匹配 `tools` tag）
3. 缺少**成本感知**（自动在"效果"和"成本"之间做 Pareto 最优选择）

#### 升级方案：Capability Scoring Matrix

在 `BackendRegistry` 中引入**能力评分矩阵**，替代二元 tags：

```python
@dataclasses.dataclass
class BackendSpec:
    # ... 现有字段 ...
    
    # 新增：能力评分（0.0 - 1.0）
    capability_scores: dict[str, float] = dataclasses.field(default_factory=dict)
    # 示例：{"code": 0.95, "vision": 0.9, "chinese": 0.7, "math": 0.85, "speed": 0.6}
    
    # 新增：成本指标（每 1K tokens，USD）
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    # 新增：延迟分级
    latency_tier: str = "medium"  # "instant" | "fast" | "medium" | "slow"
```

**任务-模型匹配算法**（`router.py` 的 `resolve_backend` 扩展）：

```python
def resolve_backend_for_task(
    chat_id: str,
    task_profile: TaskProfile,
) -> BackendSpec:
    """
    多目标优化：在满足 capability 约束的前提下，
    根据复杂度、成本预算、延迟要求选择最优 backend。
    """
    candidates = BACKEND_REGISTRY.match_capabilities(
        required=task_profile.required_capabilities,
        min_score=task_profile.min_quality_score,
    )
    
    # 简单任务 → 优先 cheap + fast（成本敏感）
    if task_profile.complexity == "simple":
        candidates = sort_by(candidates, key=lambda s: (
            -s.capability_scores.get("speed", 0),
            s.cost_per_1k_input,
        ))
    
    # 复杂任务 → 优先代码能力 + 推理深度（质量敏感）
    elif task_profile.complexity == "complex":
        candidates = sort_by(candidates, key=lambda s: (
            -s.capability_scores.get("code", 0),
            -s.capability_scores.get("reasoning", 0),
        ))
    
    # 中等任务 → Pareto 最优（质量/成本平衡）
    else:
        candidates = sort_by(candidates, key=lambda s: (
            -(s.capability_scores.get("general", 0.5) / max(s.cost_per_1k_input, 0.001)),
        ))
    
    return candidates[0] if candidates else fallback_orchestrator()
```

**与现有代码的结合**：
- `BackendRegistry.get_by_tag()` 可保留作为快速路径（向后兼容）。
- 新增 `BackendRegistry.match_capabilities()` 和 `BackendRegistry.score_for_task()` 方法。
- `resolve_backend()` 增加 `task_profile` 可选参数，未传时保持现有行为。

---

### 1.4 Agent 插件化架构

#### 现状诊断

当前 `/dev`、`/crew`、`/plan` 是**硬编码的内置 Agent**：

```
larkhelm/crew/_commands.py   → /crew 命令
larkhelm/crew/_pipeline.py   → /dev 命令
larkhelm/cmd_plan.py         → /plan 命令
```

每个 Agent 的输入输出、状态管理、卡片交互都是**独立实现**的，没有统一接口。

#### 目标架构：Agent Plugin Interface

定义标准接口 `AgentExecutor`，所有 Agent（内置 + 第三方）必须实现：

```python
# larkhelm/agent_base.py（新建）
import abc
from dataclasses import dataclass
from typing import Any

@dataclass
class AgentContext:
    chat_id: str
    user_msg_id: str
    cwd: str
    message: str
    images: list[str] | None
    memory_ctx: str
    # 可扩展：user_profile, project_context, etc.

@dataclass
class AgentResult:
    success: bool
    output: str
    artifacts: list[dict]  # 产出物：文件路径、飞书文档 token 等
    follow_up: str | None  # 如需用户确认，设置此字段
    cost_usd: float
    duration_sec: float

class AgentExecutor(abc.ABC):
    """所有 Agent（内置/插件）必须实现的接口。"""
    
    @property
    @abc.abstractmethod
    def agent_type(self) -> str:
        """唯一标识，如 'dev', 'crew', 'plan', 'search'"""
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """用于意图识别的自然语言描述。"""
        pass
    
    @property
    @abc.abstractmethod
    def required_capabilities(self) -> list[str]:
        """执行此 Agent 所需的 backend 能力标签。"""
        pass
    
    @abc.abstractmethod
    def can_handle(self, intent: IntentResult) -> float:
        """返回 0.0-1.0 的置信度，表示能否处理该意图。"""
        pass
    
    @abc.abstractmethod
    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        """执行 Agent 任务。阻塞/异步均可。"""
        pass
    
    @abc.abstractmethod
    def abort(self, chat_id: str) -> bool:
        """中断指定 chat 的正在执行的任务。"""
        pass
```

**内置 Agent 的改造**：

| 现有模块 | 改造为 AgentExecutor 子类 | 说明 |
|----------|--------------------------|------|
| `crew/_commands.py` | `CrewAgent(AgentExecutor)` | 封装 `cmd_crew()` 逻辑 |
| `crew/_pipeline.py` | `DevAgent(AgentExecutor)` | 封装 `cmd_dev()` 逻辑 |
| `cmd_plan.py` | `PlanAgent(AgentExecutor)` | 封装 `cmd_plan()` 逻辑 |
| `cmd_doc.py` | `DocAgent(AgentExecutor)` | 封装文档读写逻辑 |
| `handlers/_query.py` | `ChatAgent(AgentExecutor)` | 封装普通对话逻辑 |

**第三方 Agent 接入示例**：

```python
# 用户自定义插件：my_search_agent.py
from larkhelm.agent_base import AgentExecutor, AgentContext, AgentResult

class SearchAgent(AgentExecutor):
    agent_type = "search"
    description = "网络搜索与信息检索 Agent，适合获取实时信息、新闻、技术文档"
    required_capabilities = ["tools", "fast"]
    
    def can_handle(self, intent):
        keywords = ["搜索", "查一下", "最新", "新闻", "资料"]
        return 0.9 if any(kw in intent.message for kw in keywords) else 0.1
    
    def execute(self, intent, ctx):
        # 调用 Search API / MCP 工具
        results = web_search(intent.message)
        return AgentResult(success=True, output=results, artifacts=[])
```

**注册机制**（在 `config.py` 初始化时加载）：

```python
# config.py 或新模块 agent_registry.py
from larkhelm.agent_base import AGENT_REGISTRY

# 注册内置 Agent
AGENT_REGISTRY.register(CrewAgent())
AGENT_REGISTRY.register(DevAgent())
AGENT_REGISTRY.register(PlanAgent())
AGENT_REGISTRY.register(ChatAgent())
AGENT_REGISTRY.register(DocAgent())

# 从配置文件加载第三方 Agent
for plugin_path in config.get("agent_plugins", []):
    AGENT_REGISTRY.load_plugin(plugin_path)
```

---

## 二、价值分析

### 2.1 用户体验：从"命令记忆"到"自然对话"

#### 现状痛点

用户需要记忆以下命令词汇：
- `/dev <需求>` — 软件工程
- `/crew <需求>` — 动态多 Agent
- `/plan <需求>` — 多阶段编排
- `/doc read/write/append` — 文档操作
- `/c`、`/g`、`/k` — 模型切换
- `/btw` — 快问

共约 **15+ 条命令**，对新用户不友好。

#### 升级后体验

| 用户输入 | 当前系统 | 自动分发系统 |
|----------|----------|-------------|
| "帮我写个登录模块" | 需输入 `/dev 帮我写个登录模块` | 直接输入，自动识别 → DevAgent |
| "这个代码有 bug 帮我修一下" | 需输入 `/dev 修复代码...` | 直接输入，自动识别 → DevAgent(fix mode) |
| "对比 React 和 Vue 的优劣" | 需输入 `/crew 对比 React 和 Vue` | 直接输入，自动识别 → CrewAgent(hermes_race) |
| "先实现功能，再测试，再修复" | 需输入 `/plan [dev]...[test]...[fix]` | 直接输入，自动识别 → PlanAgent |
| "搜索一下最新的 AI 新闻" | 无对应命令，只能普通对话 | 自动识别 → SearchAgent（插件） |
| "分析一下这张架构图" | 发图片 + 普通对话 | 自动识别图片 + vision backend |

**核心价值**：
- **认知负荷降低 70%**：用户无需记忆命令，像与人对话一样与 AI 协作。
- **错误输入减少**：避免因命令拼写错误（如 `/cre` `/deve`）导致的功能失效。
- **发现性增强**：用户看到"自动选择了 DevAgent"的反馈后，逐渐理解系统能力边界。

### 2.2 效率提升：智能模型切换

#### 现状

当前模型选择依赖：
1. 用户手动 `/model` 或 `/lock`
2. 简单的规则路由（图片→vision、短消息→cheap）

**问题**：用户往往"一锁到底"，用 Claude 回答"1+1="，或用 Gemini Flash 做复杂架构设计。

#### 升级后

```
用户："1+1 等于几？"
系统：Intent=chat, complexity=simple
      → 路由到 gemini-3-flash-lite (cheap, fast)
      → 响应时间 < 1s，成本 ≈ $0.0001

用户："设计一个支持 10万 QPS 的订单系统"
系统：Intent=dev, complexity=complex
      → 路由到 claude / gemini-3-pro (worker, tools, reasoning)
      → DevAgent 启动完整流水线
      → 质量最优，成本 ≈ $0.05-0.20

用户："审查一下这段代码的安全问题"（附带截图）
系统：Intent=dev/review, complexity=medium, has_vision=true
      → 路由到 claude (worker, vision, tools)
      → 能看图 + 能调用安全扫描工具
```

**量化收益估算**（基于 `backend_registry.py` 中的模型成本梯度）：

| 场景 | 当前平均成本 | 优化后平均成本 | 节省 |
|------|-------------|---------------|------|
| 日常简单问答（60% 消息） | $0.003/次（Claude） | $0.0003/次（Flash Lite） | **90%** |
| 中等复杂度任务（30%） | $0.01/次 | $0.005/次（匹配最优 worker） | **50%** |
| 复杂工程任务（10%） | $0.05/次 | $0.05/次（保持最高质量） | 0% |
| **加权平均** | **$0.008/次** | **$0.002/次** | **75%** |

### 2.3 生态扩展性：从"桥接工具"到"企业级 Agent Hub"

#### 战略定位升级

```
当前定位：Feishu ↔ AI CLI 桥接工具
    │
    ├── 功能：多模型接入、命令执行、文档读写
    ├── 用户：个人开发者、小团队
    └── 扩展性：硬编码，每新增功能需改核心代码
    
未来定位：企业级 Agent Hub（智能体编排平台）
    │
    ├── 功能：意图识别 + Agent 市场 + 模型自适应 + 工作流编排
    ├── 用户：企业研发团队、PM、运维、数据分析师
    └── 扩展性：插件化，第三方 Agent 可无缝接入
```

#### 企业场景扩展

| 企业角色 | 所需 Agent | 插件化价值 |
|----------|-----------|-----------|
| **开发工程师** | DevAgent、CodeReviewAgent | 集成 SonarQube、GitHub PR 插件 |
| **产品经理** | PRDAgent、DataAnalysisAgent | 集成 SQL 查询、BI 报表插件 |
| **运维工程师** | AlertAgent、RunbookAgent | 集成 Prometheus、Kubernetes 插件 |
| **安全工程师** | SecurityAuditAgent | 集成 SAST/DAST 扫描插件 |
| **数据分析师** | SQLAgent、VisualizationAgent | 集成 BigQuery、Tableau 插件 |

**竞争壁垒**：
- 飞书生态深度集成（文档、审批、卡片）是 LarkHelm 的核心护城河。
- 自动分发 + 插件化使 LarkHelm 从"个人工具"进化为"团队基础设施"。

---

## 三、实现建议

### 3.1 架构分层建议

```
larkhelm/
├── agent_hub/                    # ← 新增：Agent 编排中心
│   ├── __init__.py
│   ├── intent_router.py          # 意图识别 + 映射
│   ├── agent_dispatcher.py       # Agent 调度器
│   ├── agent_base.py             # AgentExecutor ABC + AgentRegistry
│   ├── model_selector.py         # 基于 TaskProfile 的模型选择（扩展 router.py）
│   └── builtin/                  # 内置 Agent 实现
│       ├── __init__.py
│       ├── chat_agent.py         # 普通对话（从 _query.py 迁移）
│       ├── dev_agent.py          # /dev 流水线（从 crew/_pipeline.py 迁移）
│       ├── crew_agent.py         # /crew 动态规划（从 crew/_commands.py 迁移）
│       ├── plan_agent.py         # /plan 编排（从 cmd_plan.py 迁移）
│       └── doc_agent.py          # 文档操作（从 cmd_doc.py 迁移）
│
├── handlers/
│   ├── _message.py               # 改造：插入 IntentRouter，解耦命令硬编码
│   └── _query.py                 # 改造：ChatAgent 的底层执行器
│
├── router.py                     # 扩展：新增 resolve_backend_for_task()
├── orchestration.py              # 扩展：支持 AGENT 协议（超越 DELEGATE）
├── backend_registry.py           # 扩展：Capability Scoring Matrix
│
└── plugins/                      # ← 新增：第三方 Agent 插件目录（可选）
    └── example_search_agent.py
```

### 3.2 针对现有代码的改造方向

#### 改造 1：`handlers/_message.py` — 解耦命令硬编码

**现状**（line 206-283）：大段 if-elif 命令匹配，新增命令需修改此处。

**改造方案**：

```python
# 新增：在消息处理入口引入 IntentRouter
def handle_message(data):
    # ... 消息解析逻辑不变 ...
    
    # 替代现有的硬编码命令解析：
    intent = resolve_intent(text, images=_msg_images)
    
    # 保留精确命令的向后兼容（/crew /dev /plan 仍可用）
    if intent.is_explicit_command:
        # 用户明确输入了 /dev /crew，直接信任
        pass
    
    # 分发到 AgentDispatcher
    from larkhelm.agent_hub import AGENT_DISPATCHER
    result = AGENT_DISPATCHER.dispatch(intent, ctx=AgentContext(...))
    
    # 根据 AgentResult 更新卡片
    if result.follow_up:
        send_confirm_card(chat_id, result.follow_up)
```

**风险与缓解**：
- **风险**：意图误判导致用户期望的功能未触发。
- **缓解**：在卡片中显式展示"已选择 DevAgent（置信度 92%）"，提供"切换为普通对话"按钮。

#### 改造 2：`router.py` — 扩展为 Task-Aware Router

**现状**：`resolve_backend()` 基于消息特征（图片、文档、长度）做简单匹配。

**改造方案**：

```python
# router.py 新增
def resolve_backend_for_task(
    chat_id: str,
    task_profile: TaskProfile,
) -> BackendSpec:
    """面向 Agent 任务的后端选择。"""
    
    # 1. 用户锁定优先（保留现有行为）
    locked = _get_locked_backend(chat_id)
    if locked: return locked
    
    # 2. 按 TaskProfile 匹配 capability scores
    candidates = BACKEND_REGISTRY.rank_for_task(task_profile)
    
    # 3. 健康检查 + 故障转移（保留现有行为）
    for spec in candidates:
        if spec.healthy and spec.enabled:
            return spec
    
    raise RuntimeError("No healthy backend for task")

# 保留现有 resolve_backend() 作为普通查询的兼容入口
def resolve_backend(chat_id, message, ...):
    # ... 现有逻辑不变 ...
```

#### 改造 3：`orchestration.py` — 从 DELEGATE 升级到 AGENT 协议

**现状**：Orchestrator 只能做简单的 "backend 转发"（DELEGATE → Worker）。

**改造方案**：

```python
# orchestration.py 扩展

_AGENT_PROTOCOL_RE = re.compile(
    r'AGENT\s+(\w+)\s*\n'
    r'(?:BACKEND\s+(\S+)\s*\n)?'
    r'(?:MODE\s+(\w+)\s*\n)?'
    r'TASK\s+(.*?)\s*\n'
    r'END_AGENT',
    re.DOTALL
)

def _detect_agent_dispatch(buffer: str) -> AgentDispatch | None:
    """检测 AGENT 协议，返回结构化调度指令。"""
    m = _AGENT_PROTOCOL_RE.search(buffer)
    if m:
        return AgentDispatch(
            agent_type=m.group(1),
            backend_id=m.group(2),
            mode=m.group(3),
            task=m.group(4).strip(),
        )
    return None
```

这使得 Orchestrator 不仅能选择 backend，还能选择**执行策略**（`MODE=full_pipeline` vs `MODE=quick_fix`）。

#### 改造 4：`backend_registry.py` — 引入 Capability Scoring

**最小侵入式改造**：

```python
# 在现有 BackendSpec 中新增可选字段（不破坏现有配置）
@dataclass
class BackendSpec:
    # ... 现有字段 ...
    
    # 新增：能力评分（配置文件中可选，未设置则基于 tags 推断）
    capability_scores: dict[str, float] = field(default_factory=dict)
    
    # 新增：成本与延迟（用于智能路由）
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    latency_tier: str = "medium"

class BackendRegistry:
    # ... 现有方法 ...
    
    def rank_for_task(self, task: TaskProfile) -> list[BackendSpec]:
        """按任务匹配度排序返回 candidates。"""
        candidates = []
        for spec in self.all_enabled():
            if not spec.healthy:
                continue
            score = self._score_match(spec, task)
            if score > 0:
                candidates.append((score, spec))
        candidates.sort(key=lambda x: -x[0])
        return [s for _, s in candidates]
    
    def _score_match(self, spec: BackendSpec, task: TaskProfile) -> float:
        score = 0.0
        for cap, weight in task.required_capabilities.items():
            spec_score = spec.capability_scores.get(cap, 0.0)
            # 若 capability_scores 未配置，fallback 到 tags
            if spec_score == 0.0 and cap in spec.tags:
                spec_score = 0.7  # tag 匹配给予基础分
            score += spec_score * weight
        return score
```

**配置兼容性**：
- 现有 `larkhelm_config.example.json` 中的 `probe_models` 无需修改即可工作。
- 新增字段为可选，老用户升级无感知。
- 新用户可通过 `capability_scores` 获得更精细的路由体验。

#### 改造 5：`ai_runner.py` / `runner_base.py` — 支持 Agent 级 Session 隔离

**现状**：Crew Agent 已使用 `sid=None` 做会话隔离（防止失败历史污染），但机制是散落在各模块中的。

**改造方案**：
在 `AgentContext` 中统一封装 session 策略：

```python
@dataclass
class AgentContext:
    # ...
    session_policy: str = "inherit"  # "inherit" | "isolated" | "ephemeral"
    
    def get_sid(self, backend_id: str) -> str | None:
        if self.session_policy == "isolated":
            return None  # 强制新会话
        if self.session_policy == "ephemeral":
            return f"agent_{self.chat_id[:8]}_{uuid4().hex[:6]}"
        return _load_sid(self.chat_id, backend_id)  # 继承主会话
```

---

### 3.3 推荐实施路线图

#### Phase 1：意图识别层（2-3 周）

1. **新建 `agent_hub/intent_router.py`**
   - 实现双层识别（规则 + Cheap Model）
   - 输出结构化 `IntentResult`

2. **改造 `handlers/_message.py`**
   - 在命令解析前插入 `resolve_intent()`
   - 保留 `/crew`、`/dev`、`/plan` 精确命令的向后兼容
   - 对自然语言输入，展示"意图识别结果 + 置信度"卡片

3. **A/B 测试**
   - 对 20% 流量开启自动意图识别
   - 收集误判数据，迭代关键词和 Prompt

#### Phase 2：Agent 插件化（3-4 周）

1. **新建 `agent_hub/agent_base.py`**
   - 定义 `AgentExecutor` ABC + `AgentRegistry`

2. **迁移现有功能为内置 Agent**
   - `ChatAgent`（`_query.py` 封装）
   - `DevAgent`（`crew/_pipeline.py` 封装）
   - `CrewAgent`（`crew/_commands.py` 封装）
   - `PlanAgent`（`cmd_plan.py` 封装）

3. **改造 `handlers/_message.py`**
   - 将硬编码的命令分发替换为 `AgentDispatcher.dispatch()`

4. **灰度发布**
   - 先对 `/crew` 和 `/dev` 做自动识别灰度
   - 稳定后全量

#### Phase 3：智能模型选择（2-3 周）

1. **扩展 `backend_registry.py`**
   - 新增 `capability_scores`、`cost_per_1k_*`、`latency_tier`
   - 实现 `rank_for_task()`

2. **扩展 `router.py`**
   - 新增 `resolve_backend_for_task()`
   - 在 `AgentExecutor.execute()` 中调用

3. **配置文件升级**
   - `larkhelm_config.example.json` 中补充 scoring 示例

#### Phase 4：生态扩展（持续）

1. **文档 + Plugin SDK**
   - 发布《LarkHelm Agent 插件开发指南》
   - 提供 `cookiecutter` 模板

2. **内置 Agent 扩展**
   - SearchAgent（搜索插件）
   - SQLAgent（数据库查询插件）
   - AlertAgent（告警处理插件）

3. **企业功能**
   - Agent 权限管理（哪些 Agent 对哪些用户可用）
   - Agent 使用审计与成本分摊
   - Agent 市场（飞书应用商店模式）

---

## 四、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **意图误判** | 用户要求写代码，系统错误识别为普通聊天 | 卡片中展示识别结果+置信度+"切换"按钮；保留 `/dev` 等精确命令作为 Override |
| **模型选择不当** | 复杂任务被路由到 cheap 模型导致质量差 | 复杂度阈值保守设置（宁可过配，不可欠配）；用户可 `/lock` 锁定 |
| **延迟增加** | 意图识别增加一次 LLM 调用（~500ms） | 使用 cheap/fast 模型做识别；规则命中时跳过 LLM 识别；异步预识别 |
| **插件安全风险** | 第三方 Agent 执行恶意代码 | Agent 运行在沙箱中；权限审批继承现有 `perm.py` 机制；插件签名验证 |
| **维护复杂度** | 抽象层过多，调试困难 | 保留详细的 `_debug_log`；IntentResult 全链路透传；卡片中展示路由决策路径 |

---

## 五、结论

**LarkHelm 实现"自动选择 Agent + 自动匹配模型"不仅是可行的，而且是战略上必要的下一步。**

### 可行性总结

| 维度 | 评估 | 关键依据 |
|------|------|----------|
| **意图识别** | ✅ 高度可行 | 现有 `_detect_hermes_mode` + `orchestration.py` 的 DELEGATE 协议已证明 Prompt 引导模型输出结构化决策的可靠性；cheap backend（Gemini Flash Lite）成本极低 |
| **自动分发** | ✅ 高度可行 | 现有 `orchestrator → worker` 的委托机制可直接升级为 `Intent → Agent → Backend` 的三层分发；`crew` 的 DAG 调度器可复用 |
| **模型自适应** | ✅ 高度可行 | `backend_registry.py` 的 role/tags 体系是极佳的基础；只需增加 scoring 维度 |
| **Agent 插件化** | ✅ 可行 | 现有 `/dev`、`/crew`、`/plan` 的抽象层次足够，封装为 `AgentExecutor` 的改造成本中等 |

### 核心价值

1. **用户体验跃迁**：从"学习命令"到"自然对话"，降低使用门槛 70% 以上。
2. **运营成本优化**：简单任务自动路由到 cheap 模型，预估节省 50-75% 的 Token 成本。
3. **战略定位升级**：从"个人 AI 桥接工具"进化为"企业级 Agent Hub"，具备插件生态扩展能力。

### 建议立即启动的 POC

1. **POC-1**：在 `handlers/_message.py` 中插入基于规则的 Intent Router（不调用 LLM，仅用关键词），将 `"帮我写代码"` → 自动触发 `cmd_dev`，验证分发链路。
2. **POC-2**：扩展 `orchestration.py`，让 Orchestrator 输出 `AGENT dev` 协议而非仅 `DELEGATE <backend_id>`，验证 Agent 级调度。
3. **POC-3**：在 `backend_registry.py` 中为 2-3 个 backend 增加 `capability_scores`，实现简单的 `rank_for_task()`，验证模型自适应选择。

三个 POC 可并行推进，预计 1-2 周内即可验证核心假设。
