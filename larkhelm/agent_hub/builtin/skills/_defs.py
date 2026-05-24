"""larkhelm · agent_hub.builtin.skills._defs — built-in Skill definitions.

All five skills previously implemented as Python Agent classes are defined here
as plain :class:`~larkhelm.agent_hub.skill_types.SkillDef` dicts.  They are
loaded by :func:`register_builtin_skills` in ``builtin/__init__.py``.

Why dicts instead of YAML files?
  - No PyYAML dependency required.
  - Importable directly; easy to patch in tests.
  - Still fully data-driven: adding a new skill = adding a dict entry here
    (or creating a JSON file in DATA_DIR/skills/).

Each dict maps 1:1 to :class:`~larkhelm.agent_hub.skill_types.SkillDef` fields.
See that module for field documentation.
"""
from __future__ import annotations

_BUILTIN_SKILL_DICTS: list[dict] = [
    # ─────────────────────────────────────────────────────────────────
    # translate
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "translate",
        "name": "中英互译",
        "description": "中英互译 / 多语言翻译，走 cheap 后端，仅输出译文",
        "system_prompt": (
            "请将以下内容翻译成目标语言（中文↔英文自动互译；"
            "如明确指定语言则翻译到该语言）。只输出翻译结果，不要解释。"
        ),
        "backend_profile": "chat",
        "context_injectors": [],
        # Regex: strip "翻译XX" / "translate:" prefix so the AI sees clean source text.
        "strip_trigger_pattern": r"^\s*(翻译[一下成到]?|帮我翻译|请翻译|translate[:\s]?)",
        "l1_keywords": [
            {"pattern": "翻译",     "strength": 0.85, "note": "直接翻译请求"},
            {"pattern": "帮我翻译", "strength": 0.90, "note": "高精度"},
            {"pattern": "请翻译",   "strength": 0.88},
            {"pattern": "re:translate[\\s:\\-]", "strength": 0.80, "note": "英文翻译指令"},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # reviewer
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "reviewer",
        "name": "代码审查",
        "description": "代码审查 / checklist review / diff 分析，不跑完整 /dev 流水线",
        "system_prompt": "",           # no extra prefix; model sees raw user text
        "backend_profile": "reviewer", # uses reviewer TaskProfile via model_selector
        "context_injectors": [],
        "strip_trigger_pattern": "",
        "l1_keywords": [
            {"pattern": "帮我review",  "strength": 0.88, "note": "直接指令"},
            {"pattern": "快速review",  "strength": 0.90},
            {"pattern": "看看这段代码", "strength": 0.70},
            {"pattern": "代码有问题吗", "strength": 0.72},
            {"pattern": "review一下",   "strength": 0.85},
            {"pattern": "re:(?:code\\s+)?review", "strength": 0.80, "note": "英文指令"},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # search
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "search",
        "name": "联网搜索",
        "description": "联网搜索（DuckDuckGo / Brave）并将结果注入 AI 上下文，适合「最新 X 是什么」类问题",
        "system_prompt": "",
        "backend_profile": "chat",
        "context_injectors": ["web_search"],
        "strip_trigger_pattern": r"^\s*(帮我搜[一下]?|搜索[一下]?|查[一下]?|search[:\s]?)",
        "l1_keywords": [
            {"pattern": "帮我搜",       "strength": 0.88},
            {"pattern": "搜索一下",     "strength": 0.87},
            {"pattern": "最新版本是什么", "strength": 0.82},
            {"pattern": "最新消息",     "strength": 0.80},
            {"pattern": "re:search\\s+for", "strength": 0.80},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # shell
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "shell",
        "name": "Shell 执行",
        "description": "执行 shell 命令并用 AI 解读输出，适合「运行 X 并告诉我 Y」类请求",
        "system_prompt": "",
        "backend_profile": "chat",
        "context_injectors": ["shell_exec"],
        "strip_trigger_pattern": r"^\s*(帮我运行|运行[一下]?|执行[一下]?|run[:\s]?)",
        "l1_keywords": [
            {"pattern": "帮我运行",       "strength": 0.88},
            {"pattern": "运行这个脚本",   "strength": 0.90},
            {"pattern": "执行命令",       "strength": 0.86},
            {"pattern": "re:^[\\$>]\\s*\\w", "strength": 0.80, "note": "shell prompt prefix"},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # history_search
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "history_search",
        "name": "历史对话搜索",
        "description": "搜索过往对话历史：找之前讨论的内容、脚本、结论等，使用 BM25 检索",
        "system_prompt": "",
        "backend_profile": "chat",
        "context_injectors": ["bm25_history"],
        "strip_trigger_pattern": "",
        "l1_keywords": [
            {"pattern": "上次讨论",   "strength": 0.85},
            {"pattern": "之前说过",   "strength": 0.82},
            {"pattern": "上次你写的", "strength": 0.85},
            {"pattern": "历史记录",   "strength": 0.78},
            {"pattern": "帮我找找之前", "strength": 0.88},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # quick-fix
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "quick-fix",
        "name": "快速定点修复",
        "description": "直接修复指定 bug 或小问题，跳过 PM/架构规划，适合已知定点问题",
        "system_prompt": (
            "请直接修复用户描述的问题。\n"
            "1. 用 Read/Grep/Glob 定位相关代码\n"
            "2. 精准修复，不要顺手重构其他代码\n"
            "3. 修复后运行相关测试验证（如有）\n"
            "4. 简要说明改了什么文件的什么位置"
        ),
        "backend_profile": "engineer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(帮我修[复一下]*|快速修复|定点修复|fix\s+(?:this\s+)?(?:bug\s+)?)",
        "l1_keywords": [
            {"pattern": "帮我修复",      "strength": 0.88, "note": "直接修复请求"},
            {"pattern": "帮我修一下",    "strength": 0.85},
            {"pattern": "快速修复",      "strength": 0.88},
            {"pattern": "定点修复",      "strength": 0.90},
            {"pattern": "修复这个bug",   "strength": 0.92},
            {"pattern": "修复一下这个",  "strength": 0.85},
            {"pattern": "re:fix\\s+(?:this\\s+)?bug\\b", "strength": 0.88, "note": "英文修复指令"},
            {"pattern": "re:quick[\\s-]?fix\\b",         "strength": 0.85},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # explain-code
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "explain-code",
        "name": "代码解释",
        "description": "深度解释代码逻辑、调用链、设计意图，适合「这段代码是做什么的」类请求",
        "system_prompt": (
            "请深度解释用户提供的代码或代码片段，按以下结构输出：\n"
            "1. **功能概述**：一句话说明这段代码做什么\n"
            "2. **关键逻辑**：逐步解释核心流程\n"
            "3. **设计意图**：说明为什么这样写（如有非显而易见之处）\n"
            "4. **注意事项**：潜在边界条件或陷阱"
        ),
        "backend_profile": "reviewer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(解释(一下|下)?(这段|这个|这份)?代码|帮我(读|看)(一下)?代码|explain\s+(?:this\s+)?code[:\s]?)",
        "l1_keywords": [
            {"pattern": "解释这段代码",       "strength": 0.90, "note": "高精度代码解释"},
            {"pattern": "解释一下代码",       "strength": 0.88},
            {"pattern": "解释一下这段",       "strength": 0.82},
            {"pattern": "帮我读一下代码",     "strength": 0.85},
            {"pattern": "帮我看懂",           "strength": 0.80},
            {"pattern": "这段代码是做什么的", "strength": 0.92},
            {"pattern": "这段代码是什么意思", "strength": 0.92},
            {"pattern": "re:explain\\s+(?:this\\s+)?(?:code|function|class|method)\\b", "strength": 0.88},
            {"pattern": "re:what\\s+does\\s+this\\s+(?:code|function|class)\\s+do\\b", "strength": 0.90},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # test-gen
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "test-gen",
        "name": "测试生成",
        "description": "为指定代码生成测试用例（单元测试/集成测试），不跑完整 /dev 流水线",
        "system_prompt": (
            "请为用户指定的代码生成测试用例，要求：\n"
            "1. 覆盖正常路径、边界条件、异常路径\n"
            "2. 遵循现有测试框架风格（pytest/unittest/jest 等，根据项目判断）\n"
            "3. 测试用例命名清晰，注释说明每个用例的意图\n"
            "4. 直接写入文件或输出，不做完整流水线规划"
        ),
        "backend_profile": "qa",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(帮我写(单元)?测试|生成测试用例|写(一下|一个|个)?(单元)?测试|补充(测试用例|单测)|generate\s+(?:unit\s+)?tests?\s*(?:for\s+)?|write\s+(?:unit\s+)?tests?\s*(?:for\s+)?|add\s+(?:unit\s+)?tests?\s*(?:for\s+)?)",
        "l1_keywords": [
            {"pattern": "帮我写测试",     "strength": 0.90},
            {"pattern": "帮我写单元测试", "strength": 0.92},
            {"pattern": "生成测试用例",   "strength": 0.90},
            {"pattern": "写单元测试",     "strength": 0.88},
            {"pattern": "写个测试",       "strength": 0.88, "note": "比 dev 的 写个(0.80) 更强"},
            {"pattern": "写一个测试",     "strength": 0.88},
            {"pattern": "补充测试用例",   "strength": 0.88},
            {"pattern": "补充单测",       "strength": 0.88},
            {"pattern": "re:generate\\s+(?:unit\\s+)?tests?(?:\\s+for)?\\b", "strength": 0.88},
            {"pattern": "re:write\\s+(?:unit\\s+)?tests?(?:\\s+for)?\\b",    "strength": 0.85},
            {"pattern": "re:add\\s+(?:unit\\s+)?tests?(?:\\s+for)?\\b",      "strength": 0.82},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # tech-doc
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "tech-doc",
        "name": "技术文档生成",
        "description": "生成技术文档、README、docstring、接口注释，区别于写回飞书文档（DocAgent）",
        "system_prompt": (
            "请生成用户要求的技术文档，要求：\n"
            "- README：包含项目概述、安装、快速开始、配置说明\n"
            "- docstring：符合语言规范（Python-Google-style / JSDoc / 等）\n"
            "- 接口文档：字段类型、默认值、约束、调用示例\n"
            "保持简洁、准确，面向开发者。"
        ),
        "backend_profile": "chat",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(帮我写(一份)?(README|readme|接口文档|docstring|代码注释)|生成(README|readme|接口文档|注释)|补充(docstring|注释|文档注释)|write\s+(?:a\s+)?(?:README|docstring|api\s+doc)|generate\s+(?:README|docstrings?\s+for|api\s+doc))",
        "l1_keywords": [
            {"pattern": "帮我写README",   "strength": 0.92},
            {"pattern": "帮我写接口文档", "strength": 0.90},
            {"pattern": "帮我写docstring","strength": 0.92},
            {"pattern": "生成接口文档",   "strength": 0.90},
            {"pattern": "补充docstring",  "strength": 0.90},
            {"pattern": "补充注释",       "strength": 0.78, "note": "稍弱，避免拦截通用问题"},
            {"pattern": "写代码注释",     "strength": 0.85},
            {"pattern": "re:(?:write|generate)\\s+(?:a\\s+)?README\\b",      "strength": 0.90},
            {"pattern": "re:(?:write|generate|add)\\s+docstrings?\\b",       "strength": 0.90},
            {"pattern": "re:(?:write|generate)\\s+api\\s+docs?\\b",          "strength": 0.88},
        ],
        "source": "builtin",
    },
]

__all__ = ["_BUILTIN_SKILL_DICTS"]
