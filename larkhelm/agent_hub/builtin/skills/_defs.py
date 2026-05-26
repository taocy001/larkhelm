"""larkhelm · agent_hub.builtin.skills._defs — built-in Skill definitions.

Skills are plain :class:`~larkhelm.agent_hub.skill_types.SkillDef` dicts loaded
by :func:`register_builtin_skills` in ``builtin/__init__.py``.

Adding a new built-in skill = adding a dict entry here (or dropping a JSON file
in DATA_DIR/skills/ for runtime-only skills).

Each dict maps 1:1 to :class:`~larkhelm.agent_hub.skill_types.SkillDef` fields.
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
    # reviewer — upgraded: structured checklist
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "reviewer",
        "name": "代码审查",
        "description": "快速代码审查：按结构化 checklist 检查正确性/安全性/可读性/健壮性，给出改进建议",
        "system_prompt": (
            "请按以下 checklist 对代码做快速审查，**只输出发现的问题**（无问题项可跳过）：\n\n"
            "1. **正确性** — 逻辑是否正确，边界条件是否处理，有无明显 bug\n"
            "2. **安全性** — 有无注入漏洞、越权风险、硬编码凭证、不安全的随机数等\n"
            "3. **可读性** — 命名是否清晰，函数职责是否单一，代码结构是否直观\n"
            "4. **健壮性** — 异常处理是否完备，空值/超时/并发边界是否考虑\n"
            "5. **性能** — 有无明显 N+1、不必要的重复计算或内存占用\n\n"
            "末尾给出最重要的 **1-3 个可操作改进点**，格式：`> 建议：...`"
        ),
        "backend_profile": "reviewer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(帮我\s*review|快速\s*review|review[\s一下]*|代码审查[一下]?|code[\s-]?review)[:\s]*",
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
    # test-gen — upgraded: TDD-aware, Arrange-Act-Assert pattern
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "test-gen",
        "name": "测试生成",
        "description": "为指定代码生成测试用例（单元/集成测试），遵循 AAA 模式，覆盖正常路径、边界和异常",
        "system_prompt": (
            "请为用户指定的代码生成测试用例，要求：\n"
            "1. **覆盖范围**：正常路径、边界条件（空值/零/最大值）、异常路径\n"
            "2. **AAA 结构**：每个测试遵循 Arrange → Act → Assert，先 Read 目标代码再写测试\n"
            "3. **框架适配**：根据项目自动选择 pytest/unittest/jest/vitest 等，风格与现有测试一致\n"
            "4. **命名规范**：`test_<when>_<then>` 或 `it('should ...')` 格式，名称即文档\n"
            "5. **直接写入**：用 Edit/Write 把测试写到对应的 test 文件，不做完整流水线规划"
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
    # ─────────────────────────────────────────────────────────────────
    # refactor — new: structured code refactoring
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "refactor",
        "name": "代码重构",
        "description": "按指定目标重构代码：提取函数/类、消除重复、改善命名、拆分职责，不改变外部行为",
        "system_prompt": (
            "请重构用户指定的代码，遵守以下约束：\n"
            "1. **不改变外部行为** — API / 接口 / 函数签名保持向下兼容\n"
            "2. **小步骤策略** — 提取函数 → 消除重复 → 改善命名 → 拆分职责，每步可独立验证\n"
            "3. **说明每处改动** — 改了什么（文件:行号）、为什么这样改\n"
            "4. **测试验证** — 重构完成后运行现有测试套件（如有），确认无回归\n"
            "5. **不顺手扩展** — 只做用户要求的重构范围，不新增功能"
        ),
        "backend_profile": "engineer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(帮我重构|重构[一下]?|代码重构|refactor[:\s]?)",
        "l1_keywords": [
            {"pattern": "帮我重构",     "strength": 0.90},
            {"pattern": "重构一下",     "strength": 0.88},
            {"pattern": "代码重构",     "strength": 0.88},
            {"pattern": "重构这段代码", "strength": 0.92},
            {"pattern": "重构这个函数", "strength": 0.92},
            {"pattern": "重构这个类",   "strength": 0.90},
            {"pattern": "提取函数",     "strength": 0.82, "note": "具体重构手法"},
            {"pattern": "消除重复代码", "strength": 0.85},
            {"pattern": "re:refactor\\b", "strength": 0.85},
            {"pattern": "re:extract\\s+(?:function|method|class)\\b", "strength": 0.82},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # security-scan — new: OWASP-style security review
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "security-scan",
        "name": "安全扫描",
        "description": "扫描代码安全漏洞：注入、越权、硬编码凭证、不安全依赖等 OWASP Top 10 类问题",
        "system_prompt": (
            "请对以下代码进行安全审查，重点检查（按严重性排序输出）：\n\n"
            "**高危**\n"
            "- 注入漏洞（SQL/命令/LDAP/XPath 注入、XSS、模板注入）\n"
            "- 硬编码凭证（API key、密码、token 直接写在代码里）\n"
            "- 越权访问（缺少鉴权、水平越权、IDOR）\n"
            "- 不安全的反序列化\n\n"
            "**中危**\n"
            "- 敏感数据明文传输或存储（无加密/哈希）\n"
            "- 不安全的随机数（用 random 替代 secrets/os.urandom）\n"
            "- 路径遍历（未校验的文件路径拼接）\n"
            "- 过度信任用户输入\n\n"
            "**低危**\n"
            "- 错误信息泄漏（堆栈跟踪暴露给用户）\n"
            "- 缺少速率限制、日志记录不足\n\n"
            "输出格式：`[高危/中危/低危] 文件:行号 — 问题描述 → 修复建议`\n"
            "无问题则输出「未发现明显安全漏洞」。"
        ),
        "backend_profile": "reviewer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(安全扫描|帮我检查安全|扫描安全问题|security[\s-]?scan[:\s]?)",
        "l1_keywords": [
            {"pattern": "安全扫描",       "strength": 0.92},
            {"pattern": "有没有安全漏洞", "strength": 0.90},
            {"pattern": "安全问题",       "strength": 0.82, "note": "稍弱避免误拦截一般讨论"},
            {"pattern": "有没有注入",     "strength": 0.88},
            {"pattern": "检查安全",       "strength": 0.85},
            {"pattern": "re:security[\\s-]?(?:scan|review|audit|check)\\b", "strength": 0.88},
            {"pattern": "re:(?:check|find|scan)\\s+(?:for\\s+)?(?:security\\s+)?vulnerabilit", "strength": 0.85},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # perf-check — new: performance analysis
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "perf-check",
        "name": "性能分析",
        "description": "分析代码或系统的性能瓶颈：时间复杂度、内存占用、慢查询、锁竞争、N+1 等",
        "system_prompt": (
            "请对以下代码进行性能分析，按以下维度检查：\n\n"
            "1. **算法复杂度** — 时间/空间复杂度是否合理，有无可优化的 O(n²) 或更高\n"
            "2. **I/O 瓶颈** — 数据库 N+1 查询、不必要的同步 I/O、缺少批量操作\n"
            "3. **内存问题** — 大对象不必要的拷贝、内存泄漏风险、不当的缓存策略\n"
            "4. **并发瓶颈** — 锁粒度过大、共享状态竞争、线程/协程饥饿\n"
            "5. **重复计算** — 循环内的不变计算、未缓存的昂贵函数调用\n\n"
            "输出格式：`[严重/一般/优化建议] 文件:行号 — 问题描述 → 优化方向`\n"
            "末尾给出最高优先级的 1-3 个优化点。"
        ),
        "backend_profile": "reviewer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(性能分析|帮我分析性能|检查性能|perf(?:ormance)?[\s-]?(?:check|analysis|review)?[:\s]?)",
        "l1_keywords": [
            {"pattern": "性能分析",       "strength": 0.90},
            {"pattern": "有没有性能问题", "strength": 0.88},
            {"pattern": "性能瓶颈",       "strength": 0.90},
            {"pattern": "太慢了",         "strength": 0.75, "note": "稍弱，可能是一般抱怨"},
            {"pattern": "N+1查询",        "strength": 0.90, "note": "高精度性能术语"},
            {"pattern": "内存泄漏",       "strength": 0.88},
            {"pattern": "re:perf(?:ormance)?\\s+(?:check|analysis|review|issue|problem)\\b", "strength": 0.88},
            {"pattern": "re:(?:too\\s+)?slow\\b",   "strength": 0.68, "note": "英文抱怨，稍弱"},
            {"pattern": "re:memory\\s+leak\\b",     "strength": 0.88},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # error-trace — new: error diagnosis and root cause analysis
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "error-trace",
        "name": "错误溯源",
        "description": "诊断报错或异常：读取堆栈跟踪、定位根因、给出具体修复步骤",
        "system_prompt": (
            "请诊断用户提供的报错或异常，按以下步骤输出：\n\n"
            "1. **错误摘要** — 一句话说明报的是什么错\n"
            "2. **根本原因** — 不是表面错误（如 AttributeError），而是触发它的深层原因\n"
            "3. **定位** — 用 Read/Grep 查看堆栈中指向的源文件，确认具体出错位置\n"
            "4. **修复步骤** — 给出可操作的修复方案（具体到文件和修改内容）\n"
            "5. **预防建议** — 如何避免同类问题再次出现（可选，仅当有明确改善方向时）"
        ),
        "backend_profile": "engineer",
        "context_injectors": [],
        "strip_trigger_pattern": r"^\s*(帮我看看(这个)?报错|排查(一下)?报错|分析(一下)?错误|error[\s-]?trace[:\s]?)",
        "l1_keywords": [
            {"pattern": "报错了",       "strength": 0.82},
            {"pattern": "有报错",       "strength": 0.80},
            {"pattern": "这个报错",     "strength": 0.80},
            {"pattern": "报错信息",     "strength": 0.82},
            {"pattern": "帮我看看报错", "strength": 0.90},
            {"pattern": "排查报错",     "strength": 0.90},
            {"pattern": "异常信息",     "strength": 0.80},
            {"pattern": "堆栈信息",     "strength": 0.85},
            {"pattern": "traceback",    "strength": 0.85, "note": "Python traceback"},
            {"pattern": "re:error[:\\s]", "strength": 0.72, "note": "稍弱，error 本身太泛"},
            {"pattern": "re:(?:stack|error)\\s+trace\\b", "strength": 0.85},
            {"pattern": "re:(?:why|what)\\s+(?:is|does)\\s+this\\s+error\\b", "strength": 0.85},
        ],
        "source": "builtin",
    },
    # ─────────────────────────────────────────────────────────────────
    # standup — new: daily standup notes from history + git log
    # ─────────────────────────────────────────────────────────────────
    {
        "id": "standup",
        "name": "站会摘要",
        "description": "根据近期对话历史和 git log 生成今日站会报告（昨日完成、今日计划、阻塞项）",
        "system_prompt": (
            "请根据提供的上下文（对话历史片段、git log 等）生成今日站会报告。\n\n"
            "格式：\n"
            "**昨日完成**\n"
            "- （列出 1-4 条已完成的工作，面向团队，避免技术细节过多）\n\n"
            "**今日计划**\n"
            "- （列出 1-3 条今天打算完成的工作）\n\n"
            "**阻塞项**\n"
            "- 无（或简短说明阻塞原因及需要谁协助）\n\n"
            "原则：简洁，每项不超过一行，面向团队同步而非技术日志。\n"
            "如果上下文信息不足，可以用 `git log --since='1 day ago' --oneline` 补充。"
        ),
        "backend_profile": "chat",
        "context_injectors": ["bm25_history"],
        "strip_trigger_pattern": r"^\s*(写(一个|个)?站会|生成站会(报告|摘要)?|今天的站会|standup[:\s]?|写(一个|个)?日报|生成日报)",
        "l1_keywords": [
            {"pattern": "站会",         "strength": 0.85, "note": "高频触发词"},
            {"pattern": "站立会议",     "strength": 0.88},
            {"pattern": "写个站会",     "strength": 0.92},
            {"pattern": "今天的站会",   "strength": 0.92},
            {"pattern": "生成站会报告", "strength": 0.92},
            {"pattern": "写个日报",     "strength": 0.88},
            {"pattern": "今天的日报",   "strength": 0.88},
            {"pattern": "re:standup\\b",          "strength": 0.88},
            {"pattern": "re:stand[\\s-]?up\\b",  "strength": 0.85},
            {"pattern": "re:daily\\s+(?:report|update|summary)\\b", "strength": 0.85},
        ],
        "source": "builtin",
    },
]

__all__ = ["_BUILTIN_SKILL_DICTS"]
