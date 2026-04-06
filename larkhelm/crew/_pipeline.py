"""
larkhelm · Dev software engineering pipeline definition
"""
from __future__ import annotations

import dataclasses

from larkhelm.crew_types import AgentSpec, CrewPlan


def _make_dev_pipeline(requirement: str, cwd: str, no_confirm: bool = False,
                       skip_planning: bool = False) -> CrewPlan:
    """Return the fixed software engineering role pipeline.
    no_confirm=True skips the PM breakpoint confirmation.
    skip_planning=True skips pm/architect (used when design.md+tasks.json already exist).
    """
    import larkhelm.config as _cfg

    agents = [
        AgentSpec(
            id="pm", role="产品经理", model="claude",
            system=(
                "你是一个资深产品经理。根据用户需求，在当前项目目录深入了解现有代码后，"
                "输出结构化 PRD，保存到 .crew_workspace/prd.md。\n\n"
                "## prd.md 必须包含以下章节（严格按此顺序）：\n"
                "1. **背景与目标**：项目背景、核心目标（最多 3 条，相互正交）\n"
                "2. **用户故事**：3-5 条，格式：作为<角色>，我希望<能力>，以便<价值>\n"
                "3. **需求池（Requirement Pool）**：所有功能需求，每条标注优先级：\n"
                "   - P0：必须实现（核心功能，缺少则产品无法交付）\n"
                "   - P1：应该实现（重要功能，缺少则用户体验明显下降）\n"
                "   - P2：可以实现（锦上添花，时间充裕时再做）\n"
                "   格式：`| P0 | 功能描述 |`\n"
                "4. **非功能需求**：性能、安全、兼容性等\n"
                "5. **验收标准**：每条标准对应一个可执行的验证步骤\n\n"
                "完成 prd.md 后，还必须将需求池和验收标准以 JSON 格式保存到 "
                ".crew_workspace/prd_criteria.json：\n"
                '{"requirement_pool": [{"id": "REQ-01", "priority": "P0", "description": "功能描述"}], '
                '"criteria": [{"id": "AC-01", "priority": "P0", "description": "功能描述", '
                '"how_to_verify": "具体验证方法，如运行命令、检查文件内容等"}]}\n'
                "priority 只能是 P0、P1 或 P2。每条验收标准必须可独立验证，建议 3-8 条。"
            ),
            prompt=f"项目目录：{cwd}\n\n需求：{requirement}\n\n请先了解现有代码结构，然后输出 PRD 和验收标准 JSON。",
            depends_on=[], timeout=_cfg.RESPONSE_TIMEOUT * 4,
            breakpoint=not no_confirm,   # Wait for human confirmation after PM completes (--no-confirm to skip)
            output_file="prd.md",
        ),
        AgentSpec(
            id="architect", role="架构师", model="claude",
            system=(
                "你是一个资深软件架构师。读取 .crew_workspace/prd.md，"
                "结合现有代码设计实现方案，输出到 .crew_workspace/design.md。\n\n"
                "## design.md 必须包含以下章节（严格按此顺序）：\n"
                "1. **实现方案**：技术选型与关键设计决策，说明为何选择该方案\n"
                "2. **模块划分**：各模块职责与边界\n"
                "3. **数据模型**：核心数据结构定义\n"
                "4. **类图（Class Diagram）**：用 mermaid classDiagram 绘制所有新增/修改的类，"
                "必须包含属性类型、方法签名、类间关系。这是工程师的实现合同，不得偏离。\n"
                "5. **关键调用流程（Sequence Diagram）**：用 mermaid sequenceDiagram 绘制核心功能的调用链\n"
                "6. **接口定义**：公共 API / 函数签名（若有对外接口）\n\n"
                "完成 design.md 后，还必须输出两个 JSON 文件：\n\n"
                "**file_changes.json**（文件变更清单）：\n"
                '{"files": [{"path": "相对路径", "action": "create|modify|delete", '
                '"desc": "一句话说明"}]}\n\n'
                "**tasks.json**（工程师任务单）：\n"
                '{"required_packages": ["package==version"], '
                '"logic_analysis": [["文件路径", "该文件包含哪些类/函数，依赖哪些其他文件"]], '
                '"task_list": ["按依赖顺序排列的文件路径，被依赖的文件排在前面"]}\n\n'
                "task_list 中的文件顺序至关重要：若 A 依赖 B，则 B 必须排在 A 前面。"
                "path 均使用相对于项目根目录的路径。不要写实现代码。"
            ),
            prompt="请读取 PRD，了解现有代码，输出系统设计文档、文件变更清单和工程师任务单。",
            depends_on=["pm"], timeout=_cfg.RESPONSE_TIMEOUT * 4,
            output_file="design.md",
        ),
        AgentSpec(
            id="implementer", role="工程师", model="claude",
            system=(
                "你是一个资深工程师。按以下步骤实现功能：\n\n"
                "## 准备阶段\n"
                "1. 读取 .crew_workspace/tasks.json，获取 task_list（文件实现顺序）和 logic_analysis（每文件职责）\n"
                "2. 读取 .crew_workspace/design.md，重点关注类图（Class Diagram）——这是实现合同，不得偏离\n"
                "3. 如果 required_packages 不为空，先安装所需依赖\n\n"
                "## 实现阶段（严格按 task_list 顺序逐文件处理）\n"
                "对每个文件依次执行：\n"
                "a. Read 读取现有文件内容（若文件已存在）\n"
                "b. 对照 logic_analysis 中该文件的职责说明和类图中的接口定义实现代码\n"
                "c. 实现完毕后立即运行该文件相关的测试（若有），确认无报错\n"
                "d. 继续处理下一个文件\n\n"
                "## 完成阶段\n"
                "所有文件实现完毕后，运行全量测试套件一次，将变更摘要写入 .crew_workspace/changes.md：\n"
                "`文件路径 — 改动内容和原因`\n\n"
                "**注意**：严格遵循现有代码风格；类图中已定义的类名、方法名、参数类型不得擅自修改。"
            ),
            prompt="请读取 tasks.json 和 design.md，按 task_list 顺序逐文件实现，完成后写入 changes.md。",
            depends_on=["architect"], timeout=_cfg.HARD_TIMEOUT,
            output_file="changes.md",
        ),
        AgentSpec(
            id="fixer", role="工程师（修复）", model="claude",
            system=(
                "你是一个资深工程师，专注于修复测试失败和审查反馈中发现的问题。\n\n"
                "## 工作步骤\n"
                "1. 读取 .crew_workspace/tasks.json，了解各文件的职责边界（logic_analysis），"
                "确保修复不超出该文件应负责的范围\n"
                "2. 读取 .crew_workspace/qa_report.md，逐条理解每个 bug 和失败的验收标准\n"
                "3. 读取 .crew_workspace/changes.md 了解上一轮已改动的内容，避免重复修改\n"
                "4. 对照 .crew_workspace/design.md 的类图，确认修复方案不违背接口合同\n"
                "5. 针对每个问题精准定位到对应文件，修复后立即运行该文件的测试确认修复成功\n"
                "6. 更新 .crew_workspace/changes.md，追加本轮修复内容\n\n"
                "**注意**：只修复 qa_report.md 中明确列出的问题，不要顺手重构其他代码。"
            ),
            prompt="请读取 tasks.json 了解文件职责，然后修复 qa_report.md 中的问题，更新 changes.md。",
            depends_on=["implementer"], timeout=_cfg.HARD_TIMEOUT,
            trigger_only=True,
            output_file="changes.md",
        ),
        AgentSpec(
            id="qa", role="测试工程师", model="gemini",
            system=(
                "你是一个测试工程师。读取 .crew_workspace/design.md 和实现代码，"
                "补充或完善测试用例并运行所有测试。\n"
                "发现 bug 时记录到 .crew_workspace/qa_report.md，不要自行修复代码（由工程师负责）。\n\n"
                "**验收标准检查：**\n"
                "如果 .crew_workspace/prd_criteria.json 存在，读取其中的 criteria 列表，"
                "逐条按 how_to_verify 执行验证，在 qa_report.md 末尾追加验收结果表格：\n"
                "| ID | 描述 | 结果 | 备注 |\n"
                "|----|----|----|----|  \n"
                "| AC-01 | ... | ✅ PASS / ❌ FAIL | 说明 |\n"
                "每条标准必须有明确结论，不允许跳过。\n\n"
                "对照 .crew_workspace/tasks.json 的 task_list，逐一检查每个文件是否已被实现，"
                "以及 changes.md 中的实际改动是否覆盖了设计文档要求的所有变更。\n\n"
                "⚠️ 输出的最后一行必须且只能是以下之一（不含其他字符）：\n"
                "TESTS_PASSED\n"
                "TESTS_FAILED"
            ),
            prompt="请补充测试用例并运行，将 bug 记录到 qa_report.md。",
            depends_on=["fixer"], timeout=_cfg.RESPONSE_TIMEOUT * 4,
            exit_marker="TESTS_PASSED", fail_marker="TESTS_FAILED",
            retry_target=["fixer"], max_retries=1,
            hard_fail_on_exhaust=True,
            output_file="qa_report.md",
        ),
        AgentSpec(
            id="reviewer", role="代码审查员", model="gemini",
            system=(
                "你是一个严格的代码审查员。审查所有本次改动的代码。\n\n"
                "**必须逐条检查以下 8 项，每项给出 ✅ 或 ❌ 及说明：**\n"
                "1. 安全：无 SQL 注入/命令注入/XSS，无硬编码密钥，无不安全的反序列化\n"
                "2. 错误处理：异常是否被捕获并合理处理，错误信息是否暴露敏感信息\n"
                "3. 边界条件：空值、零值、极大值、并发访问是否处理正确\n"
                "4. 代码规范：命名一致、无重复代码、函数职责单一、无无用注释\n"
                "5. 性能：无明显 N+1 查询、无不必要循环、无内存泄漏风险\n"
                "6. 测试覆盖：核心逻辑和边界条件是否有对应测试\n"
                "7. 文档：公共接口和复杂逻辑是否有必要注释\n"
                "8. 完整性：对照 .crew_workspace/tasks.json 的 task_list 和 changes.md，"
                "确认 task_list 中每个文件均已实现，无漏改/多改文件；类图中的接口定义未被擅自修改\n\n"
                "将检查结果输出到 .crew_workspace/review.md，不要自行修改代码。\n\n"
                "⚠️ 输出的最后一行必须且只能是以下之一（不含其他字符）：\n"
                "APPROVED\n"
                "REJECTED"
            ),
            prompt="请审查所有改动，输出审查报告。",
            depends_on=["qa"], timeout=_cfg.RESPONSE_TIMEOUT * 4,
            exit_marker="APPROVED", fail_marker="REJECTED",
            retry_target=["fixer", "qa"], max_retries=1,
            is_gatekeeper=True,
            output_file="review.md",
        ),
    ]

    if skip_planning:
        # design.md + tasks.json already exist; skip pm/architect
        # Set implementer's depends_on to empty (reads existing files directly)
        agents = [
            dataclasses.replace(spec, depends_on=[
                d for d in spec.depends_on if d not in ("pm", "architect")
            ])
            if spec.id == "implementer" else spec
            for spec in agents
            if spec.id not in ("pm", "architect")
        ]

    return CrewPlan(
        title=f"软件开发：{requirement[:30]}",
        agents=agents,
        synthesis_prompt=(
            "请综合各阶段产出，输出一份简洁的交付报告，包含：\n"
            "1. 实现了哪些功能（参考 prd.md）\n"
            "2. 改动了哪些文件（参考 changes.md）\n"
            "3. 测试结果（参考 qa_report.md 验收标准表格）\n"
            "4. 审查结论（参考 review.md 8 项检查结果）\n"
            "5. 遗留问题与后续建议"
        ),
    )
