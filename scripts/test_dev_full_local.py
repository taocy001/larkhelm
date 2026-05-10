#!/usr/bin/env python3
"""
本地模拟完整 /dev 流水线测试
不发送飞书消息，打印进度到控制台
"""
import sys
import os
import time
import threading
import shutil
from pathlib import Path

# Auto-detect repo root: <repo>/scripts/test_dev_full_local.py → <repo>
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 先 patch card_builder 避免 Python 3.9 union syntax 问题
import larkhelm.card_builder as cb
original = cb._make_card
cb._make_card = lambda *a, **k: original(*a, **k)

from larkhelm.config import _init_runtime
from larkhelm.crew._pipeline import _make_dev_pipeline
from larkhelm.crew_types import AgentState, AgentStatus, CrewState
from larkhelm.crew._runner import _run_agent, _workspace_dir

_init_runtime()

# 清理之前的工作区
TEST_CWD = "/tmp/test_dev_full"
if os.path.exists(TEST_CWD):
    shutil.rmtree(TEST_CWD)
os.makedirs(TEST_CWD, exist_ok=True)

print("=" * 70)
print("完整 /dev 流水线测试 (本地模拟模式)")
print("=" * 70)
print(f"工作目录: {TEST_CWD}")
print()

# 创建 dev pipeline
plan = _make_dev_pipeline(
    requirement="实现一个用户登录模块，包含用户名密码验证和 JWT token 生成",
    cwd=TEST_CWD,
    no_confirm=True,
    skip_planning=False,
)

print(f"流水线: {plan.title}")
print(f"Agent 数量: {len(plan.agents)}")
for a in plan.agents:
    print(f"  - {a.id}: {a.role} [model={a.model}, depends_on={a.depends_on}]")
print()

# 创建 state
state = CrewState(
    crew_id="full_dev_test", chat_id="test_chat", plan=plan,
    agents={a.id: AgentState(spec=a) for a in plan.agents},
    cancel_ev=threading.Event(),
    phase="running", kind="dev",
)

# 按依赖顺序执行
completed = set()
total_start = time.time()

for spec in plan.agents:
    # 检查依赖是否完成
    for dep in spec.depends_on:
        if dep not in completed:
            print(f"❌ 依赖 {dep} 未完成，跳过 {spec.id}")
            continue
    
    print(f"\n{'='*70}")
    print(f"🚀 {spec.id}: {spec.role} [model={spec.model}]")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        result = _run_agent(state, spec.id)
        elapsed = time.time() - start
        
        state.agents[spec.id].status = AgentStatus.DONE
        state.agents[spec.id].result = result
        state.agents[spec.id].end_time = time.time()
        completed.add(spec.id)
        
        print(f"\n✅ DONE in {elapsed:.1f}s ({len(result)} chars)")
        # 打印结果前 200 字符
        preview = result[:200].replace('\n', ' ')
        print(f"   Preview: {preview}...")
        
        # 检查是否有输出文件
        workspace = _workspace_dir("test_chat", "full_dev_test")
        if spec.output_file:
            out_path = workspace / spec.output_file
            if out_path.exists():
                print(f"   📄 Output: {out_path} ({out_path.stat().st_size} bytes)")
        
        # reviewer 的特殊处理
        if spec.id == "reviewer":
            summary_file = workspace / "reviewer_summary.md"
            if summary_file.exists():
                print(f"   📄 Hermes Summary: {summary_file} ({summary_file.stat().st_size} bytes)")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        break

total_elapsed = time.time() - total_start

# 最终报告
print(f"\n{'='*70}")
print("PIPELINE SUMMARY")
print(f"{'='*70}")
for spec in plan.agents:
    a = state.agents[spec.id]
    icon = "✅" if a.status == AgentStatus.DONE else "❌" if a.status == AgentStatus.FAILED else "⏳"
    print(f"{icon} {spec.id}: {spec.role} [{a.status.value}]")

print(f"\nTotal time: {total_elapsed:.1f}s")

# 列出所有产出文件
workspace = _workspace_dir("test_chat", "full_dev_test")
print(f"\nWorkspace: {workspace}")
for f in sorted(workspace.iterdir()):
    size = f.stat().st_size
    print(f"  📄 {f.name}: {size} bytes")

# 列出项目目录下的文件
print(f"\nProject files:")
for root, dirs, files in os.walk(TEST_CWD):
    level = root.replace(TEST_CWD, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"{subindent}{file} ({size} bytes)")
