#!/usr/bin/env python3
"""
Local simulation runner for /dev pipeline testing.
Does NOT send Feishu messages; prints progress to console instead.
"""
import sys
import os
import time
import threading
from pathlib import Path

sys.path.insert(0, '/path/to/larkhelm')

from larkhelm.config import _init_runtime
from larkhelm.crew._pipeline import _make_dev_pipeline
from larkhelm.crew_types import AgentSpec, AgentState, AgentStatus, CrewPlan, CrewState
from larkhelm.crew._runner import _run_agent, _workspace_dir


def mock_send_card(chat_id, title, content, color="blue"):
    """Mock send_card that prints to console instead of sending to Feishu."""
    print(f"\n{'='*60}")
    print(f"📨 [{chat_id}] {title}")
    print(f"{'='*60}")
    print(content[:500])
    if len(content) > 500:
        print("...")
    print()


def mock_patch_card_raw(card_mid, card_json):
    """Mock patch_card_raw that prints card update to console."""
    import json
    try:
        data = json.loads(card_json)
        title = data.get("header", {}).get("title", {}).get("content", "?")
        print(f"🔄 Card update: {title}")
    except:
        print(f"🔄 Card update (raw)")


def run_dev_local(requirement: str, cwd: str = "/tmp/test_dev", no_confirm: bool = True):
    """Run /dev pipeline locally without Feishu integration."""
    
    # Initialize config
    _init_runtime()
    
    # Monkey-patch lark_client functions to avoid Feishu API calls
    import larkhelm.lark_client as lark_client
    lark_client.send_card = mock_send_card
    lark_client._reply_card_raw = lambda *a, **k: None
    lark_client._send_card_raw = lambda *a, **k: None
    lark_client._patch_card_raw = mock_patch_card_raw
    lark_client._pin_task_card = lambda *a, **k: None
    
    # Create plan
    plan = _make_dev_pipeline(requirement, cwd, no_confirm=no_confirm, skip_planning=False)
    
    print(f"\n{'#'*70}")
    print(f"# DEV PIPELINE: {plan.title}")
    print(f"# Requirement: {requirement}")
    print(f"# Agents: {len(plan.agents)}")
    print(f"{'#'*70}\n")
    
    for i, spec in enumerate(plan.agents, 1):
        print(f"  {i}. {spec.id}: {spec.role} [model={spec.model}] → {spec.depends_on or '（无依赖）'}")
    
    # Create CrewState
    crew_id = "local_test"
    chat_id = "local_chat"
    cancel_ev = threading.Event()
    
    state = CrewState(
        crew_id=crew_id,
        chat_id=chat_id,
        plan=plan,
        agents={spec.id: AgentState(spec=spec) for spec in plan.agents},
        cancel_ev=cancel_ev,
        phase="planned",
        kind="dev",
    )
    
    # Execute agents in topological order (simplified, no parallel execution)
    print(f"\n{'='*70}")
    print("STARTING EXECUTION")
    print(f"{'='*70}\n")
    
    completed = set()
    failed = set()
    
    for wave in _topological_waves(plan.agents):
        print(f"\n--- Wave: {[s.id for s in wave]} ---\n")
        
        for spec in wave:
            if cancel_ev.is_set():
                print("CANCELLED")
                return
            
            # Check dependencies
            if any(dep in failed for dep in spec.depends_on):
                print(f"⏭️  {spec.id}: SKIPPED (upstream failed)")
                state.agents[spec.id].status = AgentStatus.FAILED
                state.agents[spec.id].error = "upstream failed"
                failed.add(spec.id)
                continue
            
            # Skip trigger_only on first pass
            if spec.trigger_only:
                print(f"⏭️  {spec.id}: SKIPPED (trigger_only, will run if needed)")
                state.agents[spec.id].status = AgentStatus.DONE
                completed.add(spec.id)
                continue
            
            print(f"🚀 {spec.id}: {spec.role} [model={spec.model}] STARTING...")
            print(f"   timeout={spec.timeout}s")
            
            start = time.time()
            try:
                result = _run_agent(state, spec.id)
                elapsed = time.time() - start
                
                state.agents[spec.id].status = AgentStatus.DONE
                state.agents[spec.id].result = result
                state.agents[spec.id].end_time = time.time()
                completed.add(spec.id)
                
                print(f"✅ {spec.id}: DONE in {elapsed:.1f}s ({len(result)} chars)")
                print(f"   Preview: {result[:200]}...")
                
                # Check exit markers for QA/reviewer
                if spec.exit_marker:
                    if spec.exit_marker in result:
                        print(f"   🎯 Exit marker found: {spec.exit_marker}")
                    else:
                        print(f"   ⚠️  Exit marker NOT found: {spec.exit_marker}")
                        
            except Exception as e:
                elapsed = time.time() - start
                state.agents[spec.id].status = AgentStatus.FAILED
                state.agents[spec.id].error = str(e)
                failed.add(spec.id)
                print(f"❌ {spec.id}: FAILED after {elapsed:.1f}s: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("EXECUTION SUMMARY")
    print(f"{'='*70}\n")
    
    for spec in plan.agents:
        a = state.agents[spec.id]
        icon = "✅" if a.status == AgentStatus.DONE else "❌" if a.status == AgentStatus.FAILED else "⏭️"
        print(f"{icon} {spec.id}: {spec.role} [{a.status.value}]")
        if a.result:
            print(f"   Result: {a.result[:100]}...")
        if a.error:
            print(f"   Error: {a.error[:100]}")
    
    print(f"\nCompleted: {len(completed)}/{len(plan.agents)}")
    print(f"Failed: {len(failed)}/{len(plan.agents)}")
    
    return state


def _topological_waves(agents: list[AgentSpec]):
    """Simple topological sort into waves."""
    remaining = {a.id: set(a.depends_on) for a in agents}
    waves = []
    
    while remaining:
        wave = [a for a in agents if a.id in remaining and not remaining[a.id]]
        if not wave:
            # Cycle or missing dependency
            break
        waves.append(wave)
        for spec in wave:
            del remaining[spec.id]
        for deps in remaining.values():
            deps.difference_update({spec.id for spec in wave})
    
    return waves


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("requirement", help="Task requirement")
    parser.add_argument("--cwd", default="/tmp/test_dev")
    parser.add_argument("--no-confirm", action="store_true", default=True)
    args = parser.parse_args()
    
    run_dev_local(args.requirement, args.cwd, args.no_confirm)
