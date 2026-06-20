"""
larkhelm · background task runner

Starts long-running shell commands detached from the main process and
schedules periodic AI progress reports via _do_query.

Task lifecycle:
  start_bg_task()        — called from MCP tool; forks a detached subprocess,
                           writes DATA_DIR/bg_tasks/<id>.json
  _start_bg_watcher()    — (bridge.py) scans task files every 60 s, fires
                           _do_query for periodic / final checks
  remove_task()          — called by watcher after the final check is fired
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BgTask:
    id: str
    pid: int
    cmd: str
    description: str
    log_file: str
    chat_id: str
    model: str
    check_interval_sec: int
    start_time: float
    last_check_time: float = 0.0
    cwd: str = ""


def _task_dir(data_dir: Path) -> Path:
    d = data_dir / "bg_tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_file(data_dir: Path, task_id: str) -> Path:
    return _task_dir(data_dir) / f"{task_id}.json"


def start_bg_task(
    chat_id: str,
    cmd: str,
    description: str,
    model: str,
    cwd: str,
    data_dir: Path,
    check_interval_sec: int = 1800,
) -> BgTask:
    """Fork a detached subprocess, write task descriptor to disk, return BgTask."""
    task_id = uuid.uuid4().hex[:8]
    log_file = str(_task_dir(data_dir) / f"{task_id}.log")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=cwd or None,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach: survive MCP server / terminal exit
            close_fds=True,
        )

    task = BgTask(
        id=task_id,
        pid=proc.pid,
        cmd=cmd,
        description=description,
        log_file=log_file,
        chat_id=chat_id,
        model=model,
        check_interval_sec=check_interval_sec,
        start_time=time.time(),
        last_check_time=0.0,
        cwd=cwd,
    )
    _task_file(data_dir, task_id).write_text(
        json.dumps(asdict(task), ensure_ascii=False, indent=2)
    )
    return task


def load_active_tasks(data_dir: Path) -> list[BgTask]:
    """Return all BgTask descriptors found on disk (one .json file per task)."""
    tasks: list[BgTask] = []
    for f in _task_dir(data_dir).glob("*.json"):
        try:
            d = json.loads(f.read_text())
            fields = BgTask.__dataclass_fields__
            tasks.append(BgTask(**{k: d[k] for k in fields if k in d}))
        except Exception:
            pass
    return tasks


def pid_alive(pid: int) -> bool:
    """True if the process exists and is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_log_tail(log_file: str, max_chars: int = 3000) -> str:
    try:
        text = Path(log_file).read_text(errors="replace")
        return text[-max_chars:] if len(text) > max_chars else text
    except Exception:
        return "(日志文件不可读)"


def build_progress_query(task: BgTask, *, alive: bool) -> str:
    """Build the AI query string for a periodic or final progress check."""
    elapsed = int(time.time() - task.start_time)
    h, m = divmod(elapsed // 60, 60)
    elapsed_str = f"{h}h{m:02d}m" if h else f"{m}m"
    status_str = "仍在运行" if alive else "已结束（进程不再存在）"
    log_tail = _read_log_tail(task.log_file)
    label = task.description or task.cmd

    lines = [
        f"【后台任务{'进度检查' if alive else '完成汇报'}】",
        f"任务描述：{label}",
        f"命令：`{task.cmd}`",
        f"已运行：{elapsed_str}    PID={task.pid}    状态：{status_str}",
        "",
        "最新日志（末尾输出）：",
        "```",
        log_tail,
        "```",
        "",
    ]
    if alive:
        lines.append(
            "请简要汇报当前执行进度，重点关注：① 已完成步骤 ② 是否有报错或警告 "
            "③ 如日志显示了进度百分比或数量请指出。3-5 条即可。"
        )
    else:
        lines.append(
            "任务已结束，请根据日志给出最终结论：成功 / 失败 / 异常，"
            "列出关键输出或错误信息，告知用户任务已完成。"
        )

    return "\n".join(lines)


def update_last_check(task: BgTask, data_dir: Path) -> None:
    """Persist updated last_check_time to the task descriptor file."""
    task.last_check_time = time.time()
    try:
        _task_file(data_dir, task.id).write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2)
        )
    except Exception:
        pass


def remove_task(task: BgTask, data_dir: Path) -> None:
    """Delete the task descriptor file (task is done or manually cancelled)."""
    try:
        _task_file(data_dir, task.id).unlink(missing_ok=True)
    except Exception:
        pass
