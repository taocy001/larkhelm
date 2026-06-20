"""Tests for larkhelm.bg_runner background task management."""
import json
import os
import time
from pathlib import Path

import pytest

from larkhelm.bg_runner import (
    BgTask,
    build_progress_query,
    load_active_tasks,
    pid_alive,
    remove_task,
    start_bg_task,
    update_last_check,
    _task_file,
)


@pytest.fixture()
def data_dir(tmp_path):
    return tmp_path


def _make_task(data_dir, cmd="sleep 999", description="test task"):
    return start_bg_task(
        chat_id="chat_abc",
        cmd=cmd,
        description=description,
        model="claude",
        cwd=str(data_dir),
        data_dir=data_dir,
        check_interval_sec=1800,
    )


class TestStartBgTask:
    def test_creates_task_file(self, data_dir):
        task = _make_task(data_dir)
        try:
            assert _task_file(data_dir, task.id).exists()
        finally:
            os.kill(task.pid, 9)

    def test_task_file_is_valid_json(self, data_dir):
        task = _make_task(data_dir)
        try:
            d = json.loads(_task_file(data_dir, task.id).read_text())
            assert d["id"] == task.id
            assert d["pid"] == task.pid
            assert d["cmd"] == "sleep 999"
        finally:
            os.kill(task.pid, 9)

    def test_creates_log_file(self, data_dir):
        task = _make_task(data_dir)
        try:
            assert Path(task.log_file).exists()
        finally:
            os.kill(task.pid, 9)

    def test_process_actually_running(self, data_dir):
        task = _make_task(data_dir)
        try:
            assert pid_alive(task.pid)
        finally:
            os.kill(task.pid, 9)

    def test_unique_ids(self, data_dir):
        t1 = _make_task(data_dir)
        t2 = _make_task(data_dir)
        try:
            assert t1.id != t2.id
        finally:
            os.kill(t1.pid, 9)
            os.kill(t2.pid, 9)


class TestPidAlive:
    def test_alive_process(self, data_dir):
        task = _make_task(data_dir)
        try:
            assert pid_alive(task.pid) is True
        finally:
            os.kill(task.pid, 9)

    def test_dead_process(self):
        import subprocess, time
        p = subprocess.Popen(["true"])
        p.wait()
        time.sleep(0.05)
        assert pid_alive(p.pid) is False

    def test_nonexistent_pid(self):
        # Very large PID unlikely to exist
        assert pid_alive(999999999) is False


class TestLoadActiveTasks:
    def test_returns_written_task(self, data_dir):
        task = _make_task(data_dir)
        try:
            tasks = load_active_tasks(data_dir)
            ids = [t.id for t in tasks]
            assert task.id in ids
        finally:
            os.kill(task.pid, 9)

    def test_empty_when_no_tasks(self, data_dir):
        assert load_active_tasks(data_dir) == []

    def test_skips_corrupt_json(self, data_dir):
        (data_dir / "bg_tasks").mkdir(exist_ok=True)
        (data_dir / "bg_tasks" / "bad.json").write_text("{not json}")
        assert load_active_tasks(data_dir) == []

    def test_multiple_tasks(self, data_dir):
        t1 = _make_task(data_dir, cmd="sleep 999", description="t1")
        t2 = _make_task(data_dir, cmd="sleep 998", description="t2")
        try:
            tasks = load_active_tasks(data_dir)
            assert len(tasks) == 2
        finally:
            os.kill(t1.pid, 9)
            os.kill(t2.pid, 9)


class TestBuildProgressQuery:
    def _fake_task(self, data_dir):
        return BgTask(
            id="abc12345",
            pid=12345,
            cmd="pytest tests/",
            description="完整测试套件",
            log_file=str(data_dir / "test.log"),
            chat_id="chat_x",
            model="claude",
            check_interval_sec=1800,
            start_time=time.time() - 300,
        )

    def test_alive_query_contains_task_info(self, data_dir):
        task = self._fake_task(data_dir)
        Path(task.log_file).write_text("PASSED 42 tests\n")
        q = build_progress_query(task, alive=True)
        assert "pytest tests/" in q
        assert "完整测试套件" in q
        assert "仍在运行" in q
        assert "PASSED 42 tests" in q
        assert "进度检查" in q

    def test_dead_query_says_finished(self, data_dir):
        task = self._fake_task(data_dir)
        Path(task.log_file).write_text("FAILED 1 error\n")
        q = build_progress_query(task, alive=False)
        assert "已结束" in q
        assert "完成汇报" in q
        assert "FAILED 1 error" in q

    def test_missing_log_file_handled(self, data_dir):
        task = self._fake_task(data_dir)
        # log_file does not exist
        q = build_progress_query(task, alive=True)
        assert "日志文件不可读" in q


class TestUpdateAndRemove:
    def test_update_last_check_persists(self, data_dir):
        task = _make_task(data_dir)
        try:
            before = task.last_check_time
            update_last_check(task, data_dir)
            d = json.loads(_task_file(data_dir, task.id).read_text())
            assert d["last_check_time"] > before
        finally:
            os.kill(task.pid, 9)

    def test_remove_task_deletes_file(self, data_dir):
        task = _make_task(data_dir)
        try:
            assert _task_file(data_dir, task.id).exists()
            remove_task(task, data_dir)
            assert not _task_file(data_dir, task.id).exists()
        finally:
            try:
                os.kill(task.pid, 9)
            except ProcessLookupError:
                pass

    def test_remove_idempotent(self, data_dir):
        task = _make_task(data_dir)
        os.kill(task.pid, 9)
        remove_task(task, data_dir)
        remove_task(task, data_dir)  # should not raise
