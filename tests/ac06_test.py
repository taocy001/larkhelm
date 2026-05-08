import atexit
import json
import shutil
import tempfile
import threading
from pathlib import Path

import larkhelm.config as _cfg

_TMP = tempfile.mkdtemp(prefix="larkhelm_ac06_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.memory import save_memory, _session_memory_file


def test_concurrency():
    chat_id = "test_concurrency"

    def worker(i):
        content = f"content from worker {i}\n" * 10
        save_memory(chat_id, content)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    memory_file = _session_memory_file(chat_id)
    data = memory_file.read_text()
    print(f"File size: {len(data)}")
    assert len(data) > 200
    print("AC-06 OK")


if __name__ == "__main__":
    test_concurrency()
