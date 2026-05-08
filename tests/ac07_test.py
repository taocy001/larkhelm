import atexit
import json
import shutil
import tempfile
from pathlib import Path

import larkhelm.config as _cfg

_TMP = tempfile.mkdtemp(prefix="larkhelm_ac07_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.memory import save_memory, inject_memory, _session_memory_file


def test_memory_injection():
    chat_id = "ac07_test"

    _session_memory_file(chat_id).unlink(missing_ok=True)

    msg = "Hello"
    assert inject_memory(chat_id, msg) == msg

    save_memory(chat_id, "This is a project about testing.")

    enriched = inject_memory(chat_id, msg)
    print(f"Enriched message: {enriched}")
    assert "[SESSION MEMORY]" in enriched
    assert "This is a project about testing." in enriched
    assert "[/SESSION MEMORY]" in enriched
    assert enriched.endswith(msg)

    _session_memory_file(chat_id).unlink(missing_ok=True)
    print("AC-07 OK")


if __name__ == "__main__":
    test_memory_injection()
