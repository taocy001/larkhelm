
import larkhelm.config as _cfg
from larkhelm.memory import save_memory, inject_memory, _session_memory_file

def test_memory_injection():
    _cfg._init_runtime("config.json", ".data")
    chat_id = "ac07_test"

    # 1. Clear existing session memory (new location)
    _session_memory_file(chat_id).unlink(missing_ok=True)

    # 2. Inject without memory (should be unchanged)
    msg = "Hello"
    assert inject_memory(chat_id, msg) == msg

    # 3. Create session memory
    save_memory(chat_id, "This is a project about testing.")

    # 4. Inject with memory — new tags: [SESSION MEMORY] / [/SESSION MEMORY]
    enriched = inject_memory(chat_id, msg)
    print(f"Enriched message: {enriched}")
    assert "[SESSION MEMORY]" in enriched
    assert "This is a project about testing." in enriched
    assert "[/SESSION MEMORY]" in enriched
    assert enriched.endswith(msg)

    # Cleanup
    _session_memory_file(chat_id).unlink(missing_ok=True)
    print("AC-07 OK")

if __name__ == "__main__":
    test_memory_injection()
