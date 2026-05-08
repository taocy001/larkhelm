
import larkhelm.config as _cfg
from larkhelm.api_session import truncate_history

def test_truncation():
    # 1. 42 messages, first is system — all short, well within 80K token budget
    history = [{"role": "system", "content": "I am a bot"}]
    for i in range(41):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"msg {i}"})

    assert len(history) == 42

    # 2. Truncate
    truncated = truncate_history(history)

    # 3. Verify: short messages are within the 80K token budget, so nothing is removed
    print(f"Truncated length: {len(truncated)}")
    assert len(truncated) == 42, f"Expected 42 (no truncation for short messages), got {len(truncated)}"
    assert truncated[0]["role"] == "system"
    assert truncated[0]["content"] == "I am a bot"

    print("AC-09 OK")

if __name__ == "__main__":
    test_truncation()
