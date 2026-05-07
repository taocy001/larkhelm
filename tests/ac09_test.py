
import larkhelm.config as _cfg
from larkhelm.api_session import truncate_history

def test_truncation():
    # 1. 42 messages, first is system
    history = [{"role": "system", "content": "I am a bot"}]
    for i in range(41):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"msg {i}"})
    
    assert len(history) == 42
    
    # 2. Truncate
    truncated = truncate_history(history)
    
    # 3. Verify
    print(f"Truncated length: {len(truncated)}")
    assert len(truncated) <= 40
    assert truncated[0]["role"] == "system"
    assert truncated[0]["content"] == "I am a bot"
    # The first few user/assistant messages after system should have been removed
    # If we keep 40, and total was 42, we remove 2.
    # index 1 and 2 should be gone.
    assert truncated[1]["content"] == "msg 2"
    
    print("AC-09 OK")

if __name__ == "__main__":
    test_truncation()
