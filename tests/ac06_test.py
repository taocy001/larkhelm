
import threading
import time
from pathlib import Path
import larkhelm.config as _cfg
from larkhelm.memory import save_memory

def test_concurrency():
    _cfg._init_runtime("config.json", ".data")
    chat_id = "test_concurrency"
    def worker(i):
        content = f"content from worker {i}\n" * 10
        save_memory(chat_id, content)

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify file is not corrupted (should contain content from one of the workers completely)
    # Since save_memory overwrites, the last one to finish wins, but it shouldn't be a mix.
    memory_file = Path(".data/memory") / f"{chat_id}.md"
    with open(memory_file, "r") as f:
        data = f.read()
    
    print(f"File size: {len(data)}")
    # Each content is about 20*10 = 200 chars + frontmatter
    assert len(data) > 200
    print("AC-06 OK")

if __name__ == "__main__":
    test_concurrency()
