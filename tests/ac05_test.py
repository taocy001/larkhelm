
import json
from pathlib import Path
from larkhelm.backend_registry import BackendSpec, BackendRegistry

def test_health_check_failure():
    registry = BackendRegistry()
    # Mock the dict that registry.load expects
    specs_data = [
        {
            "id": "bad-cli",
            "provider": "claude_cli",
            "command": "non-existent-command-xyz",
            "enabled": True
        }
    ]
    registry.load(specs_data)
    registry.health_check()
    
    bad_spec = registry.get("bad-cli")
    print(f"Spec healthy: {bad_spec.healthy}")
    assert bad_spec.healthy == False
    print("AC-05 OK")

if __name__ == "__main__":
    test_health_check_failure()
