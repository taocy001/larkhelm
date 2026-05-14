"""Test-mode bootstrap: must run before any test module imports
larkhelm.config or larkhelm.bridge. Sets LARKHELM_TEST_MODE=1 so
_init_runtime skips network probes / daemon threads, making pytest --co
fast (REQ-07)."""

import os

os.environ.setdefault("LARKHELM_TEST_MODE", "1")
