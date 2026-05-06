import os
import sys

# Setup env
os.environ["LARKHELM_CONFIG"] = os.path.abspath("config.json")
os.environ["LARKHELM_DATA_DIR"] = os.path.abspath(".data")

try:
    import larkhelm.config as cfg
    cfg._init_runtime()
    print("✅ larkhelm.config initialized")

    import larkhelm.dedup as dedup
    assert hasattr(dedup, 'DEDUP_CAP')
    assert hasattr(dedup, '_is_duplicate')
    print("✅ larkhelm.dedup imported")

    import larkhelm.concurrency as concurrency
    assert hasattr(concurrency, 'wait_for_idle')
    print("✅ larkhelm.concurrency imported")

    import larkhelm.log as log
    assert hasattr(log, 'log_entry')
    print("✅ larkhelm.log imported")

    import larkhelm.token_stats as token_stats
    assert hasattr(token_stats, 'record_token_usage')
    print("✅ larkhelm.token_stats imported")

    import larkhelm.chat_state as chat_state
    assert hasattr(chat_state, '_set_chat_field')
    print("✅ larkhelm.chat_state imported")

    import larkhelm.state as state
    # state should re-export everything
    assert hasattr(state, 'DEDUP_CAP')
    assert hasattr(state, 'wait_for_idle')
    assert hasattr(state, 'log_entry')
    assert hasattr(state, 'record_token_usage')
    assert hasattr(state, '_set_chat_field')
    print("✅ larkhelm.state (re-export layer) imported")

    import larkhelm.lark_client as lark_client
    print("✅ larkhelm.lark_client imported")

    print("SMOKE_TEST_PASSED")
except Exception as e:
    print(f"❌ Smoke test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
