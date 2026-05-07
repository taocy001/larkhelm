
import sys
from unittest.mock import MagicMock, patch
from larkhelm.backend_registry import BackendSpec
from larkhelm.backend_api import run_anthropic


def test_run_anthropic_mock():
    spec = BackendSpec(
        id="sonnet-api",
        provider="anthropic_api",
        display_name="Sonnet API",
        role="worker",
        tags=["api"],
        api_key="test-key",
        model="claude-3-5-sonnet-20240620"
    )

    chat_id = "ac10_test"
    message = "Hello API"
    history = []

    # anthropic may not be installed; inject a mock module so the dynamic
    # `import anthropic` inside run_anthropic() succeeds under test.
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    mock_stream = MagicMock()
    mock_stream.__enter__.return_value.text_stream = ["Hello ", "from ", "mock ", "API"]
    mock_client.messages.stream.return_value = mock_stream

    on_text_calls = []
    def on_text(text, status):
        on_text_calls.append(text)

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        output, updated_history = run_anthropic(
            spec, chat_id, message, history, on_text=on_text
        )

    assert output == "Hello from mock API"
    assert len(updated_history) == 2
    assert updated_history[0]["role"] == "user"
    assert updated_history[1]["content"] == "Hello from mock API"
    assert len(on_text_calls) > 0
    assert on_text_calls[-1] == "Hello from mock API"


if __name__ == "__main__":
    test_run_anthropic_mock()
