"""MEM-C1 tests — AC-04.

Verifies that MemoryContextBuilder propagates sender_open_id to the global
memory layer functions so each builder instance reads the correct file.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock


def test_builder_passes_sender_open_id_to_global_layer(tmp_path, monkeypatch):
    """MemoryContextBuilder stores sender_open_id and passes it to _global_memory_file."""
    import larkhelm.memory as mem
    import larkhelm.memory_context as ctx_mod

    monkeypatch.setattr(mem, "MEMORY_HOME_DIR", tmp_path)
    monkeypatch.setattr(mem, "_ensure_dir", lambda: None)

    # Write a global memory file for user_A
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "global_user_A.md").write_text("user A preference\n")

    # Ensure ContextVar and chat_state won't interfere
    token = mem._query_sender_open_id.set("")
    try:
        with patch("larkhelm.memory._get_chat_state", return_value={}):
            # Build with explicit sender_open_id="user_A"
            builder_a = ctx_mod.MemoryContextBuilder(
                chat_id="chat1", cwd=str(tmp_path),
                sender_open_id="user_A",
            )
            # Build with a different explicit sender_open_id="user_B" (no file exists)
            builder_b = ctx_mod.MemoryContextBuilder(
                chat_id="chat1", cwd=str(tmp_path),
                sender_open_id="user_B",
            )

            assert builder_a.sender_open_id == "user_A"
            assert builder_b.sender_open_id == "user_B"

            # _layer_global should produce different results
            ctx_a = builder_a._layer_global_uncached()
            ctx_b = builder_b._layer_global_uncached()

            assert "user A preference" in ctx_a
            assert not ctx_b  # no file for user_B
    finally:
        mem._query_sender_open_id.reset(token)


def test_builder_default_sender_open_id_is_none(tmp_path, monkeypatch):
    """MemoryContextBuilder with no sender_open_id stores None."""
    import larkhelm.memory_context as ctx_mod

    builder = ctx_mod.MemoryContextBuilder(
        chat_id="chat1", cwd=str(tmp_path),
    )
    assert builder.sender_open_id is None
