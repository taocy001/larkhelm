"""MEM-C1 tests — AC-05.

Verifies that _do_query accepts and propagates sender_open_id to
get_memory_context_v2, so the correct global memory is read during query.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock


def test_do_query_accepts_sender_open_id_param():
    """_do_query has a sender_open_id keyword parameter."""
    from larkhelm.handlers._query import _do_query
    sig = inspect.signature(_do_query)
    assert "sender_open_id" in sig.parameters


def test_do_query_source_passes_sender_open_id_to_memory():
    """Verify _do_query source code passes sender_open_id to get_memory_context_v2."""
    import ast
    import pathlib

    src = pathlib.Path("larkhelm/handlers/_query.py").read_text()
    tree = ast.parse(src)

    found_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Look for get_memory_context_v2(..., sender_open_id=...)
        func = node.func
        func_name = (func.attr if isinstance(func, ast.Attribute) else
                     func.id if isinstance(func, ast.Name) else "")
        if func_name != "get_memory_context_v2":
            continue
        kw_names = [kw.arg for kw in node.keywords]
        if "sender_open_id" in kw_names:
            found_call = True
            break

    assert found_call, (
        "_do_query does not call get_memory_context_v2 with sender_open_id keyword"
    )
