"""Pin the P1-2c ``/lock`` ↔ ``/model`` split (AC-05).

Before P1-2c, ``/lock`` was registered as an ``aliases=("/lock",)`` entry
on the ``/model`` ``CommandSpec``. The help renderer iterates the registry
by primary name, so the alias route never surfaced as its own help row.

Pulling ``/lock`` out as an independent spec (sharing the same handler)
keeps user-visible dispatch identical while making both entries
discoverable. These tests verify that the default registry resolves both
names to ``_cmd_lock`` exactly as before.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from larkhelm.command_registry import (
    COMMAND_REGISTRY,
    DispatchContext,
)


def _ctx(text: str) -> DispatchContext:
    return DispatchContext(
        chat_id="oc_lock_split",
        msg_id="m_split",
        text=text,
        tl=text.lower().strip(),
    )


class LockAliasSplitTests(unittest.TestCase):

    def test_both_specs_registered_independently(self):
        model_spec = COMMAND_REGISTRY.lookup("/model")
        lock_spec = COMMAND_REGISTRY.lookup("/lock")
        self.assertIsNotNone(model_spec, "/model must be registered")
        self.assertIsNotNone(lock_spec, "/lock must be registered as its own spec")
        # The two specs share the same handler closure (both call _cmd_lock).
        self.assertIs(model_spec.handler, lock_spec.handler,
                      "/model and /lock should share the same handler shim")
        # /lock must not appear as an alias of /model anymore (split, not alias).
        self.assertNotIn("/lock", model_spec.aliases,
                         "/lock should be an independent spec, not an alias of /model")

    def test_lock_bare_dispatches_to_cmd_lock(self):
        with patch("larkhelm.commands._cmd_lock") as m:
            result = COMMAND_REGISTRY.dispatch(_ctx("/lock"))
        self.assertEqual(result, "handled")
        m.assert_called_once()
        # signature: _cmd_lock(chat_id, raw_args, msg_id)
        self.assertEqual(m.call_args.args[0], "oc_lock_split")
        self.assertEqual(m.call_args.args[1], "")

    def test_lock_with_arg_dispatches_with_raw_args(self):
        with patch("larkhelm.commands._cmd_lock") as m:
            result = COMMAND_REGISTRY.dispatch(_ctx("/lock claude"))
        self.assertEqual(result, "handled")
        m.assert_called_once()
        self.assertEqual(m.call_args.args[1], "claude")

    def test_lock_off_dispatches_with_off_arg(self):
        with patch("larkhelm.commands._cmd_lock") as m:
            result = COMMAND_REGISTRY.dispatch(_ctx("/lock off"))
        self.assertEqual(result, "handled")
        m.assert_called_once()
        self.assertEqual(m.call_args.args[1], "off")

    def test_model_still_dispatches_to_cmd_lock(self):
        """The ``/model`` route must continue to reach ``_cmd_lock``
        (they are the same command under two names)."""
        with patch("larkhelm.commands._cmd_lock") as m:
            result = COMMAND_REGISTRY.dispatch(_ctx("/model claude"))
        self.assertEqual(result, "handled")
        m.assert_called_once()
        self.assertEqual(m.call_args.args[1], "claude")


if __name__ == "__main__":
    unittest.main()
