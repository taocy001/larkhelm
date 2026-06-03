"""Tests for workspace_snapshot MCP tool registration (AC-06)."""
import inspect
import unittest


class TestMcpSnapshotImport(unittest.TestCase):
    def test_import_no_error(self):
        """Importing mcp_server must not raise ImportError (AC-06 import check)."""
        try:
            import larkhelm.mcp_server  # noqa: F401
        except ImportError as e:
            self.fail(f"ImportError when importing larkhelm.mcp_server: {e}")

    def test_workspace_snapshot_defined_in_run(self):
        """workspace_snapshot function is defined inside mcp_server.run (AC-06)."""
        import larkhelm.mcp_server as ms

        # Check that 'workspace_snapshot' appears in the source of run()
        source = inspect.getsource(ms.run)
        self.assertIn("workspace_snapshot", source,
                      "workspace_snapshot tool is not defined in mcp_server.run()")

    def test_generate_workspace_snapshot_importable(self):
        """generate_workspace_snapshot is importable from workspace_finalize."""
        from larkhelm.workspace_finalize import generate_workspace_snapshot
        self.assertTrue(callable(generate_workspace_snapshot))


if __name__ == "__main__":
    unittest.main()
