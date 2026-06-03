"""Tests for add_doc_collaborators and the feishu_doc_collaborators config hook
wired into create_doc / create_wiki_node (M-DOC-PERM).
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch, call


class AddDocCollaboratorsTests(unittest.TestCase):
    """Direct tests for FeishuDocClient.add_doc_collaborators."""

    def _make_client(self):
        from larkhelm.lark_client import FeishuDocClient
        return FeishuDocClient.__new__(FeishuDocClient)

    def test_add_doc_collaborators_calls_api_per_open_id(self):
        client = self._make_client()
        with patch.object(client, "_call_api") as mock_api:
            mock_api.return_value = {}
            client.add_doc_collaborators("tok123", ["oc_alice", "oc_bob"], perm="view", doc_type="docx")
        self.assertEqual(mock_api.call_count, 2)
        urls = [c.args[1] for c in mock_api.call_args_list]
        self.assertTrue(all("tok123" in u for u in urls))

    def test_add_doc_collaborators_warns_on_api_error(self):
        from larkhelm.lark_client import DocAPIError
        client = self._make_client()
        with patch.object(client, "_call_api", side_effect=DocAPIError(403, "no perm")):
            with patch("larkhelm.lark_client.warn") as mock_warn:
                # Should NOT raise — errors are warned and skipped
                client.add_doc_collaborators("tok", ["oc_x"], perm="view", doc_type="docx")
        mock_warn.assert_called_once()

    def test_add_doc_collaborators_skips_empty_open_ids(self):
        client = self._make_client()
        with patch.object(client, "_call_api") as mock_api:
            mock_api.return_value = {}
            client.add_doc_collaborators("tok", ["", "oc_real", ""], perm="view", doc_type="docx")
        # Only the non-empty open_id should be processed
        self.assertEqual(mock_api.call_count, 1)


class CreateDocCallsCollaboratorsTests(unittest.TestCase):
    """Integration-style tests: create_doc / create_wiki_node should call
    add_doc_collaborators when FEISHU_DOC_COLLABORATORS is non-empty."""

    def _make_client(self):
        from larkhelm.lark_client import FeishuDocClient
        return FeishuDocClient.__new__(FeishuDocClient)

    def _stub_tenant_create_doc(self, client, doc_id: str = "docXYZ"):
        """Patch out the heavy SDK + HTTP parts of create_doc (tenant path)."""
        import larkhelm.lark_client as _lc
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.document.document_id = doc_id
        mock_create = MagicMock(return_value=mock_resp)
        return mock_create

    def test_create_doc_calls_add_collaborators_when_configured(self):
        """With FEISHU_DOC_COLLABORATORS=['oc_alice'], create_doc calls add_doc_collaborators."""
        import larkhelm.lark_client as _lc
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.document.document_id = "doc123"

        with patch.object(client, "_should_use_user_token", return_value=False), \
             patch.object(_lc, "client") as mock_sdk_client, \
             patch.object(client, "transfer_doc_owner"), \
             patch.object(client, "add_doc_collaborators") as mock_add, \
             patch("larkhelm.lark_client._cfg") as mock_cfg:
            mock_cfg.FEISHU_DOC_COLLABORATORS = ["oc_alice"]
            mock_sdk_client.docx.v1.document.create.return_value = mock_resp
            result = client.create_doc("Test Doc", owner_open_id="oc_owner")

        mock_add.assert_called_once()
        args = mock_add.call_args
        self.assertEqual(args[0][0], "doc123")
        self.assertIn("oc_alice", args[0][1])

    def test_create_doc_skips_collaborators_when_empty(self):
        """With FEISHU_DOC_COLLABORATORS=[], add_doc_collaborators is NOT called."""
        import larkhelm.lark_client as _lc
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.document.document_id = "doc456"

        with patch.object(client, "_should_use_user_token", return_value=False), \
             patch.object(_lc, "client") as mock_sdk_client, \
             patch.object(client, "transfer_doc_owner"), \
             patch.object(client, "add_doc_collaborators") as mock_add, \
             patch("larkhelm.lark_client._cfg") as mock_cfg:
            mock_cfg.FEISHU_DOC_COLLABORATORS = []
            mock_sdk_client.docx.v1.document.create.return_value = mock_resp
            client.create_doc("Test Doc", owner_open_id="oc_owner")

        mock_add.assert_not_called()

    def test_create_wiki_node_calls_add_collaborators(self):
        """create_wiki_node also calls add_doc_collaborators when configured."""
        import larkhelm.lark_client as _lc
        client = self._make_client()

        api_response = {
            "data": {
                "node": {
                    "node_token": "wnk_abc",
                    "obj_token": "wiki_obj_123",
                }
            }
        }

        with patch.object(client, "_should_use_user_token", return_value=False), \
             patch.object(client, "_call_api", return_value=api_response), \
             patch.object(client, "transfer_doc_owner"), \
             patch.object(client, "add_doc_collaborators") as mock_add, \
             patch("larkhelm.lark_client._cfg") as mock_cfg:
            mock_cfg.FEISHU_DOC_COLLABORATORS = ["oc_bob"]
            client.create_wiki_node("space_001", "Wiki Page", owner_open_id="oc_owner")

        mock_add.assert_called_once()
        args = mock_add.call_args
        self.assertEqual(args[0][0], "wiki_obj_123")
        self.assertIn("oc_bob", args[0][1])


if __name__ == "__main__":
    unittest.main()
