"""
Integration tests for ``FeishuDocClient.create_doc`` / ``create_wiki_node`` /
``create_folder`` user-token branch.

Critical invariant being defended: when ``owner_open_id`` matches the
``LOGGED_IN_OPEN_ID`` and a valid user_token exists, the three create methods
must POST as the user **and skip ``transfer_doc_owner``**. Without this test,
a future regression that silently falls back to tenant + transfer would be
invisible — ``create_doc`` would still succeed and return a doc URL.

We mock ``urllib.request.urlopen`` rather than the higher-level
``_http_request`` so we can also verify the Authorization header carries the
user token (not the tenant token).
"""
from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import larkhelm.config as _cfg
import larkhelm.lark_client as lc
import larkhelm.oauth_user as ou


# Tenant-token endpoint that must NOT be hit on the user-token path.
TRANSFER_ENDPOINT = "/drive/v1/permissions"


class _FakeResp:
    """Minimal duck-type for ``urllib.request.urlopen`` context manager."""

    def __init__(self, body: dict):
        self._payload = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


class UserTokenCreatePathTests(unittest.TestCase):
    """The three create_* must take the user-token path under the right setup."""

    LOGGED_IN_OPEN_ID = "ou_logged_in_user"
    USER_ACCESS_TOKEN = "u-fresh-XXX"

    def setUp(self):
        # Inject deterministic config so _should_use_user_token returns True.
        self._orig_logged = getattr(_cfg, "LOGGED_IN_OPEN_ID", "")
        _cfg.LOGGED_IN_OPEN_ID = self.LOGGED_IN_OPEN_ID

        # Force is_token_valid → True and get_user_token → constant.
        self._is_valid_patch = patch.object(ou, "is_token_valid",
                                             return_value=True)
        self._get_token_patch = patch.object(ou, "get_user_token",
                                              return_value=self.USER_ACCESS_TOKEN)
        self._is_valid_patch.start()
        self._get_token_patch.start()

        self.client = lc.FeishuDocClient()
        self.recorded_calls: list[tuple[str, str, dict, dict]] = []

    def tearDown(self):
        self._is_valid_patch.stop()
        self._get_token_patch.stop()
        _cfg.LOGGED_IN_OPEN_ID = self._orig_logged

    # ── Helpers ─────────────────────────────────────────────────────────

    def _patched_urlopen(self, response_body: dict):
        """Return an urlopen mock that records every request and returns ``response_body``."""

        def _fn(req, timeout=None):
            method = req.get_method()
            url = req.full_url
            headers = {k.lower(): v for k, v in req.header_items()}
            try:
                body = json.loads(req.data.decode("utf-8")) if req.data else {}
            except Exception:
                body = {}
            self.recorded_calls.append((method, url, body, headers))
            return _FakeResp(response_body)

        return _fn

    def _assert_no_transfer_called(self):
        """No request must have touched the transfer_owner endpoint."""
        for method, url, _body, _hdr in self.recorded_calls:
            self.assertNotIn(
                TRANSFER_ENDPOINT, url,
                f"unexpected transfer call on user-token path: {method} {url}")

    # ── create_doc ──────────────────────────────────────────────────────

    def test_create_doc_uses_user_token_and_skips_transfer(self):
        resp = {"code": 0, "data": {"document": {"document_id": "DOCXTOKEN123"}}}
        with patch.object(lc._urllib_req, "urlopen",
                          side_effect=self._patched_urlopen(resp)):
            ref = self.client.create_doc(
                "Test Title",
                owner_open_id=self.LOGGED_IN_OPEN_ID,
            )
        self.assertEqual(ref.token, "DOCXTOKEN123")
        self.assertEqual(ref.doc_type, "docx")

        # 1) Exactly one HTTP call — to the user-token docx endpoint
        self.assertEqual(len(self.recorded_calls), 1,
                         f"expected 1 call, got {[c[1] for c in self.recorded_calls]}")
        method, url, body, headers = self.recorded_calls[0]
        self.assertEqual(method, "POST")
        self.assertIn("/docx/v1/documents", url)
        self.assertEqual(body, {"title": "Test Title"})
        # 2) Authorization header carries the *user* token, not tenant
        self.assertEqual(headers.get("authorization"),
                         f"Bearer {self.USER_ACCESS_TOKEN}")
        # 3) No transfer call was attempted
        self._assert_no_transfer_called()

    def test_create_doc_falls_back_to_tenant_when_user_call_errors(self):
        """API-level failure on user path → tenant path takes over (and transfers)."""
        # Make the user-token HTTP call return an error code so DocAPIError fires.
        user_resp = {"code": 99999, "msg": "user-token api failed"}

        # Track which path actually returned a doc_id. SDK ``client.docx.v1.document.create``
        # is the tenant path; mock it to succeed.
        sdk_create = MagicMock()
        sdk_create.return_value = MagicMock(
            success=lambda: True,
            data=MagicMock(document=MagicMock(document_id="TENANTDOC456")),
        )
        # Also mock transfer_doc_owner so we can assert it was called (tenant path).
        with patch.object(lc._urllib_req, "urlopen",
                          side_effect=self._patched_urlopen(user_resp)), \
             patch.object(lc, "client", MagicMock()) as fake_client, \
             patch.object(lc.FeishuDocClient, "transfer_doc_owner") as fake_transfer:
            fake_client.docx.v1.document.create = sdk_create
            ref = self.client.create_doc(
                "Fallback Title",
                owner_open_id=self.LOGGED_IN_OPEN_ID,
            )
        self.assertEqual(ref.token, "TENANTDOC456",
                         "must have fallen back to tenant SDK path")
        fake_transfer.assert_called_once_with("TENANTDOC456", self.LOGGED_IN_OPEN_ID)

    # ── create_wiki_node ────────────────────────────────────────────────

    def test_create_wiki_node_uses_user_token_and_skips_transfer(self):
        resp = {"code": 0, "data": {"node": {
            "node_token": "NODETOKEN1", "obj_token": "OBJTOKEN1",
        }}}
        with patch.object(lc._urllib_req, "urlopen",
                          side_effect=self._patched_urlopen(resp)):
            ref = self.client.create_wiki_node(
                "spc_xxx", "Wiki Title",
                owner_open_id=self.LOGGED_IN_OPEN_ID,
            )
        self.assertEqual(ref.token, "OBJTOKEN1")
        self.assertEqual(ref.raw_url, "wiki/NODETOKEN1")
        self.assertEqual(len(self.recorded_calls), 1)
        _method, url, body, headers = self.recorded_calls[0]
        self.assertIn("/wiki/v2/spaces/spc_xxx/nodes", url)
        # node_type defaults to "origin" — the user path must use the same value
        self.assertEqual(body.get("node_type"), "origin")
        self.assertEqual(body.get("title"),     "Wiki Title")
        self.assertEqual(headers.get("authorization"),
                         f"Bearer {self.USER_ACCESS_TOKEN}")
        self._assert_no_transfer_called()

    # ── create_folder ───────────────────────────────────────────────────

    def test_create_folder_uses_user_token_and_skips_transfer(self):
        resp = {"code": 0, "data": {"token": "FOLDERTOK1"}}
        with patch.object(lc._urllib_req, "urlopen",
                          side_effect=self._patched_urlopen(resp)):
            tok = self.client.create_folder(
                "My Folder",
                owner_open_id=self.LOGGED_IN_OPEN_ID,
            )
        self.assertEqual(tok, "FOLDERTOK1")
        self.assertEqual(len(self.recorded_calls), 1)
        _method, url, body, headers = self.recorded_calls[0]
        self.assertIn("/drive/v1/files/create_folder", url)
        self.assertEqual(body.get("name"), "My Folder")
        self.assertEqual(headers.get("authorization"),
                         f"Bearer {self.USER_ACCESS_TOKEN}")
        self._assert_no_transfer_called()


class UserTokenGatingTests(unittest.TestCase):
    """``_should_use_user_token`` boundary cases — the second line of defense."""

    def setUp(self):
        self._orig_logged = getattr(_cfg, "LOGGED_IN_OPEN_ID", "")
        self.client = lc.FeishuDocClient()

    def tearDown(self):
        _cfg.LOGGED_IN_OPEN_ID = self._orig_logged

    def test_no_logged_in_user_disables_user_path(self):
        _cfg.LOGGED_IN_OPEN_ID = ""
        with patch.object(ou, "is_token_valid", return_value=True):
            self.assertFalse(self.client._should_use_user_token("ou_anyone"))

    def test_empty_owner_disables_user_path(self):
        _cfg.LOGGED_IN_OPEN_ID = "ou_a"
        with patch.object(ou, "is_token_valid", return_value=True):
            self.assertFalse(self.client._should_use_user_token(""))

    def test_owner_mismatch_disables_user_path(self):
        """User token can only create *as* the authorized user — transferring
        to someone else must still go through tenant + transfer."""
        _cfg.LOGGED_IN_OPEN_ID = "ou_a"
        with patch.object(ou, "is_token_valid", return_value=True):
            self.assertFalse(self.client._should_use_user_token("ou_b"))

    def test_invalid_token_disables_user_path(self):
        _cfg.LOGGED_IN_OPEN_ID = "ou_a"
        with patch.object(ou, "is_token_valid", return_value=False):
            self.assertFalse(self.client._should_use_user_token("ou_a"))

    def test_owner_matches_and_token_valid_enables_user_path(self):
        _cfg.LOGGED_IN_OPEN_ID = "ou_a"
        with patch.object(ou, "is_token_valid", return_value=True):
            self.assertTrue(self.client._should_use_user_token("ou_a"))


if __name__ == "__main__":
    unittest.main()
