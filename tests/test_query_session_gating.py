"""AC-02 — P3 REQ-02 query_session_v2 hash-bucket gating.

Verifies :func:`larkhelm._gating.hash_bucket_allows` is deterministic
(same chat_id always lands in the same bucket) and matches the
requested traffic ratio within a reasonable statistical band.
"""
from __future__ import annotations

import unittest

from larkhelm._gating import hash_bucket_allows


class TestHashBucketAllows(unittest.TestCase):

    def test_traffic_zero_always_false(self) -> None:
        for i in range(100):
            self.assertFalse(hash_bucket_allows(f"chat_{i}", 0.0))

    def test_traffic_one_always_true(self) -> None:
        for i in range(100):
            self.assertTrue(hash_bucket_allows(f"chat_{i}", 1.0))

    def test_same_chat_id_stable(self) -> None:
        for chat_id in ("chat_abc", "user_42", "团队 chat 7"):
            first = hash_bucket_allows(chat_id, 0.5)
            for _ in range(99):
                self.assertEqual(hash_bucket_allows(chat_id, 0.5), first)

    def test_traffic_half_lands_in_band(self) -> None:
        allowed = 0
        n = 1000
        for i in range(n):
            if hash_bucket_allows(f"chat_id_{i}", 0.5):
                allowed += 1
        ratio = allowed / n
        # sha256 distribution over 1000 ids should sit in [0.45, 0.55].
        self.assertGreaterEqual(ratio, 0.45)
        self.assertLessEqual(ratio, 0.55)

    def test_invalid_traffic_returns_false(self) -> None:
        self.assertFalse(hash_bucket_allows("chat_1", "not a number"))
        self.assertFalse(hash_bucket_allows("chat_1", None))


if __name__ == "__main__":
    unittest.main()
