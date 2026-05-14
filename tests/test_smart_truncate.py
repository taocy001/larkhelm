"""Unit tests for memory_context.smart_truncate (S42).

When a memory layer overflows the combined budget the legacy implementation
used ``content[:budget] + "…"`` — a raw character cut that can chop a
Chinese sentence mid-character, split a markdown code fence, or leave a
half-finished numbered list item. ``smart_truncate`` prefers a semantic
boundary (paragraph > sentence > line > word) within a 15% slack window.
"""
import unittest

from larkhelm.memory_context import smart_truncate


class SmartTruncateTests(unittest.TestCase):
    def test_under_budget_returned_unchanged(self):
        text = "short text"
        self.assertEqual(smart_truncate(text, 100), text)

    def test_zero_budget_returns_empty(self):
        self.assertEqual(smart_truncate("anything", 0), "")
        self.assertEqual(smart_truncate("anything", -5), "")

    def test_paragraph_boundary_preferred(self):
        text = (
            "First paragraph with some content here.\n\n"
            "Second paragraph that will probably get cut off because budget is small."
        )
        result = smart_truncate(text, 50)
        # Must end at the paragraph boundary, not mid-sentence
        self.assertTrue(result.endswith("…"))
        self.assertIn("First paragraph", result)
        # The second paragraph should be entirely cut (paragraph boundary
        # falls within the slack window).
        self.assertNotIn("Second paragraph", result)

    def test_sentence_boundary_chinese(self):
        text = "这是第一句话。这是第二句话。这是第三句话非常长非常长非常长。"
        # Budget 14 chars → backtrack to "。" within 15% slack (≈2 chars)
        result = smart_truncate(text, 14)
        self.assertTrue(result.endswith("…"))
        # Result should end on a 。 boundary (not mid-character)
        body = result.rstrip("…")
        self.assertTrue(body.endswith("。"), f"expected sentence boundary, got {result!r}")

    def test_sentence_boundary_ascii_requires_trailing_space(self):
        # Bare "." is too ambiguous (version numbers, file names) — only
        # ". " counts as a sentence boundary.
        text = "First sentence here. Second sentence here. Third one is longer than budget allows."
        result = smart_truncate(text, 45)
        self.assertTrue(result.endswith("…"))
        body = result.rstrip("…")
        # Body should end on a sentence boundary (with the trailing space
        # stripped by rstrip).
        self.assertTrue(body.rstrip().endswith("."),
                        f"expected sentence boundary, got {result!r}")

    def test_line_boundary_when_no_paragraph_or_sentence(self):
        text = "item one\nitem two\nitem three is longer than fits"
        result = smart_truncate(text, 20)
        self.assertTrue(result.endswith("…"))
        # Line break preserves item-list shape
        self.assertIn("\n", result)

    def test_word_boundary_for_english_runs(self):
        text = "the quick brown fox jumps over the lazy dog and several more words follow"
        result = smart_truncate(text, 30)
        self.assertTrue(result.endswith("…"))
        # Should not break mid-word
        body = result.rstrip("…").rstrip()
        last_word = body.split()[-1] if body else ""
        # The original text up to a word boundary must contain this last
        # word as a whole token.
        self.assertTrue(
            last_word and (" " + last_word + " " in text or text.startswith(last_word + " ")),
            f"mid-word cut detected: last token {last_word!r} in {result!r}",
        )

    def test_falls_back_to_char_cut_when_no_boundary(self):
        # No spaces, no punctuation, no newlines — must fall back to raw cut.
        text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = smart_truncate(text, 20)
        self.assertEqual(result, text[:20] + "…")

    def test_slack_window_does_not_overshrink(self):
        # Paragraph break exists but FAR before budget. slack_pct=0.15
        # of budget=200 = 30, so a paragraph break at position 50 is
        # outside the window — we should fall through to line / word /
        # char cuts instead of trimming away 150 chars.
        text = (
            "intro.\n\n"  # paragraph break at position 7
            + "x" * 300   # no further boundaries
        )
        result = smart_truncate(text, 200)
        # Result should be close to 200 chars, not 7
        self.assertGreater(len(result), 150,
                           f"slack chase shrunk output too far: len={len(result)}")

    def test_ellipsis_always_appended_when_trimmed(self):
        for budget in (5, 50, 500):
            with self.subTest(budget=budget):
                text = "x" * (budget * 3)
                result = smart_truncate(text, budget)
                self.assertTrue(result.endswith("…"),
                                f"missing ellipsis for budget={budget}: {result!r}")

    def test_cut_between_period_and_space_falls_back_gracefully(self):
        # Round-2 review #5: the sentence boundary search uses rfind(". ",
        # floor, budget). If the budget cut lands EXACTLY between "." and
        # " " (e.g. budget points to the index of the space), the two-char
        # sequence isn't entirely inside [floor, budget) and the search
        # misses. We should still emit a sensible result via word / char
        # fallback, not crash, and not produce an empty body.
        text = "A sentence here. Another sentence follows here."
        for budget in range(14, 22):
            with self.subTest(budget=budget):
                result = smart_truncate(text, budget)
                self.assertTrue(result.endswith("…"),
                                f"missing ellipsis at budget={budget}: {result!r}")
                # Must produce non-trivial output (more than just "…")
                self.assertGreater(len(result), 1)

    def test_ellipsis_style_consistent_after_rstrip(self):
        # Round-2 review #4: every boundary path now rstrip()s before
        # the ellipsis. Verify no trailing whitespace appears between
        # the body and the ellipsis (which used to leak through on
        # line/word paths).
        cases = [
            # paragraph: cut at "\n\n"
            "first paragraph here.\n\nsecond paragraph contents here are long enough to overflow",
            # line break with trailing spaces before "\n"
            "line one with trailing space   \nline two contents extend beyond the budget",
            # word break with multiple spaces
            "word one     word two contents continue past the budget",
        ]
        for text in cases:
            with self.subTest(text=text[:30]):
                result = smart_truncate(text, 30)
                self.assertTrue(result.endswith("…"))
                body = result.rstrip("…").rstrip("\n")
                # No trailing whitespace before the ellipsis marker
                self.assertFalse(body != body.rstrip(),
                                 f"trailing whitespace before ellipsis: {result!r}")

    def test_markdown_code_fence_not_split_midline(self):
        # A common pain case: budget cuts mid-fence. With line-boundary
        # fallback (no paragraph / sentence here), we should at least cut
        # at a newline so the rendered output doesn't show ``…``  inside a
        # broken ``` fence.
        text = (
            "Here is some code:\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n"
            "Trailing prose that overflows the budget by a wide margin."
        )
        result = smart_truncate(text, 60)
        self.assertTrue(result.endswith("…"))
        # The cut should not land in the middle of any line: every line in
        # the result body (except possibly the last "…" placeholder) must
        # be a prefix of an original line.
        body_lines = result.rstrip("…").rstrip("\n").splitlines()
        orig_lines = text.splitlines()
        for ln in body_lines:
            self.assertIn(ln, orig_lines, f"mid-line cut: {ln!r}")


if __name__ == "__main__":
    unittest.main()
