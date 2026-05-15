"""Tests for ``larkhelm.log.redact_error``."""
from __future__ import annotations

import pytest

from larkhelm.log import redact_error


# ── Positive matches ─────────────────────────────────────────────────

def test_redacts_api_key_eq():
    assert "***" in redact_error("err: api_key=abcdef12345 some other text")


def test_redacts_api_key_colon():
    out = redact_error("err: API_KEY: my-secret-token-1234")
    assert "my-secret-token-1234" not in out
    assert "***" in out


def test_redacts_authorization_bearer():
    out = redact_error("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.something.fake")
    assert "eyJhbGciOiJIUzI1NiJ9.something.fake" not in out
    assert "Bearer ***" in out


def test_redacts_sk_prefix():
    raw = "Failed call: sk-1234567890abcdefghijklmnopq trailer"
    out = redact_error(raw)
    assert "sk-1234567890abcdefghijklmnopq" not in out
    assert "sk-***" in out


def test_redacts_sk_long_realistic():
    # sk- key with 40+ chars (Anthropic-style)
    sk = "sk-ant-api03-" + "x" * 80
    out = redact_error(f"401 invalid key {sk}")
    assert sk not in out
    assert "sk-***" in out


# ── Negative matches (no false positives) ────────────────────────────

def test_does_not_redact_short_sk_id():
    # 5-char sk- is below the 20-char floor
    raw = "stage env: sk-stage-001 reachable"
    out = redact_error(raw)
    assert out == raw


def test_does_not_redact_random_text():
    raw = "TimeoutError: deadline 30s exceeded"
    out = redact_error(raw)
    assert out == raw


def test_does_not_redact_url_path():
    raw = "fetched https://example.com/api/v1/users"
    out = redact_error(raw)
    assert out == raw


def test_does_not_redact_uuid_like():
    raw = "trace_id=550e8400-e29b-41d4-a716-446655440000"
    # api_key is the keyword we filter on; trace_id is unrelated.
    assert redact_error(raw) == raw


def test_does_not_redact_normal_word_with_sk_substring():
    raw = "ask-some-question"
    assert redact_error(raw) == raw


# ── Idempotency / robustness ─────────────────────────────────────────

def test_idempotent():
    once = redact_error("api_key=verysecretverysecret token123")
    twice = redact_error(once)
    assert once == twice


def test_handles_none():
    assert redact_error(None) == ""


def test_handles_non_string():
    # Exception → str() yields the message; redact_error coerces and returns it
    out = redact_error(Exception("oops"))
    assert isinstance(out, str)
    assert "oops" in out


def test_handles_empty_string():
    assert redact_error("") == ""


def test_handles_multiple_secrets_in_one_message():
    raw = ("api_key=alpha1 Authorization: Bearer beta2 "
           "and sk-this_is_a_long_enough_key_xxxxx")
    out = redact_error(raw)
    assert "alpha1" not in out
    assert "beta2" not in out
    assert "sk-this_is_a_long_enough_key_xxxxx" not in out
    # All three placeholders present
    assert out.count("***") >= 2  # api_key=*** + Bearer ***
    assert "sk-***" in out
