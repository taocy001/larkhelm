"""larkhelm · agent_hub · feedback-driven micro-learn classifier (D).

Goal: turn ``intent_feedback.jsonl`` (user corrections) into a small
local classifier whose predictions add a third vote to ``resolve_intent``
alongside the L1 keyword tier and the L2 embedding / LLM JSON tier. The
classifier itself is intentionally tiny — a logistic-regression head on
top of a frozen sentence-bert (or any 384-d) embedding — so it loads in
<100 ms and predicts in <10 ms with zero network round-trips.

This module is INFERENCE-ONLY. Training is a separate offline script
(``scripts/train_intent_classifier.py``) that:

  1. Reads ``intent_feedback.jsonl`` (corrections) + a sampled labeled
     slice of ``intent_audit.jsonl`` (high-confidence dispatches that
     never got corrected).
  2. Embeds each text with the local ONNX backend used by Phase D
     embedding L2 (no separate model — same vector space → cross-tier
     reasoning).
  3. Fits scikit-learn LogisticRegression(C=1.0, multi_class="auto").
  4. Saves ``(coef, intercept, classes, sklearn_version)`` to
     ``DATA_DIR/intent_microlearn.pkl``.

At runtime ``predict(text)`` lazily loads the checkpoint on first call.
Until a checkpoint exists OR ``intent_microlearn_enabled=false`` the
classifier returns ``None`` and ``resolve_intent`` carries on with L1 +
L2 only — zero risk to existing behaviour.

The classifier is INTENTIONALLY conservative: it returns a label only
when its softmax confidence ≥ ``intent_microlearn_min_confidence`` (0.65
default). Below that, it abstains so the L2 ensemble's vote wins.

Failure modes (all graceful):
  * No checkpoint   → return None
  * Stale sklearn   → log once, return None
  * Embedding fails → return None
  * Math overflow   → return None
"""
from __future__ import annotations

import math
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from larkhelm.log import safe_log


# ── Checkpoint shape ────────────────────────────────────────────────────


@dataclass
class _MicroLearnCheckpoint:
    """Frozen state shipped from the offline training script.

    ``coef`` is ``(n_classes, n_features)``; ``intercept`` is
    ``(n_classes,)``; ``classes`` lists agent names in the same column
    order as ``coef``. ``meta`` carries training metadata (date,
    sklearn version, sample counts) for diagnostics.
    """

    coef: Any            # numpy.ndarray (n_classes, n_features)
    intercept: Any       # numpy.ndarray (n_classes,)
    classes: list[str]   # ordered: ["chat", "crew", "dev", "doc", "plan"]
    meta: dict


# ── Singleton state ─────────────────────────────────────────────────────


_LOCK = threading.Lock()
_CHECKPOINT: "_MicroLearnCheckpoint | None" = None
_CHECKPOINT_LOAD_TRIED: bool = False
_CHECKPOINT_MTIME: float = 0.0
_LAST_WARN_TS: float = 0.0


def _checkpoint_path() -> Path:
    """Return ``DATA_DIR/intent_microlearn.pkl`` — the agreed location.

    Falls back to a tempdir path when ``DATA_DIR`` isn't configured
    (early bootstrap / single-file debug); inference simply finds no
    checkpoint and returns None, which is the desired no-op.
    """
    try:
        import larkhelm.config as _cfg
        data_dir = getattr(_cfg, "DATA_DIR", None)
        if data_dir:
            return Path(data_dir) / "intent_microlearn.pkl"
    except Exception:
        pass
    import tempfile
    return Path(tempfile.gettempdir()) / "intent_microlearn.pkl"


def _load_checkpoint() -> "_MicroLearnCheckpoint | None":
    """Lazy-load (and hot-reload on mtime change) the inference state."""
    global _CHECKPOINT, _CHECKPOINT_LOAD_TRIED, _CHECKPOINT_MTIME

    path = _checkpoint_path()
    try:
        mtime = path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        mtime = 0.0

    with _LOCK:
        # First call OR file refreshed → reload.
        if _CHECKPOINT_LOAD_TRIED and mtime == _CHECKPOINT_MTIME:
            return _CHECKPOINT

        _CHECKPOINT_LOAD_TRIED = True
        _CHECKPOINT_MTIME = mtime
        _CHECKPOINT = None

        if mtime == 0.0:
            return None

        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
        except Exception as e:
            _warn_throttled(f"[IntentMicroLearn] checkpoint load failed: {e}")
            return None

        # Validate shape; checkpoint is offline-trained, so be defensive
        # in case the schema drifts.
        try:
            coef = raw["coef"]
            intercept = raw["intercept"]
            classes = list(raw["classes"])
            meta = dict(raw.get("meta", {}))
        except (KeyError, TypeError) as e:
            _warn_throttled(f"[IntentMicroLearn] checkpoint shape invalid: {e}")
            return None

        _CHECKPOINT = _MicroLearnCheckpoint(
            coef=coef, intercept=intercept, classes=classes, meta=meta,
        )
        return _CHECKPOINT


def _warn_throttled(msg: str) -> None:
    """Warn at most once every 60s so a broken checkpoint doesn't spam
    the debug log on every classification call."""
    global _LAST_WARN_TS
    now = time.time()
    if now - _LAST_WARN_TS < 60.0:
        return
    _LAST_WARN_TS = now
    safe_log(msg)


# ── Public inference API ────────────────────────────────────────────────


def predict(text: str) -> "tuple[str, float] | None":
    """Return ``(agent_type, confidence)`` or ``None`` to abstain.

    ``confidence`` is the softmax probability of the winning class
    (0.0–1.0). Caller compares against
    ``config.INTENT_MICROLEARN_MIN_CONFIDENCE`` (default 0.65).

    Hard contract: never raises. Returns None on:
      * feature flag off
      * missing checkpoint
      * embedding backend unavailable
      * any internal math / shape error
    """
    if not text or not isinstance(text, str):
        return None

    try:
        import larkhelm.config as _cfg
        if not bool(getattr(_cfg, "INTENT_MICROLEARN_ENABLED", False)):
            return None
    except Exception:
        return None

    ckpt = _load_checkpoint()
    if ckpt is None:
        return None

    # Embed query via the same ONNX backend used by the L2 embedding
    # classifier so the two tiers operate in identical vector space and
    # offline training can be replayed against historical L2 outputs.
    try:
        from larkhelm.memory_embedding import get_embedding_backend
        backend = get_embedding_backend()
    except Exception as e:
        _warn_throttled(f"[IntentMicroLearn] embedding backend import failed: {e}")
        return None
    if backend is None:
        return None
    try:
        vec = backend.embed(text)
    except Exception as e:
        _warn_throttled(f"[IntentMicroLearn] embed() raised: {e}")
        return None
    if vec is None:
        return None

    # Avoid numpy import at module top so a stripped install (no numpy /
    # onnxruntime) just collapses to None without ImportError.
    try:
        import numpy as _np
    except Exception:
        return None

    try:
        v = _np.asarray(vec, dtype=_np.float32).reshape(-1)
        logits = ckpt.coef @ v + ckpt.intercept
        # Stable softmax
        m = float(logits.max())
        exps = _np.exp(logits - m)
        probs = exps / float(exps.sum())
        idx = int(probs.argmax())
        conf = float(probs[idx])
        cls = ckpt.classes[idx]
        return cls, conf
    except Exception as e:
        _warn_throttled(f"[IntentMicroLearn] inference math failed: {e}")
        return None


def reset_for_tests() -> None:
    """Test-only hook: drop the singleton + mtime so a re-load happens."""
    global _CHECKPOINT, _CHECKPOINT_LOAD_TRIED, _CHECKPOINT_MTIME
    with _LOCK:
        _CHECKPOINT = None
        _CHECKPOINT_LOAD_TRIED = False
        _CHECKPOINT_MTIME = 0.0


def status() -> dict:
    """Diagnostic snapshot for ``/stats intent`` / debug tooling."""
    path = _checkpoint_path()
    try:
        import larkhelm.config as _cfg
        enabled = bool(getattr(_cfg, "INTENT_MICROLEARN_ENABLED", False))
        min_conf = float(getattr(_cfg, "INTENT_MICROLEARN_MIN_CONFIDENCE", 0.65))
    except Exception:
        enabled, min_conf = False, 0.65

    ckpt = _CHECKPOINT
    out: dict = {
        "enabled": enabled,
        "min_confidence": min_conf,
        "checkpoint_path": str(path),
        "checkpoint_exists": path.exists(),
        "classes": list(ckpt.classes) if ckpt else [],
        "trained_at": (ckpt.meta.get("trained_at") if ckpt else None),
        "sample_count": (ckpt.meta.get("sample_count") if ckpt else None),
    }
    return out


__all__ = ["predict", "reset_for_tests", "status"]
