#!/usr/bin/env python3
"""Offline trainer for the Phase D-D feedback-driven intent classifier.

Reads ``intent_feedback.jsonl`` (user corrections, gold-standard) +
``intent_audit.jsonl`` (high-confidence dispatches as semi-supervised
positives), embeds each text with the same local ONNX backend used by
the L2 embedding classifier, fits scikit-learn LogisticRegression, and
writes the inference checkpoint to ``DATA_DIR/intent_microlearn.pkl``.

Inference-side contract is in ``larkhelm/agent_hub/intent_microlearn.py``.

Usage:
    python3 scripts/train_intent_classifier.py            # uses default DATA_DIR
    python3 scripts/train_intent_classifier.py --data-dir /var/lib/larkhelm
    python3 scripts/train_intent_classifier.py --dry-run  # report counts, don't write

Caveats:
  * Until ``intent_feedback.jsonl`` has >= ~200 corrections you'll have
    too little data for a useful classifier. Run a dry-run to see the
    label distribution first.
  * Audit rows DON'T carry the user's text (only a trace_id). The
    trainer joins them against ``logs/all.jsonl`` by trace_id to recover
    the prompt; rows that can't be joined are skipped silently.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def _find_data_dir() -> Path:
    """Mirror the runtime DATA_DIR priority from CLAUDE.md."""
    env = os.environ.get("LARKHELM_DATA_DIR")
    if env:
        return Path(env)
    for candidate in ("/var/lib/larkhelm", os.path.expanduser("~/.local/share/larkhelm")):
        if Path(candidate).is_dir():
            return Path(candidate)
    raise SystemExit("could not locate DATA_DIR; pass --data-dir explicitly")


def _load_feedback(path: Path) -> list[dict]:
    """``corrected_intent`` is the gold label."""
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get("text") or "").strip()
            label = (r.get("corrected_intent") or "").strip()
            if text and label and 5 <= len(text) <= 2000:
                out.append({"text": text, "label": label, "source": "feedback"})
    return out


def _load_audit_joined(audit_path: Path, log_path: Path,
                        max_per_label: int = 200) -> list[dict]:
    """Audit rows lack text; join with logs/all.jsonl on trace_id."""
    if not audit_path.exists() or not log_path.exists():
        return []

    # Build trace_id → text map from all.jsonl user rows. (User rows
    # come BEFORE assistant rows, so we keep the most recent text per
    # trace_id we encounter.)
    text_by_trace: dict[str, str] = {}
    pending_user_text: "str | None" = None
    with open(log_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = r.get("role")
            if role == "user":
                pending_user_text = (r.get("content") or "").strip()
            elif role == "assistant":
                trace = r.get("trace_id")
                if trace and pending_user_text:
                    text_by_trace[trace] = pending_user_text
                pending_user_text = None

    out_by_label: dict[str, list[dict]] = defaultdict(list)
    with open(audit_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("success"):
                continue
            label = (r.get("agent_type") or "").strip()
            conf = float(r.get("confidence") or 0.0)
            layer = r.get("layer", "")
            # Only take high-confidence dispatches; skip "override"
            # (those are user-corrections already in feedback).
            if not label or conf < 0.80 or layer == "override":
                continue
            trace = r.get("trace_id")
            text = text_by_trace.get(trace, "").strip() if trace else ""
            if not text or not (5 <= len(text) <= 2000):
                continue
            if len(out_by_label[label]) >= max_per_label:
                continue
            out_by_label[label].append({
                "text": text, "label": label, "source": "audit",
            })
    return [r for rows in out_by_label.values() for r in rows]


def _embed_texts(texts: list[str]) -> "tuple[Any, Any]":
    """Use the same ONNX backend as L2 embedding so the trained classifier
    and the L2 classifier live in the same vector space."""
    # Adding the project to sys.path so this script can import
    # ``larkhelm`` without requiring an editable install.
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from larkhelm.memory_embedding import get_embedding_backend  # noqa: E402
    backend = get_embedding_backend()
    if backend is None:
        raise SystemExit(
            "embedding backend unavailable; install onnxruntime + the model "
            "(see memory_embedding.py for the resolver order)"
        )

    import numpy as np  # noqa: E402
    vecs: list = []
    for i, t in enumerate(texts):
        v = backend.embed(t)
        if v is None:
            raise SystemExit(f"backend returned None on sample {i}: {t[:60]!r}")
        vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
    return np.vstack(vecs), np


def _fit_and_save(samples: list[dict], out_path: Path, *, dry_run: bool) -> None:
    if not samples:
        raise SystemExit("no samples found")

    labels = [s["label"] for s in samples]
    label_counts = Counter(labels)
    print("# Label distribution:")
    for lbl, n in label_counts.most_common():
        print(f"    {lbl}: {n}")
    if min(label_counts.values()) < 5:
        print(
            "WARNING: at least one class has < 5 samples; the model will "
            "probably overfit. Collect more feedback before relying on it."
        )

    if dry_run:
        print("--dry-run: skipping fit + write")
        return

    print("# Embedding...")
    X, np = _embed_texts([s["text"] for s in samples])
    y = np.array(labels)

    print("# Fitting LogisticRegression...")
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise SystemExit(f"scikit-learn required for training: {e}")

    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)
    clf.fit(X, y)
    acc = float((clf.predict(X) == y).mean())
    print(f"# Training accuracy: {acc:.3f} (in-sample; not a real eval)")

    ckpt = {
        "coef":      clf.coef_,
        "intercept": clf.intercept_,
        "classes":   list(clf.classes_),
        "meta": {
            "trained_at":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sample_count":       len(samples),
            "label_distribution": dict(label_counts),
            "sklearn_version":    sklearn.__version__,
            "training_accuracy":  acc,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.chmod(out_path, 0o600)   # checkpoint is local-only training output
    print(f"# Wrote checkpoint to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="override DATA_DIR")
    ap.add_argument("--dry-run", action="store_true",
                    help="report sample counts, skip fit / write")
    ap.add_argument("--max-audit-per-label", type=int, default=200,
                    help="cap semi-supervised audit samples per label")
    args = ap.parse_args()

    data_dir = args.data_dir or _find_data_dir()
    feedback_path = data_dir / "intent_feedback.jsonl"
    audit_path = data_dir / "intent_audit.jsonl"
    log_path = data_dir / "logs" / "all.jsonl"
    out_path = data_dir / "intent_microlearn.pkl"

    feedback_samples = _load_feedback(feedback_path)
    audit_samples = _load_audit_joined(
        audit_path, log_path, max_per_label=args.max_audit_per_label,
    )
    samples = feedback_samples + audit_samples
    print(f"# Loaded {len(feedback_samples)} feedback + {len(audit_samples)} audit "
          f"= {len(samples)} samples total")

    _fit_and_save(samples, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
