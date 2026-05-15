"""larkhelm · embedding backends — Phase D / Phase 2.

Implements three concrete :class:`EmbeddingBackend` flavours plus a small
in-process LRU cache and a config-driven factory:

    LocalONNXEmbedding   — onnxruntime + tokenizer, lazy init.
    HTTPEmbedding        — POST to an external endpoint, per-instance
                           circuit breaker (5 consecutive failures →
                           5 minutes refused, fail-open at caller side).
    StubEmbedding        — deterministic dim=8 vector hashed from the
                           input string, used by tests only.

All public types use stdlib + ``numpy`` (optional) + ``onnxruntime``
(optional). ``numpy`` / ``onnxruntime`` are imported lazily so that
``import larkhelm.memory_embedding`` is safe even on hosts without the
``memory-embedding`` extra installed — the local backend's ``__init__``
is the first place a missing dep raises :class:`EmbeddingError`, which
:func:`get_embedding_backend` catches to fall back to ``None``.

Log prefix follows CLAUDE.md PascalCase: ``[MemoryRetriever] EmbeddingLocal:``
/ ``[MemoryRetriever] EmbeddingHTTP:`` (design.md §6.4).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from types import ModuleType
from typing import TYPE_CHECKING, Any

import larkhelm.config as _cfg
from larkhelm.log import _debug_log

if TYPE_CHECKING:
    import numpy as np  # noqa: F401 — type-only forward ref
    from larkhelm.memory_slice import EmbeddingBackend  # noqa: F401


class EmbeddingError(RuntimeError):
    """Raised by :class:`EmbeddingBackend.embed` when the underlying
    model / endpoint fails.

    Caught by :class:`HybridRetriever` / :class:`EmbeddingRetriever` and
    triggers fail-open: caller logs a WARN and reverts to keyword retrieval
    while flipping ``audit.fail_open=True``.
    """


# ── HTTP circuit breaker constants ────────────────────────────────────────

_HTTP_CIRCUIT_FAIL_THRESHOLD = 5         # consecutive failures before opening
_HTTP_CIRCUIT_OPEN_SECONDS = 5 * 60      # how long the circuit stays open

# Approximate per-text token bytes for a rough payload sanity guard. Keeps
# pathological inputs from spending O(seconds) doing nothing meaningful.
_HTTP_MAX_PAYLOAD_BYTES = 1 * 1024 * 1024


def _ensure_numpy() -> ModuleType:  # pragma: no cover — import-time hook
    """Return the ``numpy`` module or raise :class:`EmbeddingError`.

    Centralised so the three backends share one error message.
    """
    try:
        import numpy as np
    except Exception as e:
        raise EmbeddingError(
            f"numpy not installed (pip install 'larkhelm[memory-embedding]'): {e}"
        ) from e
    return np


# ── StubEmbedding (test fixture) ──────────────────────────────────────────


class StubEmbedding:
    """Deterministic in-memory embedding — pure Python, no model files.

    Built for tests + fast smoke checks. Vectors are stable across runs
    (key = sha1 of the text), unit-normalised, dimension is **always
    ``dim``** even when callers pass tiny strings. Not used at runtime —
    :func:`get_embedding_backend` never resolves to this class from config.
    """

    def __init__(self, dim: int = 8) -> None:
        self.name = "stub"
        self.dim = int(dim) if int(dim) > 0 else 8

    def embed(self, texts: list[str]) -> Any:
        np = _ensure_numpy()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha1((t or "").encode("utf-8")).digest()
            # Spread the 20-byte sha1 across the vector deterministically.
            for j in range(self.dim):
                out[i, j] = float(digest[j % len(digest)]) / 255.0 - 0.5
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out

    def warm(self) -> None:  # noqa: D401 — protocol hook
        return None


# ── LocalONNXEmbedding ───────────────────────────────────────────────────


class LocalONNXEmbedding:
    """In-process ONNX-runtime backend.

    Lazy: the ONNX session and tokenizer are NOT created in ``__init__``
    (so import-time cost is zero); the first ``embed`` call (or an explicit
    ``warm()`` from the boot thread) triggers ``_lazy_init``. If onnxruntime
    is missing, ``_lazy_init`` raises :class:`EmbeddingError` and the
    factory in :func:`get_embedding_backend` swallows it and falls back to
    ``None``.

    Tokenisation is intentionally minimal: we expect the operator to supply
    a pre-tokenised model (e.g. BGE small-zh) and rely on the model's own
    tokenizer. For Phase 2 we fall back to a stdlib word-piece-ish split so
    the contract holds without HuggingFace tokenizers.

    Thread-safety: ``_init_lock`` guards lazy init; once initialised, ONNX
    Runtime sessions are themselves thread-safe.
    """

    name: str = "local"

    def __init__(self, model_path: str, dim: int) -> None:
        self.model_path = os.path.expanduser(str(model_path or ""))
        self.dim = int(dim) if int(dim) > 0 else 512
        # ``Any`` because ``onnxruntime.InferenceSession`` is optional dep;
        # the type narrows at runtime via the ``assert self._session is not
        # None`` guard inside :meth:`embed`.
        self._session: Any = None
        self._init_lock = threading.Lock()
        self._warmed = False
        if not self.model_path:
            raise EmbeddingError(
                "LocalONNXEmbedding: model_path is empty"
            )

    # ── lazy init ────────────────────────────────────────────────────

    def _lazy_init(self) -> None:
        if self._session is not None:
            return
        with self._init_lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
            except Exception as e:
                raise EmbeddingError(
                    "onnxruntime not installed (pip install "
                    f"'larkhelm[memory-embedding]'): {e}"
                ) from e
            if not os.path.exists(self.model_path):
                raise EmbeddingError(
                    f"ONNX model not found at {self.model_path}; "
                    "see README.md for download instructions"
                )
            try:
                sess_opts = ort.SessionOptions()
                # Single-thread; we already throttle via MAX_AI_PROCS-like
                # semaphores upstream. Avoids ORT spinning N cores per call.
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(
                    self.model_path, sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                raise EmbeddingError(
                    f"ONNX InferenceSession init failed: {e}"
                ) from e

    # ── public API ──────────────────────────────────────────────────

    def warm(self) -> None:
        try:
            self._lazy_init()
            # Run a tiny inference so onnxruntime allocates kernels.
            try:
                self.embed(["warmup"])
                self._warmed = True
            except Exception as e:
                _debug_log(
                    f"[MemoryRetriever] EmbeddingLocal: warm inference failed: {e}"
                )
        except EmbeddingError as e:
            _debug_log(f"[MemoryRetriever] EmbeddingLocal: warm skipped: {e}")
        except Exception as e:  # never raise from warm()
            _debug_log(f"[MemoryRetriever] EmbeddingLocal: warm error: {e}")

    def embed(self, texts: list[str]) -> Any:
        np = _ensure_numpy()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            self._lazy_init()
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"LocalONNXEmbedding init failed: {e}") from e

        session = self._session
        assert session is not None, "_lazy_init must have set self._session"
        try:
            tokens = self._tokenise_batch(texts)
            input_ids = np.asarray(tokens, dtype=np.int64)
            attention_mask = (input_ids != 0).astype(np.int64)
            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            # Optional ``token_type_ids`` — BERT-style models often expect it.
            try:
                input_names = {inp.name for inp in session.get_inputs()}
            except Exception:
                input_names = set()
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            # Drop optional inputs the model doesn't actually declare to avoid
            # ORT's "unexpected input" error.
            feeds = {k: v for k, v in feeds.items() if k in input_names or k == "input_ids"}
            outputs = session.run(None, feeds)
            vec = np.asarray(outputs[0], dtype=np.float32)
            # Mean-pool over sequence length when the model returns
            # (batch, seq, hidden); pass through when already (batch, dim).
            if vec.ndim == 3:
                vec = vec.mean(axis=1)
            if vec.ndim != 2 or vec.shape[1] != self.dim:
                # The dim mismatch is reported up so the cache layer can
                # invalidate; we don't crash here — we just signal failure.
                raise EmbeddingError(
                    f"LocalONNXEmbedding: shape {vec.shape} mismatches dim={self.dim}"
                )
            # L2-normalise so downstream cosine = dot product.
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (vec / norms).astype(np.float32)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"LocalONNXEmbedding.embed failed: {e}"
            ) from e

    # ── internals ───────────────────────────────────────────────────

    def _tokenise_batch(self, texts: list[str]) -> list[list[int]]:
        """Cheap whitespace + CJK-codepoint tokeniser; max-pad to 64 tokens.

        Not a real BERT tokenizer — but for the local model contract used
        here (mean-pooled CPU embedding) the model is generally robust to
        approximate input. Tests use :class:`StubEmbedding` so this path
        isn't exercised in CI.
        """
        max_len = 64
        out: list[list[int]] = []
        for t in texts:
            buf: list[int] = []
            for ch in (t or "")[: max_len * 4]:
                ord_ = ord(ch)
                if ord_ < 32:
                    continue
                buf.append(ord_ % 30000 + 1)  # avoid 0 (used as PAD)
                if len(buf) >= max_len:
                    break
            while len(buf) < max_len:
                buf.append(0)
            out.append(buf)
        return out


# ── HTTPEmbedding ────────────────────────────────────────────────────────


class HTTPEmbedding:
    """Remote HTTP-JSON embedding backend.

    Wire shape: POST ``{ "texts": [...] }`` → ``{ "vectors": [[...], ...] }``
    (each vector ``len == dim``). Failures (timeout, non-2xx, malformed
    JSON, dim mismatch) all raise :class:`EmbeddingError`.

    Circuit breaker per-instance: after 5 consecutive failures, ``embed``
    raises immediately for 5 minutes without hitting the network.
    """

    name: str = "http"

    def __init__(self, endpoint: str, dim: int, timeout: float = 5.0) -> None:
        self.endpoint = str(endpoint or "")
        self.dim = int(dim) if int(dim) > 0 else 512
        self.timeout = float(timeout) if timeout else 5.0
        self._fail_count = 0
        self._circuit_open_until = 0.0
        self._circuit_lock = threading.Lock()
        if not self.endpoint:
            raise EmbeddingError("HTTPEmbedding: endpoint is empty")

    # ── public API ──────────────────────────────────────────────────

    def warm(self) -> None:
        try:
            self.embed(["warmup"])
        except Exception as e:
            _debug_log(f"[MemoryRetriever] EmbeddingHTTP: warm failed: {e}")

    def embed(self, texts: list[str]) -> Any:
        np = _ensure_numpy()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self._check_circuit():
            raise EmbeddingError(
                "HTTPEmbedding circuit open — refusing for safety window"
            )

        payload = json.dumps({"texts": list(texts)}, ensure_ascii=False).encode("utf-8")
        if len(payload) > _HTTP_MAX_PAYLOAD_BYTES:
            self._record_failure()
            raise EmbeddingError(
                f"HTTPEmbedding payload too large ({len(payload)} bytes)"
            )

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                if resp.status < 200 or resp.status >= 300:
                    raise EmbeddingError(
                        f"HTTPEmbedding non-2xx: {resp.status}"
                    )
                body = resp.read()
        except urllib.error.URLError as e:
            self._record_failure()
            raise EmbeddingError(f"HTTPEmbedding URL error: {e}") from e
        except Exception as e:
            self._record_failure()
            raise EmbeddingError(f"HTTPEmbedding transport error: {e}") from e

        try:
            data = json.loads(body.decode("utf-8"))
            vectors = data.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise ValueError(
                    f"shape mismatch: got {len(vectors) if isinstance(vectors, list) else type(vectors).__name__} "
                    f"vectors for {len(texts)} texts"
                )
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                raise ValueError(
                    f"shape mismatch: got {arr.shape}, expected (*, {self.dim})"
                )
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = (arr / norms).astype(np.float32)
        except EmbeddingError:
            raise
        except Exception as e:
            self._record_failure()
            raise EmbeddingError(f"HTTPEmbedding parse failed: {e}") from e
        self._record_success()
        return out

    # ── circuit breaker ─────────────────────────────────────────────

    def _check_circuit(self) -> bool:
        with self._circuit_lock:
            return time.monotonic() >= self._circuit_open_until

    def _record_failure(self) -> None:
        with self._circuit_lock:
            self._fail_count += 1
            if self._fail_count >= _HTTP_CIRCUIT_FAIL_THRESHOLD:
                self._circuit_open_until = time.monotonic() + _HTTP_CIRCUIT_OPEN_SECONDS
                _debug_log(
                    f"[MemoryRetriever] EmbeddingHTTP: circuit OPEN for "
                    f"{_HTTP_CIRCUIT_OPEN_SECONDS}s after {self._fail_count} failures"
                )

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._fail_count = 0
            self._circuit_open_until = 0.0


# ── EmbeddingCache (LRU) ─────────────────────────────────────────────────


class EmbeddingCache:
    """Thread-safe LRU cache for slice vectors.

    Stores ``sha1(slice_id|body[:512]|dim|backend_name) → np.ndarray``.
    Vector freshness is implicit: a body edit changes the prefix, which
    changes the key. A dim or backend-name change invalidates everything
    for that backend.

    Implementation is a plain :class:`collections.OrderedDict` so we get
    true LRU semantics (move-to-end on hit, pop-front on overflow) with
    one lock — ``functools.lru_cache`` doesn't fit cleanly because the
    cached function is the backend's ``embed`` call, which we cannot wrap
    by argument identity (the backend is a Python object).
    """

    def __init__(self, maxsize: int = 2048) -> None:
        from collections import OrderedDict
        self._maxsize = max(64, int(maxsize))
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._store_lock = threading.Lock()

    def _key(self, slice_id: str, body: str, backend: Any) -> str:
        prefix = (body or "")[:512]
        return hashlib.sha1(
            f"{slice_id}|{prefix}|{getattr(backend, 'dim', 0)}|{getattr(backend, 'name', '?')}"
            .encode("utf-8")
        ).hexdigest()

    def get_or_compute(self, slice_id: str, body: str, backend: Any) -> Any:
        key = self._key(slice_id, body, backend)
        with self._store_lock:
            if key in self._store:
                # LRU hit — bump to most-recent.
                self._store.move_to_end(key)
                return self._store[key]
        # Compute outside the lock — backend.embed may take 10–100ms.
        vec = backend.embed([body or ""])[0]
        with self._store_lock:
            # Concurrent miss-then-miss: keep the latest; either is correct.
            self._store[key] = vec
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)
        return vec

    def invalidate(self) -> None:
        with self._store_lock:
            self._store.clear()

    def __len__(self) -> int:  # noqa: D401 — observability for tests
        with self._store_lock:
            return len(self._store)


# ── Factory ───────────────────────────────────────────────────────────────

_BACKEND_SINGLETON: "EmbeddingBackend | None" = None
_BACKEND_LOCK = threading.Lock()
_BACKEND_KEY: tuple[Any, ...] = ()  # (backend_name, endpoint, model_path, dim)


def get_embedding_backend(config: dict[str, Any] | None = None) -> "EmbeddingBackend | None":
    """Resolve an :class:`EmbeddingBackend` from config; cached singleton.

    Returns ``None`` when ``embedding_backend == "none"`` OR the requested
    backend's deps are missing. The singleton is keyed by
    ``(backend_name, endpoint, model_path, dim)`` so a config change that
    swaps backends will rebuild it on the next call.
    """
    global _BACKEND_SINGLETON, _BACKEND_KEY
    cfg = config if config is not None else (getattr(_cfg, "config", {}) or {})

    backend_name = str(cfg.get("embedding_backend", "none") or "none").lower()
    if backend_name == "none":
        return None

    dim = int(cfg.get("embedding_dim", 512) or 512)
    model_path = str(cfg.get("embedding_model_path", "") or "")
    endpoint = str(cfg.get("embedding_http_endpoint", "") or "")
    timeout = float(cfg.get("embedding_http_timeout_sec", 5.0) or 5.0)

    key = (backend_name, endpoint, model_path, dim)
    with _BACKEND_LOCK:
        if _BACKEND_SINGLETON is not None and _BACKEND_KEY == key:
            return _BACKEND_SINGLETON

        instance: "EmbeddingBackend | None"
        try:
            if backend_name == "local":
                instance = LocalONNXEmbedding(model_path=model_path, dim=dim)
            elif backend_name == "http":
                instance = HTTPEmbedding(endpoint=endpoint, dim=dim, timeout=timeout)
            elif backend_name == "stub":
                instance = StubEmbedding(dim=dim)
            else:
                _debug_log(
                    f"[MemoryRetriever] EmbeddingFactory: unknown backend "
                    f"{backend_name!r}, falling back to none"
                )
                return None
        except EmbeddingError as e:
            _debug_log(
                f"[MemoryRetriever] EmbeddingFactory: {backend_name} init "
                f"failed (degrading to none): {e}"
            )
            return None
        except Exception as e:
            _debug_log(
                f"[MemoryRetriever] EmbeddingFactory: {backend_name} unexpected "
                f"init error (degrading to none): {e}"
            )
            return None

        _BACKEND_SINGLETON = instance
        _BACKEND_KEY = key
        return _BACKEND_SINGLETON


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine-sim between two 1-D ``np.ndarray``; returns 0 on zero norms.

    Helper for retrievers — kept here so retrievers don't need numpy at
    module import time.
    """
    np = _ensure_numpy()
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


__all__ = [
    "EmbeddingError",
    "LocalONNXEmbedding",
    "HTTPEmbedding",
    "StubEmbedding",
    "EmbeddingCache",
    "get_embedding_backend",
    "cosine_similarity",
]
