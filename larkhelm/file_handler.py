"""larkhelm · File processing handler.

Validates, downloads, and extracts text content from Feishu file attachments,
then builds prompt blocks suitable for injection into an AI query.

Public API:
  - ExtractedFile      dataclass: result for a single file
  - FileProcessResult  dataclass: result for a batch of files
  - FileProcessor      class:     main processing engine
  - process_file()     convenience wrapper around FileProcessor.process()
  - build_file_prompt_blocks()  convert FileProcessResult → prompt prefix string
  - files_to_prompt_fragment()  convert list[ExtractedFile] → prompt prefix string
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, warn

# ── Session-level file dedup cache ──────────────────────────────────────────
# Keyed by (chat_id, file_key); evicts LRU entries beyond the cap.
# Prevents repeated download + extraction for the same attachment.
_DEDUP_CACHE_MAX = 200
_file_cache: "OrderedDict[tuple[str, str], ExtractedFile]" = OrderedDict()
_file_cache_lock = threading.Lock()


def _cache_get(chat_id: str, file_key: str) -> "ExtractedFile | None":
    key = (chat_id, file_key)
    with _file_cache_lock:
        entry = _file_cache.get(key)
        if entry is None:
            return None
        _file_cache.move_to_end(key)
        # Return a shallow copy so the caller cannot mutate the cached instance.
        import dataclasses as _dc
        return _dc.replace(entry)


def _cache_set(chat_id: str, file_key: str, extracted: "ExtractedFile") -> None:
    key = (chat_id, file_key)
    with _file_cache_lock:
        _file_cache[key] = extracted
        _file_cache.move_to_end(key)
        while len(_file_cache) > _DEDUP_CACHE_MAX:
            _file_cache.popitem(last=False)


# ── AI output file marker regex ──────────────────────────────────────────────
_FILE_MARKER_RE = re.compile(
    r"<!--FILE:([^\s>]+?)-->([\s\S]*?)<!--/FILE-->",
    re.DOTALL,
)


@dataclass
class ExtractedFile:
    """Processing result for a single file attachment."""
    path: str
    file_name: str
    ext: str          # lowercase extension without leading dot, e.g. "py"
    size: int         # bytes
    content: Optional[str] = None   # extracted text; None when extraction failed
    error: Optional[str] = None     # error description; None on success


@dataclass
class FileProcessResult:
    """Aggregate result for a batch of file attachments in one message turn."""
    files: list[ExtractedFile] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)     # ready-to-inject markdown code blocks
    warnings: list[str] = field(default_factory=list)   # user-visible warning strings

    @property
    def has_content(self) -> bool:
        return any(f.content is not None for f in self.files)

    @property
    def prompt_fragment(self) -> str:
        """Merge all blocks into a single prompt prefix string.

        Empty result returns empty string so callers can unconditionally prepend.
        """
        if not self.blocks:
            return ""
        header = "[用户上传了以下文件]\n\n"
        return header + "\n\n".join(self.blocks) + "\n\n---\n\n"


class FileProcessor:
    """Validates, downloads, and extracts content from Feishu file attachments."""

    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    TEXT_EXTENSIONS: frozenset[str] = frozenset({
        "txt", "md", "py", "js", "json", "yaml", "yml", "csv", "log",
        "sh", "go", "rs", "java", "c", "cpp", "h", "ts", "tsx", "jsx",
        "css", "html", "xml", "sql", "dockerfile", "toml", "ini", "cfg", "conf",
    })
    PDF_EXTENSIONS: frozenset[str] = frozenset({"pdf"})
    LANG_MAP: dict[str, str] = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "tsx": "typescript", "jsx": "javascript", "sh": "bash",
        "go": "go", "rs": "rust", "java": "java", "c": "c", "cpp": "cpp",
        "h": "c", "cs": "csharp", "rb": "ruby", "php": "php",
        "html": "html", "css": "css", "sql": "sql",
        "json": "json", "yaml": "yaml", "yml": "yaml",
        "toml": "toml", "xml": "xml", "md": "markdown",
    }

    def process(
        self,
        file_key: str,
        file_name: str,
        chat_id: str,
        msg_id: str,
    ) -> FileProcessResult:
        """Process a single Feishu file attachment end-to-end.

        Flow: format whitelist check → download → size validation → content
        extraction → prompt block construction. Any step failure logs via
        _debug_log / warn and writes to ExtractedFile.error or
        FileProcessResult.warnings; exceptions are never propagated.
        """
        result = FileProcessResult()
        try:
            if not getattr(_cfg, "FILE_ENABLED", True):
                return result

            # Check session-level dedup cache before downloading.
            _cached = _cache_get(chat_id, file_key)
            if _cached is not None:
                _debug_log(f"[FileProcessor] cache hit file_key={file_key!r} file={file_name!r}")
                result.files.append(_cached)
                if _cached.content is not None:
                    result.blocks.append(
                        self._build_block(_cached.file_name, _cached.ext, _cached.content)
                    )
                return result

            ext = Path(file_name).suffix.lower().lstrip(".")
            text_exts: frozenset[str] = getattr(_cfg, "FILE_TEXT_EXTENSIONS", self.TEXT_EXTENSIONS)
            pdf_enabled: bool = getattr(_cfg, "FILE_PDF_ENABLED", True)
            max_size: int = getattr(_cfg, "MAX_FILE_SIZE_BYTES", self.MAX_FILE_SIZE)

            is_text = ext in text_exts
            is_pdf = (ext in self.PDF_EXTENSIONS) and pdf_enabled

            if not is_text and not is_pdf:
                warn_msg = (
                    f"暂不支持该文件格式（.{ext}）。"
                    "支持：常见代码/文本文件（.py/.md/.json 等）及 PDF。"
                )
                result.warnings.append(warn_msg)
                _debug_log(f"[FileProcessor] rejected ext={ext!r} file={file_name!r}")
                _bump_download(ext, "rejected")
                return result

            local_path = self._download(file_key, chat_id, msg_id)
            if not local_path:
                _debug_log(f"[FileProcessor] download failed file_key={file_key!r} file={file_name!r}")
                _bump_download(ext, "failed")
                return result

            try:
                actual_size = os.path.getsize(local_path)
            except Exception:
                actual_size = 0

            size_err = self._validate(file_name, actual_size, max_size)
            if size_err:
                result.warnings.append(size_err)
                _debug_log(f"[FileProcessor] size validation failed: {size_err}")
                _bump_download(ext, "rejected")
                return result

            _bump_download(ext, "success")

            content = self._extract(local_path, ext)
            extracted = ExtractedFile(
                path=local_path,
                file_name=file_name,
                ext=ext,
                size=actual_size,
                content=content,
                error=None if content is not None else "内容提取失败",
            )
            _cache_set(chat_id, file_key, extracted)
            result.files.append(extracted)

            if content is not None:
                result.blocks.append(self._build_block(file_name, ext, content))
            else:
                result.warnings.append(
                    f"文件 `{file_name}` 内容提取失败，请尝试发送截图或粘贴文本。"
                )
        except Exception as e:
            _debug_log(f"[FileProcessor] unexpected error processing {file_name!r}: {e}")
            result.warnings.append("文件处理过程中发生意外错误，请重试。")

        return result

    def _validate(self, file_name: str, size: int, max_size: int) -> Optional[str]:
        if size > max_size:
            limit_mb = max_size / (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            return (
                f"文件 `{file_name}` 超过大小限制"
                f"（{actual_mb:.1f} MB > {limit_mb:.0f} MB）。"
            )
        return None

    def _download(self, file_key: str, chat_id: str, msg_id: str) -> Optional[str]:
        try:
            from larkhelm.lark_client import _download_message_file
            return _download_message_file(file_key, chat_id, msg_id, kind="file")
        except Exception as e:
            _debug_log(f"[FileProcessor] download exception: {e}")
            return None

    def _extract(self, path: str, ext: str) -> Optional[str]:
        if ext == "pdf":
            return self._extract_pdf(path)
        try:
            return Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return Path(path).read_text(encoding="latin-1")
            except Exception as e:
                _debug_log(f"[FileProcessor] text decode failed ext={ext!r}: {e}")
                _bump_extract_error(ext, "decode")
                return None
        except Exception as e:
            _debug_log(f"[FileProcessor] read failed ext={ext!r}: {e}")
            _bump_extract_error(ext, "io")
            return None

    def _extract_pdf(self, path: str) -> Optional[str]:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            pages = []
            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text)
                except Exception:
                    pass
            return "\n\n".join(pages) if pages else None
        except ImportError:
            lib = getattr(_cfg, "FILE_PDF_LIB", "PyPDF2")
            _debug_log(f"[FileProcessor] PDF lib {lib!r} not installed")
            _bump_extract_error("pdf", "missing_lib")
            return None
        except Exception as e:
            _debug_log(f"[FileProcessor] PDF extract failed: {e}")
            _bump_extract_error("pdf", "unknown")
            return None

    def _build_block(self, file_name: str, ext: str, content: str) -> str:
        lang = self._lang_tag(ext)
        return f"### {file_name}\n```{lang}\n{content}\n```"

    def _lang_tag(self, ext: str) -> str:
        return self.LANG_MAP.get(ext, "text")


def _bump_download(ext: str, outcome: str) -> None:
    try:
        from larkhelm.metrics import inc_file_download
        inc_file_download(ext, outcome)
    except Exception:
        pass


def _bump_extract_error(ext: str, error_type: str) -> None:
    try:
        from larkhelm.metrics import inc_file_extract_error
        inc_file_extract_error(ext, error_type)
    except Exception:
        pass


def process_file(
    file_key: str,
    file_name: str,
    chat_id: str,
    msg_id: str,
) -> FileProcessResult:
    """Process a single Feishu file attachment (module-level convenience wrapper)."""
    return FileProcessor().process(file_key, file_name, chat_id, msg_id)


def build_file_prompt_blocks(result: FileProcessResult) -> str:
    """Convert a FileProcessResult into a prompt prefix string.

    Empty or content-less result returns empty string.
    """
    return result.prompt_fragment


def files_to_prompt_fragment(files: "list[ExtractedFile]") -> str:
    """Build a prompt prefix string from a list of already-extracted files.

    Rebuilds the code blocks from ExtractedFile.content so callers that only
    have the files list (e.g. ``_do_query``) don't need a FileProcessResult.
    Returns empty string when no file has extractable content.
    """
    if not files:
        return ""
    fp = FileProcessor()
    blocks = [
        fp._build_block(f.file_name, f.ext, f.content)
        for f in files
        if f.content is not None
    ]
    if not blocks:
        return ""
    return "[用户上传了以下文件]\n\n" + "\n\n".join(blocks) + "\n\n---\n\n"


def extract_file_markers(text: str) -> "tuple[str, list[tuple[str, str]]]":
    """Scan AI output for ``<!--FILE:name-->content<!--/FILE-->`` markers.

    Returns ``(cleaned_text, [(filename, content), ...])``. Markers are removed
    from the returned text; the content of each marker becomes a file to be
    sent via ``send_text_as_file``. Empty filename or empty content are ignored.
    """
    found: list[tuple[str, str]] = []

    def _replace(m: re.Match) -> str:
        # Path.name strips any directory components (../../etc/passwd → passwd)
        # so a misbehaving model cannot traverse outside the temp dir.
        filename = Path(m.group(1).strip()).name
        content = m.group(2).strip()
        if filename and content:
            found.append((filename, content))
        return ""

    cleaned = _FILE_MARKER_RE.sub(_replace, text).strip()
    return cleaned, found


__all__ = [
    "ExtractedFile",
    "FileProcessResult",
    "FileProcessor",
    "process_file",
    "build_file_prompt_blocks",
    "files_to_prompt_fragment",
    "extract_file_markers",
]
