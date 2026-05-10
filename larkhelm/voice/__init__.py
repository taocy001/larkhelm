"""larkhelm voice subpackage — STT via faster-whisper (M3.2)."""
from larkhelm.voice.transcribe import is_model_loaded, is_ready, transcribe

__all__ = ["transcribe", "is_ready", "is_model_loaded"]
