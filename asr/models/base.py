"""Base classes and protocols for ASR models"""

from typing import Protocol, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ChunkInfo:
    """Timestamp chunk information"""
    text: str
    start_time: float
    end_time: float


@dataclass
class TranscriptionResult:
    """Unified transcription result across all models"""
    text: str
    confidence: Optional[float] = None
    chunks: Optional[List[ChunkInfo]] = None
    metadata: Optional[Dict[str, Any]] = None


class ASRModel(Protocol):
    """Protocol defining ASR model interface (duck typing)"""

    def transcribe(
        self,
        audio_path: str,
        language: str = "ja",
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio file and return unified result"""
        ...

    @property
    def model_name(self) -> str:
        """Return model identifier for reporting"""
        ...
