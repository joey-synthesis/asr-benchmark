"""Kotoba Whisper v2.2 ASR model implementation"""

import torch
from transformers import pipeline
from typing import Optional
from .base import TranscriptionResult, ChunkInfo


class KotobaWhisperModel:
    """Kotoba Whisper v2.2 local ASR model"""

    def __init__(
        self,
        device: Optional[str] = None,
        flash_attention: bool = True,
        chunk_length_s: int = 15,
        batch_size: int = 8
    ):
        """
        Initialize Kotoba Whisper model.

        Args:
            device: "cuda:0", "cpu", or None (auto-detect)
            flash_attention: Try to use Flash Attention 2 if available
            chunk_length_s: Audio chunk size for processing
            batch_size: Batch size for parallel processing
        """
        self._pipe = None
        self._device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._flash_attention = flash_attention
        self._chunk_length = chunk_length_s
        self._batch_size = batch_size

    def _lazy_load_pipeline(self):
        """Lazy-load pipeline on first transcribe() call"""
        if self._pipe is not None:
            return

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_kwargs = {}
        if self._flash_attention:
            try:
                import flash_attn
                model_kwargs = {"attn_implementation": "flash_attention_2"}
            except ImportError:
                pass  # Fall back to standard attention

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model="kotoba-tech/kotoba-whisper-v2.2",
            torch_dtype=torch_dtype,
            device=self._device,
            model_kwargs=model_kwargs,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "ja",
        return_timestamps: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file using Kotoba Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "ja" for Japanese)
            return_timestamps: Whether to return timestamp chunks
            **kwargs: Additional arguments

        Returns:
            TranscriptionResult with text and optional chunks
        """
        self._lazy_load_pipeline()

        result = self._pipe(
            audio_path,
            chunk_length_s=self._chunk_length,
            batch_size=self._batch_size,
            return_timestamps=return_timestamps,
            generate_kwargs={"language": language, "task": "transcribe"}
        )

        # Convert to unified format
        chunks = None
        if return_timestamps and 'chunks' in result:
            chunks = [
                ChunkInfo(
                    text=chunk['text'],
                    start_time=chunk['timestamp'][0],
                    end_time=chunk['timestamp'][1]
                )
                for chunk in result['chunks']
            ]

        return TranscriptionResult(
            text=result['text'],
            chunks=chunks,
            metadata={'device': self._device}
        )

    @property
    def model_name(self) -> str:
        """Return model identifier"""
        return "kotoba-whisper-v2.2"
