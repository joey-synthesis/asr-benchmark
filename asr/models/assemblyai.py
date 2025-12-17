"""AssemblyAI cloud-based ASR model implementation"""

import os
from typing import Optional
from dotenv import load_dotenv
import assemblyai as aai
from .base import TranscriptionResult


class AssemblyAIModel:
    """AssemblyAI cloud-based ASR model"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        speech_model: str = "best"
    ):
        """
        Initialize AssemblyAI model.

        Args:
            api_key: AssemblyAI API key (or load from env)
            speech_model: "best" or "nano"
        """
        load_dotenv()
        self._api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "AssemblyAI API key not found!\n"
                "Set ASSEMBLYAI_API_KEY in .env or pass api_key parameter"
            )

        aai.settings.api_key = self._api_key
        self._speech_model = (
            aai.SpeechModel.best if speech_model == "best"
            else aai.SpeechModel.nano
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "ja",
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file via AssemblyAI API.

        Args:
            audio_path: Path to audio file or URL
            language: Language code (default: "ja" for Japanese)
            **kwargs: Additional arguments

        Returns:
            TranscriptionResult with text and metadata
        """
        config = aai.TranscriptionConfig(
            speech_model=self._speech_model,
            language_code=language
        )

        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        return TranscriptionResult(
            text=transcript.text,
            confidence=getattr(transcript, 'confidence', None),
            metadata={
                'transcript_id': transcript.id,
                'audio_duration': getattr(transcript, 'audio_duration', None)
            }
        )

    @property
    def model_name(self) -> str:
        """Return model identifier"""
        return f"assemblyai-{self._speech_model.value}"
