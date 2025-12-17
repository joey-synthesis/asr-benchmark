"""Batch evaluation framework for ASR models"""

import time
import os
import json
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from ..models.base import ASRModel
from ..metrics.accuracy import calculate_accuracy_metrics
from ..metrics.performance import get_audio_duration, calculate_rtf


@dataclass
class EvaluationResult:
    """Result from evaluating a single sample"""
    index: int
    audio_file: str
    duration: float
    reference: str
    hypothesis: Optional[str]
    cer: Optional[float]
    wer: Optional[float]
    processing_time: Optional[float]
    rtf: Optional[float]
    status: str  # 'success' or 'error'
    error: Optional[str]


class BatchEvaluator:
    """Batch evaluation engine for ASR models"""

    def __init__(self, model: ASRModel):
        """
        Initialize batch evaluator.

        Args:
            model: ASR model implementing ASRModel protocol
        """
        self.model = model

    def evaluate_sample(
        self,
        sample: Dict,
        audio_dir: str
    ) -> EvaluationResult:
        """
        Evaluate a single audio sample.

        Args:
            sample: Sample metadata dict with 'audio_file', 'transcript', 'index', 'duration_seconds'
            audio_dir: Directory containing audio files

        Returns:
            EvaluationResult object
        """
        audio_path = os.path.join(audio_dir, sample['audio_file'])
        ground_truth = sample['transcript']

        try:
            # Transcribe
            start_time = time.perf_counter()
            result = self.model.transcribe(audio_path, language="ja")
            processing_time = time.perf_counter() - start_time

            # Calculate metrics
            hypothesis = result.text
            accuracy = calculate_accuracy_metrics(hypothesis, ground_truth)
            audio_duration = get_audio_duration(audio_path)
            rtf = calculate_rtf(processing_time, audio_duration)

            return EvaluationResult(
                index=sample['index'],
                audio_file=sample['audio_file'],
                duration=sample['duration_seconds'],
                reference=ground_truth,
                hypothesis=hypothesis,
                cer=accuracy['cer'] if accuracy else None,
                wer=accuracy['wer'] if accuracy else None,
                processing_time=processing_time,
                rtf=rtf,
                status='success',
                error=None
            )

        except Exception as e:
            return EvaluationResult(
                index=sample['index'],
                audio_file=sample['audio_file'],
                duration=sample['duration_seconds'],
                reference=ground_truth,
                hypothesis=None,
                cer=None,
                wer=None,
                processing_time=None,
                rtf=None,
                status='error',
                error=str(e)
            )

    def evaluate_dataset(
        self,
        metadata_path: str,
        audio_dir: str,
        progress_callback: Optional[Callable] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate all samples in dataset.

        Args:
            metadata_path: Path to metadata.json file
            audio_dir: Directory containing audio files
            progress_callback: Optional callback(current, total, filename) for progress updates

        Returns:
            List of EvaluationResult objects
        """
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        results = []
        for i, sample in enumerate(metadata, 1):
            if progress_callback:
                progress_callback(i, len(metadata), sample['audio_file'])

            result = self.evaluate_sample(sample, audio_dir)
            results.append(result)

        return results
