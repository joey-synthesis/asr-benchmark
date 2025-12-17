"""Performance metrics calculation (RTF, latency)"""

import librosa
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    audio_duration: float
    processing_time: float
    rtf: float  # Real-Time Factor
    num_chunks: Optional[int] = None
    avg_chunk_latency: Optional[float] = None


def calculate_rtf(processing_time: float, audio_duration: float) -> float:
    """
    Calculate Real-Time Factor.

    Args:
        processing_time: Total processing time in seconds
        audio_duration: Audio duration in seconds

    Returns:
        RTF value (< 1.0 = faster than real-time)
    """
    if audio_duration == 0:
        return 0.0
    return processing_time / audio_duration


def get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    return librosa.get_duration(path=audio_path)


def measure_performance(
    audio_path: str,
    processing_time: float,
    num_chunks: Optional[int] = None,
    chunk_latencies: Optional[list] = None
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    Args:
        audio_path: Path to audio file
        processing_time: Total processing time in seconds
        num_chunks: Number of chunks processed (optional)
        chunk_latencies: List of per-chunk latencies (optional)

    Returns:
        PerformanceMetrics object
    """
    duration = get_audio_duration(audio_path)
    rtf = calculate_rtf(processing_time, duration)

    avg_latency = None
    if chunk_latencies:
        avg_latency = sum(chunk_latencies) / len(chunk_latencies)

    return PerformanceMetrics(
        audio_duration=duration,
        processing_time=processing_time,
        rtf=rtf,
        num_chunks=num_chunks,
        avg_chunk_latency=avg_latency
    )
