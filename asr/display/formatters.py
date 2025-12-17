"""Formatting utilities for ASR results display"""

from typing import Optional
from ..models.base import TranscriptionResult
from ..metrics.performance import PerformanceMetrics


def format_section_header(title: str, width: int = 60) -> str:
    """Format a section header"""
    return f"\n{'=' * width}\n{title}\n{'=' * width}\n"


def format_transcription_result(result: TranscriptionResult, show_timestamps: bool = True) -> str:
    """
    Format transcription result for display.

    Args:
        result: TranscriptionResult object
        show_timestamps: Whether to show timestamp chunks

    Returns:
        Formatted string for display
    """
    output = format_section_header("TRANSCRIPTION RESULT")
    output += f"\n{result.text}\n"

    if show_timestamps and result.chunks:
        output += "\n" + "-" * 60 + "\n"
        output += "Timestamps:\n"
        output += "-" * 60 + "\n"
        for chunk in result.chunks:
            output += f"[{chunk.start_time:.2f}s - {chunk.end_time:.2f}s]: {chunk.text}\n"
        output += "\n" + "=" * 60

    return output


def format_performance_metrics(metrics: PerformanceMetrics) -> str:
    """
    Format performance metrics table.

    Args:
        metrics: PerformanceMetrics object

    Returns:
        Formatted string for display
    """
    output = format_section_header("PERFORMANCE METRICS")

    if metrics.audio_duration:
        output += f"\nAudio Duration:        {metrics.audio_duration:.2f} seconds\n"
    output += f"Total Processing Time: {metrics.processing_time:.2f} seconds\n"

    if metrics.rtf is not None:
        rtf_status = "faster than real-time" if metrics.rtf < 1.0 else "slower than real-time"
        output += f"Real-Time Factor (RTF): {metrics.rtf:.2f}x ({rtf_status})\n"

    if metrics.num_chunks is not None:
        output += f"\nNumber of Chunks:      {metrics.num_chunks}\n"
    if metrics.avg_chunk_latency is not None:
        output += f"Avg Chunk Latency:     {metrics.avg_chunk_latency*1000:.2f} ms\n"

    output += "\n" + "=" * 60

    return output


def format_accuracy_metrics(
    cer: float,
    wer: float,
    reference: str,
    hypothesis: str
) -> str:
    """
    Format accuracy comparison.

    Args:
        cer: Character Error Rate (0.0 to 1.0)
        wer: Word Error Rate (0.0 to 1.0)
        reference: Ground truth transcript
        hypothesis: ASR output transcript

    Returns:
        Formatted string for display
    """
    output = format_section_header("ACCURACY METRICS (vs Ground Truth)")
    output += f"\nReference:  {reference}\n"
    output += f"Hypothesis: {hypothesis}\n"
    output += f"\nCharacter Error Rate (CER): {cer*100:.2f}%\n"
    output += f"Word Error Rate (WER):       {wer*100:.2f}%\n"
    output += "\n" + "=" * 60

    return output


def format_per_sample_table_header() -> str:
    """Format the header for per-sample results table"""
    output = "=" * 80 + "\n"
    output += "PER-SAMPLE RESULTS\n"
    output += "=" * 80 + "\n\n"
    output += f"{'Index':<7} {'Audio File':<18} {'Duration':<10} {'CER(%)':<9} {'WER(%)':<9} "
    output += f"{'RTF':<8} {'Time(s)':<9} {'Status':<7}\n"
    output += "-" * 80 + "\n"
    return output


def format_per_sample_row(
    index: int,
    audio_file: str,
    duration: float,
    cer: Optional[float],
    wer: Optional[float],
    rtf: Optional[float],
    processing_time: Optional[float],
    status: str
) -> str:
    """
    Format a single row in the per-sample results table.

    Args:
        index: Sample index
        audio_file: Audio filename
        duration: Audio duration in seconds
        cer: Character Error Rate (or None)
        wer: Word Error Rate (or None)
        rtf: Real-Time Factor (or None)
        processing_time: Processing time in seconds (or None)
        status: 'success' or 'error'

    Returns:
        Formatted table row string
    """
    status_symbol = "✓" if status == 'success' else "✗"
    cer_str = f"{cer*100:.2f}" if cer is not None else "N/A"
    wer_str = f"{wer*100:.2f}" if wer is not None else "N/A"
    rtf_str = f"{rtf:.2f}" if rtf is not None else "N/A"
    time_str = f"{processing_time:.2f}" if processing_time is not None else "N/A"

    return (f"{index:<7} {audio_file:<18} {duration:.2f}s{'':<6} "
            f"{cer_str:<9} {wer_str:<9} {rtf_str:<8} {time_str:<9} {status_symbol:<7}\n")


def format_summary_header(model_name: str, total: int, successful: int, failed: int) -> str:
    """
    Format the summary statistics header.

    Args:
        model_name: Name of the ASR model
        total: Total number of samples
        successful: Number of successful transcriptions
        failed: Number of failed transcriptions

    Returns:
        Formatted header string
    """
    output = "\n" + "=" * 80 + "\n"
    output += "SUMMARY STATISTICS\n"
    output += "=" * 80 + "\n\n"

    output += f"Evaluation Model: {model_name}\n"
    output += f"Total Samples:    {total}\n"
    output += f"Successful:       {successful} ({successful/total*100:.1f}%)\n"
    output += f"Failed:           {failed} ({failed/total*100:.1f}%)\n\n"

    return output


def format_statistics_table(
    cer_stats: dict,
    wer_stats: dict,
    rtf_stats: dict,
    time_stats: dict
) -> str:
    """
    Format the statistics table.

    Args:
        cer_stats: Dict with 'mean', 'median', 'std', 'min', 'max' for CER
        wer_stats: Dict with 'mean', 'median', 'std', 'min', 'max' for WER
        rtf_stats: Dict with 'mean', 'median', 'std', 'min', 'max' for RTF
        time_stats: Dict with 'mean', 'median', 'std', 'min', 'max' for processing time

    Returns:
        Formatted statistics table string
    """
    output = f"{'Metric':<21} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}\n"
    output += "-" * 80 + "\n"

    output += (f"{'CER (%)':<21} {cer_stats['mean']*100:<10.2f} {cer_stats['median']*100:<10.2f} "
               f"{cer_stats['std']*100:<10.2f} {cer_stats['min']*100:<10.2f} {cer_stats['max']*100:<10.2f}\n")
    output += (f"{'WER (%)':<21} {wer_stats['mean']*100:<10.2f} {wer_stats['median']*100:<10.2f} "
               f"{wer_stats['std']*100:<10.2f} {wer_stats['min']*100:<10.2f} {wer_stats['max']*100:<10.2f}\n")
    output += (f"{'RTF':<21} {rtf_stats['mean']:<10.2f} {rtf_stats['median']:<10.2f} "
               f"{rtf_stats['std']:<10.2f} {rtf_stats['min']:<10.2f} {rtf_stats['max']:<10.2f}\n")
    output += (f"{'Processing Time (s)':<21} {time_stats['mean']:<10.2f} {time_stats['median']:<10.2f} "
               f"{time_stats['std']:<10.2f} {time_stats['min']:<10.2f} {time_stats['max']:<10.2f}\n")
    output += "-" * 80 + "\n"

    return output


def format_performance_assessment(rtf_mean: float, cer_mean: float) -> str:
    """
    Format the performance assessment text.

    Args:
        rtf_mean: Mean Real-Time Factor
        cer_mean: Mean Character Error Rate

    Returns:
        Formatted assessment string
    """
    output = "\nPerformance Assessment:\n"

    if rtf_mean < 1.0:
        output += "  ✓ Average RTF < 1.0 (faster than real-time)\n"
    else:
        output += "  ✗ Average RTF > 1.0 (slower than real-time)\n"

    if cer_mean < 0.10:
        output += "  ✓ Average CER < 10% (excellent accuracy)\n"
    elif cer_mean < 0.20:
        output += "  ⚠ Average CER < 20% (good accuracy)\n"
    else:
        output += "  ✗ Average CER > 20% (needs improvement)\n"

    return output
