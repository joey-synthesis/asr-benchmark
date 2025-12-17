#!/usr/bin/env python3
"""
AssemblyAI ASR Test Script with Performance Metrics
Tests AssemblyAI's transcription API and measures latency/RTF
"""

import sys
import time
import os
import argparse
import librosa
from dotenv import load_dotenv
import assemblyai as aai
from asr_metrics_utils import load_ground_truth, calculate_accuracy_metrics


def load_api_key():
    """Load AssemblyAI API key from environment"""
    # Load from .env file if it exists
    load_dotenv()

    api_key = os.getenv("ASSEMBLYAI_API_KEY")

    if not api_key:
        raise ValueError(
            "AssemblyAI API key not found!\n"
            "Please set ASSEMBLYAI_API_KEY in .env file or environment variable"
        )

    return api_key


def transcribe_with_assemblyai(audio_path, config=None):
    """
    Transcribe audio file using AssemblyAI API

    Args:
        audio_path: Path to local audio file or URL
        config: AssemblyAI TranscriptionConfig (optional)

    Returns:
        transcript: AssemblyAI Transcript object
    """
    print(f"Transcribing: {audio_path}")
    print("Uploading to AssemblyAI and processing...\n")

    # Use default config if not provided
    if config is None:
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,  # Use best model
            language_code="ja",  # Japanese language
        )

    # Create transcriber and transcribe
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    # Check for errors
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript


def transcribe_with_metrics(audio_path, config=None, ground_truth=None):
    """
    Transcribe audio and measure performance metrics

    Args:
        audio_path: Path to audio file or URL
        config: AssemblyAI TranscriptionConfig (optional)
        ground_truth: Ground truth transcript for accuracy (optional)

    Returns:
        tuple: (transcript, metrics_dict)
    """
    # Get audio duration for RTF calculation
    audio_duration = None
    if not audio_path.startswith("http"):
        # Only for local files
        try:
            audio_duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            print(f"Warning: Could not get audio duration: {e}")

    # Measure processing time
    start_time = time.perf_counter()

    transcript = transcribe_with_assemblyai(audio_path, config)

    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time

    metrics = {
        "total_time": total_time,
        "audio_duration": audio_duration,
        "rtf": total_time / audio_duration if audio_duration else None,
        "upload_time": None,  # AssemblyAI doesn't expose this separately
        "processing_time": None,  # Combined in total_time
    }

    # Try to get audio duration from transcript if not available
    if audio_duration is None and hasattr(transcript, 'audio_duration'):
        metrics["audio_duration"] = transcript.audio_duration / 1000.0  # ms to seconds
        metrics["rtf"] = total_time / metrics["audio_duration"]

    # Calculate accuracy metrics if ground truth available
    if ground_truth:
        hypothesis = transcript.text
        accuracy = calculate_accuracy_metrics(hypothesis, ground_truth)
        if accuracy:
            metrics['accuracy'] = {
                'cer': accuracy['cer'],
                'wer': accuracy['wer'],
                'reference': ground_truth,
                'hypothesis': hypothesis
            }

    return transcript, metrics


def display_results(transcript, metrics):
    """Display transcription results and performance metrics"""
    print("=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print()
    print(transcript.text)
    print()
    print("=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    if metrics["audio_duration"]:
        print(f"Audio duration:    {metrics['audio_duration']:.2f}s")

    print(f"Total time:        {metrics['total_time']:.2f}s")

    if metrics["rtf"]:
        rtf_status = "faster than real-time" if metrics["rtf"] < 1.0 else "slower than real-time"
        print(f"Real-Time Factor:  {metrics['rtf']:.2f}x ({rtf_status})")

    print()
    print("=" * 60)
    print("TRANSCRIPT DETAILS")
    print("=" * 60)
    print(f"Status:            {transcript.status}")
    print(f"ID:                {transcript.id}")

    if hasattr(transcript, 'confidence'):
        print(f"Confidence:        {transcript.confidence:.2%}")

    if hasattr(transcript, 'audio_duration'):
        print(f"Audio duration:    {transcript.audio_duration / 1000.0:.2f}s")

    if hasattr(transcript, 'words') and transcript.words:
        print(f"Word count:        {len(transcript.words)}")

    print("=" * 60)

    # Display accuracy metrics if available
    if 'accuracy' in metrics:
        print()
        print("=" * 60)
        print("ACCURACY METRICS (vs Ground Truth)")
        print("=" * 60)

        acc = metrics['accuracy']
        print(f"\nReference:  {acc['reference']}")
        print(f"Hypothesis: {acc['hypothesis']}")
        print(f"\nCharacter Error Rate (CER): {acc['cer']*100:.2f}%")
        print(f"Word Error Rate (WER):       {acc['wer']*100:.2f}%")

        print("=" * 60)


def compare_with_kotoba(assemblyai_result, metrics):
    """
    Display comparison with Kotoba Whisper results

    Based on previous test results:
    - Kotoba Whisper (CPU): RTF ~1.53x, Latency ~3.56s per 3s chunk
    - Kotoba Whisper (GPU): RTF ~0.2-0.4x (estimated)
    """
    print()
    print("=" * 60)
    print("COMPARISON WITH KOTOBA WHISPER v2.2")
    print("=" * 60)
    print()

    if metrics["rtf"]:
        print(f"{'Model':<25} {'RTF':<15} {'Status':<25}")
        print("-" * 60)
        print(f"{'AssemblyAI (Cloud)':<25} {metrics['rtf']:.2f}x{'':<12} {'(This test)':<25}")
        print(f"{'Kotoba Whisper (CPU)':<25} {'1.53x':<15} {'(Previous test)':<25}")
        print(f"{'Kotoba Whisper (GPU)':<25} {'0.2-0.4x':<15} {'(Estimated)':<25}")
        print()

        if metrics['rtf'] < 1.0:
            print("✓ AssemblyAI is FASTER than real-time (RTF < 1.0)")
        else:
            print("⚠ AssemblyAI is SLOWER than real-time (RTF > 1.0)")

        if metrics['rtf'] < 1.53:
            print("✓ AssemblyAI is FASTER than Kotoba Whisper on CPU")
        else:
            print("⚠ AssemblyAI is SLOWER than Kotoba Whisper on CPU")

    print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test AssemblyAI ASR with performance metrics"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="dataset/sample_audio.mp3",
        help="Path to audio file or URL (default: dataset/sample_audio.mp3)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language code (default: ja for Japanese)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["best", "nano"],
        default="best",
        help="Speech model to use (default: best)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison with Kotoba Whisper"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AssemblyAI ASR Performance Test")
    print("=" * 60)
    print()

    # Load API key
    try:
        api_key = load_api_key()
        aai.settings.api_key = api_key
        print("✓ API key loaded successfully")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Configure transcription
    speech_model = aai.SpeechModel.best if args.model == "best" else aai.SpeechModel.nano

    config = aai.TranscriptionConfig(
        speech_model=speech_model,
        language_code=args.language,
    )

    print(f"Model:             {args.model}")
    print(f"Language:          {args.language}")
    print(f"Audio:             {args.audio}")
    print()

    # Load ground truth if available
    ground_truth = load_ground_truth(args.audio)
    if ground_truth:
        print("✓ Ground truth loaded for accuracy measurement\n")

    # Transcribe and measure
    try:
        transcript, metrics = transcribe_with_metrics(args.audio, config, ground_truth=ground_truth)

        # Display results
        display_results(transcript, metrics)

        # Show comparison if requested
        if args.compare:
            compare_with_kotoba(transcript, metrics)

        return 0

    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
