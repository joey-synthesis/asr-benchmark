#!/usr/bin/env python3
"""Simple batch audio transcription"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from asr.models import KotobaWhisperModel, AssemblyAIModel
from asr.metrics import load_ground_truth, calculate_accuracy_metrics
from asr.metrics.performance import measure_performance
from asr.display.formatters import (
    format_transcription_result,
    format_performance_metrics,
    format_accuracy_metrics
)


def main():
    parser = argparse.ArgumentParser(description="Batch audio transcription")
    parser.add_argument('audio', help='Audio file path')
    parser.add_argument('--model', choices=['kotoba', 'assemblyai'],
                       default='kotoba', help='ASR model to use')
    parser.add_argument('--language', default='ja', help='Language code')
    parser.add_argument('--show-timestamps', action='store_true',
                       help='Show timestamp chunks')
    args = parser.parse_args()

    # Create model
    print(f"Loading {args.model} model...")
    if args.model == 'kotoba':
        model = KotobaWhisperModel()
    else:
        model = AssemblyAIModel()

    # Transcribe
    print(f"Transcribing: {args.audio}\n")
    start_time = time.perf_counter()
    result = model.transcribe(args.audio, language=args.language,
                             return_timestamps=args.show_timestamps)
    processing_time = time.perf_counter() - start_time

    # Display result
    print(format_transcription_result(result, show_timestamps=args.show_timestamps))

    # Performance metrics
    perf_metrics = measure_performance(args.audio, processing_time)
    print(format_performance_metrics(perf_metrics))

    # Accuracy metrics (if ground truth available)
    ground_truth = load_ground_truth(args.audio)
    if ground_truth:
        accuracy = calculate_accuracy_metrics(result.text, ground_truth)
        if accuracy:
            print(format_accuracy_metrics(
                accuracy['cer'], accuracy['wer'],
                ground_truth, result.text
            ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
