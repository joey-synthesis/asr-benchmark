#!/usr/bin/env python3
"""Batch evaluation of ASR models on test dataset"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asr.models import KotobaWhisperModel, AssemblyAIModel
from asr.evaluation import BatchEvaluator, ResultsReporter

METADATA_PATH = "dataset/sample_audios/metadata.json"
AUDIO_DIR = "dataset/sample_audios"


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of ASR models"
    )
    parser.add_argument('--model', choices=['kotoba', 'assemblyai'],
                       default='kotoba', help='ASR model to evaluate')
    parser.add_argument('--output', type=str, metavar='FILE',
                       help='Export results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show per-sample transcription details')
    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print("BATCH ASR EVALUATION")
    print("=" * 80)
    print()

    # Create model
    print(f"Initializing {args.model} model...")
    if args.model == 'kotoba':
        model = KotobaWhisperModel()
    elif args.model == 'assemblyai':
        model = AssemblyAIModel()
    else:
        print(f"Error: Unknown model '{args.model}'")
        return 1
    print()

    # Evaluate
    evaluator = BatchEvaluator(model)
    reporter = ResultsReporter()

    print(f"Processing samples from {METADATA_PATH}...")
    print()

    results = evaluator.evaluate_dataset(
        METADATA_PATH,
        AUDIO_DIR,
        progress_callback=reporter.display_progress
    )

    print()  # New line after progress

    # Display results
    if args.verbose:
        print("\n")
        reporter.display_per_sample(results, verbose=True)

    reporter.display_summary(results, model.model_name)

    # Export if requested
    if args.output:
        reporter.export_json(results, model.model_name, args.output)
        print(f"\n✓ Results exported to {args.output}")

    # Exit code
    successful = sum(1 for r in results if r.status == 'success')
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
