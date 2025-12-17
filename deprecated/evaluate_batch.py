#!/usr/bin/env python3
"""
Batch Evaluation Script for ASR Models
Processes all samples in dataset/sample_audios and generates comprehensive metrics
"""

import sys
import argparse
import json
import time
import os
from datetime import datetime
from statistics import mean, median, stdev

import torch
import librosa
from dotenv import load_dotenv
import assemblyai as aai

# Import existing utilities
from asr_metrics_utils import load_ground_truth, calculate_accuracy_metrics
from test_asr_optimized import load_pipeline, transcribe_with_latency
from test_asr_assemblyai import load_api_key, transcribe_with_metrics


# Constants
METADATA_PATH = "dataset/sample_audios/metadata.json"
AUDIO_DIR = "dataset/sample_audios"


# Helper Functions

def load_metadata():
    """Load test samples from metadata.json"""
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if not isinstance(metadata, list) or len(metadata) == 0:
            raise ValueError("Metadata must be a non-empty list")

        return metadata

    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in metadata file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        sys.exit(1)


def calculate_statistics(values):
    """Calculate mean, median, std, min, max for a list of numbers"""
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }

    return {
        'mean': mean(values),
        'median': median(values),
        'std': stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values)
    }


def display_progress(current, total, filename):
    """Show progress indicator"""
    percentage = (current / total) * 100
    print(f"\rProcessing samples... [{current}/{total}] ({percentage:.0f}%) - {filename}", end='', flush=True)


def display_per_sample_results(results, verbose=False):
    """Display detailed per-sample table"""
    print("=" * 80)
    print("PER-SAMPLE RESULTS")
    print("=" * 80)
    print()

    # Table header
    print(f"{'Index':<7} {'Audio File':<18} {'Duration':<10} {'CER(%)':<9} {'WER(%)':<9} {'RTF':<8} {'Time(s)':<9} {'Status':<7}")
    print("-" * 80)

    # Table rows
    for r in results:
        status_symbol = "✓" if r['status'] == 'success' else "✗"
        cer_str = f"{r['cer']*100:.2f}" if r['cer'] is not None else "N/A"
        wer_str = f"{r['wer']*100:.2f}" if r['wer'] is not None else "N/A"
        rtf_str = f"{r['rtf']:.2f}" if r['rtf'] is not None else "N/A"
        time_str = f"{r['processing_time']:.2f}" if r['processing_time'] is not None else "N/A"

        print(f"{r['index']:<7} {r['audio_file']:<18} {r['duration']:.2f}s{'':<6} {cer_str:<9} {wer_str:<9} {rtf_str:<8} {time_str:<9} {status_symbol:<7}")

    # Show transcription details if verbose
    if verbose:
        print("\n" + "=" * 80)
        print("TRANSCRIPTION DETAILS")
        print("=" * 80)
        for r in results:
            print(f"\nSample {r['index']}:")
            print(f"  Reference:  {r['reference']}")
            print(f"  Hypothesis: {r['hypothesis']}")
            print(f"  Status: {'✓ Success' if r['status'] == 'success' else '✗ Error: ' + str(r['error'])}")


def display_summary_statistics(results, model_name):
    """Display aggregate statistics table"""
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # Model and sample info
    model_display = "Kotoba Whisper v2.2" if model_name == 'kotoba' else "AssemblyAI"
    print(f"Evaluation Model: {model_display}")
    print(f"Total Samples:    {len(results)}")
    print(f"Successful:       {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed:           {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print()

    if not successful:
        print("No successful transcriptions to show statistics.")
        return

    # Calculate statistics
    cer_values = [r['cer'] for r in successful if r['cer'] is not None]
    wer_values = [r['wer'] for r in successful if r['wer'] is not None]
    rtf_values = [r['rtf'] for r in successful if r['rtf'] is not None]
    time_values = [r['processing_time'] for r in successful if r['processing_time'] is not None]

    cer_stats = calculate_statistics(cer_values)
    wer_stats = calculate_statistics(wer_values)
    rtf_stats = calculate_statistics(rtf_values)
    time_stats = calculate_statistics(time_values)

    # Display statistics table
    print(f"{'Metric':<21} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    print(f"{'CER (%)':<21} {cer_stats['mean']*100:<10.2f} {cer_stats['median']*100:<10.2f} {cer_stats['std']*100:<10.2f} {cer_stats['min']*100:<10.2f} {cer_stats['max']*100:<10.2f}")
    print(f"{'WER (%)':<21} {wer_stats['mean']*100:<10.2f} {wer_stats['median']*100:<10.2f} {wer_stats['std']*100:<10.2f} {wer_stats['min']*100:<10.2f} {wer_stats['max']*100:<10.2f}")
    print(f"{'RTF':<21} {rtf_stats['mean']:<10.2f} {rtf_stats['median']:<10.2f} {rtf_stats['std']:<10.2f} {rtf_stats['min']:<10.2f} {rtf_stats['max']:<10.2f}")
    print(f"{'Processing Time (s)':<21} {time_stats['mean']:<10.2f} {time_stats['median']:<10.2f} {time_stats['std']:<10.2f} {time_stats['min']:<10.2f} {time_stats['max']:<10.2f}")
    print("-" * 80)

    # Summary metrics
    total_time = sum(time_values) if time_values else 0
    avg_time = mean(time_values) if time_values else 0
    print(f"\nTotal Processing Time: {total_time:.2f} seconds")
    print(f"Average Time per Sample: {avg_time:.2f} seconds")

    # Performance assessment
    print("\nPerformance Assessment:")
    if rtf_stats['mean'] < 1.0:
        print(f"  ✓ Average RTF < 1.0 (faster than real-time)")
    else:
        print(f"  ✗ Average RTF > 1.0 (slower than real-time)")

    if cer_stats['mean'] < 0.10:
        print(f"  ✓ Average CER < 10% (excellent accuracy)")
    elif cer_stats['mean'] < 0.20:
        print(f"  ⚠ Average CER < 20% (good accuracy)")
    else:
        print(f"  ✗ Average CER > 20% (needs improvement)")


def export_to_json(results, model_name, output_path):
    """Export results to JSON file"""
    try:
        # Filter successful results for statistics
        successful = [r for r in results if r['status'] == 'success']

        # Calculate statistics
        cer_values = [r['cer'] for r in successful if r['cer'] is not None]
        wer_values = [r['wer'] for r in successful if r['wer'] is not None]
        rtf_values = [r['rtf'] for r in successful if r['rtf'] is not None]
        time_values = [r['processing_time'] for r in successful if r['processing_time'] is not None]

        # Prepare export data
        export_data = {
            "metadata": {
                "model": "kotoba-whisper-v2.2" if model_name == 'kotoba' else "assemblyai",
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful)
            },
            "samples": results,
            "statistics": {
                "cer": calculate_statistics(cer_values),
                "wer": calculate_statistics(wer_values),
                "rtf": calculate_statistics(rtf_values),
                "processing_time": calculate_statistics(time_values)
            }
        }

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"\nError exporting to JSON: {e}")


# Core Evaluation Functions

def evaluate_sample_kotoba(sample, pipe, verbose=False):
    """Evaluate single sample with Kotoba Whisper"""
    audio_path = os.path.join(AUDIO_DIR, sample['audio_file'])
    ground_truth = sample['transcript']

    try:
        # Transcribe with metrics
        result, latency_metrics = transcribe_with_latency(
            audio_path,
            pipe,
            ground_truth=ground_truth,
            language="ja",
            task="transcribe",
            chunk_length_s=15,
            batch_size=8
        )

        # Extract metrics
        hypothesis = result.get('text', '')
        cer = None
        wer = None

        if 'accuracy' in latency_metrics:
            cer = latency_metrics['accuracy']['cer']
            wer = latency_metrics['accuracy']['wer']

        return {
            'index': sample['index'],
            'audio_file': sample['audio_file'],
            'duration': sample['duration_seconds'],
            'reference': ground_truth,
            'hypothesis': hypothesis,
            'cer': cer,
            'wer': wer,
            'processing_time': latency_metrics.get('processing_time'),
            'rtf': latency_metrics.get('rtf'),
            'status': 'success',
            'error': None
        }

    except Exception as e:
        return {
            'index': sample['index'],
            'audio_file': sample['audio_file'],
            'duration': sample['duration_seconds'],
            'reference': ground_truth,
            'hypothesis': None,
            'cer': None,
            'wer': None,
            'processing_time': None,
            'rtf': None,
            'status': 'error',
            'error': str(e)
        }


def evaluate_sample_assemblyai(sample, config, verbose=False):
    """Evaluate single sample with AssemblyAI"""
    audio_path = os.path.join(AUDIO_DIR, sample['audio_file'])
    ground_truth = sample['transcript']

    try:
        # Transcribe with metrics
        transcript, metrics = transcribe_with_metrics(
            audio_path,
            config,
            ground_truth=ground_truth
        )

        # Extract metrics
        hypothesis = transcript.text
        cer = None
        wer = None

        if 'accuracy' in metrics:
            cer = metrics['accuracy']['cer']
            wer = metrics['accuracy']['wer']

        return {
            'index': sample['index'],
            'audio_file': sample['audio_file'],
            'duration': sample['duration_seconds'],
            'reference': ground_truth,
            'hypothesis': hypothesis,
            'cer': cer,
            'wer': wer,
            'processing_time': metrics.get('total_time'),
            'rtf': metrics.get('rtf'),
            'status': 'success',
            'error': None
        }

    except Exception as e:
        return {
            'index': sample['index'],
            'audio_file': sample['audio_file'],
            'duration': sample['duration_seconds'],
            'reference': ground_truth,
            'hypothesis': None,
            'cer': None,
            'wer': None,
            'processing_time': None,
            'rtf': None,
            'status': 'error',
            'error': str(e)
        }


def evaluate_batch(metadata, model='kotoba', verbose=False):
    """Process all samples and collect results"""
    results = []

    # Initialize model
    if model == 'kotoba':
        print("Initializing Kotoba Whisper v2.2 pipeline...")
        pipe = load_pipeline()
        print()
    else:
        print("Initializing AssemblyAI...")
        api_key = load_api_key()
        aai.settings.api_key = api_key
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_code="ja"
        )
        print()

    # Process all samples
    print(f"Processing {len(metadata)} samples...")
    print()

    try:
        for i, sample in enumerate(metadata, 1):
            display_progress(i, len(metadata), sample['audio_file'])

            if model == 'kotoba':
                result = evaluate_sample_kotoba(sample, pipe, verbose)
            else:
                result = evaluate_sample_assemblyai(sample, config, verbose)

            results.append(result)

        print()  # New line after progress

    except KeyboardInterrupt:
        print("\n\nBatch evaluation interrupted by user.")
        print(f"Processed {len(results)} out of {len(metadata)} samples.")

    return results


# Main Function

def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Batch evaluation of ASR models on Japanese audio dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_batch.py                           # Kotoba Whisper, summary only
  python evaluate_batch.py --verbose                 # Show per-sample details
  python evaluate_batch.py --output results.json     # Export to JSON
  python evaluate_batch.py --model assemblyai        # Test AssemblyAI instead
        """
    )
    parser.add_argument('--model', choices=['kotoba', 'assemblyai'], default='kotoba',
                        help='ASR model to evaluate (default: kotoba)')
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

    # Load metadata
    metadata = load_metadata()
    print(f"Loaded {len(metadata)} samples from metadata.json")
    print()

    # Run batch evaluation
    results = evaluate_batch(metadata, model=args.model, verbose=args.verbose)

    # Display per-sample results if verbose
    if args.verbose:
        print("\n")
        display_per_sample_results(results, verbose=True)

    # Display summary statistics (always)
    display_summary_statistics(results, args.model)

    # Export to JSON if requested
    if args.output:
        export_to_json(results, args.model, args.output)
        print(f"\n✓ Results exported to {args.output}")

    # Return exit code
    successful = sum(1 for r in results if r['status'] == 'success')
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
