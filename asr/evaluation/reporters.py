"""Results reporting and export for batch evaluation"""

import json
from datetime import datetime
from statistics import mean, median, stdev
from typing import List, Optional
from .evaluator import EvaluationResult


def calculate_statistics(values: List[float]) -> dict:
    """
    Calculate mean, median, std, min, max.

    Args:
        values: List of numerical values

    Returns:
        Dict with statistical measures
    """
    if not values:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    return {
        'mean': mean(values),
        'median': median(values),
        'std': stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values)
    }


class ResultsReporter:
    """Generate reports and statistics from evaluation results"""

    @staticmethod
    def display_progress(current: int, total: int, filename: str):
        """
        Show progress indicator.

        Args:
            current: Current sample number
            total: Total number of samples
            filename: Current filename being processed
        """
        percentage = (current / total) * 100
        print(f"\rProcessing samples... [{current}/{total}] ({percentage:.0f}%) - {filename}",
              end='', flush=True)

    @staticmethod
    def display_per_sample(results: List[EvaluationResult], verbose: bool = False):
        """
        Display per-sample results table.

        Args:
            results: List of EvaluationResult objects
            verbose: If True, show transcription details
        """
        print("=" * 80)
        print("PER-SAMPLE RESULTS")
        print("=" * 80)
        print()

        # Table header
        print(f"{'Index':<7} {'Audio File':<18} {'Duration':<10} {'CER(%)':<9} {'WER(%)':<9} "
              f"{'RTF':<8} {'Time(s)':<9} {'Status':<7}")
        print("-" * 80)

        # Table rows
        for r in results:
            status_symbol = "✓" if r.status == 'success' else "✗"
            cer_str = f"{r.cer*100:.2f}" if r.cer is not None else "N/A"
            wer_str = f"{r.wer*100:.2f}" if r.wer is not None else "N/A"
            rtf_str = f"{r.rtf:.2f}" if r.rtf is not None else "N/A"
            time_str = f"{r.processing_time:.2f}" if r.processing_time is not None else "N/A"

            print(f"{r.index:<7} {r.audio_file:<18} {r.duration:.2f}s{'':<6} "
                  f"{cer_str:<9} {wer_str:<9} {rtf_str:<8} {time_str:<9} {status_symbol:<7}")

        # Verbose details
        if verbose:
            print("\n" + "=" * 80)
            print("TRANSCRIPTION DETAILS")
            print("=" * 80)
            for r in results:
                print(f"\nSample {r.index}:")
                print(f"  Reference:  {r.reference}")
                print(f"  Hypothesis: {r.hypothesis}")
                status_text = '✓ Success' if r.status == 'success' else f'✗ Error: {r.error}'
                print(f"  Status: {status_text}")

    @staticmethod
    def display_summary(results: List[EvaluationResult], model_name: str):
        """
        Display aggregate statistics.

        Args:
            results: List of EvaluationResult objects
            model_name: Name of the ASR model
        """
        successful = [r for r in results if r.status == 'success']
        failed = [r for r in results if r.status == 'error']

        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print()

        print(f"Evaluation Model: {model_name}")
        print(f"Total Samples:    {len(results)}")
        print(f"Successful:       {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed:           {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print()

        if not successful:
            print("No successful transcriptions to show statistics.")
            return

        # Calculate statistics
        cer_values = [r.cer for r in successful if r.cer is not None]
        wer_values = [r.wer for r in successful if r.wer is not None]
        rtf_values = [r.rtf for r in successful if r.rtf is not None]
        time_values = [r.processing_time for r in successful if r.processing_time is not None]

        cer_stats = calculate_statistics(cer_values)
        wer_stats = calculate_statistics(wer_values)
        rtf_stats = calculate_statistics(rtf_values)
        time_stats = calculate_statistics(time_values)

        # Display table
        print(f"{'Metric':<21} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        print(f"{'CER (%)':<21} {cer_stats['mean']*100:<10.2f} {cer_stats['median']*100:<10.2f} "
              f"{cer_stats['std']*100:<10.2f} {cer_stats['min']*100:<10.2f} {cer_stats['max']*100:<10.2f}")
        print(f"{'WER (%)':<21} {wer_stats['mean']*100:<10.2f} {wer_stats['median']*100:<10.2f} "
              f"{wer_stats['std']*100:<10.2f} {wer_stats['min']*100:<10.2f} {wer_stats['max']*100:<10.2f}")
        print(f"{'RTF':<21} {rtf_stats['mean']:<10.2f} {rtf_stats['median']:<10.2f} "
              f"{rtf_stats['std']:<10.2f} {rtf_stats['min']:<10.2f} {rtf_stats['max']:<10.2f}")
        print(f"{'Processing Time (s)':<21} {time_stats['mean']:<10.2f} {time_stats['median']:<10.2f} "
              f"{time_stats['std']:<10.2f} {time_stats['min']:<10.2f} {time_stats['max']:<10.2f}")
        print("-" * 80)

        # Performance assessment
        print("\nPerformance Assessment:")
        if rtf_stats['mean'] < 1.0:
            print("  ✓ Average RTF < 1.0 (faster than real-time)")
        else:
            print("  ✗ Average RTF > 1.0 (slower than real-time)")

        if cer_stats['mean'] < 0.10:
            print("  ✓ Average CER < 10% (excellent accuracy)")
        elif cer_stats['mean'] < 0.20:
            print("  ⚠ Average CER < 20% (good accuracy)")
        else:
            print("  ✗ Average CER > 20% (needs improvement)")

    @staticmethod
    def export_json(
        results: List[EvaluationResult],
        model_name: str,
        output_path: str
    ):
        """
        Export results to JSON file.

        Args:
            results: List of EvaluationResult objects
            model_name: Name of the ASR model
            output_path: Path to output JSON file
        """
        successful = [r for r in results if r.status == 'success']

        # Calculate statistics
        cer_values = [r.cer for r in successful if r.cer is not None]
        wer_values = [r.wer for r in successful if r.wer is not None]
        rtf_values = [r.rtf for r in successful if r.rtf is not None]
        time_values = [r.processing_time for r in successful if r.processing_time is not None]

        export_data = {
            "metadata": {
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful)
            },
            "samples": [
                {
                    'index': r.index,
                    'audio_file': r.audio_file,
                    'duration': r.duration,
                    'reference': r.reference,
                    'hypothesis': r.hypothesis,
                    'cer': r.cer,
                    'wer': r.wer,
                    'processing_time': r.processing_time,
                    'rtf': r.rtf,
                    'status': r.status,
                    'error': r.error
                }
                for r in results
            ],
            "statistics": {
                "cer": calculate_statistics(cer_values),
                "wer": calculate_statistics(wer_values),
                "rtf": calculate_statistics(rtf_values),
                "processing_time": calculate_statistics(time_values)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
