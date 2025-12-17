#!/usr/bin/env python3
"""
Azure GPU Benchmark Script for ASR Models

Runs batch evaluation with baseline vs Flash Attention comparison.
Collects all 4 required metrics:
1. RTF & Latency (per-sample)
2. Throughput (files/hour, audio hours/hour)
3. Flash Attention Comparison (baseline vs Flash Attention)
4. GPU Utilization (GPU %, VRAM, temperature, power)

Usage:
    # Run both baseline and Flash Attention
    python scripts/benchmark_azure.py

    # Run baseline only
    python scripts/benchmark_azure.py --baseline-only

    # Run Flash Attention only
    python scripts/benchmark_azure.py --flash-only

    # Custom output directory
    python scripts/benchmark_azure.py --output-dir my_results
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dotenv import load_dotenv

# Import refactored ASR library
from asr.models.kotoba import KotobaWhisperModel
from asr.evaluation.evaluator import BatchEvaluator, EvaluationResult
from asr.evaluation.reporters import ResultsReporter, calculate_statistics
from asr.metrics.gpu_metrics import GPUMonitor


class AzureBenchmark:
    """
    Azure GPU benchmark orchestrator

    Responsibilities:
    1. Run batch evaluation with multiple configs (baseline + Flash Attention)
    2. Monitor GPU metrics during execution
    3. Calculate throughput metrics
    4. Generate comparison reports
    5. Export comprehensive JSON results
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize benchmark orchestrator.

        Args:
            output_dir: Directory for results output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)

    def run_benchmark(
        self,
        metadata_path: str,
        audio_dir: str,
        configs: List[Dict]
    ) -> Dict:
        """
        Run benchmark for all configurations.

        Args:
            metadata_path: Path to metadata.json
            audio_dir: Directory with audio files
            configs: List of config dicts [{"name": "baseline", "flash_attention": False}, ...]

        Returns:
            Dict with comprehensive results for all configs
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {
            "timestamp": timestamp,
            "model": "kotoba-whisper-v2.2",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "configs": []
        }

        for config in configs:
            print(f"\n{'='*70}")
            print(f"RUNNING CONFIG: {config['name'].upper()}")
            print(f"Flash Attention: {config['flash_attention']}")
            print(f"{'='*70}\n")

            # Start GPU monitoring
            gpu_monitor = GPUMonitor(
                output_dir=str(self.output_dir / "logs"),
                interval=1.0
            )
            gpu_monitor.start()

            try:
                # Create model with config
                print(f"Loading model...")
                model = KotobaWhisperModel(
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                    flash_attention=config['flash_attention']
                )
                print(f"Model loaded on {model._device}")
                print()

                # Create evaluator
                evaluator = BatchEvaluator(model=model)

                # Run batch evaluation (METRIC 1: RTF & Latency)
                print(f"Starting batch evaluation...")
                start_time = time.perf_counter()

                results = evaluator.evaluate_dataset(
                    metadata_path=metadata_path,
                    audio_dir=audio_dir,
                    progress_callback=ResultsReporter.display_progress
                )

                total_time = time.perf_counter() - start_time
                print(f"\n\nBatch completed in {total_time:.2f}s\n")

                # Calculate METRIC 2: Throughput
                successful = [r for r in results if r.status == 'success']
                total_audio_duration = sum(r.duration for r in successful)
                total_processing_time = sum(r.processing_time for r in successful if r.processing_time)

                throughput_metrics = {
                    "files_per_hour": (len(successful) / total_processing_time) * 3600 if total_processing_time > 0 else 0,
                    "audio_hours_per_hour": (total_audio_duration / 3600) / (total_processing_time / 3600) if total_processing_time > 0 else 0,
                    "average_rtf": total_processing_time / total_audio_duration if total_audio_duration > 0 else 0,
                    "total_files": len(results),
                    "successful_files": len(successful),
                    "total_audio_duration": total_audio_duration,
                    "total_processing_time": total_processing_time
                }

                # Display throughput
                print(f"\nThroughput Metrics:")
                print(f"  Files/hour: {throughput_metrics['files_per_hour']:.2f}")
                print(f"  Audio hours/hour: {throughput_metrics['audio_hours_per_hour']:.2f}")
                print(f"  Average RTF: {throughput_metrics['average_rtf']:.3f}x\n")

                # Stop GPU monitoring (METRIC 4: GPU utilization)
                gpu_summary = gpu_monitor.stop()

                # Package results for this config
                config_results = {
                    "config_name": config['name'],
                    "flash_attention": config['flash_attention'],
                    "evaluation_results": [
                        {
                            "index": r.index,
                            "audio_file": r.audio_file,
                            "duration": r.duration,
                            "reference": r.reference,
                            "hypothesis": r.hypothesis,
                            "cer": r.cer,
                            "wer": r.wer,
                            "processing_time": r.processing_time,
                            "rtf": r.rtf,
                            "status": r.status,
                            "error": r.error,
                            "flash_attention": r.flash_attention,
                            "timestamp": r.timestamp
                        }
                        for r in results
                    ],
                    "throughput_metrics": throughput_metrics,
                    "accuracy_statistics": self._calculate_accuracy_stats(successful)
                }

                # Add GPU metrics if available
                if gpu_summary:
                    config_results["gpu_metrics"] = {
                        "csv_file": gpu_summary.csv_file,
                        "json_file": gpu_summary.json_file,
                        "summary": {
                            "duration_seconds": gpu_summary.duration_seconds,
                            "samples": gpu_summary.samples,
                            "gpu_utilization": gpu_summary.gpu_utilization,
                            "memory_usage_mb": gpu_summary.memory_usage_mb,
                            "temperature_c": gpu_summary.temperature_c,
                            "power_draw_w": gpu_summary.power_draw_w
                        }
                    }
                else:
                    config_results["gpu_metrics"] = None

                all_results["configs"].append(config_results)

            finally:
                # Clean up
                if gpu_monitor.running:
                    gpu_monitor.stop()

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Cool down between configs
                print("\nCooling down...")
                time.sleep(5)

        # Save comprehensive results
        results_file = self.output_dir / f"benchmark_azure_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n\nResults saved to: {results_file}\n")

        # Generate METRIC 3: Flash Attention comparison report
        if len(configs) > 1:
            self.generate_comparison_report(all_results)

        return all_results

    def _calculate_accuracy_stats(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate accuracy statistics from results.

        Args:
            results: List of successful EvaluationResult objects

        Returns:
            Dict with CER and WER statistics
        """
        cer_values = [r.cer for r in results if r.cer is not None]
        wer_values = [r.wer for r in results if r.wer is not None]

        return {
            "cer": calculate_statistics(cer_values) if cer_values else None,
            "wer": calculate_statistics(wer_values) if wer_values else None
        }

    def generate_comparison_report(self, all_results: Dict) -> None:
        """
        Generate METRIC 3: Flash Attention vs Baseline comparison report.

        Args:
            all_results: Complete benchmark results dict
        """
        print(f"\n{'='*70}")
        print("FLASH ATTENTION vs BASELINE COMPARISON")
        print(f"{'='*70}\n")

        configs = all_results["configs"]

        # Find baseline and flash configs
        baseline = next((c for c in configs if not c['flash_attention']), None)
        flash = next((c for c in configs if c['flash_attention']), None)

        if not baseline or not flash:
            print("ERROR: Missing baseline or flash attention config")
            return

        # RTF Comparison
        baseline_rtf = baseline['throughput_metrics']['average_rtf']
        flash_rtf = flash['throughput_metrics']['average_rtf']
        rtf_improvement = ((baseline_rtf - flash_rtf) / baseline_rtf) * 100 if baseline_rtf > 0 else 0

        print("1. Real-Time Factor (RTF)")
        print(f"   Baseline:         {baseline_rtf:.3f}x")
        print(f"   Flash Attention:  {flash_rtf:.3f}x")
        print(f"   Improvement:      {rtf_improvement:+.1f}%")
        print()

        # Throughput Comparison
        baseline_throughput = baseline['throughput_metrics']['files_per_hour']
        flash_throughput = flash['throughput_metrics']['files_per_hour']
        throughput_improvement = ((flash_throughput - baseline_throughput) / baseline_throughput) * 100 if baseline_throughput > 0 else 0

        print("2. Throughput (files/hour)")
        print(f"   Baseline:         {baseline_throughput:.2f}")
        print(f"   Flash Attention:  {flash_throughput:.2f}")
        print(f"   Improvement:      {throughput_improvement:+.1f}%")
        print()

        # Accuracy Comparison
        if baseline['accuracy_statistics']['cer'] and flash['accuracy_statistics']['cer']:
            baseline_cer = baseline['accuracy_statistics']['cer']['mean']
            flash_cer = flash['accuracy_statistics']['cer']['mean']

            print("3. Accuracy (CER)")
            print(f"   Baseline:         {baseline_cer:.2%}")
            print(f"   Flash Attention:  {flash_cer:.2%}")
            print(f"   Delta:            {abs(baseline_cer - flash_cer):.2%}")
            print()

        # GPU Metrics Comparison
        if baseline.get('gpu_metrics') and flash.get('gpu_metrics'):
            baseline_gpu = baseline['gpu_metrics']['summary']['gpu_utilization']['mean']
            flash_gpu = flash['gpu_metrics']['summary']['gpu_utilization']['mean']

            baseline_vram = baseline['gpu_metrics']['summary']['memory_usage_mb']['max']
            flash_vram = flash['gpu_metrics']['summary']['memory_usage_mb']['max']

            print("4. GPU Utilization")
            print(f"   Baseline:         {baseline_gpu:.1f}%")
            print(f"   Flash Attention:  {flash_gpu:.1f}%")
            print()

            print("5. Peak VRAM Usage")
            print(f"   Baseline:         {baseline_vram:.0f} MB")
            print(f"   Flash Attention:  {flash_vram:.0f} MB")
            print()

        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Azure GPU Benchmark for ASR Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--metadata', default="dataset/sample_audios/metadata.json",
                       help='Path to metadata.json (default: dataset/sample_audios/metadata.json)')
    parser.add_argument('--audio-dir', default="dataset/sample_audios",
                       help='Directory with audio files (default: dataset/sample_audios)')
    parser.add_argument('--output-dir', default="results",
                       help='Output directory for results (default: results)')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run baseline only (skip Flash Attention)')
    parser.add_argument('--flash-only', action='store_true',
                       help='Run Flash Attention only (skip baseline)')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Define test configurations
    configs = []
    if not args.flash_only:
        configs.append({"name": "baseline", "flash_attention": False})
    if not args.baseline_only:
        configs.append({"name": "flash_attention", "flash_attention": True})

    if not configs:
        print("ERROR: Must run at least one configuration")
        print("Use --baseline-only or --flash-only, not both")
        return 1

    # Display configuration
    print(f"\n{'='*70}")
    print("AZURE GPU BENCHMARK")
    print(f"{'='*70}")
    print(f"Metadata: {args.metadata}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Configurations: {', '.join(c['name'] for c in configs)}")
    print(f"{'='*70}\n")

    # Run benchmark
    benchmark = AzureBenchmark(output_dir=args.output_dir)
    results = benchmark.run_benchmark(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        configs=configs
    )

    print("Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
