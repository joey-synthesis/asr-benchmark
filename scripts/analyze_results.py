#!/usr/bin/env python3
"""
Results Analysis and Visualization Script

Generates comparison plots and summary reports from Azure benchmark results.

Usage:
    python scripts/analyze_results.py results/benchmark_azure_*.json

Outputs:
    - results/plots/rtf_comparison.png
    - results/plots/latency_distribution.png
    - results/plots/gpu_utilization_timeline.png
    - results/plots/throughput_comparison.png
    - results/plots/summary_report.txt
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results"""

    def __init__(self, results_json: str):
        """
        Initialize analyzer.

        Args:
            results_json: Path to benchmark results JSON file
        """
        self.results_path = Path(results_json)
        self.output_dir = self.results_path.parent / "plots"
        self.output_dir.mkdir(exist_ok=True)

        # Load results
        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

    def analyze(self):
        """Run full analysis suite"""
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS ANALYSIS")
        print(f"{'='*70}")
        print(f"Results file: {self.results_path}")
        print(f"Output directory: {self.output_dir}\n")

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

        # Generate all visualizations
        self.plot_rtf_comparison()
        self.plot_latency_distribution()
        self.plot_gpu_utilization_timeline()
        self.plot_throughput_comparison()
        self.generate_summary_report()

        print(f"\n{'='*70}")
        print("Analysis complete!")
        print(f"{'='*70}\n")

    def plot_rtf_comparison(self):
        """Plot RTF comparison across configs (per-sample)"""
        fig, ax = plt.subplots()

        for config in self.results['configs']:
            eval_results = config['evaluation_results']
            successful = [r for r in eval_results if r['status'] == 'success']

            indices = [r['index'] for r in successful]
            rtfs = [r['rtf'] for r in successful]

            label = 'Flash Attention' if config['flash_attention'] else 'Baseline'
            marker = 's' if config['flash_attention'] else 'o'
            color = 'green' if config['flash_attention'] else 'steelblue'

            ax.plot(indices, rtfs, marker=marker, label=label, linewidth=2,
                   markersize=8, color=color)

        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
                  label='Real-time threshold', alpha=0.7)
        ax.set_xlabel('Audio Sample Index', fontsize=12)
        ax.set_ylabel('Real-Time Factor (RTF)', fontsize=12)
        ax.set_title('Per-Sample RTF Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "rtf_comparison.png"
        plt.savefig(output_file, dpi=150)
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_latency_distribution(self):
        """Plot processing time distribution histogram"""
        fig, ax = plt.subplots()

        data = []
        labels = []
        colors = []

        for config in self.results['configs']:
            eval_results = config['evaluation_results']
            successful = [r for r in eval_results if r['status'] == 'success']

            processing_times = [r['processing_time'] for r in successful
                              if r['processing_time'] is not None]
            data.append(processing_times)

            label = 'Flash Attention' if config['flash_attention'] else 'Baseline'
            labels.append(label)
            colors.append('green' if config['flash_attention'] else 'steelblue')

        ax.hist(data, bins=20, label=labels, alpha=0.7, color=colors,
               edgecolor='black')
        ax.set_xlabel('Processing Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Processing Time Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / "latency_distribution.png"
        plt.savefig(output_file, dpi=150)
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_gpu_utilization_timeline(self):
        """Plot GPU metrics timeline for each config"""
        configs_with_gpu = [c for c in self.results['configs']
                           if c.get('gpu_metrics') is not None]

        if not configs_with_gpu:
            print("⚠ Skipping GPU timeline plot (no GPU metrics available)")
            return

        fig, axes = plt.subplots(len(configs_with_gpu), 1,
                                figsize=(12, 4 * len(configs_with_gpu)))

        if len(configs_with_gpu) == 1:
            axes = [axes]

        for i, config in enumerate(configs_with_gpu):
            csv_file = config['gpu_metrics']['csv_file']

            try:
                df = pd.read_csv(csv_file)

                ax = axes[i]
                ax2 = ax.twinx()

                # GPU utilization
                ax.plot(df['gpu_util_%'], label='GPU Util %', color='blue',
                       linewidth=1.5)
                ax.plot(df['memory_util_%'], label='Memory Util %', color='orange',
                       linewidth=1.5)

                # Temperature on second axis
                ax2.plot(df['temperature_c'], label='Temperature', color='red',
                        linewidth=1.5, linestyle='--', alpha=0.7)

                # Labels
                label = 'Flash Attention' if config['flash_attention'] else 'Baseline'
                ax.set_title(f'{label}: GPU Metrics Over Time',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Sample #', fontsize=11)
                ax.set_ylabel('Utilization (%)', fontsize=11)
                ax2.set_ylabel('Temperature (°C)', fontsize=11, color='red')

                # Legends
                ax.legend(loc='upper left', fontsize=10)
                ax2.legend(loc='upper right', fontsize=10)

                ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"⚠ Warning: Could not plot GPU metrics for {config['config_name']}: {e}")

        plt.tight_layout()
        output_file = self.output_dir / "gpu_utilization_timeline.png"
        plt.savefig(output_file, dpi=150)
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_throughput_comparison(self):
        """Bar chart comparing throughput metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        configs = self.results['configs']
        labels = ['Flash Attention' if c['flash_attention'] else 'Baseline'
                 for c in configs]

        # Files per hour
        files_per_hour = [c['throughput_metrics']['files_per_hour'] for c in configs]
        colors = ['green' if c['flash_attention'] else 'steelblue' for c in configs]

        ax1.bar(labels, files_per_hour, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Files/Hour', fontsize=12)
        ax1.set_title('Throughput: Files per Hour', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Audio hours per hour
        audio_hours = [c['throughput_metrics']['audio_hours_per_hour'] for c in configs]

        ax2.bar(labels, audio_hours, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Audio Hours/Hour', fontsize=12)
        ax2.set_title('Throughput: Audio Hours per Hour', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / "throughput_comparison.png"
        plt.savefig(output_file, dpi=150)
        print(f"✓ Saved: {output_file}")
        plt.close()

    def generate_summary_report(self):
        """Generate text summary report"""
        output_file = self.output_dir / "summary_report.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("AZURE GPU BENCHMARK SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Model: {self.results['model']}\n")
            f.write(f"Device: {self.results['device']}\n")
            f.write(f"GPU: {self.results['gpu_name']}\n\n")

            for config in self.results['configs']:
                label = 'FLASH ATTENTION' if config['flash_attention'] else 'BASELINE'
                f.write(f"\n{label}\n")
                f.write("-"*70 + "\n")

                # Throughput
                metrics = config['throughput_metrics']
                f.write(f"  Files/hour:         {metrics['files_per_hour']:.2f}\n")
                f.write(f"  Audio hours/hour:   {metrics['audio_hours_per_hour']:.2f}\n")
                f.write(f"  Average RTF:        {metrics['average_rtf']:.3f}x\n")
                f.write(f"  Total files:        {metrics['total_files']}\n")
                f.write(f"  Successful files:   {metrics['successful_files']}\n")

                # Accuracy
                stats = config['accuracy_statistics']
                if stats.get('cer'):
                    f.write(f"  Average CER:        {stats['cer']['mean']:.2%}\n")
                    f.write(f"  Average WER:        {stats['wer']['mean']:.2%}\n")

                # GPU
                if config.get('gpu_metrics'):
                    gpu = config['gpu_metrics']['summary']
                    f.write(f"  GPU Util (mean):    {gpu['gpu_utilization']['mean']:.1f}%\n")
                    f.write(f"  GPU Util (max):     {gpu['gpu_utilization']['max']:.1f}%\n")
                    f.write(f"  Peak VRAM:          {gpu['memory_usage_mb']['max']:.0f} MB\n")
                    f.write(f"  VRAM utilization:   {gpu['memory_usage_mb']['peak_util_%']:.1f}%\n")
                    f.write(f"  Avg Temperature:    {gpu['temperature_c']['mean']:.1f}°C\n")
                    f.write(f"  Max Temperature:    {gpu['temperature_c']['max']:.1f}°C\n")
                    f.write(f"  Avg Power Draw:     {gpu['power_draw_w']['mean']:.1f} W\n")
                    f.write(f"  Max Power Draw:     {gpu['power_draw_w']['max']:.1f} W\n")

            # Comparison
            if len(self.results['configs']) == 2:
                baseline = next((c for c in self.results['configs']
                               if not c['flash_attention']), None)
                flash = next((c for c in self.results['configs']
                             if c['flash_attention']), None)

                if baseline and flash:
                    f.write(f"\n\nCOMPARISON\n")
                    f.write("-"*70 + "\n")

                    # RTF improvement
                    baseline_rtf = baseline['throughput_metrics']['average_rtf']
                    flash_rtf = flash['throughput_metrics']['average_rtf']
                    rtf_improvement = ((baseline_rtf - flash_rtf) / baseline_rtf) * 100 if baseline_rtf > 0 else 0

                    f.write(f"  RTF Improvement:    {rtf_improvement:+.1f}%\n")

                    # Throughput improvement
                    baseline_tp = baseline['throughput_metrics']['files_per_hour']
                    flash_tp = flash['throughput_metrics']['files_per_hour']
                    throughput_improvement = ((flash_tp - baseline_tp) / baseline_tp) * 100 if baseline_tp > 0 else 0

                    f.write(f"  Throughput Gain:    {throughput_improvement:+.1f}%\n")

                    # VRAM comparison
                    if baseline.get('gpu_metrics') and flash.get('gpu_metrics'):
                        baseline_vram = baseline['gpu_metrics']['summary']['memory_usage_mb']['max']
                        flash_vram = flash['gpu_metrics']['summary']['memory_usage_mb']['max']
                        vram_delta = flash_vram - baseline_vram

                        f.write(f"  VRAM Delta:         {vram_delta:+.0f} MB\n")

            f.write("\n" + "="*70 + "\n")

        print(f"✓ Saved: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Azure benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('results_json', help='Path to benchmark results JSON file')

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.results_json).exists():
        print(f"Error: Results file not found: {args.results_json}")
        return 1

    # Run analysis
    analyzer = BenchmarkAnalyzer(args.results_json)
    analyzer.analyze()

    return 0


if __name__ == "__main__":
    sys.exit(main())
