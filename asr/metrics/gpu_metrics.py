"""GPU metrics monitoring and collection"""

import time
import json
import csv
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class GPUSnapshot:
    """Single point-in-time GPU metrics"""
    timestamp: str
    gpu_util_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_util_percent: float
    temperature_c: float
    power_draw_w: float
    power_limit_w: float


@dataclass
class GPUMetricsSummary:
    """Aggregated GPU metrics summary"""
    duration_seconds: float
    samples: int
    csv_file: str
    json_file: str
    gpu_utilization: Dict[str, float]  # mean, max, min, std
    memory_usage_mb: Dict[str, float]  # mean, max, peak_util_%
    temperature_c: Dict[str, float]    # mean, max
    power_draw_w: Dict[str, float]     # mean, max


class GPUMonitor:
    """
    Background GPU monitoring with CSV logging and summary statistics.

    Usage:
        # As context manager
        with GPUMonitor(output_dir='results/logs') as monitor:
            # Your GPU-intensive code here
            pass

        # Manual control
        monitor = GPUMonitor(output_dir='results/logs', interval=1.0)
        monitor.start()
        # ... do work ...
        summary = monitor.stop()
    """

    def __init__(self, output_dir: str = "results/logs", interval: float = 1.0):
        """
        Initialize GPU monitor.

        Args:
            output_dir: Directory to save CSV and JSON files
            interval: Sampling interval in seconds (default: 1.0)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Initialize log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"gpu_metrics_{timestamp}.csv"
        self.json_file = self.output_dir / f"gpu_summary_{timestamp}.json"

        # CSV headers
        self.csv_headers = [
            "timestamp", "gpu_util_%", "memory_used_mb", "memory_total_mb",
            "memory_util_%", "temperature_c", "power_draw_w", "power_limit_w"
        ]

        # Store collected data for summary
        self._data: List[GPUSnapshot] = []

    def get_gpu_stats(self) -> Optional[GPUSnapshot]:
        """
        Query current GPU stats using nvidia-smi.

        Returns:
            GPUSnapshot with current metrics, or None if query fails
        """
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True, timeout=5)

            values = result.stdout.strip().split(', ')

            return GPUSnapshot(
                timestamp=datetime.now().isoformat(),
                gpu_util_percent=float(values[0]),
                memory_used_mb=float(values[1]),
                memory_total_mb=float(values[2]),
                memory_util_percent=(float(values[1]) / float(values[2])) * 100,
                temperature_c=float(values[3]),
                power_draw_w=float(values[4]),
                power_limit_w=float(values[5])
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError) as e:
            # nvidia-smi not available or GPU query failed
            return None

    def _monitor_loop(self):
        """Background monitoring loop that runs in separate thread"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()

            while self.running:
                stats = self.get_gpu_stats()
                if stats:
                    # Write to CSV
                    row = {
                        "timestamp": stats.timestamp,
                        "gpu_util_%": stats.gpu_util_percent,
                        "memory_used_mb": stats.memory_used_mb,
                        "memory_total_mb": stats.memory_total_mb,
                        "memory_util_%": stats.memory_util_percent,
                        "temperature_c": stats.temperature_c,
                        "power_draw_w": stats.power_draw_w,
                        "power_limit_w": stats.power_limit_w
                    }
                    writer.writerow(row)
                    f.flush()

                    # Store for summary calculation
                    self._data.append(stats)

                time.sleep(self.interval)

    def start(self) -> None:
        """Start background GPU monitoring thread"""
        if self.running:
            print("Warning: GPU monitoring already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"GPU monitoring started (logging to {self.csv_file})")

    def stop(self) -> Optional[GPUMetricsSummary]:
        """
        Stop monitoring and generate summary statistics.

        Returns:
            GPUMetricsSummary with aggregated statistics, or None if no data collected
        """
        if not self.running:
            print("Warning: GPU monitoring not running")
            return None

        self.running = False

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)

        # Generate summary
        summary = self._generate_summary()

        if summary:
            print(f"GPU monitoring stopped (summary: {self.json_file})")
        else:
            print("GPU monitoring stopped (no data collected)")

        return summary

    def _generate_summary(self) -> Optional[GPUMetricsSummary]:
        """
        Generate summary statistics from collected data.

        Returns:
            GPUMetricsSummary or None if no data
        """
        if not self._data:
            return None

        # Calculate statistics
        gpu_utils = [s.gpu_util_percent for s in self._data]
        memory_used = [s.memory_used_mb for s in self._data]
        memory_total = self._data[0].memory_total_mb  # Constant
        temps = [s.temperature_c for s in self._data]
        powers = [s.power_draw_w for s in self._data]

        summary = GPUMetricsSummary(
            duration_seconds=len(self._data) * self.interval,
            samples=len(self._data),
            csv_file=str(self.csv_file),
            json_file=str(self.json_file),
            gpu_utilization={
                "mean": sum(gpu_utils) / len(gpu_utils),
                "max": max(gpu_utils),
                "min": min(gpu_utils),
                "std": self._calculate_std(gpu_utils)
            },
            memory_usage_mb={
                "mean": sum(memory_used) / len(memory_used),
                "max": max(memory_used),
                "peak_util_%": (max(memory_used) / memory_total) * 100
            },
            temperature_c={
                "mean": sum(temps) / len(temps),
                "max": max(temps)
            },
            power_draw_w={
                "mean": sum(powers) / len(powers),
                "max": max(powers)
            }
        )

        # Save to JSON
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2)

        return summary

    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    # Context manager support
    def __enter__(self):
        """Enter context manager"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        self.stop()
        return False  # Don't suppress exceptions
