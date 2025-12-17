"""Live display for streaming transcription"""

import sys
import time
from typing import Optional
from ..metrics.accuracy import calculate_accuracy_metrics


class LiveDisplay:
    """Unified live display for streaming transcription"""

    def __init__(self, ground_truth: Optional[str] = None, display_mode: str = "chunks"):
        """
        Initialize live display.

        Args:
            ground_truth: Optional reference transcript for accuracy
            display_mode: "chunks" for chunk-based, "turns" for turn-based
        """
        self.accumulated_text = ""
        self.start_time = time.perf_counter()
        self.item_count = 0  # chunks or turns
        self.latencies = []
        self.ground_truth = ground_truth
        self.display_mode = display_mode

    def update(self, new_text: str, item_num: int, latency: float, rtf: Optional[float] = None):
        """
        Update the live display.

        Args:
            new_text: New text to append
            item_num: Chunk or turn number
            latency: Processing latency in seconds
            rtf: Real-Time Factor (optional)
        """
        self.accumulated_text += new_text + " "
        self.item_count = item_num
        self.latencies.append(latency)

        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print("=" * 60)
        print("[LIVE TRANSCRIPTION]")
        print("=" * 60)
        print()

        # Accumulated text
        print(self.accumulated_text)
        print()

        # Status
        item_label = "Chunk" if self.display_mode == "chunks" else "Turn"
        status = f"[{item_label} {item_num} | Latency: {latency:.2f}s"
        if rtf:
            status += f" | RTF: {rtf:.2f}x"
        status += "] ⚡ LIVE"
        print(status)
        print("=" * 60)

        sys.stdout.flush()

    def finalize(self, total_time: float, audio_duration: Optional[float] = None):
        """
        Display final summary.

        Args:
            total_time: Total processing time in seconds
            audio_duration: Audio duration in seconds (optional)
        """
        print('\033[2J\033[H', end='')

        print("=" * 60)
        print("STREAMING TRANSCRIPTION COMPLETE")
        print("=" * 60)
        print()
        print("Final Text:")
        print(self.accumulated_text.strip())
        print()
        print("=" * 60)
        print("Statistics:")
        print("-" * 60)

        item_label = "chunks" if self.display_mode == "chunks" else "turns"
        print(f"  Total {item_label}:      {self.item_count}")

        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            print(f"  Average latency:   {avg_latency:.2f}s per {item_label[:-1]}")

        print(f"  Total time:        {total_time:.2f}s")

        if audio_duration:
            rtf = total_time / audio_duration
            rtf_status = "faster" if rtf < 1.0 else "slower"
            print(f"  Audio duration:    {audio_duration:.2f}s")
            print(f"  Real-Time Factor:  {rtf:.2f}x ({rtf_status} than real-time)")

        # Accuracy metrics
        if self.ground_truth:
            accuracy = calculate_accuracy_metrics(
                self.accumulated_text.strip(),
                self.ground_truth
            )
            if accuracy:
                print("-" * 60)
                print(f"  Reference:         {self.ground_truth}")
                print(f"  Hypothesis:        {self.accumulated_text.strip()}")
                print(f"  CER:               {accuracy['cer']*100:.2f}%")
                print(f"  WER:               {accuracy['wer']*100:.2f}%")

        print("=" * 60)
