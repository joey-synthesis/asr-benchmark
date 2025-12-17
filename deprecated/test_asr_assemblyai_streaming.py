#!/usr/bin/env python3
"""
AssemblyAI Real-Time Streaming ASR Test Script
Supports live microphone transcription with real-time display and metrics
"""

import sys
import time
import os
import argparse
import logging
from typing import Type
from dotenv import load_dotenv
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
from asr_metrics_utils import load_ground_truth, calculate_accuracy_metrics


# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Set to WARNING to reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveDisplay:
    """Manages live-updating terminal display for streaming transcription"""

    def __init__(self, ground_truth=None):
        self.accumulated_text = ""
        self.start_time = time.perf_counter()
        self.turn_count = 0
        self.total_latencies = []
        self.ground_truth = ground_truth

    def update(self, new_text, turn_num, is_final=False, latency=None):
        """Update the live display with new transcription"""
        # Append new text
        if is_final:
            # Add period and space for final turns
            self.accumulated_text += new_text + ". "
        else:
            self.accumulated_text += new_text + " "

        # Clear screen and move cursor to top
        print('\033[2J\033[H', end='')

        # Print header
        print("=" * 60)
        print("[LIVE TRANSCRIPTION - AssemblyAI Streaming]")
        print("=" * 60)
        print()

        # Print accumulated text with wrapping
        print(self.accumulated_text)
        print()

        # Print status
        status_parts = [f"Turn {turn_num}"]
        if latency:
            status_parts.append(f"Latency: {latency:.2f}s")
        status_parts.append("⚡ LIVE")

        print(f"[{' | '.join(status_parts)}]")
        print("=" * 60)

        sys.stdout.flush()

    def finalize(self, total_time, audio_duration=None):
        """Display final summary"""
        print('\033[2J\033[H', end='')  # Clear screen
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
        print(f"  Total turns:       {self.turn_count}")

        if self.total_latencies:
            avg_latency = sum(self.total_latencies) / len(self.total_latencies)
            print(f"  Average latency:   {avg_latency:.2f}s per turn")

        print(f"  Total time:        {total_time:.2f}s")

        if audio_duration:
            print(f"  Audio duration:    {audio_duration:.2f}s")
            rtf = total_time / audio_duration
            rtf_status = "faster than real-time" if rtf < 1.0 else "slower than real-time"
            print(f"  Real-Time Factor:  {rtf:.2f}x ({rtf_status})")

        # Display accuracy metrics if ground truth available
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


class AssemblyAIStreamingTranscriber:
    """Handles AssemblyAI streaming transcription with live display"""

    def __init__(self, api_key, language="ja"):
        self.api_key = api_key
        self.language = language
        self.display = LiveDisplay()
        self.client = None
        self.session_start_time = None
        self.audio_duration = 0
        self.last_turn_time = None

    def on_begin(self, client: Type[StreamingClient], event: BeginEvent):
        """Called when streaming session begins"""
        self.session_start_time = time.perf_counter()
        logger.info(f"Session started: {event.id}")
        print(f"Session ID: {event.id}")
        print("Listening... (Press Ctrl+C to stop)")
        print()

    def on_turn(self, client: Type[StreamingClient], event: TurnEvent):
        """Called when a turn (speech segment) is detected"""
        current_time = time.perf_counter()

        # Calculate latency (time since last turn or session start)
        if self.last_turn_time:
            latency = current_time - self.last_turn_time
        else:
            latency = current_time - self.session_start_time if self.session_start_time else 0

        self.last_turn_time = current_time

        # Update display
        self.display.turn_count += 1
        self.display.total_latencies.append(latency)
        self.display.update(
            event.transcript,
            self.display.turn_count,
            is_final=event.end_of_turn,
            latency=latency
        )

        # Enable formatting after first unformatted turn
        if event.end_of_turn and not event.turn_is_formatted:
            params = StreamingSessionParameters(
                format_turns=True,
            )
            client.set_params(params)
            logger.info("Enabled turn formatting")

    def on_terminated(self, client: Type[StreamingClient], event: TerminationEvent):
        """Called when session terminates"""
        self.audio_duration = event.audio_duration_seconds
        logger.info(f"Session terminated: {self.audio_duration}s of audio processed")

    def on_error(self, client: Type[StreamingClient], error: StreamingError):
        """Called when an error occurs"""
        print(f"\nError occurred: {error}")
        logger.error(f"Streaming error: {error}")

    def stream_from_microphone(self, sample_rate=16000):
        """Stream transcription from microphone"""
        print("=" * 60)
        print("AssemblyAI Real-Time Streaming - Microphone Mode")
        print("=" * 60)
        print()

        # Create streaming client
        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        # Register event handlers
        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_terminated)
        self.client.on(StreamingEvents.Error, self.on_error)

        # Connect with streaming parameters
        params = StreamingParameters(
            sample_rate=sample_rate,
            format_turns=True,
            language=self.language,
        )

        self.client.connect(params)

        try:
            # Stream from microphone
            self.client.stream(
                aai.extras.MicrophoneStream(sample_rate=sample_rate)
            )

        except KeyboardInterrupt:
            print("\n\nStopping microphone...")

        finally:
            # Disconnect and show final stats
            self.client.disconnect(terminate=True)

            if self.session_start_time:
                total_time = time.perf_counter() - self.session_start_time
                self.display.finalize(total_time, self.audio_duration)

    def stream_from_file(self, file_path, sample_rate=16000):
        """
        File transcription is not supported in streaming mode.
        Use test_asr_assemblyai.py for file transcription instead.
        """
        print("=" * 60)
        print("ERROR: File Mode Not Supported")
        print("=" * 60)
        print()
        print("AssemblyAI's streaming API is designed for real-time audio input")
        print("(microphone) only. For file transcription, please use:")
        print()
        print(f"  python test_asr_assemblyai.py --audio {file_path}")
        print()
        print("The regular API provides better performance and accuracy for")
        print("pre-recorded audio files.")
        print("=" * 60)
        return


def load_api_key():
    """Load AssemblyAI API key from environment"""
    load_dotenv()

    api_key = os.getenv("ASSEMBLYAI_API_KEY")

    if not api_key:
        raise ValueError(
            "AssemblyAI API key not found!\n"
            "Please set ASSEMBLYAI_API_KEY in .env file or environment variable"
        )

    return api_key


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="AssemblyAI Real-Time Streaming ASR Test (Microphone Only)",
        epilog="Note: For file transcription, use test_asr_assemblyai.py instead"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language code (default: ja for Japanese)"
    )

    args = parser.parse_args()

    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create transcriber
    transcriber = AssemblyAIStreamingTranscriber(
        api_key=api_key,
        language=args.language
    )

    # Run microphone streaming
    try:
        transcriber.stream_from_microphone(sample_rate=args.sample_rate)
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
