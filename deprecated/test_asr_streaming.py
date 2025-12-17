#!/usr/bin/env python3
"""
Streaming ASR Test Script for Kotoba Whisper v2.2
Supports both file streaming and live microphone input with real-time display
"""

import sys
import time
import argparse
import queue
import numpy as np
import torch
import librosa
import sounddevice as sd
from transformers import pipeline
from asr_metrics_utils import load_ground_truth, calculate_accuracy_metrics


class LiveDisplay:
    """Manages live-updating terminal display for streaming transcription"""

    def __init__(self, ground_truth=None):
        self.accumulated_text = ""
        self.chunk_count = 0
        self.ground_truth = ground_truth

    def update(self, new_text, chunk_num, latency, rtf=None):
        """Update display with new transcription chunk"""
        self.accumulated_text += new_text
        self.chunk_count = chunk_num

        # Clear screen and move cursor to home
        print('\033[2J\033[H', end='')

        # Display header
        print("=" * 60)
        print("[LIVE TRANSCRIPTION]")
        print("=" * 60)
        print()

        # Display accumulated text
        print(self.accumulated_text)
        print()

        # Display real-time metrics
        rtf_str = f" | RTF: {rtf:.2f}x" if rtf is not None else ""
        print(f"[Chunk {chunk_num} | Latency: {latency:.2f}s{rtf_str}] ⚡ LIVE")
        print("=" * 60)

        sys.stdout.flush()

    def finalize(self, total_time, avg_latency, total_chunks, audio_duration=None):
        """Display final summary"""
        print('\033[2J\033[H', end='')  # Clear screen

        print("=" * 60)
        print("STREAMING TRANSCRIPTION COMPLETE")
        print("=" * 60)
        print()
        print("Final Text:")
        print(self.accumulated_text)
        print()
        print("=" * 60)
        print("Statistics:")
        print("-" * 60)
        print(f"  Total chunks:      {total_chunks}")
        print(f"  Average latency:   {avg_latency:.2f}s per chunk")
        print(f"  Total time:        {total_time:.2f}s")
        if audio_duration:
            rtf = total_time / audio_duration
            rtf_status = "faster" if rtf < 1.0 else "slower"
            print(f"  Audio duration:    {audio_duration:.2f}s")
            print(f"  Real-Time Factor:  {rtf:.2f}x ({rtf_status} than real-time)")

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


class AudioChunker:
    """Handles audio chunking for streaming processing"""

    def __init__(self, chunk_duration=3.0, overlap=0.5, sample_rate=16000):
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.stride = self.chunk_samples - self.overlap_samples

    def chunk_from_file(self, audio_path):
        """Generator that yields overlapping audio chunks from file"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Yield chunks with overlap
        start = 0
        while start < len(audio):
            end = min(start + self.chunk_samples, len(audio))
            chunk = audio[start:end]

            # Pad last chunk if needed
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

            yield chunk
            start += self.stride

    def chunk_from_microphone(self):
        """Generator that yields audio chunks from microphone"""
        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Status: {status}", file=sys.stderr)
            audio_queue.put(indata.copy())

        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=self.chunk_samples
        )

        with stream:
            print("🎤 Microphone active - speak now! (Press Ctrl+C to stop)")
            print()
            try:
                while True:
                    chunk = audio_queue.get()
                    yield chunk.flatten()
            except KeyboardInterrupt:
                print("\n\nStopping microphone...")


class StreamingTranscriber:
    """Handles streaming transcription with Whisper pipeline"""

    def __init__(self, pipe, device, language="ja"):
        self.pipe = pipe
        self.device = device
        self.language = language

    def transcribe_chunk(self, audio_chunk):
        """Transcribe a single audio chunk and return (text, latency)"""
        start_time = time.perf_counter()

        # Use pipeline (simpler and faster than manual processing)
        result = self.pipe(
            audio_chunk,
            return_timestamps=False,  # For streaming, just get text
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
            }
        )

        latency = time.perf_counter() - start_time
        text = result['text']

        return text, latency

    def transcribe_stream(self, audio_generator):
        """Process audio stream and yield (text, latency) tuples"""
        for chunk in audio_generator:
            text, latency = self.transcribe_chunk(chunk)
            yield text, latency


def load_model():
    """Load Kotoba Whisper pipeline with optimizations"""
    print("Loading Kotoba Whisper v2.2 pipeline...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device}")
    print(f"Data type: {torch_dtype}")

    # Flash Attention detection
    model_kwargs = {}
    try:
        import flash_attn
        model_kwargs = {"attn_implementation": "flash_attention_2"}
        print("Flash Attention 2 enabled")
    except ImportError:
        print("Flash Attention not available - using standard attention")

    # Load pipeline (not individual components)
    pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.2",
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
    )

    print("Pipeline loaded successfully!\n")

    return pipe, device


def stream_from_file(file_path, pipe, device, chunk_duration=3.0):
    """Stream transcription from audio file"""
    print(f"Streaming from file: {file_path}\n")

    # Get audio duration
    audio_duration = librosa.get_duration(path=file_path)

    # Load ground truth if available
    ground_truth = load_ground_truth(file_path)
    if ground_truth:
        print("✓ Ground truth loaded for accuracy measurement\n")

    # Initialize components
    chunker = AudioChunker(chunk_duration=chunk_duration)
    transcriber = StreamingTranscriber(pipe, device)
    display = LiveDisplay(ground_truth=ground_truth)

    # Process stream
    latencies = []
    chunk_num = 0
    start_time = time.perf_counter()

    for text, latency in transcriber.transcribe_stream(chunker.chunk_from_file(file_path)):
        chunk_num += 1
        latencies.append(latency)

        # Calculate current RTF
        elapsed = time.perf_counter() - start_time
        current_rtf = elapsed / (chunk_num * chunk_duration) if chunk_num > 0 else 0

        # Update display
        display.update(text + " ", chunk_num, latency, current_rtf)

        # Small delay for visual effect
        time.sleep(0.1)

    total_time = time.perf_counter() - start_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Show final summary
    display.finalize(total_time, avg_latency, chunk_num, audio_duration)


def stream_from_microphone(pipe, device, chunk_duration=3.0):
    """Stream transcription from microphone"""
    print("Streaming from microphone\n")

    # Initialize components
    chunker = AudioChunker(chunk_duration=chunk_duration)
    transcriber = StreamingTranscriber(pipe, device)
    display = LiveDisplay()

    # Process stream
    latencies = []
    chunk_num = 0
    start_time = time.perf_counter()

    try:
        for text, latency in transcriber.transcribe_stream(chunker.chunk_from_microphone()):
            chunk_num += 1
            latencies.append(latency)

            # Update display
            display.update(text + " ", chunk_num, latency)

    except KeyboardInterrupt:
        pass

    total_time = time.perf_counter() - start_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Show final summary
    display.finalize(total_time, avg_latency, chunk_num)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Streaming ASR with Kotoba Whisper v2.2"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "mic"],
        default="file",
        help="Streaming mode: file or microphone (default: file)"
    )
    parser.add_argument(
        "--audio",
        default="dataset/sample_audio.mp3",
        help="Audio file path for file mode (default: dataset/sample_audio.mp3)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Chunk duration in seconds (default: 3.0)"
    )

    args = parser.parse_args()

    # Load model
    pipe, device = load_model()

    # Run appropriate mode
    if args.mode == "file":
        stream_from_file(args.audio, pipe, device, args.chunk_duration)
    else:
        stream_from_microphone(pipe, device, args.chunk_duration)


if __name__ == "__main__":
    main()
