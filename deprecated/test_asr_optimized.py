#!/usr/bin/env python3
"""
Test script for Kotoba Whisper v2.2 ASR model
Optimized with Flash Attention and recommended parameters for better accuracy and speed
"""

import sys
import time
import torch
import librosa
from transformers import pipeline
from asr_metrics_utils import load_ground_truth, calculate_accuracy_metrics

def load_pipeline():
    """Load the Kotoba Whisper pipeline with optimized settings"""
    print("Loading Kotoba Whisper v2.2 pipeline...")

    # Determine device and data type
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device}")
    print(f"Data type: {torch_dtype}")

    # Model configuration for flash attention (if available)
    model_kwargs = {}
    try:
        # Try to use flash attention for better speed
        import flash_attn
        model_kwargs = {"attn_implementation": "flash_attention_2"}
        print("Flash Attention 2 enabled")
    except ImportError:
        print("Flash Attention not available - using standard attention")
        print("Install with: pip install flash-attn --no-build-isolation")

    # Load pipeline with optimized settings
    pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.2",
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
    )

    print("Pipeline loaded successfully!\n")
    return pipe

def transcribe_audio(audio_path, pipe, **kwargs):
    """
    Transcribe an audio file with optimized parameters

    Args:
        audio_path: Path to audio file
        pipe: The ASR pipeline
        **kwargs: Additional parameters
    """
    print(f"Transcribing: {audio_path}\n")

    # Run transcription with optimized parameters
    # chunk_length_s and batch_size are pipeline parameters, not generate_kwargs
    result = pipe(
        audio_path,
        chunk_length_s=kwargs.get("chunk_length_s", 15),
        batch_size=kwargs.get("batch_size", 8),
        return_timestamps=kwargs.get("return_timestamps", True),
        generate_kwargs={
            "language": kwargs.get("language", "ja"),
            "task": kwargs.get("task", "transcribe"),
        }
    )

    return result

def transcribe_with_latency(audio_path, pipe, ground_truth=None, **kwargs):
    """
    Transcribe audio file and measure latency metrics

    Args:
        audio_path: Path to audio file
        pipe: The ASR pipeline
        ground_truth: Optional ground truth transcript for accuracy calculation
        **kwargs: Additional parameters for transcription

    Returns:
        tuple: (transcription_result, latency_metrics)
    """
    # Get audio duration for RTF calculation
    try:
        audio_duration = librosa.get_duration(path=audio_path)
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        audio_duration = None

    # Measure transcription time
    start_time = time.perf_counter()
    result = transcribe_audio(audio_path, pipe, **kwargs)
    end_time = time.perf_counter()

    # Calculate metrics
    processing_time = end_time - start_time
    rtf = processing_time / audio_duration if audio_duration else None

    # Calculate per-chunk latency (estimated)
    num_chunks = len(result.get('chunks', [])) if 'chunks' in result else 1
    avg_chunk_latency = processing_time / num_chunks if num_chunks > 0 else processing_time

    # Build latency metrics dictionary
    latency_metrics = {
        'audio_duration': audio_duration,
        'processing_time': processing_time,
        'rtf': rtf,
        'num_chunks': num_chunks,
        'avg_chunk_latency': avg_chunk_latency,
        'chunks': result.get('chunks', [])
    }

    # Calculate accuracy metrics if ground truth available
    if ground_truth:
        hypothesis = result.get('text', '')
        accuracy = calculate_accuracy_metrics(hypothesis, ground_truth)
        if accuracy:
            latency_metrics['accuracy'] = {
                'cer': accuracy['cer'],
                'wer': accuracy['wer'],
                'reference': ground_truth,
                'hypothesis': hypothesis
            }

    return result, latency_metrics

def main():
    """Main function to test the ASR model"""
    print("="*60)
    print("Kotoba Whisper v2.2 ASR Test (Optimized)")
    print("="*60 + "\n")

    # Load pipeline
    pipe = load_pipeline()

    # Use command-line argument if provided, otherwise use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Using audio file from command line: {audio_file}\n")
    else:
        audio_file = "dataset/sample_audio.mp3"
        print(f"Using default audio file: {audio_file}\n")

    # Load ground truth if available
    ground_truth = load_ground_truth(audio_file)
    if ground_truth:
        print(f"✓ Ground truth loaded for accuracy measurement\n")

    if audio_file:
        try:
            # Transcribe with latency measurement
            result, latency_metrics = transcribe_with_latency(
                audio_file,
                pipe,
                ground_truth=ground_truth,
                # You can customize these parameters:
                language="ja",  # Japanese
                task="transcribe",
                chunk_length_s=15,
                batch_size=8,
            )

            print("\n" + "="*60)
            print("TRANSCRIPTION RESULT")
            print("="*60)
            print(f"\nText: {result['text']}\n")

            # Print timestamps if available
            if 'chunks' in result and result['chunks']:
                print("-"*60)
                print("Timestamps:")
                print("-"*60)
                for chunk in result['chunks']:
                    timestamp = chunk.get('timestamp', (None, None))
                    text = chunk.get('text', '')
                    if timestamp[0] is not None:
                        print(f"[{timestamp[0]:.2f}s - {timestamp[1]:.2f}s]: {text}")
                    else:
                        print(f"{text}")

            print("\n" + "="*60)

            # Display latency metrics
            print("\n" + "="*60)
            print("LATENCY METRICS")
            print("="*60)

            if latency_metrics['audio_duration']:
                print(f"\nAudio Duration:        {latency_metrics['audio_duration']:.2f} seconds")
            print(f"Total Processing Time: {latency_metrics['processing_time']:.2f} seconds")

            if latency_metrics['rtf']:
                rtf = latency_metrics['rtf']
                rtf_status = "faster than real-time" if rtf < 1.0 else "slower than real-time"
                print(f"Real-Time Factor (RTF): {rtf:.2f}x ({rtf_status})")

            print(f"\nNumber of Chunks:      {latency_metrics['num_chunks']}")
            print(f"Avg Chunk Latency:     {latency_metrics['avg_chunk_latency']*1000:.2f} ms")

            # Show estimated per-chunk processing times
            if latency_metrics['chunks'] and latency_metrics['num_chunks'] > 1:
                print("\nEstimated Per-Chunk Processing Times:")
                print("-"*60)
                for i, chunk in enumerate(latency_metrics['chunks'], 1):
                    timestamp = chunk.get('timestamp', (None, None))
                    est_time = latency_metrics['avg_chunk_latency'] * 1000  # in ms
                    if timestamp[0] is not None:
                        chunk_duration = timestamp[1] - timestamp[0]
                        print(f"  Chunk {i} [{timestamp[0]:.2f}s - {timestamp[1]:.2f}s] "
                              f"(duration: {chunk_duration:.2f}s): ~{est_time:.2f} ms")

            print("\n" + "="*60)

            # Display accuracy metrics if available
            if 'accuracy' in latency_metrics:
                print("\n" + "="*60)
                print("ACCURACY METRICS (vs Ground Truth)")
                print("="*60)

                acc = latency_metrics['accuracy']
                print(f"\nReference:  {acc['reference']}")
                print(f"Hypothesis: {acc['hypothesis']}")
                print(f"\nCharacter Error Rate (CER): {acc['cer']*100:.2f}%")
                print(f"Word Error Rate (WER):       {acc['wer']*100:.2f}%")

                print("\n" + "="*60)

        except FileNotFoundError:
            print(f"\nError: Audio file '{audio_file}' not found!")
            print("Please make sure the file exists in the current directory.")
        except Exception as e:
            print(f"\nError during transcription: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo audio file provided. Pipeline loaded successfully.")

    print("\nTest completed!")

if __name__ == "__main__":
    main()
