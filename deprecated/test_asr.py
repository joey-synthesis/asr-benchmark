#!/usr/bin/env python3
"""
Test script for Kotoba Whisper v2.2 ASR model
"""

import sys
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def load_model():
    """Load the Kotoba Whisper model and processor"""
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("kotoba-tech/kotoba-whisper-v2.2")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("kotoba-tech/kotoba-whisper-v2.2")

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    return processor, model, device

def transcribe_audio(audio_path, processor, model, device):
    """Transcribe an audio file"""
    print(f"\nTranscribing: {audio_path}")

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(**inputs)

    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription

def main():
    """Main function to test the ASR model"""
    # Load model
    processor, model, device = load_model()

    print("\n" + "="*50)
    print("Kotoba Whisper v2.2 ASR Test")
    print("="*50 + "\n")

    # Use command-line argument if provided, otherwise use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Using audio file from command line: {audio_file}\n")
    else:
        audio_file = "dataset/sample_audio.mp3"
        print(f"Using default audio file: {audio_file}\n")

    if audio_file:
        try:
            transcription = transcribe_audio(audio_file, processor, model, device)
            print("\n" + "-"*50)
            print("Transcription:")
            print(transcription)
            print("-"*50)
        except FileNotFoundError:
            print(f"\nError: Audio file '{audio_file}' not found!")
            print("Please make sure the file exists in the current directory.")
        except Exception as e:
            print(f"\nError during transcription: {e}")
            import traceback
            traceback.print_exc()

    print("\nTest completed!")

if __name__ == "__main__":
    main()
