# Kotoba Whisper v2.2 ASR Test

This project tests the Kotoba Whisper v2.2 Japanese ASR model.

## Setup

The virtual environment has been created using `uv` and all dependencies are installed.

## Usage

### Option 1: Using the helper script (Recommended)
```bash
./run_test.sh
```

### Option 2: Manual activation
```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the test script
python test_asr.py

# Deactivate when done
deactivate
```

### Option 3: Direct execution with uv
```bash
uv run python test_asr.py
```

## Testing with your audio file

When you run the test script, you'll be prompted to enter the path to an audio file. The script supports common audio formats (WAV, MP3, FLAC, etc.).

Example:
```
Enter the path to an audio file: /path/to/your/audio.wav
```

## Dependencies

- transformers: Hugging Face transformers library
- torch: PyTorch framework
- torchaudio: Audio processing for PyTorch
- librosa: Audio loading and processing
- soundfile: Audio file I/O
- accelerate: For faster model loading

## Model Information

- Model: kotoba-tech/kotoba-whisper-v2.2
- Type: Speech-to-Text (ASR)
- Language: Japanese
- Based on: OpenAI Whisper architecture

## Notes

- The model will automatically download on first run
- If you have a CUDA-compatible GPU, the model will use it automatically
- Audio files are resampled to 16kHz for processing
