# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Japanese ASR (Automatic Speech Recognition) testing repository that evaluates multiple transcription approaches:
- **Kotoba Whisper v2.2**: Local Japanese ASR model (kotoba-tech/kotoba-whisper-v2.2)
- **AssemblyAI**: Cloud-based transcription API

The project contains various test scripts for comparing performance, latency metrics, and real-time factor (RTF) across different modes (batch, streaming, file-based, microphone).

## Architecture

The ASR system is organized into reusable modules:

### Core Library (`asr/`)
- **models/**: Model implementations (Kotoba Whisper, AssemblyAI)
  - `base.py`: TranscriptionResult, ChunkInfo, ASRModel protocol
  - `kotoba.py`: Local Kotoba Whisper implementation
  - `assemblyai.py`: Cloud AssemblyAI implementation

- **metrics/**: Performance and accuracy metrics
  - `accuracy.py`: CER/WER calculation with Japanese MeCab support
  - `performance.py`: RTF, latency calculations

- **display/**: UI and formatting
  - `formatters.py`: Result formatting utilities
  - `live_display.py`: Streaming display (unified)

- **evaluation/**: Batch evaluation framework
  - `evaluator.py`: BatchEvaluator for dataset processing
  - `reporters.py`: Statistics and export

- **streaming/**: Streaming components (future expansion)

- **utils/**: Common utilities (future expansion)

### User Scripts (`scripts/`)
- `asr_transcribe.py`: Single audio file transcription
- `asr_batch_evaluate.py`: Batch evaluation with statistics (processes multiple files)

### Legacy Test Scripts
The following scripts are retained for backward compatibility:
- `test_asr.py`, `test_asr_optimized.py`: Kotoba Whisper tests
- `test_asr_streaming.py`: Streaming transcription
- `test_asr_assemblyai.py`, `test_asr_assemblyai_streaming.py`: AssemblyAI tests
- `evaluate_batch.py`: Batch evaluation (legacy)

## Environment Setup

This project uses Python 3.13 with a virtual environment managed by `uv`. The virtual environment is located at `.venv/`.

### Activate the virtual environment:
```bash
source .venv/bin/activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables

Required API keys are stored in `.env`:
- `ASSEMBLYAI_API_KEY`: For AssemblyAI cloud transcription
- `HF_TOKEN` / `HF_API_TOKEN`: For Hugging Face model downloads

## Common Commands

### Single Audio Transcription (Recommended)
```bash
# Kotoba Whisper (default)
python scripts/asr_transcribe.py dataset/sample_audios/audio_000.wav

# AssemblyAI
python scripts/asr_transcribe.py dataset/sample_audios/audio_000.wav --model assemblyai

# With timestamps
python scripts/asr_transcribe.py audio.mp3 --show-timestamps
```

### Batch Evaluation (Recommended)
```bash
# Evaluate Kotoba Whisper on all 10 samples
python scripts/asr_batch_evaluate.py

# Evaluate AssemblyAI on all samples
python scripts/asr_batch_evaluate.py --model assemblyai

# Verbose output + JSON export
python scripts/asr_batch_evaluate.py --verbose --output results.json
```

### Legacy Scripts

#### Run Basic ASR Test (Kotoba Whisper)
```bash
# Using default audio file (dataset/sample_audio.mp3)
python test_asr.py

# With custom audio file
python test_asr.py path/to/audio.wav

# Using the helper script
./scripts/run_test.sh
```

#### Run Optimized ASR Test with Metrics
```bash
# Shows detailed latency metrics, RTF, and timestamps
python test_asr_optimized.py

# With custom audio file
python test_asr_optimized.py path/to/audio.mp3
```

#### Run Streaming ASR Test (Kotoba Whisper)
```bash
# File streaming mode (default)
python test_asr_streaming.py --mode file --audio dataset/sample_audio.mp3

# Microphone streaming mode
python test_asr_streaming.py --mode mic

# With custom chunk duration
python test_asr_streaming.py --mode file --chunk-duration 5.0
```

#### Run AssemblyAI Tests
```bash
# File transcription with AssemblyAI
python test_asr_assemblyai.py --audio dataset/sample_audio.mp3

# Show comparison with Kotoba Whisper
python test_asr_assemblyai.py --compare

# Use different model
python test_asr_assemblyai.py --model nano

# Real-time microphone streaming (AssemblyAI only)
python test_asr_assemblyai_streaming.py
```

## Code Architecture

### Test Script Organization

1. **test_asr.py**: Basic ASR test using manual model loading
   - Uses `AutoProcessor` and `AutoModelForSpeechSeq2Seq` from transformers
   - Loads model explicitly, moves to GPU if available
   - Simple batch transcription workflow

2. **test_asr_optimized.py**: Enhanced version with performance metrics
   - Uses transformers `pipeline` API (simpler, optimized)
   - Flash Attention 2 support (auto-detected)
   - Detailed latency and RTF metrics
   - Timestamp extraction from chunks
   - Configurable chunk length and batch size

3. **test_asr_streaming.py**: Streaming ASR for both files and microphone
   - **AudioChunker**: Handles audio segmentation with overlap
   - **StreamingTranscriber**: Processes chunks with Whisper pipeline
   - **LiveDisplay**: Terminal UI with real-time updates
   - Supports both file simulation and live microphone input
   - Calculates per-chunk latency and overall RTF

4. **test_asr_assemblyai.py**: Cloud-based transcription via AssemblyAI API
   - File upload and transcription workflow
   - Performance comparison with Kotoba Whisper
   - Metrics: total time, RTF, confidence scores

5. **test_asr_assemblyai_streaming.py**: Real-time AssemblyAI streaming
   - Uses AssemblyAI's streaming v3 API
   - Microphone-only (file mode not supported in streaming)
   - Event-driven architecture (BeginEvent, TurnEvent, TerminationEvent)
   - Live display with turn-based updates
   - **Important**: For file transcription with AssemblyAI, use `test_asr_assemblyai.py` instead

### Key Design Patterns

**Model Loading (Kotoba Whisper)**:
- Basic approach: `AutoProcessor` + `AutoModelForSpeechSeq2Seq` (test_asr.py)
- Pipeline approach: `pipeline("automatic-speech-recognition", ...)` (test_asr_optimized.py, test_asr_streaming.py)
- Pipeline is preferred for production use (simpler, optimized, Flash Attention support)

**Audio Processing**:
- All scripts use 16kHz sample rate (Whisper requirement)
- librosa for audio loading and duration calculation
- sounddevice for microphone capture

**Streaming Architecture**:
- Overlapping chunks (default: 3s chunks with 0.5s overlap)
- Generator pattern for chunk processing
- Real-time terminal display with metrics

**Performance Metrics**:
- **Latency**: Processing time per chunk
- **RTF (Real-Time Factor)**: `processing_time / audio_duration`
  - RTF < 1.0 = faster than real-time
  - RTF > 1.0 = slower than real-time

### Dataset Structure

- `dataset/sample_audio.mp3`: Default test audio file
- `dataset/sample_audios/`: Multiple test audio files with transcriptions
  - Format: `audio_NNN.wav` and `audio_NNN.txt` pairs
  - Used for batch testing and evaluation
- `dataset/audio_downloader.ipynb`: Jupyter notebook for downloading audio samples

## Device and Optimization

- **GPU Detection**: All scripts auto-detect CUDA availability
- **Data Types**: FP16 on GPU, FP32 on CPU
- **Flash Attention 2**: Auto-enabled if `flash_attn` package is installed
  - Install: `pip install flash-attn --no-build-isolation`
  - Significantly improves speed on compatible GPUs

## Language and Task Configuration

All Kotoba Whisper scripts are configured for:
- **Language**: Japanese (`language="ja"`)
- **Task**: Transcription (`task="transcribe"`)

When working with audio in other languages, update the `language` parameter accordingly.

## Performance Evaluation with Ground Truth

The `dataset/sample_audios/` directory contains 10 audio-transcription pairs for accuracy evaluation:
- Audio files: `audio_000.wav` through `audio_009.wav`
- Ground truth: `audio_000.txt` through `audio_009.txt`
- Metadata: `metadata.json` (contains all transcripts, durations, sample rates)

### Metadata Structure

The `metadata.json` file contains:
```json
{
  "index": 0,
  "audio_file": "audio_000.wav",
  "transcript_file": "audio_000.txt",
  "transcript": "これまたジミーさん",
  "duration_seconds": 1.39,
  "sample_rate": 16000
}
```

### Accuracy Metrics for Japanese ASR

Common metrics for evaluating Japanese transcription accuracy:

1. **CER (Character Error Rate)**: Primary metric for Japanese
   - Measures character-level errors (insertions, deletions, substitutions)
   - Formula: `(S + D + I) / N` where S=substitutions, D=deletions, I=insertions, N=total characters
   - Lower is better (0% = perfect)

2. **WER (Word Error Rate)**: Japanese-aware word segmentation
   - Automatically detects Japanese text and uses MeCab tokenizer
   - Provides accurate word-level segmentation for Japanese
   - Falls back to character-based WER if MeCab unavailable
   - For English/mixed text, uses standard whitespace tokenization

### Evaluating Transcription Accuracy

To evaluate model accuracy against ground truth, use the `torchmetrics` library (already installed):

```python
from torchmetrics.text import CharErrorRate, WordErrorRate
import json

# Load ground truth
with open("dataset/sample_audios/metadata.json") as f:
    metadata = json.load(f)

# Initialize metrics
cer = CharErrorRate()
wer = WordErrorRate()

# For each test sample
for item in metadata:
    audio_path = f"dataset/sample_audios/{item['audio_file']}"
    ground_truth = item['transcript']

    # Get prediction from your model
    prediction = transcribe_audio(audio_path, ...)

    # Calculate metrics
    cer_score = cer([prediction], [ground_truth])
    wer_score = wer([prediction], [ground_truth])

    print(f"CER: {cer_score:.2%}, WER: {wer_score:.2%}")
```

### Japanese Tokenization Setup

For accurate Japanese WER calculation, MeCab tokenizer is required:

```bash
pip install mecab-python3 unidic-lite
```

The system will automatically:
- Detect Japanese text (>50% Japanese characters)
- Use MeCab tokenization for word segmentation
- Fall back gracefully if MeCab is not installed

**Note:** If MeCab is unavailable, WER will use character-based approximation and a one-time warning will be displayed.

### Creating an Evaluation Script

When creating a new evaluation script, include:
1. Load all test samples from `metadata.json`
2. Transcribe each audio file with the target model
3. Calculate CER and WER for each sample
4. Compute aggregate statistics (mean, median, std)
5. Compare multiple models (Kotoba Whisper vs AssemblyAI)
6. Generate a report with per-sample and overall metrics

Example structure:
```bash
python evaluate_accuracy.py --model kotoba --output results.json
python evaluate_accuracy.py --model assemblyai --compare results.json
```

## Important Notes

- The repository is NOT a git repository (version control not initialized)
- Model weights auto-download on first run from Hugging Face
- AssemblyAI streaming API is microphone-only; use regular API for files
- Streaming mode displays live updates by clearing terminal screen
