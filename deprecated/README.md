# Deprecated Scripts

These scripts have been replaced by the new modular architecture in `asr/` and `scripts/`.

## Why These Files Are Here

The legacy test scripts contained duplicate code and mixed concerns (transcription + metrics + display). They have been replaced with a clean, modular architecture that eliminates code duplication and provides better separation of concerns.

## Migration Guide

| Old Script | New Script | Notes |
|------------|------------|-------|
| test_asr.py | scripts/asr_transcribe.py | Simpler interface using new model abstraction |
| test_asr_optimized.py | scripts/asr_transcribe.py | Same functionality, unified interface |
| test_asr_streaming.py | *(future)* scripts/asr_streaming.py | Not yet migrated - streaming functionality to be added |
| test_asr_assemblyai.py | scripts/asr_transcribe.py --model assemblyai | Unified model interface |
| test_asr_assemblyai_streaming.py | *(future)* scripts/asr_streaming.py | Not yet migrated |
| evaluate_batch.py | scripts/asr_batch_evaluate.py | Same functionality with cleaner code |

## New Architecture Benefits

- **No Code Duplication**: Single implementation of each component
- **Unified Model Interface**: Both Kotoba and AssemblyAI use the same API
- **Modular Design**: Separate modules for models, metrics, display, and evaluation
- **Easier Testing**: Each module can be tested independently
- **Future Expansion**: Easy to add new ASR models by implementing the ASRModel protocol

## Quick Start with New Scripts

### Single File Transcription
```bash
# Kotoba Whisper
python scripts/asr_transcribe.py dataset/sample_audios/audio_000.wav

# AssemblyAI
python scripts/asr_transcribe.py audio.wav --model assemblyai

# With timestamps
python scripts/asr_transcribe.py audio.wav --show-timestamps
```

### Batch Evaluation
```bash
# Evaluate all samples
python scripts/asr_batch_evaluate.py

# Verbose output with JSON export
python scripts/asr_batch_evaluate.py --verbose --output results.json

# Test AssemblyAI
python scripts/asr_batch_evaluate.py --model assemblyai
```

## Documentation

See [../CLAUDE.md](../CLAUDE.md) for complete documentation of the new architecture.

## Can I Still Use These Scripts?

Yes, these legacy scripts still work, but they are no longer maintained. We recommend migrating to the new scripts for:
- Cleaner code
- Better performance
- Unified interface
- Future updates and features
