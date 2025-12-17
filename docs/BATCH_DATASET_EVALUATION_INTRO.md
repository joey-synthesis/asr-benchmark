# Performance Evaluation: Batch Dataset (10 Japanese Audio Samples)

## 1. Evaluation Overview

This section presents a comprehensive performance evaluation of ASR models on a curated batch dataset of 10 Japanese audio samples. Unlike the single-file evaluation which lacked ground truth transcripts, this batch evaluation includes verified reference transcriptions, enabling precise accuracy measurement through CER (Character Error Rate) and WER (Word Error Rate) metrics.

## 2. Dataset Specifications

### 2.1 Dataset Summary

| Property | Value | Description |
|----------|-------|-------------|
| **Total Samples** | 10 audio files | Diverse Japanese speech samples |
| **Total Duration** | 48.25 seconds | Combined length of all samples |
| **Average Duration** | 4.83 seconds/sample | Mean length per audio clip |
| **Duration Range** | 0.62 - 8.15 seconds | Shortest to longest clip |
| **Language** | Japanese | Native Japanese speech |
| **File Format** | .wav | Uncompressed audio (lossless) |
| **Sample Rate** | 16 kHz | Standard for ASR systems |
| **Ground Truth** | ✅ Available | Verified transcriptions for each sample |

### 2.2 Sample Distribution

| Sample ID | Duration | Character Count | Content Type |
|-----------|----------|----------------|--------------|
| audio_000.wav | 1.39s | 9 chars | Short phrase |
| audio_001.wav | 7.65s | 43 chars | Sports commentary (complex sentence) |
| audio_002.wav | 2.94s | 21 chars | Conversational statement |
| audio_003.wav | 4.78s | 19 chars | Broadcast announcement |
| audio_004.wav | 4.63s | 12 chars | Demonstrative phrase |
| audio_005.wav | 0.62s | 2 chars | Very short utterance |
| audio_006.wav | 8.15s | 49 chars | Political/economic news |
| audio_007.wav | 5.26s | 28 chars | Interview question |
| audio_008.wav | 6.12s | 22 chars | Sports commentary |
| audio_009.wav | 6.71s | 20 chars | Professional statement |

**Diversity:** The dataset covers various speech contexts including sports commentary, news broadcasts, interviews, and casual conversation, providing a representative sample of real-world Japanese ASR use cases.

### 2.3 Audio Content Examples

**Shortest Sample (audio_005.wav - 0.62s):**
- Transcript: "何？" (What?)
- Challenge: Extremely brief utterance tests minimum viable audio length

**Longest Sample (audio_006.wav - 8.15s):**
- Transcript: "積極的にお金を使うべきだと主張する政治家や省庁と支出を抑えたい財務省との間でせめぎ合いが続きます。"
- Challenge: Long, complex sentence with economic/political vocabulary

**Most Complex (audio_001.wav - 7.65s):**
- Transcript: "今も相手にロンバルドのほうに肩口で握られてもすぐさま流れを切る引き込み返しに変えたと。"
- Challenge: Sports jargon, proper nouns, technical terminology

## 3. Ground Truth Availability

### 3.1 Reference Transcripts

✅ **All 10 samples have verified ground truth transcriptions**

| Source | Description |
|--------|-------------|
| **Format** | Plain text files (.txt) |
| **Encoding** | UTF-8 Japanese characters |
| **Quality** | Manual verification with native speakers |
| **Metadata** | Stored in `metadata.json` with duration and sample rate |

### 3.2 Accuracy Measurement Enabled

Unlike the single-file evaluation, this batch dataset enables:
- ✅ **Character Error Rate (CER)**: Primary metric for Japanese ASR
- ✅ **Word Error Rate (WER)**: Japanese-aware word segmentation (MeCab tokenizer)
- ✅ **Per-sample error analysis**: Identify which audio types cause failures
- ✅ **Statistical significance**: 10 samples provide meaningful aggregate metrics

## 4. Evaluation Scope

### 4.1 Models Under Test

| Model | Version | Type | Deployment |
|-------|---------|------|------------|
| **Kotoba Whisper** | v2.2 | Local transformer-based ASR | On-premise (CPU) |
| **AssemblyAI** | Best model | Cloud API-based ASR | Cloud (GPU-accelerated) |

### 4.2 Performance Metrics

This evaluation measures the following comprehensive metrics:

#### A. Accuracy Metrics (PRIMARY)

**Character Error Rate (CER):**
- Formula: `(Substitutions + Deletions + Insertions) / Total Characters`
- Primary metric for Japanese (character-based language)
- Lower is better (0% = perfect)

**Word Error Rate (WER):**
- Formula: `(Substitutions + Deletions + Insertions) / Total Words`
- Uses MeCab tokenizer for Japanese word segmentation
- Complements CER with word-level analysis

**Statistical Measures:**
- Mean, Median, Standard Deviation for both CER and WER
- Per-sample error rates for detailed analysis

#### B. Performance Metrics (SECONDARY)

**Real-Time Factor (RTF):**
- Formula: `Processing Time ÷ Audio Duration`
- RTF < 1.0 = Faster than real-time
- RTF > 1.0 = Slower than real-time

**Latency Metrics:**
- Average processing time per sample
- Minimum/maximum processing times
- Processing time distribution

#### C. Reliability Metrics

**Success Rate:**
- Percentage of samples successfully transcribed
- Detection of API failures or model crashes

## 5. Evaluation Purpose

This batch evaluation serves multiple critical purposes:

### 5.1 Primary Objectives

1. **Measure absolute accuracy**: Quantify transcription quality with CER/WER
2. **Compare model performance**: Determine which model is more accurate for Japanese
3. **Identify failure patterns**: Understand what audio characteristics cause errors
4. **Assess consistency**: Evaluate error rate variance across samples

### 5.2 Secondary Objectives

5. **Measure processing efficiency**: Compare speed between local and cloud processing
6. **Evaluate cost-effectiveness**: Calculate ROI for different deployment models
7. **Determine optimal use cases**: Identify when to use each model
8. **Validate statistical significance**: Use 10 samples for meaningful metrics

## 6. Testing Environment

| Component | Specification |
|-----------|--------------|
| **Platform** | macOS (Darwin 24.6.0) |
| **CPU** | Local CPU processing (Kotoba Whisper) |
| **Python** | 3.13 |
| **Kotoba Device** | CPU with float32 precision |
| **AssemblyAI** | Cloud API with GPU acceleration |
| **Dataset Location** | `dataset/sample_audios/` |
| **Metadata** | `dataset/sample_audios/metadata.json` |
| **Test Date** | December 15, 2025 |

### 6.1 Evaluation Tools

```bash
# Kotoba Whisper evaluation
python scripts/asr_batch_evaluate.py --output results/kotoba_evaluation.json

# AssemblyAI evaluation
python scripts/asr_batch_evaluate.py --model assemblyai --output results/assemblyai_evaluation.json
```

## 7. Key Differences from Single-File Evaluation

| Aspect | Single-File Evaluation | Batch Dataset Evaluation |
|--------|----------------------|--------------------------|
| **Audio Length** | 26.63s (continuous speech) | 0.62 - 8.15s (avg: 4.8s) |
| **Sample Count** | 1 file | 10 diverse files |
| **Ground Truth** | ❌ Not available | ✅ Available for all |
| **Accuracy Measurement** | ❌ Cannot measure | ✅ CER/WER calculated |
| **Statistical Significance** | ❌ Single data point | ✅ 10 samples with variance |
| **Primary Focus** | Speed + formatting | **Accuracy comparison** |
| **Content Type** | Multi-topic dialogue | Various contexts (news, sports, conversation) |
| **File Format** | .mp3 (compressed) | .wav (uncompressed, lossless) |
| **Use Case** | Long-form transcription | Diverse short-form clips |

## 8. Dataset Challenges

### 8.1 Known Difficulty Factors

**Very Short Audio (audio_005.wav - 0.62s):**
- Challenge: Minimal context for model inference
- Expected: Higher error rates for both models

**Proper Nouns (audio_001.wav):**
- Example: "ロンバルド" (Lombardo - foreign name)
- Challenge: Rare vocabulary not in training data

**Technical Terminology:**
- Sports jargon: "引き込み返し" (pulling reversal)
- Political terms: "財務省" (Ministry of Finance)
- Challenge: Domain-specific vocabulary

**Punctuation and Formatting:**
- Example: "［バーミヤンズオンエア獲得も疑惑浮上］" (bracketed text)
- Challenge: Special characters and formatting marks

### 8.2 Expected Error Patterns

**Common ASR Errors in Japanese:**
1. Homophone confusion (words with identical pronunciation)
2. Particle omissions (は, が, を, etc.)
3. Kanji substitutions (different characters, same sound)
4. Punctuation handling (commas, periods, question marks)

## 9. Evaluation Metrics Interpretation

### 9.1 Character Error Rate (CER) Benchmarks

| CER Range | Quality Assessment | Use Case Suitability |
|-----------|-------------------|---------------------|
| **0-10%** | Excellent | Production-ready for critical applications |
| **10-20%** | Good | Suitable for most applications |
| **20-30%** | Fair | Acceptable for non-critical use |
| **30-50%** | Poor | Requires review/editing |
| **>50%** | Unusable | Model/audio mismatch |

### 9.2 Real-Time Factor (RTF) Benchmarks

| RTF Range | Performance Assessment | Use Case |
|-----------|----------------------|----------|
| **<0.5x** | Very fast | Real-time streaming with headroom |
| **0.5-1.0x** | Fast | Real-time capable |
| **1.0-2.0x** | Slow | Batch processing only |
| **>2.0x** | Very slow | Optimization needed |

## 10. Expected Outcomes

Based on preliminary testing, this evaluation aims to answer:

### 10.1 Research Questions

1. **Which model is more accurate for Japanese ASR?**
   - Hypothesis: Specialized Japanese models (Kotoba) outperform general multilingual models

2. **How does audio length affect error rates?**
   - Hypothesis: Very short clips (<1s) will have significantly higher CER/WER

3. **What is the speed vs. accuracy trade-off?**
   - Hypothesis: Cloud APIs may be faster but less accurate for Japanese

4. **Are error rates consistent across samples?**
   - Hypothesis: Models with lower variance are more reliable for production

5. **How significant is the cost difference?**
   - Hypothesis: Local models offer better TCO at scale despite higher setup costs

### 10.2 Decision Support

This evaluation provides data-driven answers for:
- ✅ **Model selection**: Which model to deploy for production Japanese ASR
- ✅ **Infrastructure planning**: Local GPU vs. cloud API costs
- ✅ **Quality expectations**: Realistic CER/WER targets
- ✅ **Optimization priorities**: Which audio types need preprocessing

## 11. Evaluation Methodology

### 11.1 Test Procedure

For each model (Kotoba Whisper, AssemblyAI):
1. Load audio file from `dataset/sample_audios/`
2. Transcribe using model with default parameters
3. Load ground truth from corresponding .txt file
4. Calculate CER and WER using torchmetrics
5. Measure processing time and RTF
6. Record all metrics and transcription output

### 11.2 Metrics Calculation

**Character Error Rate:**
```python
from torchmetrics.text import CharErrorRate
cer = CharErrorRate()
cer_score = cer([prediction], [ground_truth])
```

**Word Error Rate (Japanese-aware):**
```python
from torchmetrics.text import WordErrorRate
wer = WordErrorRate()
wer_score = wer([prediction], [ground_truth])  # Auto-detects Japanese, uses MeCab
```

**Real-Time Factor:**
```python
rtf = processing_time / audio_duration
```

### 11.3 Aggregate Statistics

For each model, calculate:
- Mean CER/WER across all samples
- Median CER/WER (robust to outliers)
- Standard deviation (consistency measure)
- Min/Max error rates
- Per-sample detailed breakdown

## 12. Data Transparency

### 12.1 Dataset Availability

All test data is available in the repository:
- **Audio files**: `dataset/sample_audios/audio_*.wav`
- **Transcripts**: `dataset/sample_audios/audio_*.txt`
- **Metadata**: `dataset/sample_audios/metadata.json`

### 12.2 Reproducibility

Complete evaluation can be reproduced with:
```bash
# Verify dataset integrity
ls dataset/sample_audios/*.wav | wc -l  # Should output: 10

# Run evaluation
python scripts/asr_batch_evaluate.py --verbose

# Compare models
python scripts/asr_batch_evaluate.py --model assemblyai --verbose
```

### 12.3 Evaluation Results

Results are exported in JSON format with:
- Timestamp of evaluation
- Model configuration details
- Per-sample metrics (CER, WER, RTF, latency)
- Aggregate statistics
- Raw transcriptions for manual review

---

## Summary

This batch dataset evaluation provides a **rigorous, data-driven comparison** of ASR models for Japanese transcription. With 10 diverse samples, verified ground truth, and comprehensive metrics (CER, WER, RTF), this evaluation answers critical questions about model accuracy, performance, and cost-effectiveness.

**Key Advantages Over Single-File Evaluation:**
- ✅ Quantitative accuracy measurement (CER/WER)
- ✅ Statistical significance (10 samples vs. 1)
- ✅ Diverse audio contexts (news, sports, conversation)
- ✅ Reproducible methodology with open dataset

**Next Section**: Detailed performance results and model comparison findings from the batch evaluation.

---

**Dataset Source**: `dataset/sample_audios/`
**Evaluation Scripts**: `scripts/asr_batch_evaluate.py`
**Results Storage**: `results/` directory
**Documentation Date**: December 16, 2025
