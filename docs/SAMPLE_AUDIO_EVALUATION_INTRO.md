# Performance Evaluation: Sample-audio-01

## 1. Evaluation Overview

This section presents a detailed performance evaluation of ASR models on a single, longer Japanese audio sample. Unlike the batch evaluation which used 10 short clips (0.62-8.15 seconds) with ground truth transcripts, this evaluation focuses on a longer continuous speech file to assess model performance on extended audio.

## 2. Test Audio Specifications

### Audio File: `dataset/sample_audio.mp3`

| Property | Value | Description |
|----------|-------|-------------|
| **Duration** | 26.63 seconds | Extended continuous speech (5x longer than batch average) |
| **Language** | Japanese | Native Japanese speech with natural prosody |
| **File Format** | .mp3 | Compressed audio format (common in real-world scenarios) |
| **Sample Rate** | 16 kHz | Standard rate for ASR systems |
| **Content Type** | Multi-topic dialogue | Contains 3 distinct segments covering different subjects |

### Audio Content Description

The audio file contains natural Japanese speech with three distinct segments:

1. **Segment 1 (0-10s)**: Discussion about indoor/outdoor temperature and air exchange
   - Topic: Environmental conditions and humidity management
   - Speech characteristics: Explanatory, moderate pace

2. **Segment 2 (10-20s)**: Discussion about city promotion strategy
   - Topic: PR campaigns and honest city promotion approach
   - Speech characteristics: Conversational, natural flow

3. **Segment 3 (20-26s)**: Statement about water supply
   - Topic: Water importation from Malaysia
   - Speech characteristics: Declarative statement

## 3. Evaluation Scope

### 3.1 Models Under Test

| Model | Version | Type | Deployment |
|-------|---------|------|------------|
| **Kotoba Whisper** | v2.2 | Local transformer-based ASR | On-premise (CPU) |
| **AssemblyAI** | Best model | Cloud API-based ASR | Cloud (GPU-accelerated) |

### 3.2 Performance Metrics

This evaluation measures the following objective metrics:

#### A. Processing Performance
- **Average Latency**: Total time to process the audio file (seconds)
- **Real-Time Factor (RTF)**: Ratio of processing time to audio duration
  - Formula: `RTF = Processing Time ÷ Audio Duration`
  - RTF < 1.0 = Faster than real-time
  - RTF > 1.0 = Slower than real-time

#### B. Transcription Quality
- **Formatting Quality**: Punctuation, sentence breaks, readability
- **Content Differences**: Observable variations in transcription output
- **Character Count**: Total characters including punctuation

#### C. Accuracy Assessment
⚠️ **Important Limitation**: No ground truth transcript exists for this audio file. Therefore:
- Accuracy metrics (CER/WER) **cannot be calculated**
- Transcription differences can be **observed but not verified**
- Quality assessment is limited to **formatting and contextual analysis**

## 4. Evaluation Purpose

This single-file evaluation serves multiple purposes:

1. **Assess performance on longer audio**: Determine how models handle extended speech vs. short clips
2. **Compare processing efficiency**: Measure speed differences between local and cloud processing
3. **Evaluate formatting capabilities**: Compare output quality for production use cases
4. **Identify contextual strengths**: Understand when each model performs better

## 5. Testing Environment

| Component | Specification |
|-----------|--------------|
| **Platform** | macOS (Darwin 24.6.0) |
| **CPU** | Local CPU processing (Kotoba Whisper) |
| **Python** | 3.13 |
| **Kotoba Device** | CPU with float32 precision |
| **AssemblyAI** | Cloud API with GPU acceleration |
| **Test Date** | December 15, 2025 |

## 6. Key Differences from Batch Evaluation

| Aspect | Batch Evaluation (10 samples) | Single File Evaluation |
|--------|------------------------------|----------------------|
| **Audio Length** | 0.62 - 8.15s (avg: 4.8s) | 26.63s (5x longer) |
| **Ground Truth** | ✅ Available | ❌ Not available |
| **Accuracy Measurement** | ✅ CER/WER calculated | ❌ Cannot measure |
| **Sample Count** | 10 diverse clips | 1 continuous speech |
| **Primary Focus** | Accuracy comparison | Speed + formatting comparison |

---

## 7. Expected Outcomes

Based on preliminary observations, this evaluation aims to answer:

1. **Does file length affect model performance?**
   - Hypothesis: Longer files may favor different processing strategies

2. **How do formatting capabilities differ?**
   - Hypothesis: Cloud APIs may provide better punctuation and structure

3. **What is the speed trade-off between local and cloud processing?**
   - Hypothesis: Cloud GPU may be faster despite network overhead

4. **Can we infer quality without ground truth?**
   - Hypothesis: Contextual analysis and formatting can provide quality indicators

---

**Next Section**: Performance results and detailed comparison between Kotoba Whisper v2.2 and AssemblyAI Best models.
