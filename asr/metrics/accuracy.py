"""
Utility functions for ASR accuracy metrics calculation
Provides CER/WER calculation using torchmetrics with Japanese word segmentation support
"""

import os
import json
from torchmetrics.text import CharErrorRate, WordErrorRate


# Module-level cache for MeCab tokenizer
_mecab_tagger = None
_mecab_available = None  # None = not checked, True/False = checked
_mecab_warning_shown = False


def detect_japanese(text):
    """
    Detect if text is predominantly Japanese

    Args:
        text: Input text string

    Returns:
        bool: True if >50% of characters are Japanese (Hiragana/Katakana/Kanji)
    """
    if not text:
        return False

    japanese_chars = sum(1 for c in text if
        '\u3040' <= c <= '\u309F' or  # Hiragana
        '\u30A0' <= c <= '\u30FF' or  # Katakana
        '\u4E00' <= c <= '\u9FFF')    # Kanji

    return japanese_chars / len(text) > 0.5


def get_mecab_tokenizer():
    """
    Get MeCab tokenizer with lazy initialization and error handling

    Returns:
        MeCab.Tagger or None: MeCab tagger instance if available, None otherwise
    """
    global _mecab_tagger, _mecab_available, _mecab_warning_shown

    # Return cached result
    if _mecab_available is not None:
        return _mecab_tagger

    try:
        import MeCab
        _mecab_tagger = MeCab.Tagger("-Owakati")  # Wakati mode (word boundaries)
        _mecab_available = True
        return _mecab_tagger
    except ImportError:
        if not _mecab_warning_shown:
            print("Warning: mecab-python3 not installed. Using character-based WER fallback.")
            print("Install with: pip install mecab-python3 unidic-lite")
            _mecab_warning_shown = True
        _mecab_available = False
        return None
    except Exception as e:
        if not _mecab_warning_shown:
            print(f"Warning: Could not initialize MeCab: {e}")
            print("Using character-based WER fallback.")
            _mecab_warning_shown = True
        _mecab_available = False
        return None


def tokenize_japanese(text):
    """
    Tokenize Japanese text using MeCab, with fallback to character-level

    Args:
        text: Japanese text string

    Returns:
        list: List of tokens (words if MeCab available, characters otherwise)
    """
    if not text:
        return []

    tagger = get_mecab_tokenizer()

    if tagger:
        # MeCab available - use proper word segmentation
        try:
            # Parse and get wakati (word-separated) output
            tokens = tagger.parse(text.strip()).strip().split()
            return [t for t in tokens if t]  # Remove empty strings
        except Exception as e:
            print(f"Warning: MeCab tokenization failed: {e}")
            # Fall through to character-based

    # Fallback: character-based tokenization
    # Each character becomes a "word" (better than treating entire string as one word)
    return list(text.strip())


def calculate_japanese_wer(hyp_tokens, ref_tokens):
    """
    Calculate WER from token lists

    Args:
        hyp_tokens: List of hypothesis tokens
        ref_tokens: List of reference tokens

    Returns:
        float: Word Error Rate
    """
    wer_metric = WordErrorRate()

    # Join tokens with spaces for WER calculation
    hyp_str = ' '.join(hyp_tokens)
    ref_str = ' '.join(ref_tokens)

    wer = wer_metric([hyp_str], [ref_str]).item()
    return wer


def load_ground_truth(audio_path):
    """
    Load ground truth transcript from metadata.json based on audio filename

    Args:
        audio_path: Path to audio file (absolute or relative)

    Returns:
        str: Ground truth transcript, or None if not found
    """
    try:
        audio_basename = os.path.basename(audio_path)
        metadata_path = "dataset/sample_audios/metadata.json"

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        for entry in metadata:
            if entry.get('audio_file') == audio_basename:
                return entry.get('transcript')

    except Exception:
        # Silently fail - it's normal for arbitrary audio files
        pass

    return None


def calculate_accuracy_metrics(hypothesis, reference):
    """
    Calculate CER and WER using torchmetrics
    Automatically uses Japanese tokenization for Japanese text

    Args:
        hypothesis: Predicted transcript (ASR output)
        reference: Ground truth transcript

    Returns:
        dict: {'cer': float, 'wer': float} or None if calculation fails
    """
    if not hypothesis or not reference:
        return None

    try:
        # Initialize CER metric
        cer_metric = CharErrorRate()

        # CER calculation (unchanged - character-level works well for Japanese)
        cer = cer_metric([hypothesis], [reference]).item()

        # WER calculation - use Japanese tokenization if applicable
        if detect_japanese(hypothesis) or detect_japanese(reference):
            # Japanese text detected - use MeCab tokenization
            hyp_tokens = tokenize_japanese(hypothesis)
            ref_tokens = tokenize_japanese(reference)

            # Calculate WER from token lists
            wer = calculate_japanese_wer(hyp_tokens, ref_tokens)
        else:
            # Non-Japanese text - use standard WER
            wer_metric = WordErrorRate()
            wer = wer_metric([hypothesis], [reference]).item()

        return {
            'cer': cer,
            'wer': wer
        }

    except Exception as e:
        print(f"Warning: Could not calculate accuracy metrics: {e}")
        return None
