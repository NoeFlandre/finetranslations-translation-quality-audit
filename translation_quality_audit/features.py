"""Cheap, reference-free features that compare a source text with its English translation.

Each feature is undefined (NaN) when its denominator is empty, so the value stays raw and honest.
"""

import math
import re
from collections import Counter

NUMBER = re.compile(r"\d+")


def length_ratio(source: str, translation: str) -> float:
    """Translation length over source length, in characters."""
    if not source:
        return math.nan
    return len(translation) / len(source)


def copy_rate(source: str, translation: str, n: int = 6) -> float:
    """Share of the source's distinct character n-grams that appear in the translation."""
    source_grams = _char_ngrams(source, n)
    if not source_grams:
        return math.nan
    return len(source_grams & _char_ngrams(translation, n)) / len(source_grams)


def number_mismatch(source: str, translation: str) -> int:
    """Count of distinct numbers that appear on one side only."""
    return len(_numbers(source) ^ _numbers(translation))


def structure_ratio(source: str, translation: str) -> float:
    """Translation newlines and table pipes over the source's, or NaN when the source has none."""
    source_markers = _markers(source)
    if source_markers == 0:
        return math.nan
    return _markers(translation) / source_markers


def repetition_share(translation: str, n: int = 4) -> float:
    """Share of the word n-grams taken by the most frequent one, or NaN for short texts."""
    words = translation.split()
    if len(words) < n:
        return math.nan
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) / sum(grams.values())


def _char_ngrams(text: str, n: int) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _numbers(text: str) -> set[str]:
    return set(NUMBER.findall(text))


def _markers(text: str) -> int:
    return text.count("\n") + text.count("|")
