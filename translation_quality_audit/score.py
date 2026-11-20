"""The v1 feature set for one source text and its English translation."""

from translation_quality_audit.features import (
    copy_rate,
    length_ratio,
    number_mismatch,
    repetition_share,
    structure_ratio,
)
from translation_quality_audit.language import english_probability

FEATURES = (
    "length_ratio",
    "english_probability",
    "copy_rate",
    "number_mismatch",
    "structure_ratio",
    "repetition_share",
)


def score_pair(source: str, translation: str, identifier) -> dict[str, float]:
    """Every v1 feature for one source text and its translation, keyed by name."""
    return {
        "length_ratio": length_ratio(source, translation),
        "english_probability": english_probability(identifier, translation),
        "copy_rate": copy_rate(source, translation),
        "number_mismatch": number_mismatch(source, translation),
        "structure_ratio": structure_ratio(source, translation),
        "repetition_share": repetition_share(translation),
    }
