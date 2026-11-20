import math

import pytest
from translation_quality_audit.score import FEATURES, score_pair

SOURCE = "in 1990 abcdefg"
TRANSLATION = "in 1990 in 1990 abcdefg"


class FixedModel:
    def predict(self, _text, _k):
        return ["__label__en"], [0.93]


def test_score_pair_returns_every_feature_by_name():
    scores = score_pair(SOURCE, TRANSLATION, FixedModel())

    assert list(scores) == list(FEATURES)


def test_score_pair_computes_each_feature_from_the_pair():
    scores = score_pair(SOURCE, TRANSLATION, FixedModel())

    assert scores["length_ratio"] == pytest.approx(23 / 15)
    assert scores["english_probability"] == pytest.approx(0.93)
    assert scores["copy_rate"] == pytest.approx(1.0)
    assert scores["number_mismatch"] == 0
    assert math.isnan(scores["structure_ratio"])
    assert scores["repetition_share"] == pytest.approx(0.5)


def test_score_pair_measures_structure_when_the_source_has_markers():
    scores = score_pair("a\nb", "a\nb\nc", FixedModel())

    assert scores["structure_ratio"] == pytest.approx(2.0)
