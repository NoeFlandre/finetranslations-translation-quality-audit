import fasttext
import pytest
from translation_quality_audit.language import english_probability, load_identifier


class FakeModel:
    def __init__(self, labels, probs):
        self.labels = labels
        self.probs = probs
        self.seen = []

    def predict(self, text, k):
        self.seen.append((text, k))
        return self.labels, self.probs


def test_english_probability_is_the_probability_of_the_english_label():
    model = FakeModel(["__label__fr", "__label__en"], [0.7, 0.2])

    assert english_probability(model, "Bonjour") == pytest.approx(0.2)


def test_english_probability_is_zero_when_english_is_not_among_the_candidates():
    model = FakeModel(["__label__fr"], [0.9])

    assert english_probability(model, "Bonjour") == 0.0


def test_newlines_are_flattened_before_prediction():
    model = FakeModel(["__label__en"], [0.99])

    english_probability(model, "first line\nsecond line")

    assert model.seen == [("first line second line", 5)]


def test_load_identifier_opens_the_model_file_by_path(monkeypatch, tmp_path):
    opened = []
    sentinel = object()

    def fake_load_model(path):
        opened.append(path)
        return sentinel

    monkeypatch.setattr(fasttext, "load_model", fake_load_model)

    assert load_identifier(tmp_path / "lid.176.ftz") is sentinel
    assert opened == [str(tmp_path / "lid.176.ftz")]
