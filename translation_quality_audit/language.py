"""English-ness of a text, from a fastText language-identification model."""

import urllib.request
from pathlib import Path

import fasttext

CANDIDATES = 5
ENGLISH = "__label__en"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def english_probability(model, text: str) -> float:
    """Probability that the text is English; zero when English is not among the top candidates."""
    labels, probs = model.predict(text.replace("\n", " "), CANDIDATES)
    return next((prob for label, prob in zip(labels, probs, strict=True) if label == ENGLISH), 0.0)


def load_identifier(path: Path):
    """Open a fastText language-identification model from a file."""
    return fasttext.load_model(str(path))


def ensure_model(path: Path, url: str = MODEL_URL) -> Path:
    """Download the model once into the cache; a failed download leaves nothing behind."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            partial.write_bytes(response.read())
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path
