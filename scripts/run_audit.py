"""Audit the translation quality of FineTranslations-Edu and upload the results as Parquet.

The language index (built once by scripts.build_index) plans each sample, so only the row groups
that hold the sampled documents are read. Nothing is stored locally except the small
language-identification model. Example:

    uv run python -m scripts.run_audit --languages fra_Latn deu_Latn --total 100
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from translation_quality_audit.hub import (
    DEFAULT_REPO,
    INDEX_LOCATION,
    INDEXED_LANGUAGES,
    RESULTS_PATH,
)
from translation_quality_audit.index import read_index
from translation_quality_audit.language import ensure_model, load_identifier
from translation_quality_audit.pipeline import audit_language
from translation_quality_audit.publish import results_to_parquet, upload_results
from translation_quality_audit.remote import open_remote

DEFAULT_MODEL = Path.home() / ".cache" / "translation-quality-audit" / "lid.176.ftz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages", nargs="+", default=list(INDEXED_LANGUAGES), choices=INDEXED_LANGUAGES
    )
    parser.add_argument("--total", type=int, default=100, help="documents per language")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="dataset repo for the results")
    parser.add_argument("--path", default=RESULTS_PATH, help="path inside the repo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="fastText LID model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Audit the requested languages and upload one Parquet file with every scored document."""
    args = parse_args(argv)
    index = read_index(open_remote, INDEX_LOCATION)
    identifier = load_identifier(ensure_model(args.model))
    rows = []
    for language in args.languages:
        rows.extend(audit_language(open_remote, language, args.total, args.seed, identifier, index))
    upload_results(HfApi(), args.repo, args.path, results_to_parquet(rows))
    print(f"uploaded {len(rows)} rows to {args.repo}/{args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
