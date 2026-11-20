"""Audit the translation quality of FineTranslations-Edu and upload the results as Parquet.

Only the language row groups that the sample needs are read, and nothing is stored locally except
the small language-identification model. Example:

    uv run python -m scripts.run_audit --languages fra_Latn deu_Latn --total 100
"""

import argparse
from functools import partial
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem
from translation_quality_audit.language import ensure_model, load_identifier
from translation_quality_audit.pipeline import audit_language
from translation_quality_audit.publish import results_to_parquet, upload_results

DATA_GLOB = "datasets/HuggingFaceFW/finetranslations-edu/data/*.parquet"
HIGH_RESOURCE = ("fra_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani")
DEFAULT_REPO = "NoeFlandre/finetranslations-translation-quality-audit"
DEFAULT_MODEL = Path.home() / ".cache" / "translation-quality-audit" / "lid.176.ftz"
READ_BLOCK_SIZE = 1 << 18  # 256 KB: small enough that a column read does not pull in megabytes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=list(HIGH_RESOURCE))
    parser.add_argument("--total", type=int, default=100, help="documents per language")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="dataset repo for the results")
    parser.add_argument("--path", default="v1/audit.parquet", help="path inside the repo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="fastText LID model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Audit the requested languages and upload one Parquet file with every scored document."""
    args = parse_args(argv)
    fs = HfFileSystem()
    paths = sorted(fs.glob(DATA_GLOB))
    identifier = load_identifier(ensure_model(args.model))
    rows = []
    opener = partial(fs.open, mode="rb", block_size=READ_BLOCK_SIZE)
    for language in args.languages:
        rows.extend(audit_language(opener, paths, language, args.total, args.seed, identifier))
    upload_results(HfApi(), args.repo, args.path, results_to_parquet(rows))
    print(f"uploaded {len(rows)} rows to {args.repo}/{args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
