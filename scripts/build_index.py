"""Build the language index from the Hugging Face Parquet files, once, and upload it.

Indexes only the audited languages, and only in the row groups whose footer range can hold them.
Example: uv run python -m scripts.build_index
"""

import argparse

from huggingface_hub import HfApi, HfFileSystem
from translation_quality_audit.hub import DATA_GLOB, DEFAULT_REPO, INDEX_PATH, INDEXED_LANGUAGES
from translation_quality_audit.index import build_index, index_to_parquet
from translation_quality_audit.publish import upload_results
from translation_quality_audit.remote import open_remote

DEFAULT_PATH = INDEX_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="dataset repo for the index")
    parser.add_argument("--path", default=DEFAULT_PATH, help="path inside the repo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Index the audited languages in every data file and upload the index as one Parquet file."""
    args = parse_args(argv)
    paths = sorted(HfFileSystem().glob(DATA_GLOB))
    index = build_index(open_remote, paths, INDEXED_LANGUAGES)
    upload_results(HfApi(), args.repo, args.path, index_to_parquet(index), "Add language index")
    print(f"indexed {index.num_rows} entries from {len(paths)} files")
    print(f"uploaded to {args.repo}/{args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
