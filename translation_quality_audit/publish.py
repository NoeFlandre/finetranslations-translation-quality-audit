"""Results as a Parquet file held in memory, and their upload to a Hugging Face dataset repo."""

import io

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("language", pa.string()),
        ("edu_score", pa.int64()),
        ("length_ratio", pa.float64()),
        ("english_probability", pa.float64()),
        ("copy_rate", pa.float64()),
        ("number_mismatch", pa.int64()),
        ("structure_ratio", pa.float64()),
        ("repetition_share", pa.float64()),
    ]
)
RESULT_COLUMNS = tuple(SCHEMA.names)


def results_to_parquet(rows: list[dict]) -> bytes:
    """The rows as a compressed Parquet file, built in memory with a fixed schema."""
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), buffer, compression="zstd")
    return buffer.getvalue()


def upload_results(api, repo_id: str, path_in_repo: str, data: bytes) -> None:
    """Upload a Parquet file to a dataset repo, without writing it to local disk."""
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add audit results",
    )
