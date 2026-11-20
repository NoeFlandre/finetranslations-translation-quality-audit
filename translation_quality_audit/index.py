"""A small index of how many documents each language and quality level has in each row group.

Building it reads only the language and quality columns, once. Afterwards a sample can be planned
without touching the data, and only the row groups it picks are read.
"""

import io
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from translation_quality_audit.remote import MAX_WORKERS
from translation_quality_audit.source import Opener

LANGUAGE_COLUMN = "og_language"
LEVEL_COLUMN = "edu_score"
INDEX_SCHEMA = pa.schema(
    [
        ("file", pa.string()),
        ("row_group", pa.int32()),
        ("language", pa.string()),
        ("level", pa.int64()),
        ("count", pa.int64()),
    ]
)

GroupKey = tuple[str, int]
Entry = tuple[GroupKey, int, int]


def build_index(
    opener: Opener, paths: Sequence[str], languages: Sequence[str] | None = None
) -> pa.Table:
    """Count every language and level in the row groups of the given files.

    With languages, only the row groups whose footer range can hold one of them are read. The index
    is then complete for those languages, and it has no entries for the others.
    """
    if not paths:
        return INDEX_SCHEMA.empty_table()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        per_file = list(pool.map(lambda path: _file_index(opener, path, languages), paths))
    return pa.concat_tables(per_file)


def index_to_parquet(index: pa.Table) -> bytes:
    """The index as a compressed Parquet file, held in memory."""
    buffer = io.BytesIO()
    pq.write_table(index.cast(INDEX_SCHEMA), buffer, compression="zstd")
    return buffer.getvalue()


def read_index(opener: Opener, path: str) -> pa.Table:
    """Read an index written by index_to_parquet."""
    with opener(path) as handle:
        return pq.read_table(handle)


def language_entries(index: pa.Table, language: str) -> list[Entry]:
    """The (row group, level, count) entries of one language, in index order."""
    entries = []
    columns = zip(
        index["file"].to_pylist(),
        index["row_group"].to_pylist(),
        index["language"].to_pylist(),
        index["level"].to_pylist(),
        index["count"].to_pylist(),
        strict=True,
    )
    for path, group, row_language, level, count in columns:
        if row_language == language:
            entries.append(((path, group), level, count))
    return entries


def _file_index(opener: Opener, path: str, languages: Sequence[str] | None) -> pa.Table:
    pieces = []
    with opener(path) as handle:
        parquet = pq.ParquetFile(handle)
        groups = (
            range(parquet.metadata.num_row_groups)
            if languages is None
            else _candidate_groups(parquet.metadata, languages)
        )
        for group in groups:
            table = parquet.read_row_group(group, columns=[LANGUAGE_COLUMN, LEVEL_COLUMN])
            counted = table.group_by([LANGUAGE_COLUMN, LEVEL_COLUMN]).aggregate(
                [(LANGUAGE_COLUMN, "count")]
            )
            pieces.append(
                pa.table(
                    {
                        "file": [path] * counted.num_rows,
                        "row_group": [group] * counted.num_rows,
                        "language": counted[LANGUAGE_COLUMN],
                        "level": counted[LEVEL_COLUMN],
                        "count": counted[f"{LANGUAGE_COLUMN}_count"],
                    }
                ).cast(INDEX_SCHEMA)
            )
    return pa.concat_tables(pieces) if pieces else INDEX_SCHEMA.empty_table()


def _candidate_groups(metadata: pq.FileMetaData, languages: Sequence[str]) -> list[int]:
    """The row groups whose language range, from the footer, can hold one of the languages."""
    column = _column_index(metadata, LANGUAGE_COLUMN)
    return [
        group
        for group in range(metadata.num_row_groups)
        if any(
            _may_contain(metadata.row_group(group).column(column).statistics, language)
            for language in languages
        )
    ]


def _column_index(metadata: pq.FileMetaData, name: str) -> int:
    return next(
        index for index in range(metadata.num_columns) if metadata.schema.column(index).name == name
    )


def _may_contain(statistics, language: str) -> bool:
    if statistics is None or not statistics.has_min_max:
        return True
    return statistics.min <= language <= statistics.max
