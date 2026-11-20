"""Reading the Hugging Face Parquet files row group by row group, touching only what is needed.

Each file keeps its language column sorted in row groups, so the footer statistics say which row
groups can contain a language. Only those row groups are then read, and only for the columns asked.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

LANGUAGE_COLUMN = "og_language"
MAX_WORKERS = 32

Opener = Callable[[str], BinaryIO]
RowGroupRef = tuple[str, int]


def language_row_groups(opener: Opener, paths: Sequence[str], language: str) -> list[RowGroupRef]:
    """Row groups whose language range covers the given language, read from the footers."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        per_file = pool.map(lambda path: _matching_row_groups(opener, path, language), paths)
    return [ref for refs in per_file for ref in refs]


def read_row_groups(
    opener: Opener, refs: Sequence[RowGroupRef], columns: list[str]
) -> list[pa.Table]:
    """One table per row group, with the requested columns, in the order given.

    Files are read in parallel, one thread per file, so network latency overlaps.
    """
    positions_by_path: dict[str, list[tuple[int, int]]] = {}
    for position, (path, index) in enumerate(refs):
        positions_by_path.setdefault(path, []).append((position, index))
    tables: dict[int, pa.Table] = {}

    def read_file(path: str) -> None:
        with opener(path) as handle:
            parquet = pq.ParquetFile(handle)
            for position, index in positions_by_path[path]:
                tables[position] = parquet.read_row_group(index, columns=columns)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(read_file, positions_by_path))
    return [tables[position] for position in range(len(refs))]


def select_language_rows(table: pa.Table, language: str) -> pa.Table:
    """Only the rows whose language is exactly the given one."""
    mask = pa.array([value == language for value in table[LANGUAGE_COLUMN].to_pylist()], pa.bool_())
    return table.filter(mask)


def _matching_row_groups(opener: Opener, path: str, language: str) -> list[RowGroupRef]:
    with opener(path) as handle:
        metadata = pq.ParquetFile(handle).metadata
    column = _column_index(metadata, LANGUAGE_COLUMN)
    return [
        (path, index)
        for index in range(metadata.num_row_groups)
        if _may_contain(metadata.row_group(index).column(column).statistics, language)
    ]


def _column_index(metadata: pq.FileMetaData, name: str) -> int:
    return next(
        index for index in range(metadata.num_columns) if metadata.schema.column(index).name == name
    )


def _may_contain(statistics, language: str) -> bool:
    if statistics is None or not statistics.has_min_max:
        return True
    return statistics.min <= language <= statistics.max
