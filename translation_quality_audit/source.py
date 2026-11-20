"""Reading row groups of the Hugging Face Parquet files, one file handle per file."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from translation_quality_audit.remote import MAX_WORKERS

LANGUAGE_COLUMN = "og_language"

Opener = Callable[[str], AbstractContextManager[Any]]
RowGroupRef = tuple[str, int]


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
