import pyarrow as pa
from translation_quality_audit.source import (
    language_row_groups,
    read_row_groups,
    select_language_rows,
)

from tests.helpers import make_rows, open_local, write_parquet

COLUMNS = ["id", "og_language", "edu_score", "og_full_text", "translated_text"]


def test_language_row_groups_are_found_from_the_footer_statistics(parquet_files):
    refs = language_row_groups(open_local, parquet_files, "deu_Latn")

    assert sorted(refs) == [(parquet_files[0], 1), (parquet_files[0], 2)]


def test_a_language_contained_in_one_row_group_is_found_in_that_group_only(parquet_files):
    refs = language_row_groups(open_local, parquet_files, "rus_Cyrl")

    assert refs == [(parquet_files[1], 2)]


def test_a_language_absent_from_every_file_has_no_row_groups(parquet_files):
    assert language_row_groups(open_local, parquet_files, "zho_Hans") == []


def test_a_row_group_without_statistics_is_kept_because_it_cannot_be_ruled_out(tmp_path):
    path = write_parquet(tmp_path / "bare.parquet", make_rows("fra_Latn", 3, 0), statistics=False)

    assert language_row_groups(open_local, [path], "fra_Latn") == [(path, 0)]


def test_read_row_groups_returns_one_table_per_group_with_only_the_requested_columns(
    parquet_files,
):
    tables = read_row_groups(
        open_local, [(parquet_files[1], 0), (parquet_files[1], 2)], ["id", "edu_score"]
    )

    assert [table.column_names for table in tables] == [["id", "edu_score"]] * 2
    assert [table.num_rows for table in tables] == [3, 2]


def test_read_row_groups_keeps_the_order_of_the_references(parquet_files):
    tables = read_row_groups(open_local, [(parquet_files[1], 2), (parquet_files[0], 0)], ["id"])

    assert tables[0].column("id").to_pylist() == ["rus_Cyrl-300", "rus_Cyrl-301"]
    assert tables[1].column("id").to_pylist()[0] == "abk_Cyrl-0"


def test_reading_no_row_groups_returns_no_tables():
    assert read_row_groups(open_local, [], ["id"]) == []


def test_select_language_rows_keeps_only_the_language(parquet_files):
    [table] = read_row_groups(open_local, [(parquet_files[0], 1)], COLUMNS)

    selected = select_language_rows(table, "deu_Latn")

    assert set(selected.column("og_language").to_pylist()) == {"deu_Latn"}
    assert selected.num_rows == 2
    assert isinstance(selected, pa.Table)


def test_row_groups_from_many_files_are_read_in_parallel_and_returned_in_order(tmp_path):
    paths = [
        write_parquet(tmp_path / f"{index}.parquet", make_rows("deu_Latn", 3, index * 10))
        for index in range(12)
    ]
    refs = [(path, 0) for path in reversed(paths)]

    tables = read_row_groups(open_local, refs, ["id"])

    assert [table.column("id").to_pylist()[0] for table in tables] == [
        f"deu_Latn-{index * 10}" for index in reversed(range(12))
    ]
