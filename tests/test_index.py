import pyarrow as pa
from translation_quality_audit.index import (
    build_index,
    index_to_parquet,
    language_entries,
    read_index,
)

from tests.helpers import make_rows, open_local, write_parquet


def as_tuples(table):
    return sorted(
        (row["file"], row["row_group"], row["language"], row["level"], row["count"])
        for row in table.to_pylist()
    )


def test_the_index_counts_each_language_and_level_in_each_row_group(parquet_files):
    index = build_index(open_local, parquet_files)

    first, second = parquet_files
    assert as_tuples(index) == sorted(
        [
            (first, 0, "abk_Cyrl", 2, 1),
            (first, 0, "abk_Cyrl", 3, 1),
            (first, 0, "abk_Cyrl", 4, 1),
            (first, 1, "abk_Cyrl", 2, 1),
            (first, 1, "deu_Latn", 2, 1),
            (first, 1, "deu_Latn", 3, 1),
            (first, 2, "deu_Latn", 2, 1),
            (first, 2, "deu_Latn", 4, 1),
            (second, 0, "fra_Latn", 2, 1),
            (second, 0, "fra_Latn", 3, 1),
            (second, 0, "fra_Latn", 4, 1),
            (second, 1, "fra_Latn", 2, 1),
            (second, 1, "fra_Latn", 3, 1),
            (second, 1, "fra_Latn", 4, 1),
            (second, 2, "rus_Cyrl", 2, 1),
            (second, 2, "rus_Cyrl", 3, 1),
        ]
    )


def test_language_entries_list_the_groups_holding_a_language(parquet_files):
    index = build_index(open_local, parquet_files)

    entries = language_entries(index, "deu_Latn")

    first = parquet_files[0]
    assert sorted(entries) == sorted(
        [((first, 1), 2, 1), ((first, 1), 3, 1), ((first, 2), 2, 1), ((first, 2), 4, 1)]
    )


def test_a_language_absent_from_the_index_has_no_entries(parquet_files):
    assert language_entries(build_index(open_local, parquet_files), "zho_Hans") == []


def test_an_index_round_trips_through_parquet(tmp_path, parquet_files):
    index = build_index(open_local, parquet_files)
    path = tmp_path / "groups.parquet"
    path.write_bytes(index_to_parquet(index))

    assert as_tuples(read_index(open_local, str(path))) == as_tuples(index)
    assert read_index(open_local, str(path)).schema.field("count").type == pa.int64()


def test_building_an_index_from_no_files_gives_an_empty_index():
    assert build_index(open_local, []).num_rows == 0


def test_with_languages_the_index_is_complete_for_them_and_skips_the_other_row_groups(
    parquet_files,
):
    everything = build_index(open_local, parquet_files)
    restricted = build_index(open_local, parquet_files, ["deu_Latn"])

    first = parquet_files[0]
    assert language_entries(restricted, "deu_Latn") == language_entries(everything, "deu_Latn")
    groups = {(row["file"], row["row_group"]) for row in restricted.to_pylist()}
    assert (first, 0) not in groups


def test_a_file_whose_footer_ranges_cannot_hold_the_languages_contributes_nothing(parquet_files):
    assert build_index(open_local, parquet_files, ["zho_Hans"]).num_rows == 0


def test_without_statistics_every_row_group_is_read(tmp_path):
    path = write_parquet(
        tmp_path / "no_statistics.parquet", make_rows("deu_Latn", 6, 0), statistics=False
    )

    restricted = build_index(open_local, [path], ["zho_Hans"])

    assert restricted.num_rows == build_index(open_local, [path]).num_rows
    assert restricted.num_rows > 0
