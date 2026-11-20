import pytest
from translation_quality_audit import pipeline
from translation_quality_audit.features import length_ratio
from translation_quality_audit.pipeline import audit_language
from translation_quality_audit.score import FEATURES

from tests.helpers import FixedModel, open_local


def test_audit_language_returns_one_scored_row_per_sampled_document(parquet_files):
    rows = audit_language(
        open_local, parquet_files, "deu_Latn", total=3, seed=0, identifier=FixedModel()
    )

    assert len(rows) == 3
    assert len({row["id"] for row in rows}) == 3
    for row in rows:
        assert row["id"].startswith("deu_Latn-")
        assert row["language"] == "deu_Latn"
        assert set(FEATURES) <= set(row)


def test_each_row_keeps_its_own_source_and_translation(parquet_files):
    rows = audit_language(
        open_local, parquet_files, "deu_Latn", total=4, seed=0, identifier=FixedModel()
    )

    for row in rows:
        index = int(row["id"].rsplit("-", 1)[1]) - 100
        expected = length_ratio(f"source deu_Latn {index}", f"translation deu_Latn {index}")
        assert row["length_ratio"] == pytest.approx(expected)


def test_a_request_larger_than_the_language_returns_every_document(parquet_files):
    rows = audit_language(
        open_local, parquet_files, "deu_Latn", total=100, seed=0, identifier=FixedModel()
    )

    assert sorted(row["id"] for row in rows) == [f"deu_Latn-{100 + index}" for index in range(4)]


def test_a_language_absent_from_the_files_returns_no_rows(parquet_files):
    assert (
        audit_language(
            open_local, parquet_files, "zho_Hans", total=10, seed=0, identifier=FixedModel()
        )
        == []
    )


def test_each_row_records_its_quality_level(parquet_files):
    rows = audit_language(
        open_local, parquet_files, "deu_Latn", total=4, seed=0, identifier=FixedModel()
    )

    levels = {row["id"]: row["edu_score"] for row in rows}
    assert levels["deu_Latn-100"] == 2
    assert levels["deu_Latn-101"] == 3


def test_the_seed_reaches_the_sampler(monkeypatch, parquet_files):
    seen = []
    original = pipeline.stratified_sample

    def recording(ids_by_level, total, seed):
        seen.append(seed)
        return original(ids_by_level, total, seed)

    monkeypatch.setattr(pipeline, "stratified_sample", recording)

    audit_language(open_local, parquet_files, "deu_Latn", total=2, seed=7, identifier=FixedModel())

    assert seen == [7]


def test_only_the_columns_each_read_needs_are_requested(monkeypatch, parquet_files):
    requested = []
    original = pipeline.read_row_groups

    def recording(opener, refs, columns):
        requested.append(list(columns))
        return original(opener, refs, columns)

    monkeypatch.setattr(pipeline, "read_row_groups", recording)

    audit_language(open_local, parquet_files, "deu_Latn", total=2, seed=0, identifier=FixedModel())

    assert requested == [
        ["id", "og_language", "edu_score"],
        ["id", "og_full_text", "translated_text"],
    ]
