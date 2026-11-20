import pyarrow as pa
import pytest
from translation_quality_audit import pipeline
from translation_quality_audit.features import length_ratio
from translation_quality_audit.index import INDEX_SCHEMA, build_index
from translation_quality_audit.pipeline import audit_language
from translation_quality_audit.score import FEATURES

from tests.helpers import FixedModel, open_local


@pytest.fixture
def index(parquet_files):
    return build_index(open_local, parquet_files)


def run(index, language="deu_Latn", total=3, seed=0):
    return audit_language(open_local, language, total, seed, FixedModel(), index)


def test_audit_language_returns_one_scored_row_per_sampled_document(index):
    rows = run(index, total=3)

    assert len(rows) == 3
    assert len({row["id"] for row in rows}) == 3
    for row in rows:
        assert row["id"].startswith("deu_Latn-")
        assert row["language"] == "deu_Latn"
        assert set(FEATURES) <= set(row)


def test_each_row_keeps_its_own_source_and_translation(index):
    for row in run(index, total=4):
        number = int(row["id"].rsplit("-", 1)[1]) - 100
        expected = length_ratio(f"source deu_Latn {number}", f"translation deu_Latn {number}")
        assert row["length_ratio"] == pytest.approx(expected)


def test_a_request_larger_than_the_language_returns_every_document(index):
    rows = run(index, total=100)

    assert sorted(row["id"] for row in rows) == [f"deu_Latn-{100 + number}" for number in range(4)]


def test_a_language_absent_from_the_index_returns_no_rows(index):
    assert run(index, language="zho_Hans") == []


def test_each_row_records_its_quality_level(index):
    levels = {row["id"]: row["edu_score"] for row in run(index, total=4)}

    assert levels["deu_Latn-100"] == 2
    assert levels["deu_Latn-101"] == 3


def test_only_the_row_groups_holding_the_sample_are_read(monkeypatch, index):
    read = []
    original = pipeline.read_row_groups

    def recording(opener, refs, columns):
        read.append(list(refs))
        return original(opener, refs, columns)

    monkeypatch.setattr(pipeline, "read_row_groups", recording)

    run(index, total=1)

    assert len(read) == 1
    assert len(read[0]) == 1


def test_the_seed_reaches_the_planner(monkeypatch, index):
    seen = []
    original = pipeline.plan_stratified

    def recording(entries, total, seed):
        seen.append(seed)
        return original(entries, total, seed)

    monkeypatch.setattr(pipeline, "plan_stratified", recording)

    run(index, total=2, seed=7)

    assert seen == [7]


def test_the_five_columns_each_group_needs_are_read_together(monkeypatch, index):
    requested = []
    original = pipeline.read_row_groups

    def recording(opener, refs, columns):
        requested.append(list(columns))
        return original(opener, refs, columns)

    monkeypatch.setattr(pipeline, "read_row_groups", recording)

    run(index, total=2)

    assert requested == [["id", "og_language", "edu_score", "og_full_text", "translated_text"]]


def test_an_index_that_disagrees_with_the_data_is_rejected(index):
    rows = index.to_pylist()
    for row in rows:
        if row["language"] == "deu_Latn" and row["level"] == 2 and row["row_group"] == 1:
            row["count"] = 5
    stale = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)

    with pytest.raises(ValueError, match="out of date"):
        run(stale, total=100)
