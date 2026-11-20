"""Audit one language: plan a sample from the index, then read and score only its groups."""

import pyarrow as pa

from translation_quality_audit.index import language_entries as entries_of
from translation_quality_audit.plan import plan_stratified
from translation_quality_audit.score import score_pair
from translation_quality_audit.source import Opener, read_row_groups, select_language_rows

ID = "id"
LEVEL = "edu_score"
SOURCE = "og_full_text"
TRANSLATION = "translated_text"
COLUMNS = [ID, "og_language", LEVEL, SOURCE, TRANSLATION]
STALE_INDEX = "the index is out of date for this row group"


def audit_language(
    opener: Opener, language: str, total: int, seed: int, identifier, index: pa.Table
) -> list[dict]:
    """Score a stratified sample of the language's documents, reading only the groups it needs."""
    plan = plan_stratified(entries_of(index, language), total, seed)
    groups = list(plan)
    tables = read_row_groups(opener, groups, COLUMNS)
    rows: list[dict] = []
    for group, table in zip(groups, tables, strict=True):
        rows.extend(_scored_picks(table, language, plan[group], identifier))
    return rows


def _scored_picks(
    table: pa.Table, language: str, picks: dict[int, list[int]], identifier
) -> list[dict]:
    """The planned documents of one row group, scored."""
    rows = select_language_rows(table, language)
    positions = _planned_positions(rows[LEVEL].to_pylist(), picks)
    return [_scored_row(row, language, identifier) for row in rows.take(positions).to_pylist()]


def _planned_positions(levels: list[int], picks: dict[int, list[int]]) -> list[int]:
    """The row positions of the planned ranks, checked against the rows that exist."""
    positions = []
    for level, ranks in picks.items():
        at_level = _positions_at(levels, level)
        _check_available(ranks, at_level)
        positions.extend(at_level[rank] for rank in ranks)
    return positions


def _positions_at(levels: list[int], level: int) -> list[int]:
    return [position for position, value in enumerate(levels) if value == level]


def _check_available(ranks: list[int], at_level: list[int]) -> None:
    if ranks and ranks[-1] >= len(at_level):
        raise ValueError(STALE_INDEX)


def _scored_row(row: dict, language: str, identifier) -> dict:
    return {
        ID: row[ID],
        "language": language,
        LEVEL: row[LEVEL],
        **score_pair(row[SOURCE], row[TRANSLATION], identifier),
    }
