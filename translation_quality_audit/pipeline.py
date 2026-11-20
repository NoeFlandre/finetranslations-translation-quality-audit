"""Audit one language: sample its documents by quality level, then score each pair."""

from collections import defaultdict

import pyarrow as pa

from translation_quality_audit.sampling import stratified_sample
from translation_quality_audit.score import score_pair
from translation_quality_audit.source import (
    Opener,
    language_row_groups,
    read_row_groups,
    select_language_rows,
)

ID = "id"
LEVEL = "edu_score"
SOURCE = "og_full_text"
TRANSLATION = "translated_text"


def audit_language(
    opener: Opener, paths: list[str], language: str, total: int, seed: int, identifier
) -> list[dict]:
    """Score a stratified sample of the language's documents, reading only the bytes it needs."""
    refs = language_row_groups(opener, paths, language)
    levels_by_id, ref_by_id = _index_language(opener, refs, language)
    chosen = stratified_sample(_ids_by_level(levels_by_id), total, seed)
    return [
        _scored_row(row, language, levels_by_id, identifier)
        for row in _texts_of(opener, chosen, ref_by_id)
    ]


def _texts_of(opener, chosen: list[str], ref_by_id: dict) -> list[dict]:
    """The source and translation of the chosen documents, from only the row groups holding them."""
    if not chosen:
        return []
    needed = list(dict.fromkeys(ref_by_id[doc_id] for doc_id in chosen))
    texts = pa.concat_tables(read_row_groups(opener, needed, [ID, SOURCE, TRANSLATION]))
    wanted = set(chosen)
    picked = texts.take([i for i, doc_id in enumerate(texts[ID].to_pylist()) if doc_id in wanted])
    return picked.to_pylist()


def _scored_row(row: dict, language: str, levels_by_id: dict, identifier) -> dict:
    return {
        ID: row[ID],
        "language": language,
        LEVEL: levels_by_id[row[ID]],
        **score_pair(row[SOURCE], row[TRANSLATION], identifier),
    }


def _index_language(opener, refs, language):
    levels_by_id: dict[str, int] = {}
    ref_by_id: dict[str, tuple[str, int]] = {}
    tables = read_row_groups(opener, refs, [ID, "og_language", LEVEL])
    for ref, table in zip(refs, tables, strict=True):
        rows = select_language_rows(table, language)
        for doc_id, level in zip(rows[ID].to_pylist(), rows[LEVEL].to_pylist(), strict=True):
            levels_by_id[doc_id] = level
            ref_by_id[doc_id] = ref
    return levels_by_id, ref_by_id


def _ids_by_level(levels_by_id: dict[str, int]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for doc_id, level in levels_by_id.items():
        grouped[level].append(doc_id)
    return dict(grouped)
