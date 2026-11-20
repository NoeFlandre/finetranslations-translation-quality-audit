"""Plan a stratified sample from index entries, without reading any data."""

import random
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence

from translation_quality_audit.index import Entry, GroupKey


def plan_stratified(
    entries: Sequence[Entry], total: int, seed: int
) -> dict[GroupKey, dict[int, list[int]]]:
    """Which documents to read, as ranks within each row group and level.

    Each level contributes an equal share of the total; a remainder goes to the lowest levels.
    A level with too few documents contributes all of them. The same seed gives the same plan.
    """
    members_by_level = _members_by_level(entries)
    levels = sorted(members_by_level)
    if not levels or total <= 0:
        return {}
    rng = random.Random(seed)
    plan: dict[GroupKey, dict[int, list[int]]] = {}
    for level, quota in zip(levels, _quotas(total, len(levels)), strict=True):
        _add_picks(plan, level, members_by_level[level], quota, rng)
    return plan


def _members_by_level(entries: Sequence[Entry]) -> dict[int, list[tuple[GroupKey, int]]]:
    members: dict[int, list[tuple[GroupKey, int]]] = defaultdict(list)
    for group, level, count in entries:
        if count > 0:
            members[level].append((group, count))
    return members


def _quotas(total: int, levels: int) -> list[int]:
    base, remainder = divmod(total, levels)
    return [base + (1 if position < remainder else 0) for position in range(levels)]


def _add_picks(
    plan: dict[GroupKey, dict[int, list[int]]],
    level: int,
    members: list[tuple[GroupKey, int]],
    quota: int,
    rng: random.Random,
) -> None:
    ends = _running_totals([count for _, count in members])
    available = ends[-1]
    for rank in sorted(rng.sample(range(available), min(quota, available))):
        member = bisect_right(ends, rank)
        group, count = members[member]
        within = rank - (ends[member] - count)
        plan.setdefault(group, {}).setdefault(level, []).append(within)


def _running_totals(counts: list[int]) -> list[int]:
    totals = []
    running = 0
    for count in counts:
        running += count
        totals.append(running)
    return totals
