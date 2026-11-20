"""Stratified sampling: an equal share of documents from each integer quality level."""

import random


def stratified_sample(ids_by_level: dict[int, list[str]], total: int, seed: int) -> list[str]:
    """Up to total ids, split as evenly as possible across the levels, reproducible for a seed.

    A remainder goes to the lowest levels first. A level with too few ids contributes all of them.
    """
    levels = sorted(ids_by_level)
    if not levels or total <= 0:
        return []
    base, remainder = divmod(total, len(levels))
    rng = random.Random(seed)
    sample: list[str] = []
    for index, level in enumerate(levels):
        quota = base + (1 if index < remainder else 0)
        pool = ids_by_level[level]
        sample.extend(rng.sample(pool, min(quota, len(pool))))
    return sample
