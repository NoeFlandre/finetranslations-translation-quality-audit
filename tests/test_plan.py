from translation_quality_audit.plan import plan_stratified


def entries(levels, groups_per_level, count_per_group):
    return [
        ((f"f{group}.parquet", group), level, count_per_group)
        for level in levels
        for group in range(groups_per_level)
    ]


def picks_per_level(plan):
    counts = {}
    for levels in plan.values():
        for level, ranks in levels.items():
            counts[level] = counts.get(level, 0) + len(ranks)
    return counts


def test_each_level_gets_an_equal_share_of_the_total():
    plan = plan_stratified(entries([2, 3, 4, 5], 5, 10), total=100, seed=0)

    assert picks_per_level(plan) == {2: 25, 3: 25, 4: 25, 5: 25}


def test_a_remainder_goes_to_the_lowest_levels_first():
    plan = plan_stratified(entries([1, 2, 3], 5, 10), total=10, seed=0)

    assert picks_per_level(plan) == {1: 4, 2: 3, 3: 3}


def test_a_level_with_too_few_documents_contributes_all_of_them():
    short = [(("a", 0), 1, 2), (("b", 0), 2, 10)]

    plan = plan_stratified(short, total=10, seed=0)

    assert picks_per_level(plan) == {1: 2, 2: 5}


def test_each_pick_is_a_valid_rank_inside_its_group():
    available = entries([2, 3], 4, 7)

    plan = plan_stratified(available, total=20, seed=1)

    counts = {(group, level): count for group, level, count in available}
    for group, levels in plan.items():
        for level, ranks in levels.items():
            assert len(set(ranks)) == len(ranks)
            assert all(0 <= rank < counts[(group, level)] for rank in ranks)


def test_the_same_seed_gives_the_same_plan():
    assert plan_stratified(entries([2, 3], 20, 50), 20, seed=7) == plan_stratified(
        entries([2, 3], 20, 50), 20, seed=7
    )


def test_a_different_seed_gives_a_different_plan():
    assert plan_stratified(entries([2, 3], 20, 50), 20, seed=7) != plan_stratified(
        entries([2, 3], 20, 50), 20, seed=8
    )


def test_no_entries_or_no_total_gives_an_empty_plan():
    assert plan_stratified([], total=10, seed=0) == {}
    assert plan_stratified(entries([2], 2, 5), total=0, seed=0) == {}


def test_a_single_document_is_planned_when_asked_for_one():
    plan = plan_stratified(entries([2, 3], 5, 5), total=1, seed=0)

    assert picks_per_level(plan) == {2: 1}


def test_entries_with_no_documents_are_ignored():
    plan = plan_stratified([(("a", 0), 2, 0), (("b", 0), 2, 3)], total=2, seed=0)

    assert set(plan) == {("b", 0)}
    assert picks_per_level(plan) == {2: 2}
