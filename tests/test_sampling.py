from translation_quality_audit.sampling import stratified_sample


def pools(levels, size):
    return {level: [f"{level}-{index}" for index in range(size)] for level in levels}


def test_each_level_contributes_an_equal_share_of_the_total():
    sample = stratified_sample(pools([2, 3, 4, 5], 50), total=100, seed=0)

    assert len(sample) == 100
    for level in [2, 3, 4, 5]:
        assert sum(item.startswith(f"{level}-") for item in sample) == 25


def test_a_remainder_goes_to_the_lowest_levels_first():
    sample = stratified_sample(pools([1, 2, 3], 50), total=10, seed=0)

    counts = [sum(item.startswith(f"{level}-") for item in sample) for level in [1, 2, 3]]
    assert counts == [4, 3, 3]


def test_a_level_with_too_few_documents_contributes_all_of_them():
    sample = stratified_sample({1: ["a", "b"], 2: list("cdefghijkl")}, total=10, seed=0)

    assert sum(item in {"a", "b"} for item in sample) == 2
    assert len(sample) == 7


def test_the_same_seed_gives_the_same_sample():
    first = stratified_sample(pools([2, 3], 200), total=20, seed=7)
    second = stratified_sample(pools([2, 3], 200), total=20, seed=7)

    assert first == second


def test_a_different_seed_gives_a_different_sample():
    first = stratified_sample(pools([2, 3], 200), total=20, seed=7)
    second = stratified_sample(pools([2, 3], 200), total=20, seed=8)

    assert first != second


def test_no_levels_or_no_total_gives_an_empty_sample():
    assert stratified_sample({}, total=10, seed=0) == []
    assert stratified_sample(pools([2], 5), total=0, seed=0) == []


def test_a_single_document_is_sampled_when_asked_for_one():
    sample = stratified_sample(pools([2, 3], 5), total=1, seed=0)

    assert len(sample) == 1
