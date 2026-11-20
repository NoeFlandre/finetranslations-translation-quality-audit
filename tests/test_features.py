import math

import pytest
from translation_quality_audit.features import (
    copy_rate,
    length_ratio,
    number_mismatch,
    repetition_share,
    structure_ratio,
)


def test_length_ratio_is_the_translation_length_over_the_source_length():
    assert length_ratio("abcd", "ab") == pytest.approx(0.5)
    assert length_ratio("ab", "abcd") == pytest.approx(2.0)


def test_length_ratio_is_undefined_for_an_empty_source():
    assert math.isnan(length_ratio("", "text"))


def test_copy_rate_is_the_share_of_source_character_ngrams_found_in_the_translation():
    assert copy_rate("abcdefg", "xxabcdefyy", n=6) == pytest.approx(0.5)


def test_a_verbatim_copy_has_a_copy_rate_of_one():
    assert copy_rate("hello world", "hello world") == pytest.approx(1.0)


def test_copy_rate_is_undefined_for_a_source_shorter_than_n():
    assert math.isnan(copy_rate("abc", "abc", n=6))


def test_number_mismatch_counts_numbers_present_on_one_side_only():
    assert number_mismatch("in 1990 and 2001", "in 1990") == 1
    assert number_mismatch("in 1990", "in 1990") == 0


def test_number_mismatch_counts_numbers_missing_in_either_direction():
    assert number_mismatch("1 and 2", "2 and 3") == 2


def test_structure_ratio_compares_newlines_and_table_pipes_between_the_sides():
    assert structure_ratio("a\nb\nc", "a\nb") == pytest.approx(0.5)
    assert structure_ratio("a | b", "a | b | c") == pytest.approx(2.0)
    assert structure_ratio("x\n|y", "x y") == pytest.approx(0.0)


def test_structure_ratio_is_undefined_when_the_source_has_no_structure():
    assert math.isnan(structure_ratio("plain", "plain"))


def test_repetition_share_is_the_share_of_the_most_frequent_word_4gram():
    translation = "a b c d a b c d a b c d"

    assert repetition_share(translation) == pytest.approx(3 / 9)


def test_repetition_share_is_half_when_every_4gram_is_distinct_and_there_are_two():
    assert repetition_share("one two three four five") == pytest.approx(0.5)


def test_repetition_share_is_undefined_for_fewer_than_four_words():
    assert math.isnan(repetition_share("only three words"))


def test_repetition_is_defined_for_exactly_four_words():
    assert repetition_share("one two three four") == pytest.approx(1.0)


def test_copy_rate_uses_six_character_ngrams_by_default():
    assert copy_rate("abcdef", "xxabcdefxx") == pytest.approx(1.0)
