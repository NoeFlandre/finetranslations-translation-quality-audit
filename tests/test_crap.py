import json
import re

import pytest
from scripts.crap import crap_scores, main

SOURCE = (
    "def simple():\n    return 1\n\n\ndef branchy(x):\n    if x:\n        return 1\n    return 2\n"
)
PACKAGE = ("translation_quality_audit",)


@pytest.fixture
def package_file(tmp_path):
    return write_module(tmp_path, "module.py", SOURCE)


def write_module(tmp_path, name, source):
    path = tmp_path / "translation_quality_audit" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    return path


def coverage_report(files):
    return {"files": {str(path): lines for path, lines in files.items()}}


def write_report(tmp_path, files):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage_report(files)))
    return path


def names_and_scores(report, packages=PACKAGE):
    return {name.rsplit(":", 1)[1]: score for score, name in crap_scores(report, packages)}


def test_the_score_weighs_complexity_by_the_uncovered_share(package_file):
    report = coverage_report(
        {package_file: {"executed_lines": [1, 2, 5, 6, 7], "missing_lines": [8]}}
    )

    scores = names_and_scores(report)

    assert scores["simple"] == pytest.approx(1.0)
    assert scores["branchy"] == pytest.approx(2.0625)


def test_complexity_is_squared_in_the_score(tmp_path):
    source = (
        "def sorter(x):\n    if x > 1:\n        return 1\n"
        "    elif x > 0:\n        return 2\n    return 3\n"
    )
    path = write_module(tmp_path, "sorter.py", source)
    report = coverage_report({path: {"executed_lines": [], "missing_lines": [1, 2, 3, 4, 5, 6]}})

    assert names_and_scores(report)["sorter"] == pytest.approx(12.0)


def test_scores_are_listed_highest_first(tmp_path):
    path = write_module(
        tmp_path,
        "ranked.py",
        "def calm():\n    return 1\n\n\n"
        "def tangled(x):\n    if x:\n        return 1\n    return 2\n",
    )
    report = coverage_report({path: {"executed_lines": [1, 2], "missing_lines": [5, 6, 7, 8]}})

    names = [name.rsplit(":", 1)[1] for _, name in crap_scores(report, PACKAGE)]

    assert names == ["tangled", "calm"]


def test_a_function_is_scored_only_on_its_own_lines(tmp_path):
    path = write_module(tmp_path, "bounded.py", "def simple():\n    return 1\nexpression = 2\n")
    report = coverage_report({path: {"executed_lines": [1, 2], "missing_lines": [3]}})

    assert names_and_scores(report)["simple"] == pytest.approx(1.0)


def test_a_function_without_measured_lines_counts_as_covered(tmp_path):
    path = write_module(tmp_path, "unmeasured.py", "def simple():\n    return 1\n")
    report = coverage_report({path: {"executed_lines": [], "missing_lines": []}})

    assert names_and_scores(report)["simple"] == pytest.approx(1.0)


def test_classes_are_skipped_but_their_methods_are_scored(tmp_path):
    path = write_module(
        tmp_path, "classy.py", "class Box:\n    def open(self):\n        return 1\n"
    )
    report = coverage_report({path: {"executed_lines": [1, 2, 3], "missing_lines": []}})

    assert list(names_and_scores(report)) == ["open"]


def test_files_outside_the_packages_are_ignored(tmp_path):
    other = tmp_path / "elsewhere" / "module.py"
    other.parent.mkdir()
    other.write_text(SOURCE)
    report = coverage_report({other: {"executed_lines": [], "missing_lines": [1, 2, 5, 6, 7, 8]}})

    assert crap_scores(report, PACKAGE) == []


def test_main_fails_when_a_function_reaches_the_limit(package_file, tmp_path, capsys):
    report_path = write_report(
        tmp_path, {package_file: {"executed_lines": [1, 2], "missing_lines": [5, 6, 7, 8]}}
    )

    status = main([str(report_path), "--packages", "translation_quality_audit", "--limit", "6"])

    assert status == 1
    assert "at or over 6.0: 1" in capsys.readouterr().out


def test_main_passes_when_every_function_stays_below_the_limit(package_file, tmp_path):
    report_path = write_report(
        tmp_path,
        {package_file: {"executed_lines": [1, 2, 5, 6, 7, 8], "missing_lines": []}},
    )

    status = main([str(report_path), "--packages", "translation_quality_audit", "--limit", "6"])

    assert status == 0


def test_main_uses_six_as_the_default_limit(package_file, tmp_path):
    report_path = write_report(
        tmp_path, {package_file: {"executed_lines": [1, 2], "missing_lines": [5, 6, 7, 8]}}
    )

    assert main([str(report_path), "--packages", "translation_quality_audit"]) == 1


def test_main_scans_both_packages_by_default(package_file, tmp_path):
    report_path = write_report(
        tmp_path, {package_file: {"executed_lines": [1, 2], "missing_lines": [5, 6, 7, 8]}}
    )

    assert main([str(report_path)]) == 1


def test_main_lists_the_five_worst_functions_with_their_scores(tmp_path, capsys):
    source = "".join(f"def f{index}():\n    return {index}\n\n\n" for index in range(6))
    path = write_module(tmp_path, "many.py", source)
    executed = [line for index in range(6) for line in (4 * index + 1, 4 * index + 2)]
    report_path = write_report(tmp_path, {path: {"executed_lines": executed, "missing_lines": []}})

    main([str(report_path), "--packages", "translation_quality_audit"])

    output = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("CRAP ") for line in output) == 5


def test_help_describes_the_tool_and_its_arguments(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])

    output = capsys.readouterr().out
    assert "CRAP scores of the functions" in output
    assert re.search(r"^\s+coverage\s+coverage JSON report$", output, re.MULTILINE)


def test_main_reports_zero_when_no_function_is_measured(tmp_path, capsys):
    report_path = write_report(tmp_path, {})

    status = main([str(report_path)])

    assert status == 0
    assert "max CRAP: 0.00" in capsys.readouterr().out
