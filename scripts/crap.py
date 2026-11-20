"""CRAP scores of the functions in a coverage report.

A function's CRAP score is its complexity squared times the share of its statements that are not
covered cubed, plus its complexity. Example: uv run python -m scripts.crap coverage.json --limit 6
"""

import argparse
import json
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class

DEFAULT_PACKAGES = ("translation_quality_audit", "scripts")


def crap_scores(coverage: dict, packages: tuple[str, ...]) -> list[tuple[float, str]]:
    """The CRAP score of each function in the given packages, highest first."""
    scores = []
    for file_name, data in coverage["files"].items():
        path = Path(file_name)
        if set(packages) & set(path.parts):
            scores.extend(_file_scores(path, data))
    return sorted(scores, reverse=True)


def _file_scores(path: Path, data: dict) -> list[tuple[float, str]]:
    executed = set(data["executed_lines"])
    statements = executed | set(data["missing_lines"])
    return [
        (_crap(block, executed, statements), f"{path}:{block.name}")
        for block in cc_visit(path.read_text())
        if not isinstance(block, Class)
    ]


def _crap(block, executed: set[int], statements: set[int]) -> float:
    lines = set(range(block.lineno, block.endline + 1)) & statements
    covered = len(lines & executed) / len(lines) if lines else 1.0
    return block.complexity**2 * (1 - covered) ** 3 + block.complexity


def main(argv: list[str] | None = None) -> int:
    """Print the worst functions and return 1 when any reaches the limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", type=Path, help="coverage JSON report")
    parser.add_argument("--packages", nargs="+", default=list(DEFAULT_PACKAGES))
    parser.add_argument("--limit", type=float, default=6.0)
    args = parser.parse_args(argv)
    scores = crap_scores(json.loads(args.coverage.read_text()), tuple(args.packages))
    for score, name in scores[:5]:
        print(f"CRAP {score:6.2f} {name}")
    at_limit = sum(score >= args.limit for score, _ in scores)
    highest = max((score for score, _ in scores), default=0.0)
    print(
        f"functions: {len(scores)} | max CRAP: {highest:.2f} | at or over {args.limit}: {at_limit}"
    )
    return 1 if at_limit else 0


if __name__ == "__main__":
    raise SystemExit(main())
