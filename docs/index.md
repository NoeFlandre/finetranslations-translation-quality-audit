# finetranslations-edu-quality-audit

Auditing the translation quality of FineTranslations documents. The scope is still being defined.

## Development

- `uv sync` installs the locked environment.
- `uv run pytest` runs the tests with branch coverage; coverage must stay at 100%.
- `uv run ruff check . && uv run ruff format --check .` lints and checks formatting.
- `uv run ty check` type-checks the code.
- `uv run mkdocs build --strict` builds these pages.
- `uv run mutmut run` runs mutation testing; survivors must be killed or justified.
- `uv run pytest -o addopts="" --cov=edu_quality_audit --cov=scripts --cov-branch --cov-report=json`
  followed by `uv run python -m scripts.crap coverage.json --limit 6` checks the CRAP scores.
