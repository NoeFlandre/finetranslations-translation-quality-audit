# finetranslations-edu-quality-audit

Auditing the translation quality of FineTranslations documents. The project is at an early stage;
the scope is still being defined.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run mkdocs build --strict
```
