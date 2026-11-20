# finetranslations-translation-quality-audit

Auditing the translation quality of [FineTranslations-Edu](https://huggingface.co/datasets/HuggingFaceFW/finetranslations-edu),
English translations of web documents from 500+ languages. Version 1 is a set of cheap,
reference-free features computed on a stratified sample per language. See the docs for the design.

```bash
uv sync
uv run python -m scripts.run_audit --languages fra_Latn deu_Latn --total 100
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run mkdocs build --strict
```
