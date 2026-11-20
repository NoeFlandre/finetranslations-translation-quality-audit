# finetranslations-translation-quality-audit

Auditing the translation quality of [FineTranslations-Edu](https://huggingface.co/datasets/HuggingFaceFW/finetranslations-edu):
English translations of web documents from 500+ languages.

## Version 1

Cheap, reference-free features that compare each source document with its English translation.
They run on CPU, read only the bytes they need from the Hugging Face Parquet files, and keep nothing
on the local disk except a 940 KB language-identification model.

| Feature | What it measures |
|---|---|
| `length_ratio` | Translation characters over source characters |
| `english_probability` | Probability that the translation is English (fastText `lid.176`) |
| `copy_rate` | Share of the source's character 6-grams found in the translation |
| `number_mismatch` | Distinct numbers present on one side only |
| `structure_ratio` | Translation newlines and table pipes over the source's |
| `repetition_share` | Share of the most frequent word 4-gram in the translation |

Each feature is raw and undefined (NaN) when its denominator is empty. v1 publishes no flags.

The sample per language is 100 documents, split equally across the integer `edu_score` levels,
drawn with a fixed seed.

## The language index

Reading a sample by scanning every file would take too many requests, because the Hub rate-limits
them. A small index says how many documents of each audited language and level sit in each row
group. It covers the four audited languages only (`fra_Latn`, `deu_Latn`, `rus_Cyrl`, `cmn_Hani`),
and it reads only the row groups whose footer language range can hold one of them. For those
languages it is complete; for other languages it has no entries, so the audit refuses them.

## Running it

```bash
uv sync
uv run python -m scripts.build_index   # once: uploads the language index
uv run python -m scripts.run_audit --languages fra_Latn deu_Latn --total 100
```

The results are uploaded as one Parquet file to the dataset repo, by default
`NoeFlandre/finetranslations-translation-quality-audit` at `v1/audit.parquet`.

## Development

- `uv run pytest` runs the tests with branch coverage; coverage must stay at 100%.
- `uv run ruff check . && uv run ruff format --check .` lints and checks formatting.
- `uv run ty check` type-checks the code.
- `uv run mutmut run` runs mutation testing; survivors must be killed or justified.
- `uv run pytest -o addopts="" --cov=translation_quality_audit --cov=scripts --cov-branch --cov-report=json`
  followed by `uv run python -m scripts.crap coverage.json --limit 6` checks the CRAP scores.
- `uv run mkdocs build --strict` builds these pages.
