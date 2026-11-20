---
pretty_name: FineTranslations translation quality audit
license: odc-by
language:
- en
- fr
- de
size_categories:
- n<1K
tags:
- translation
- machine-translation
- data-quality
- fineweb-edu
---

# FineTranslations translation quality audit

A reference-free audit of the English translations in
[FineTranslations-Edu](https://huggingface.co/datasets/HuggingFaceFW/finetranslations-edu). Each row
compares a source document with its English translation, using six cheap features. None of them
needs a reference translation or a GPU.

**Status:** version 1 covers French and German so far. Russian and Chinese are not audited yet.

## Results (v1)

100 documents per language, split equally across the educational-score levels 2 to 5 (25 each),
drawn with a fixed seed. Each cell is the median, with the interquartile range in brackets.

| Feature | French (`fra_Latn`) | German (`deu_Latn`) |
|---|---|---|
| `length_ratio` | 0.884 [0.852, 0.917] | 0.912 [0.880, 0.943] |
| `english_probability` | 0.961 [0.945, 0.972] | 0.959 [0.948, 0.973] |
| `copy_rate` | 0.090 [0.074, 0.107] | 0.040 [0.026, 0.068] |
| `repetition_share` | 0.004 [0.003, 0.006] | 0.004 [0.003, 0.006] |
| `number_mismatch` | 0 [0, 0]; 20 of 100 documents non-zero | 0 [0, 0]; 10 of 100 non-zero |
| `structure_ratio` | 1.000 [1.000, 1.000]; 1 undefined | 1.000 [1.000, 1.000]; 1 undefined |

- `english_probability` is at least 0.84 for every document in both languages. None falls below 0.5.
- `copy_rate` is above 0.2 for 3 German documents and for no French document.
- `number_mismatch` reaches 9 in French and 6 in German.

Medians by educational-score level, both languages together (50 documents per level):

| Feature | Level 2 | Level 3 | Level 4 | Level 5 |
|---|---|---|---|---|
| `length_ratio` | 0.922 | 0.912 | 0.885 | 0.877 |
| `english_probability` | 0.964 | 0.968 | 0.957 | 0.952 |
| `copy_rate` | 0.092 | 0.077 | 0.051 | 0.059 |
| `repetition_share` | 0.004 | 0.004 | 0.004 | 0.004 |

`number_mismatch` has median 0, and `structure_ratio` has median 1, at every level. These are
descriptive medians, not tested differences.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Id of the source document |
| `language` | string | Source language code: `fra_Latn` or `deu_Latn` |
| `edu_score` | int | Educational-score level, 2 to 5, that the sample is stratified on |
| `length_ratio` | float | Translation characters over source characters |
| `english_probability` | float | Probability that the translation is English (fastText `lid.176`) |
| `copy_rate` | float | Share of the source's character 6-grams that appear in the translation |
| `number_mismatch` | int | Distinct numbers present on one side only |
| `structure_ratio` | float | Translation newlines and table pipes over the source's |
| `repetition_share` | float | Share of the most frequent word 4-gram in the translation |

Each feature is raw. It is undefined (NaN) when its denominator is empty, for example
`structure_ratio` when the source has no newlines or pipes. v1 publishes no flags or thresholds.

## How to read the numbers

- These are raw signals, not quality judgements. A low copy rate or a length ratio near 1 does not
  prove a good translation, and the reverse does not prove a bad one.
- Some differences between languages are expected. English is usually shorter than French or
  German in characters, so length ratios below 1 are the norm.
- `copy_rate` counts shared character 6-grams, so names, numbers, code and cognates count too.
- `english_probability` comes from a language identifier, which is less reliable on short or
  mixed-language text.
- With 100 documents per language, differences of a few points are within sampling noise. No
  significance tests have been run.
- No human reference translations were used.

## Files

- `v1/audit.parquet`: the 200 audited documents, one row each, with the columns above.
- `index/language_groups.parquet`: how many documents of each audited language and level sit in
  each row group of the source, 81,301 rows. The audit uses it to plan its samples. It covers only
  the four audited languages (`fra_Latn`, `deu_Latn`, `rus_Cyrl`, `cmn_Hani`).

```python
from datasets import load_dataset

audit = load_dataset(
    "NoeFlandre/finetranslations-translation-quality-audit",
    data_files="v1/audit.parquet",
    split="train",
)
```

## Source and license

The documents come from FineTranslations-Edu, which is released under ODC-BY. This dataset keeps
only ids, languages, levels and feature values, and no document text. Please cite
FineTranslations-Edu when you use these results.

## Reproduce

```bash
uv sync
uv run python -m scripts.build_index   # once: builds and uploads the language index
uv run python -m scripts.run_audit --languages fra_Latn deu_Latn --total 100
```

Code: https://github.com/NoeFlandre/finetranslations-translation-quality-audit
