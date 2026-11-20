"""Small Parquet files and models that the tests share."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def make_rows(language, count, start):
    return [
        {
            "id": f"{language}-{start + index}",
            "og_language": language,
            "edu_score": 2 + index % 3,
            "og_full_text": f"source {language} {index}",
            "translated_text": f"translation {language} {index}",
        }
        for index in range(count)
    ]


def write_parquet(path, rows, row_group_size=3, statistics=True):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=row_group_size, write_statistics=statistics)
    return str(path)


def open_local(path):
    return Path(path).open("rb")


class FixedModel:
    """Stands in for the language-identification model: always English with probability 0.93."""

    def predict(self, _text, _k):
        return ["__label__en"], [0.93]
