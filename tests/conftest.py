import pytest

from tests.helpers import make_rows, write_parquet


@pytest.fixture
def parquet_files(tmp_path):
    first = write_parquet(
        tmp_path / "000.parquet",
        make_rows("abk_Cyrl", 4, 0) + make_rows("deu_Latn", 4, 100),
    )
    second = write_parquet(
        tmp_path / "001.parquet",
        make_rows("fra_Latn", 6, 200) + make_rows("rus_Cyrl", 2, 300),
    )
    return [first, second]
