import io
import math

import pyarrow.parquet as pq
from translation_quality_audit.publish import RESULT_COLUMNS, results_to_parquet, upload_results

ROW = {
    "id": "fra_Latn-1",
    "language": "fra_Latn",
    "edu_score": 3,
    "length_ratio": 1.1,
    "english_probability": 0.9,
    "copy_rate": 0.25,
    "number_mismatch": 0,
    "structure_ratio": math.nan,
    "repetition_share": 0.5,
}


class RecordingApi:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.uploads.append((path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message))


def test_results_round_trip_through_parquet_in_memory():
    table = pq.read_table(io.BytesIO(results_to_parquet([ROW])))

    assert table.column_names == list(RESULT_COLUMNS)
    assert table.column("id").to_pylist() == ["fra_Latn-1"]
    assert table.column("edu_score").to_pylist() == [3]
    assert math.isnan(table.column("structure_ratio").to_pylist()[0])


def test_results_are_compressed_with_zstd():
    metadata = pq.ParquetFile(io.BytesIO(results_to_parquet([ROW]))).metadata

    assert metadata.row_group(0).column(0).compression == "ZSTD"


def test_no_results_still_write_a_parquet_file_with_the_schema():
    table = pq.read_table(io.BytesIO(results_to_parquet([])))

    assert table.column_names == list(RESULT_COLUMNS)
    assert table.num_rows == 0


def test_upload_sends_the_parquet_bytes_to_the_dataset_repo_with_a_commit_message():
    api = RecordingApi()

    upload_results(api, "NoeFlandre/audit", "v1/audit.parquet", b"parquet-bytes")

    assert api.uploads == [
        (b"parquet-bytes", "v1/audit.parquet", "NoeFlandre/audit", "dataset", "Add audit results")
    ]
