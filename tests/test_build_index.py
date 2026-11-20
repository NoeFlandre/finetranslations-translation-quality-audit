import io

import pyarrow.parquet as pq
import scripts.build_index as build_index_script
from translation_quality_audit.hub import INDEX_PATH, INDEXED_LANGUAGES
from translation_quality_audit.index import build_index

from tests.helpers import open_local


class FakeFileSystem:
    def __init__(self, paths):
        self.paths = paths
        self.patterns = []

    def glob(self, pattern):
        self.patterns.append(pattern)
        return list(self.paths)


class RecordingApi:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, **kwargs):
        self.uploads.append(
            (path_or_fileobj, path_in_repo, repo_id, repo_type, kwargs["commit_message"])
        )


def test_the_defaults_upload_the_index_to_the_results_repo():
    args = build_index_script.parse_args([])

    assert args.path == INDEX_PATH
    assert args.repo == build_index_script.DEFAULT_REPO


def test_main_indexes_the_audited_languages_and_uploads_the_index(
    monkeypatch, capsys, parquet_files
):
    fs = FakeFileSystem(parquet_files)
    api = RecordingApi()
    monkeypatch.setattr(build_index_script, "HfFileSystem", lambda: fs)
    monkeypatch.setattr(build_index_script, "HfApi", lambda: api)
    monkeypatch.setattr(build_index_script, "open_remote", open_local)

    status = build_index_script.main([])

    assert status == 0
    assert fs.patterns == [build_index_script.DATA_GLOB]
    [(data, path, repo, repo_type, message)] = api.uploads
    assert (path, repo_type, message) == (INDEX_PATH, "dataset", "Add language index")
    assert repo == build_index_script.DEFAULT_REPO
    uploaded = pq.read_table(io.BytesIO(data))
    expected = build_index(open_local, parquet_files, INDEXED_LANGUAGES)
    assert uploaded.num_rows == expected.num_rows
    assert uploaded.column("count").to_pylist() == expected.column("count").to_pylist()
    assert capsys.readouterr().out.splitlines()[0] == (
        f"indexed {expected.num_rows} entries from 2 files"
    )
