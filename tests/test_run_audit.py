import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import scripts.run_audit as run_audit

from tests.helpers import FixedModel, open_local


class FakeFileSystem:
    def __init__(self, paths):
        self.paths = paths
        self.patterns = []
        self.block_sizes = []

    def glob(self, pattern):
        self.patterns.append(pattern)
        return list(self.paths)

    def open(self, path, block_size=None, **_kwargs):
        self.block_sizes.append(block_size)
        return open_local(path)


class RecordingApi:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, **_kwargs):
        self.uploads.append((path_or_fileobj, path_in_repo, repo_id, repo_type))


def test_the_defaults_audit_the_four_high_resource_languages_with_100_documents_each():
    args = run_audit.parse_args([])

    assert args.languages == list(run_audit.HIGH_RESOURCE)
    assert args.total == 100
    assert args.seed == 0
    assert args.repo == "NoeFlandre/finetranslations-translation-quality-audit"
    assert args.path == "v1/audit.parquet"
    assert args.model == run_audit.DEFAULT_MODEL


def test_numeric_and_path_options_are_parsed_to_their_types():
    args = run_audit.parse_args(["--total", "5", "--seed", "3", "--model", "lid.ftz"])

    assert (args.total, args.seed) == (5, 3)
    assert args.model == Path("lid.ftz")


def test_the_help_describes_the_tool_and_every_option(capsys):
    with pytest.raises(SystemExit):
        run_audit.parse_args(["--help"])

    output = capsys.readouterr().out
    for text in (
        "Audit the translation quality",
        "documents per language",
        "dataset repo for the results",
        "path inside the repo",
        "fastText LID model",
    ):
        assert text in output


def test_main_audits_each_language_and_uploads_one_parquet_file(monkeypatch, capsys, parquet_files):
    fs = FakeFileSystem(parquet_files)
    api = RecordingApi()
    ensured, loaded = [], []

    def ensure(path):
        ensured.append(path)
        return path

    def load(path):
        loaded.append(path)
        return FixedModel()

    monkeypatch.setattr(run_audit, "HfFileSystem", lambda: fs)
    monkeypatch.setattr(run_audit, "HfApi", lambda: api)
    monkeypatch.setattr(run_audit, "ensure_model", ensure)
    monkeypatch.setattr(run_audit, "load_identifier", load)

    status = run_audit.main(
        [
            "--languages",
            "deu_Latn",
            "--total",
            "2",
            "--model",
            "lid.176.ftz",
            "--path",
            "v1/a.parquet",
            "--repo",
            "user/repo",
        ]
    )

    assert status == 0
    assert fs.patterns == [run_audit.DATA_GLOB]
    assert ensured == [Path("lid.176.ftz")]
    assert loaded == [Path("lid.176.ftz")]
    [(data, path, repo, repo_type)] = api.uploads
    assert (path, repo, repo_type) == ("v1/a.parquet", "user/repo", "dataset")
    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == 2
    assert set(table.column("language").to_pylist()) == {"deu_Latn"}
    assert capsys.readouterr().out.strip() == "uploaded 2 rows to user/repo/v1/a.parquet"


def test_main_passes_the_seed_to_each_language_audit(monkeypatch, parquet_files):
    seen = []
    original = run_audit.audit_language

    def recording(opener, paths, language, total, seed, identifier):
        seen.append((language, seed))
        return original(opener, paths, language, total, seed, identifier)

    monkeypatch.setattr(run_audit, "audit_language", recording)
    monkeypatch.setattr(run_audit, "HfFileSystem", lambda: FakeFileSystem(parquet_files))
    monkeypatch.setattr(run_audit, "HfApi", RecordingApi)
    monkeypatch.setattr(run_audit, "ensure_model", lambda path: path)
    monkeypatch.setattr(run_audit, "load_identifier", lambda _path: FixedModel())

    run_audit.main(["--languages", "deu_Latn", "fra_Latn", "--total", "1", "--seed", "9"])

    assert seen == [("deu_Latn", 9), ("fra_Latn", 9)]


def test_remote_reads_use_small_blocks_so_each_column_read_stays_small(monkeypatch, parquet_files):
    fs = FakeFileSystem(parquet_files)
    monkeypatch.setattr(run_audit, "HfFileSystem", lambda: fs)
    monkeypatch.setattr(run_audit, "HfApi", RecordingApi)
    monkeypatch.setattr(run_audit, "ensure_model", lambda path: path)
    monkeypatch.setattr(run_audit, "load_identifier", lambda _path: FixedModel())

    run_audit.main(["--languages", "deu_Latn", "--total", "1"])

    assert fs.block_sizes
    assert set(fs.block_sizes) == {run_audit.READ_BLOCK_SIZE}
    assert run_audit.READ_BLOCK_SIZE <= 1 << 20
