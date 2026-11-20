import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import scripts.run_audit as run_audit
from translation_quality_audit.hub import DEFAULT_REPO, INDEX_PATH, INDEXED_LANGUAGES
from translation_quality_audit.index import build_index, index_to_parquet

from tests.helpers import FixedModel, open_local


class RemoteFiles:
    """Opens the index from a local copy, and every other path from its own location."""

    def __init__(self, remote_files):
        self.remote_files = remote_files
        self.opened = []

    def __call__(self, path):
        self.opened.append(path)
        return open_local(self.remote_files.get(path, path))


class RecordingApi:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, **_kwargs):
        self.uploads.append((path_or_fileobj, path_in_repo, repo_id, repo_type))


@pytest.fixture
def remote(tmp_path, parquet_files):
    index_file = tmp_path / "groups.parquet"
    index_file.write_bytes(index_to_parquet(build_index(open_local, parquet_files)))
    index_path = f"datasets/{DEFAULT_REPO}/{INDEX_PATH}"
    return {index_path: str(index_file)}


def install(monkeypatch, remote_files, api=None):
    monkeypatch.setattr(run_audit, "open_remote", remote_files)
    monkeypatch.setattr(run_audit, "HfApi", lambda: api or RecordingApi())
    monkeypatch.setattr(run_audit, "ensure_model", lambda path: path)
    monkeypatch.setattr(run_audit, "load_identifier", lambda _path: FixedModel())


def test_the_defaults_audit_the_four_indexed_languages_with_100_documents_each():
    args = run_audit.parse_args([])

    assert args.languages == list(INDEXED_LANGUAGES)
    assert (args.total, args.seed) == (100, 0)
    assert args.repo == DEFAULT_REPO
    assert args.path == "v1/audit.parquet"
    assert args.model == run_audit.DEFAULT_MODEL


def test_only_indexed_languages_can_be_audited():
    with pytest.raises(SystemExit):
        run_audit.parse_args(["--languages", "zho_Hans"])


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


def test_main_reads_the_index_audits_each_language_and_uploads_one_parquet_file(
    monkeypatch, capsys, remote
):
    remote_files = RemoteFiles(remote)
    api = RecordingApi()
    install(monkeypatch, remote_files, api)

    status = run_audit.main(
        ["--languages", "deu_Latn", "--total", "2", "--path", "v1/a.parquet", "--repo", "user/repo"]
    )

    assert status == 0
    assert f"datasets/{DEFAULT_REPO}/{INDEX_PATH}" in remote_files.opened
    [(data, path, repo, repo_type)] = api.uploads
    assert (path, repo, repo_type) == ("v1/a.parquet", "user/repo", "dataset")
    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == 2
    assert set(table.column("language").to_pylist()) == {"deu_Latn"}
    assert capsys.readouterr().out.strip() == "uploaded 2 rows to user/repo/v1/a.parquet"


def test_main_passes_the_seed_to_each_language_audit(monkeypatch, remote):
    seen = []
    original = run_audit.audit_language

    def recording(opener, language, total, seed, identifier, index):
        seen.append((language, seed))
        return original(opener, language, total, seed, identifier, index)

    monkeypatch.setattr(run_audit, "audit_language", recording)
    install(monkeypatch, RemoteFiles(remote))

    run_audit.main(["--languages", "deu_Latn", "fra_Latn", "--total", "1", "--seed", "9"])

    assert seen == [("deu_Latn", 9), ("fra_Latn", 9)]
