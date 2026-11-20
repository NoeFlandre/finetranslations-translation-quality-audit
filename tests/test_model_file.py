import io

import pytest
from translation_quality_audit.language import ensure_model

URL = "https://example.test/lid.176.ftz"


def test_a_missing_model_is_downloaded_once_into_the_cache(monkeypatch, tmp_path):
    requested = []

    def fake_urlopen(url, **kwargs):
        requested.append((url, kwargs))
        return io.BytesIO(b"model-bytes")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    target = tmp_path / "cache" / "lid.176.ftz"

    assert ensure_model(target, URL) == target
    assert ensure_model(target, URL) == target

    assert target.read_bytes() == b"model-bytes"
    assert requested == [(URL, {"timeout": 120})]


def test_the_cache_folder_can_be_nested_and_may_already_exist(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda _url, **_kwargs: io.BytesIO(b"x"))
    existing = tmp_path / "a" / "b"
    existing.mkdir(parents=True)

    assert ensure_model(existing / "lid.ftz", URL).read_bytes() == b"x"
    assert ensure_model(tmp_path / "c" / "d" / "lid.ftz", URL).exists()


def test_a_failed_download_leaves_no_partial_model_behind(monkeypatch, tmp_path):
    def broken_urlopen(_url, **_kwargs):
        message = "network down"
        raise OSError(message)

    monkeypatch.setattr("urllib.request.urlopen", broken_urlopen)

    with pytest.raises(OSError, match="network down"):
        ensure_model(tmp_path / "lid.176.ftz", URL)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_final_rename_leaves_no_partial_model_behind(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda _url, **_kwargs: io.BytesIO(b"x"))

    def broken_replace(_path, _target):
        message = "disk full"
        raise OSError(message)

    monkeypatch.setattr("pathlib.Path.replace", broken_replace)

    with pytest.raises(OSError, match="disk full"):
        ensure_model(tmp_path / "lid.176.ftz", URL)
    assert list(tmp_path.iterdir()) == []
