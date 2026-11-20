import io

import pytest
import requests
import translation_quality_audit.remote as remote
from translation_quality_audit.remote import (
    ATTEMPTS,
    BACKOFF_SECONDS,
    RangeFile,
    get_with_retries,
    new_session,
    open_remote,
)

DATA = bytes(range(256)) * 4


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    """Serves DATA for every URL, and answers the first `rate_limited` requests with 429."""

    def __init__(self, rate_limited=0):
        self.rate_limited = rate_limited
        self.requested = []
        self.headers = {}

    def get(self, url, headers, **_kwargs):
        self.requested.append((url, headers["Range"]))
        if self.rate_limited:
            self.rate_limited -= 1
            return FakeResponse(429)
        start, last = (int(value) for value in headers["Range"].removeprefix("bytes=").split("-"))
        content = DATA[start : last + 1]
        return FakeResponse(206, content, {"Content-Range": f"bytes {start}-{last}/{len(DATA)}"})


class MissingSession(FakeSession):
    def get(self, url, headers, **_kwargs):
        self.requested.append((url, headers["Range"]))
        return FakeResponse(404)


def test_a_range_file_reads_exactly_the_bytes_it_is_asked_for():
    handle = RangeFile("https://example.test/f", FakeSession(), len(DATA))

    assert handle.read(3) == DATA[0:3]
    assert handle.tell() == 3
    handle.seek(10)
    assert handle.read(4) == DATA[10:14]


def test_each_read_is_one_request_for_its_range():
    session = FakeSession()
    handle = RangeFile("https://example.test/f", session, len(DATA))

    handle.read(3)

    assert session.requested == [("https://example.test/f", "bytes=0-2")]


def test_seeking_counts_from_the_start_the_current_position_and_the_end():
    handle = RangeFile("u", FakeSession(), len(DATA))

    handle.seek(5)
    handle.seek(2, io.SEEK_CUR)
    assert handle.tell() == 7
    handle.seek(-4, io.SEEK_END)
    assert handle.read() == DATA[-4:]


def test_reading_without_a_size_reads_to_the_end():
    handle = RangeFile("u", FakeSession(), len(DATA))

    handle.seek(1000)

    assert handle.read() == DATA[1000:]
    assert handle.read(None) == b""
    assert handle.read(-1) == b""


def test_a_read_past_the_end_stops_at_the_end():
    handle = RangeFile("u", FakeSession(), len(DATA))

    handle.seek(len(DATA) - 2)

    assert handle.read(10) == DATA[-2:]


def test_a_range_file_is_readable_and_seekable():
    handle = RangeFile("u", FakeSession(), len(DATA))

    assert handle.readable()
    assert handle.seekable()


def test_a_rate_limited_request_waits_and_is_sent_again(monkeypatch):
    waits = []
    monkeypatch.setattr(remote.time, "sleep", waits.append)
    session = FakeSession(rate_limited=2)

    response = get_with_retries(session, "u", {"Range": "bytes=0-0"})

    assert response.status_code == 206
    assert waits == [BACKOFF_SECONDS, BACKOFF_SECONDS * 2]
    assert len(session.requested) == 3


def test_a_request_rate_limited_on_every_attempt_fails(monkeypatch):
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    session = FakeSession(rate_limited=ATTEMPTS + 1)

    with pytest.raises(requests.HTTPError, match="429"):
        get_with_retries(session, "u", {"Range": "bytes=0-0"})

    assert len(session.requested) == ATTEMPTS + 1


def test_an_error_other_than_rate_limiting_fails_without_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(remote.time, "sleep", waits.append)
    session = MissingSession()

    with pytest.raises(requests.HTTPError, match="404"):
        get_with_retries(session, "u", {"Range": "bytes=0-0"})

    assert waits == []
    assert len(session.requested) == 1


def test_open_remote_resolves_the_path_and_reads_the_size_from_the_first_byte(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(remote, "new_session", lambda: session)

    handle = open_remote("datasets/owner/name/data/000.parquet")

    assert handle.url == "https://huggingface.co/datasets/owner/name/resolve/main/data/000.parquet"
    assert handle.size == len(DATA)
    assert session.requested == [(handle.url, "bytes=0-0")]


def test_a_session_sends_the_hub_token_when_there_is_one(monkeypatch):
    monkeypatch.setattr(remote, "get_token", lambda: "hf_token")

    assert new_session().headers["Authorization"] == "Bearer hf_token"


def test_a_session_is_anonymous_without_a_token(monkeypatch):
    monkeypatch.setattr(remote, "get_token", lambda: None)

    assert "Authorization" not in new_session().headers
