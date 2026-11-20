"""Byte-range reads from files of the Hub, one request per read, retried while rate-limited.

A Parquet reader asks for many small byte ranges. A block cache would fetch a whole block for each
of them, so a range is requested exactly, and a request answered with 429 (too many requests) waits
and is sent again.
"""

import io
import time
from collections.abc import Mapping
from typing import Protocol

import requests
from huggingface_hub import get_token

HUB = "https://huggingface.co"
MAX_WORKERS = 8  # files read at once: enough to overlap the waits, few enough to avoid the limit
ATTEMPTS = 8
BACKOFF_SECONDS = 2
TIMEOUT_SECONDS = 120
RATE_LIMITED = 429


class HttpResponse(Protocol):
    """The part of a response that the reader uses."""

    @property
    def status_code(self) -> int:
        """The HTTP status code."""

    @property
    def content(self) -> bytes:
        """The body."""

    @property
    def headers(self) -> Mapping[str, str]:
        """The response headers."""

    def raise_for_status(self) -> None:
        """Raise an error when the status is a failure."""


class HttpSession(Protocol):
    """The part of a requests session that the reader uses."""

    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        """GET a URL with the given headers."""


class RangeFile(io.RawIOBase):
    """A read-only file whose reads are HTTP byte ranges."""

    def __init__(self, url: str, session: HttpSession, size: int):
        self.url = url
        self.session = session
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        """Reads are allowed."""
        return True

    def seekable(self) -> bool:
        """Seeks are allowed."""
        return True

    def tell(self) -> int:
        """The current position."""
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move to an offset from the start, the current position or the end."""
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = bases[whence] + offset
        return self.position

    def read(self, size: int | None = -1) -> bytes:
        """Read up to size bytes, or to the end, with one request for the range."""
        if size is None or size < 0:
            size = self.size - self.position
        last = min(self.position + size, self.size) - 1
        if last < self.position:
            return b""
        response = get_with_retries(
            self.session, self.url, {"Range": f"bytes={self.position}-{last}"}
        )
        self.position += len(response.content)
        return response.content


def open_remote(path: str) -> RangeFile:
    """Open a file by its path in a dataset repo, like datasets/<owner>/<name>/data/x.parquet."""
    _, owner, name, inside = path.split("/", 3)
    url = f"{HUB}/datasets/{owner}/{name}/resolve/main/{inside}"
    session = new_session()
    response = get_with_retries(session, url, {"Range": "bytes=0-0"})
    size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    return RangeFile(url, session, size)


def new_session() -> requests.Session:
    """A session that sends the Hub token, when there is one, so the requests are not anonymous."""
    session = requests.Session()
    token = get_token()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def get_with_retries(session: HttpSession, url: str, headers: Mapping[str, str]) -> HttpResponse:
    """GET a byte range, waiting and sending it again while the Hub answers 429."""
    response = session.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
    for attempt in range(ATTEMPTS):
        if response.status_code != RATE_LIMITED:
            break
        time.sleep(BACKOFF_SECONDS * 2**attempt)
        response = session.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response
