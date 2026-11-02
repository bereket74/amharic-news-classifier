from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from scripts import download_data


class _FakeResponse(io.BytesIO):
    def __init__(self, content: bytes):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_download_reuses_only_a_cache_matching_the_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"pinned source data"
    expected = _digest(content)
    monkeypatch.setattr(download_data, "FILES", ("train.csv",))
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", {"train.csv": expected})
    (tmp_path / "train.csv").write_bytes(content)
    monkeypatch.setattr(
        download_data.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("matching cache should not be fetched again"),
    )

    assert download_data.download(tmp_path) == {"train.csv": expected}


def test_download_refreshes_a_stale_cache_from_the_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"verified replacement"
    expected = _digest(content)
    monkeypatch.setattr(download_data, "FILES", ("train.csv",))
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", {"train.csv": expected})
    destination = tmp_path / "train.csv"
    destination.write_bytes(b"stale content")
    monkeypatch.setattr(
        download_data.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(content),
    )

    assert download_data.download(tmp_path) == {"train.csv": expected}
    assert destination.read_bytes() == content


def test_bad_remote_payload_does_not_replace_or_leave_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _digest(b"expected content")
    monkeypatch.setattr(download_data, "FILES", ("train.csv",))
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", {"train.csv": expected})
    destination = tmp_path / "train.csv"
    destination.write_bytes(b"existing cache")
    monkeypatch.setattr(
        download_data.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"wrong remote data"),
    )

    with pytest.raises(ValueError, match="pinned Hub revision"):
        download_data.download(tmp_path)

    assert destination.read_bytes() == b"existing cache"
    assert not (tmp_path / "train.csv.part").exists()
