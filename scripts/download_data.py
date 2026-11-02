from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from amharic_news.data import DATASET_REPO, DATASET_REVISION

FILES = ("train.csv", "test.csv")
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"
# SHA-256 values are the immutable Hub LFS hashes for these files at DATASET_REVISION.
EXPECTED_SHA256 = {
    "train.csv": "446bab6e0380dd3b131115186aa87168ebe17a07c92b8290543dfc6ee48fa529",
    "test.csv": "7deb14a8d01655b80e77d66ef495b1bf123b91ca450eacb3e470ed9ba220de73",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_one(name: str, destination: Path) -> None:
    request = urllib.request.Request(
        f"{BASE_URL}/{name}?download=true",
        headers={"User-Agent": "amharic-news-classifier/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            total = response.headers.get("Content-Length", "unknown")
            transferred = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                transferred += len(block)
                if transferred % (20 * 1024 * 1024) < len(block):
                    print(f"{name}: downloaded {transferred:,} / {total} bytes", flush=True)
        actual = sha256(temporary)
        expected = EXPECTED_SHA256[name]
        if actual != expected:
            raise ValueError(
                f"Downloaded {name} does not match the pinned Hub revision: "
                f"expected SHA-256 {expected}, got {actual}. The destination was not replaced."
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download(path: Path) -> dict[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in FILES:
        if name not in EXPECTED_SHA256:
            raise KeyError(f"Missing pinned SHA-256 for source file {name!r}")
        destination = path / name
        expected = EXPECTED_SHA256[name]
        actual = sha256(destination) if destination.is_file() else None
        if actual != expected:
            if actual is not None:
                print(f"{name}: cached SHA-256 mismatch; refreshing from the pinned revision")
            _download_one(name, destination)
            actual = sha256(destination)
        if actual != expected:
            raise RuntimeError(f"Source file {name} failed pinned-revision verification")
        hashes[name] = actual
        print(f"{name}: {destination.stat().st_size:,} bytes, verified sha256={actual}")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned public Amharic news benchmark"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    hashes = download(args.output_dir)
    manifest = {
        "dataset_repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "expected_sha256": EXPECTED_SHA256,
        "files": hashes,
        "source_license_declared": "CC BY 4.0 (Hugging Face dataset card)",
        "redistribute_source_text": False,
    }
    manifest_path = args.output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
