from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from amharic_news.data import DATASET_REPO, DATASET_REVISION

FILES = ("train.csv", "test.csv")
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path: Path) -> dict[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in FILES:
        destination = path / name
        if not destination.exists():
            request = urllib.request.Request(
                f"{BASE_URL}/{name}?download=true",
                headers={"User-Agent": "amharic-news-classifier/1.0"},
            )
            temporary = destination.with_suffix(destination.suffix + ".part")
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
            temporary.replace(destination)
        hashes[name] = sha256(destination)
        print(f"{name}: {destination.stat().st_size:,} bytes, sha256={hashes[name]}")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pinned public Amharic news benchmark"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    hashes = download(args.output_dir)
    manifest = {
        "dataset_repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "files": hashes,
        "source_license_declared": "CC BY 4.0 (Hugging Face dataset card)",
        "redistribute_source_text": False,
    }
    manifest_path = args.output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
