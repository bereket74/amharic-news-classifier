from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from download_data import download

from amharic_news.data import prepare_splits, save_processed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an audited Amharic news classification dataset"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--script-share", type=float, default=0.10)
    args = parser.parse_args()
    hashes = download(args.raw_dir)
    train_raw = pd.read_csv(args.raw_dir / "train.csv", low_memory=False, encoding="utf-8")
    test_raw = pd.read_csv(args.raw_dir / "test.csv", low_memory=False, encoding="utf-8")
    splits, audit = prepare_splits(train_raw, test_raw, min_ethiopic_share=args.script_share)
    audit["source_file_sha256"] = hashes
    save_processed(splits, audit, args.processed_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Processed splits written under {args.processed_dir}; source text is Git-ignored.")


if __name__ == "__main__":
    main()
