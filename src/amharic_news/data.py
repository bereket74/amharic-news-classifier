from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_REPO = "israel/Amharic-News-Text-classification-Dataset"
DATASET_REVISION = "b646090ef0d09981da9c9765c4d376b407aa5955"
MODEL_REPO = "Davlan/afro-xlmr-mini"
MODEL_REVISION = "bc04038b969667884bd83cdd37bed9559c2c3d9c"
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.20
MIN_ETHIOPIC_LETTER_SHARE = 0.10

LABEL_NAMES = {
    "ቢዝነስ": "business",
    "መዝናኛ": "entertainment",
    "ሀገር አቀፍ ዜና": "national_news",
    "ፖለቲካ": "politics",
    "ስፖርት": "sports",
    "ዓለም አቀፍ ዜና": "world_news",
}
LABEL_DESCRIPTIONS = {
    "business": "Business and economy",
    "entertainment": "Entertainment and culture",
    "national_news": "National and domestic news",
    "politics": "Politics and government",
    "sports": "Sports",
    "world_news": "International news",
}

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_SPACE = re.compile(r"\s+")
_REQUIRED_COLUMNS = {"headline", "article", "category"}


def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace without transliterating Ethiopic text."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _ZERO_WIDTH.sub("", text)
    return _SPACE.sub(" ", text).strip()


def make_text(headline: Any, article: Any) -> str:
    """Join the editorial headline and article body; exclude URL/date/view metadata."""
    title = normalize_text(headline)
    body = normalize_text(article)
    if title and body:
        return f"{title}\n{body}"
    return title or body


def ethiopic_letter_share(text: str) -> float:
    """Return the fraction of Unicode letters in the Ethiopic block."""
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    ethiopic = sum("\u1200" <= ch <= "\u137f" for ch in letters)
    return ethiopic / len(letters)


def text_fingerprint(text: str) -> str:
    canonical = normalize_text(text).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prepare_frame(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, int]:
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{split} split is missing required columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "text": [
                make_text(h, a) for h, a in zip(frame["headline"], frame["article"], strict=True)
            ],
            "label_am": frame["category"].map(normalize_text),
        }
    )
    out["row_id"] = [f"{split}-{i:06d}" for i in range(len(out))]
    missing_labels = out["label_am"].eq("")
    invalid_label_rows = int(missing_labels.sum())
    out = out.loc[~missing_labels].copy()
    unknown = sorted(set(out["label_am"]) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"Unexpected category labels in {split}: {unknown}")
    out["label"] = out["label_am"].map(LABEL_NAMES)
    out["ethiopic_letter_share"] = out["text"].map(ethiopic_letter_share)
    out["fingerprint"] = out["text"].map(text_fingerprint)
    out["source_split"] = split
    return out, invalid_label_rows


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}


def _drop_ambiguous_and_deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop text fingerprints with conflicting labels; retain one same-label copy."""
    label_counts = frame.groupby("fingerprint")["label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    conflict_rows = int(frame["fingerprint"].isin(conflicting).sum())
    unambiguous = frame[~frame["fingerprint"].isin(conflicting)].copy()
    before = len(unambiguous)
    unique = unambiguous.drop_duplicates("fingerprint", keep="first")
    return unique, conflict_rows, before - len(unique)


def prepare_splits(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    min_ethiopic_share: float = MIN_ETHIOPIC_LETTER_SHARE,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Clean, screen, deduplicate and create a stratified validation split.

    The source's train and test splits are kept separate. Exact text duplicates
    are collapsed within each split, contradictory exact duplicates are
    excluded, and any exact cross-split overlap is removed from training.
    """
    raw_counts = {"official_train": len(train_raw), "official_test": len(test_raw)}
    train_all, train_missing_labels = _prepare_frame(train_raw, "official_train")
    test_all, test_missing_labels = _prepare_frame(test_raw, "official_test")
    valid_label_counts = {"official_train": len(train_all), "official_test": len(test_all)}
    train_nonempty = train_all[train_all["text"].str.len() > 0].copy()
    test_nonempty = test_all[test_all["text"].str.len() > 0].copy()
    nonempty_counts = {"official_train": len(train_nonempty), "official_test": len(test_nonempty)}

    train_screened = train_nonempty[
        train_nonempty["ethiopic_letter_share"] >= min_ethiopic_share
    ].copy()
    test_screened = test_nonempty[
        test_nonempty["ethiopic_letter_share"] >= min_ethiopic_share
    ].copy()
    screened_counts = {"official_train": len(train_screened), "official_test": len(test_screened)}
    excluded_category_counts = {
        split: _distribution(frame[frame["ethiopic_letter_share"] < min_ethiopic_share])
        for split, frame in (("official_train", train_nonempty), ("official_test", test_nonempty))
    }

    train_screened, train_conflicts, train_duplicates = _drop_ambiguous_and_deduplicate(
        train_screened
    )
    test_screened, test_conflicts, test_duplicates = _drop_ambiguous_and_deduplicate(test_screened)
    cross_common = set(train_screened["fingerprint"]) & set(test_screened["fingerprint"])
    test_label_by_fingerprint = test_screened.set_index("fingerprint")["label"].to_dict()
    cross_label_conflicts = sum(
        int(label != test_label_by_fingerprint[fingerprint])
        for fingerprint, label in zip(
            train_screened["fingerprint"], train_screened["label"], strict=True
        )
        if fingerprint in cross_common
    )
    before_overlap = len(train_screened)
    train_screened = train_screened[~train_screened["fingerprint"].isin(cross_common)].copy()
    cross_split_overlap_removed = before_overlap - len(train_screened)

    train_part, val_part = train_test_split(
        train_screened,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        stratify=train_screened["label"],
    )
    columns = ["row_id", "text", "label", "label_am", "ethiopic_letter_share"]
    splits = {
        "train": train_part[columns].reset_index(drop=True),
        "validation": val_part[columns].reset_index(drop=True),
        "test": test_screened[columns].reset_index(drop=True),
    }
    audit: dict[str, Any] = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "source_license_declared": "CC BY 4.0 (Hugging Face dataset card)",
        "seed": seed,
        "validation_fraction_of_clean_training_data": VALIDATION_FRACTION,
        "minimum_ethiopic_letter_share": min_ethiopic_share,
        "raw_rows": raw_counts,
        "rows_with_valid_labels": valid_label_counts,
        "removed_rows_with_missing_labels": {
            "official_train": train_missing_labels,
            "official_test": test_missing_labels,
        },
        "nonempty_rows": nonempty_counts,
        "rows_after_ethiopic_script_screen": screened_counts,
        "removed_empty_rows": {
            split: valid_label_counts[split] - nonempty_counts[split]
            for split in valid_label_counts
        },
        "removed_below_script_threshold": {
            split: nonempty_counts[split] - screened_counts[split] for split in valid_label_counts
        },
        "excluded_category_counts_below_script_threshold": excluded_category_counts,
        "removed_conflicting_label_duplicate_rows": {
            "official_train": train_conflicts,
            "official_test": test_conflicts,
        },
        "removed_same_label_duplicate_rows": {
            "official_train": train_duplicates,
            "official_test": test_duplicates,
        },
        "cross_split_exact_overlaps_removed_from_training": cross_split_overlap_removed,
        "cross_split_exact_label_conflicts": cross_label_conflicts,
        "final_rows": {name: len(df) for name, df in splits.items()},
        "label_names": LABEL_NAMES,
        "label_descriptions": LABEL_DESCRIPTIONS,
        "label_counts": {name: _distribution(df) for name, df in splits.items()},
        "notes": [
            (
                "Only headline and article body are model inputs; dates, source URLs, view counts, "
                "and row IDs are excluded."
            ),
            (
                "The supplied official test partition is screened, made unique by normalized "
                "exact text, and kept out of tuning and checkpoint selection."
            ),
            (
                "Empty labels and exact text records with contradictory labels are excluded; "
                "identical same-label records are collapsed."
            ),
            (
                "Exact training records that overlap the screened test partition "
                "are removed from training."
            ),
            (
                "The 10% Ethiopic-letter threshold is a script heuristic, not proof of language "
                "identity; code-mixed text can remain."
            ),
            (
                "Near-duplicate stories, source-level leakage, and copyright ownership of "
                "underlying publisher text are not independently adjudicated."
            ),
            (
                "Source article text is not included in the public repository, "
                "model card, or Space bundle."
            ),
        ],
    }
    return splits, audit


def save_processed(
    splits: dict[str, pd.DataFrame], audit: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_processed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    required = {"text", "label", "row_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Processed data missing columns: {sorted(missing)}")
    frame["text"] = frame["text"].map(normalize_text)
    frame["label"] = frame["label"].astype(str)
    return frame
