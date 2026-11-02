from __future__ import annotations

import pandas as pd

from amharic_news.data import (
    ethiopic_letter_share,
    make_text,
    normalize_text,
    prepare_splits,
)


def test_normalize_text_preserves_ethiopic_and_removes_hidden_controls() -> None:
    assert normalize_text("\ufeff  ዜና\u200b  ነው  ") == "ዜና ነው"


def test_script_share_counts_letters_not_digits_or_punctuation() -> None:
    assert ethiopic_letter_share("አበበ ABC 123 !") == 3 / 6
    assert ethiopic_letter_share("English only") == 0.0


def test_make_text_uses_title_and_article_only() -> None:
    assert make_text(" ርዕስ ", " ይዘት ") == "ርዕስ\nይዘት"
    assert make_text(None, "ይዘት") == "ይዘት"


def _row(
    label: str, index: int, *, title: str | None = None, body: str | None = None
) -> dict[str, str]:
    return {
        "headline": title or f"ዜና ርዕስ {index}",
        "article": body or f"ይህ የአማርኛ ዜና ጽሑፍ ቁጥር {index} ነው እና ተጨማሪ መረጃ ይዟል",
        "category": label,
    }


def test_prepare_splits_filters_english_and_prevents_exact_overlap() -> None:
    labels = ["ቢዝነስ", "መዝናኛ", "ሀገር አቀፍ ዜና", "ፖለቲካ", "ስፖርት", "ዓለም አቀፍ ዜና"]
    train_rows = [_row(label, i) for i, label in enumerate(labels * 6)]
    train_rows[1] = _row("መዝናኛ", 1, title="English headline", body="Only English article text")
    shared = _row("ስፖርት", 900, title="የጋራ ርዕስ", body="የጋራ የአማርኛ ጽሑፍ ነው")
    train_rows.append(shared)
    test_rows = [_row(label, 100 + i) for i, label in enumerate(labels * 3)]
    test_rows.append(shared)
    splits, audit = prepare_splits(pd.DataFrame(train_rows), pd.DataFrame(test_rows))
    assert set(splits) == {"train", "validation", "test"}
    assert audit["cross_split_exact_overlaps_removed_from_training"] == 1
    assert audit["removed_below_script_threshold"]["official_train"] == 1
    assert len(splits["test"]) == len(test_rows)
    assert set(splits["train"]["label"]) == set(splits["validation"]["label"])
    assert not set(splits["train"]["text"]) & set(splits["test"]["text"])
