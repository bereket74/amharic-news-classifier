from __future__ import annotations

import pandas as pd

from amharic_news.train import _cap_training_rows


def test_training_cap_is_deterministic_and_per_class() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{i}" for i in range(10)],
            "label": ["a"] * 6 + ["b"] * 3 + ["c"],
        }
    )
    sampled, info = _cap_training_rows(frame, max_per_class=2, seed=42)
    sampled_again, _ = _cap_training_rows(frame, max_per_class=2, seed=42)
    assert sampled["row_id"].tolist() == sampled_again["row_id"].tolist()
    assert sampled["label"].value_counts().to_dict() == {"a": 2, "b": 2, "c": 1}
    assert info["full_clean_training_rows"] == 10
    assert info["used_training_rows"] == 5
