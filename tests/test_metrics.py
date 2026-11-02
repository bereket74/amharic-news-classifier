from __future__ import annotations

from amharic_news.metrics import classification_metrics, summarize_errors


def test_classification_metrics_reports_balanced_and_macro_scores() -> None:
    result = classification_metrics(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        labels=["a", "b"],
        probabilities=[[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]],
        bootstrap_samples=80,
    )
    assert result["accuracy"] == 0.75
    assert 0 <= result["macro_f1"] <= 1
    assert len(result["macro_f1_95pct_bootstrap_ci"]) == 2
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_summarize_errors_is_aggregate_only() -> None:
    result = summarize_errors(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "a"],
        [0.95, 0.55, 0.85, 0.51],
        [2, 8, 1, 13],
        labels=["a", "b"],
    )
    assert result["top_confusions"][0]["gold"] in {"a", "b"}
    assert result["length_bins"]
    assert "article_text" not in str(result)
