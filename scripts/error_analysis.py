# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from amharic_news.metrics import summarize_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Write aggregate-only held-out error analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("reports/error_analysis.md"))
    args = parser.parse_args()
    test = pd.read_csv(args.data_dir / "test.csv")
    predictions = pd.read_csv(args.run_dir / "transformer_test_predictions.csv")
    joined = test[["row_id", "label", "text"]].merge(
        predictions, on="row_id", validate="one_to_one", suffixes=("_data", "_pred")
    )
    if not (joined["label"] == joined["y_true"]).all():
        raise ValueError("Prediction gold labels do not match the processed test split")
    report_path = args.run_dir / "transformer_training.json"
    config = json.loads(report_path.read_text(encoding="utf-8"))
    labels = list(config["id_to_label"].values())
    summary = summarize_errors(
        joined["y_true"].tolist(),
        joined["y_pred"].tolist(),
        joined["max_softmax_score"].tolist(),
        joined["word_length"].tolist(),
        labels=labels,
    )
    lines = [
        "# Held-out error analysis",
        "",
        f"The transformer made **{summary['errors']:,} errors** in **{summary['n']:,}** eligible held-out articles "
        f"(error rate {summary['error_rate']:.1%}). The report is aggregate-only; it contains no headlines or article text.",
        "",
        "## Most frequent confusions",
        "",
        "| Gold category | Predicted category | Count |",
        "|---|---|---:|",
    ]
    if summary["top_confusions"]:
        lines.extend(
            f"| `{item['gold']}` | `{item['predicted']}` | {item['count']:,} |"
            for item in summary["top_confusions"]
        )
    else:
        lines.append("| — | — | 0 |")
    lines.extend(
        [
            "",
            "## Error rate by gold category",
            "",
            "| Category | Test examples | Error rate |",
            "|---|---:|---:|",
        ]
    )
    for label, item in summary["label_error_rates"].items():
        error_rate = "—" if item["error_rate"] is None else f"{item['error_rate']:.1%}"
        lines.append(f"| `{label}` | {item['n']:,} | {error_rate} |")
    lines.extend(
        [
            "",
            "## Error rate by article length",
            "",
            "| Length cohort | Examples | Error rate |",
            "|---|---:|---:|",
        ]
    )
    for item in summary["length_bins"]:
        lines.append(
            f"| {item['bin'].replace('_', ' ')} | {item['n']:,} | {item['error_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Error rate by model score",
            "",
            "| Maximum softmax score | Examples | Error rate |",
            "|---|---:|---:|",
        ]
    )
    for item in summary["confidence_bins"]:
        lines.append(f"| {item['range']} | {item['n']:,} | {item['error_rate']:.1%} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are measurable error patterns, not explanations of individual mistakes. Category boundaries overlap in real reporting, and the small entertainment class has less evaluation support than the larger classes. The model's maximum softmax score is not calibrated confidence. Length cohorts use the processed text's whitespace-delimited word count; long articles are truncated using a limit selected from training data only.",
            "",
            "Publisher/source-level separation, near-duplicate detection, and external-domain validation remain future work. No article examples are copied into this report because the source corpus contains publisher text.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote aggregate error analysis: {args.output}")


if __name__ == "__main__":
    main()
