# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from amharic_news.data import (
    DATASET_REPO,
    DATASET_REVISION,
    LABEL_DESCRIPTIONS,
    LABEL_NAMES,
    MODEL_REPO,
    MODEL_REVISION,
)
from amharic_news.labels import GITHUB_REPO_URL, PUBLISHED_MODEL_ID

ROOT = Path(__file__).resolve().parents[1]


def _table_rows(results: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    baselines = results.get("baselines") or {}
    for name, item in baselines.get("models", {}).items():
        val = item["validation_macro_f1"]
        test = item["test_metrics"]["macro_f1"]
        rows.append(f"| `{name}` | {val:.3f} | {test:.3f} |")
    transformer = results["transformer"]
    rows.append(
        f"| `AfroXLMR-mini` | {transformer['selected_validation_macro_f1']:.3f} | "
        f"{transformer['test_metrics']['macro_f1']:.3f} |"
    )
    return rows


def _model_card(results: dict[str, Any], audit: dict[str, Any]) -> str:
    transformer = results["transformer"]
    test = transformer["test_metrics"]
    low, high = test["macro_f1_95pct_bootstrap_ci"]
    sample = results["training_sample"]
    train_n = sample["used_training_rows"]
    full_train_n = sample["full_clean_training_rows"]
    max_per_class = sample["max_per_class"]
    val_n = transformer["validation_rows"]
    test_n = transformer["test_rows"]
    selected_name = results["selected_overall_by_validation_macro_f1"]
    method_names = {
        "baseline:majority": "majority classifier",
        "baseline:word_tfidf_linear_svc": "word TF–IDF + LinearSVC",
        "baseline:char_tfidf_linear_svc": "character TF–IDF + LinearSVC",
        "baseline:word_char_tfidf_linear_svc": "combined word/character TF–IDF + LinearSVC",
        "transformer": "AfroXLMR-mini transformer",
    }
    selected_display = method_names.get(selected_name, selected_name)
    best_baseline_name = results["baselines"]["selected_baseline_by_validation_macro_f1"]
    best_baseline_test = results["baselines"]["models"][best_baseline_name]["test_metrics"][
        "macro_f1"
    ]
    national_news_error_rate = 1.0 - test["per_class"]["national_news"]["recall"]
    truncated_fraction = transformer["max_length_profile"][
        "training_fraction_truncated_at_selected_length"
    ]
    table = "\n".join(_table_rows(results))
    labels = "\n".join(
        f"- `{english}` — {LABEL_DESCRIPTIONS[english]} (`{amharic}`)"
        for amharic, english in LABEL_NAMES.items()
    )
    return f'''---
license: cc-by-4.0
language:
  - am
pipeline_tag: text-classification
base_model: {MODEL_REPO}
datasets:
  - {DATASET_REPO}
tags:
  - amharic
  - news-classification
  - afroxlmr
  - text-classification
---

# Amharic News Classifier

A six-way Amharic news-topic classifier fine-tuned from AfroXLMR-mini. This repository contains the model and inference code, not source articles or dataset rows.

**Source code:** [Public GitHub repository]({GITHUB_REPO_URL})

## Labels

{labels}

## Evaluation

The source training partition contains {full_train_n:,} cleaned training examples. For bounded, reproducible CPU fitting, a fixed-seed sample of {train_n:,} examples (up to {max_per_class:,} per category) was used for both the baselines and transformer. The full {val_n:,}-example validation set was used for model selection. Each candidate was evaluated once on the full {test_n:,}-example held-out test partition after validation-based selection; test metrics did not determine the released checkpoint.

| Method | Validation macro-F1 | Held-out test macro-F1 |
|---|---:|---:|
{table}

Transformer test accuracy: **{test["accuracy"]:.3f}**; macro-F1: **{test["macro_f1"]:.3f}** (2,000-resample 95% row-bootstrap interval **{low:.3f}–{high:.3f}**); weighted-F1: **{test["weighted_f1"]:.3f}**; balanced accuracy: **{test["balanced_accuracy"]:.3f}**. The test set has {test_n:,} examples. The interval reflects row resampling of this fixed trained model, not variability across seeds.

The best validation system overall was **{selected_display}**. The best classical baseline scored test macro-F1 {best_baseline_test:.3f}, versus {test["macro_f1"]:.3f} for this transformer. This transformer is an experimental benchmark artifact, not the preferred classifier on this dataset. A hosted Space has not been published: free ZeroGPU creation was rejected with HTTP 402, and no converted browser weights are being distributed while source-article rights remain unresolved.

See `evaluation_metrics.json`, `data_audit.json`, and `error_analysis.md` for per-class scores, confusion counts, split audit, and aggregate-only error patterns. No error example includes article text.

## Training

- Base model: [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}), revision `{MODEL_REVISION}` (Hub metadata declares MIT).
- Dataset: [`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO}), revision `{DATASET_REVISION}`; its Hub card declares CC BY 4.0 and cites Azime & Mohammed (2021).
- Seed: `{transformer["seed"]}`; selected epoch: `{transformer["selected_epoch"]}`; validation macro-F1: `{transformer["selected_validation_macro_f1"]:.3f}`.
- The classical comparison uses majority, word-TF–IDF LinearSVC, character-TF–IDF LinearSVC, and combined word/character TF–IDF LinearSVC; linear SVM regularization is fixed at `C=1.0`.
- Maximum input length: `{transformer["max_length_profile"]["selected_max_length"]}` subword tokens, chosen from the training-only length profile; only headline and article body are inputs.
- `{truncated_fraction:.1%}` of the sampled training texts exceeded the selected input length and were truncated.
- Class-balanced cross-entropy, AdamW, learning rate `{transformer["learning_rate"]}`, batch `{transformer["batch_size"]}` with gradient accumulation `{transformer["gradient_accumulation_steps"]}`.

## Preprocessing and data quality

The pipeline NFC-normalizes text, removes zero-width controls, collapses whitespace, excludes empty labels and records with less than 10% Ethiopic letters, drops exact text records with conflicting labels, collapses same-label exact duplicates within each partition, and removes exact training records that overlap test. In this run, {audit["raw_rows"]["official_train"]:,} training and {audit["raw_rows"]["official_test"]:,} test rows were supplied; {audit["cross_split_exact_overlaps_removed_from_training"]:,} exact train/test overlaps were removed from training. The final full train/validation/test counts are {audit["final_rows"]["train"]:,} / {audit["final_rows"]["validation"]:,} / {audit["final_rows"]["test"]:,}.

A 10% Ethiopic-script share is a heuristic, not a language identifier; code-switching and English items with some Ethiopic text may remain. Dataset source/article copyright has not been independently cleared beyond the CC BY 4.0 declaration on the Hub dataset card. The raw corpus is not redistributed here.

## Intended use and limitations

For research, education, and exploratory topic triage only. It is not a substitute for editorial judgment and should not be used for high-impact decisions, political targeting, or decisions about people. On this test split, national-news recall was **{1.0 - national_news_error_rate:.1%}** (error rate {national_news_error_rate:.1%}), and the transformer substantially trailed the TF–IDF baselines. News topics can overlap, language and editorial conventions change over time, and a random split may not reveal source or time leakage. The entertainment class is much smaller than national news; macro-F1 and class-level metrics should be considered together. Long articles are truncated, and the maximum softmax score is not calibrated confidence.

## Inference

```python
from amharic_news.inference import NewsClassifier

classifier = NewsClassifier("{PUBLISHED_MODEL_ID}")
print(classifier.predict("የዜና ርዕስ", "የዜናው ይዘት", top_k=3))
```

The release includes `src/amharic_news/inference.py`, `labels.py`, and `text.py`. Install PyTorch, Transformers, and the files from this repository; the example returns label names, descriptions, and raw softmax scores.

## Attribution and license

Original project code is MIT licensed. The Hugging Face model repository declares CC BY 4.0 with attribution to the dataset authors based on the dataset repository's license statement; this does not independently establish the copyright chain for underlying publisher articles. The base model retains its MIT terms. This project's public artifacts contain no news article text. Dataset citation: Israel Abebe Azime and Nebil Mohammed, [“An Amharic News Text classification Dataset” (2021)](https://arxiv.org/abs/2103.05639).
'''


def build(run_dir: Path, data_dir: Path, release_dir: Path) -> None:
    if release_dir.exists():
        raise FileExistsError(f"Release already exists: {release_dir}; choose a new output path")
    results = json.loads((run_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "transformer_training.json").read_text(encoding="utf-8"))
    audit = json.loads((data_dir / "data_audit.json").read_text(encoding="utf-8"))
    release_dir.mkdir(parents=True, exist_ok=False)
    model_dir = release_dir / "model"
    shutil.copytree(run_dir / "transformer", model_dir)
    shutil.copy2(ROOT / "LICENSE", model_dir / "LICENSE-CODE")
    config_path = model_dir / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config["text_classification_max_length"] = training["max_length_profile"][
        "selected_max_length"
    ]
    config_path.write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
    (model_dir / "README.md").write_text(_model_card(results, audit), encoding="utf-8")
    (model_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "transformer_test": results["transformer"]["test_metrics"],
                "transformer_validation": results["transformer"]["validation_metrics"],
                "baselines": results.get("baselines"),
                "selected_overall_by_validation_macro_f1": results[
                    "selected_overall_by_validation_macro_f1"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    public_config = {
        key: value
        for key, value in training.items()
        if key not in {"device", "elapsed_seconds", "history"}
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(public_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_package = model_dir / "src" / "amharic_news"
    source_package.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "inference.py", "labels.py", "text.py"):
        source = ROOT / "src" / "amharic_news" / name
        if source.exists():
            shutil.copy2(source, source_package / name)
    (model_dir / "requirements.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "torch==2.15.0+cpu\ntransformers==4.57.6\n"
        "huggingface-hub==0.36.2\nsafetensors==0.8.0\n",
        encoding="utf-8",
    )
    error_analysis = ROOT / "reports" / "error_analysis.md"
    if error_analysis.exists():
        shutil.copy2(error_analysis, model_dir / "error_analysis.md")
    space_dir = release_dir / "space"
    shutil.copytree(
        ROOT / "space", space_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(ROOT / "LICENSE", space_dir / "LICENSE-CODE")
    (space_dir / "src" / "amharic_news").mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "inference.py", "labels.py", "text.py"):
        shutil.copy2(
            ROOT / "src" / "amharic_news" / name, space_dir / "src" / "amharic_news" / name
        )
    (space_dir / "evaluation_metrics.json").write_text(
        json.dumps(results["transformer"]["test_metrics"], indent=2) + "\n", encoding="utf-8"
    )
    print(f"Release prepared: {release_dir}")
    print(f"Model folder: {model_dir}")
    print(f"Space folder: {space_dir}")
    print("No raw data or per-article test records were copied.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-dataset-text model and Space release")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("release/project04-v1"))
    args = parser.parse_args()
    build(args.run_dir, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
