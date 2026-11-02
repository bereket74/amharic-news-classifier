from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def _expected_calibration_error(
    y_true: np.ndarray, pred: np.ndarray, confidence: np.ndarray
) -> float:
    correct = (y_true == pred).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    total = len(y_true)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += (mask.sum() / total) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute a compact, reproducible classification report."""
    truth = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    if truth.size == 0 or truth.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    label_list = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=label_list, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(label_list)
    }
    macro_f1 = float(
        f1_score(truth, predicted, labels=label_list, average="macro", zero_division=0)
    )
    metrics: dict[str, Any] = {
        "n": int(truth.size),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": macro_f1,
        "weighted_f1": float(
            f1_score(truth, predicted, labels=label_list, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=label_list)
        .astype(int)
        .tolist(),
        "labels": label_list,
    }
    if bootstrap_samples > 0:
        rng = np.random.default_rng(seed)
        scores = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            sampled = rng.integers(0, truth.size, size=truth.size)
            scores[index] = f1_score(
                truth[sampled],
                predicted[sampled],
                labels=label_list,
                average="macro",
                zero_division=0,
            )
        low, high = np.percentile(scores, [2.5, 97.5])
        metrics["macro_f1_95pct_bootstrap_ci"] = [float(low), float(high)]
        metrics["bootstrap_samples"] = int(bootstrap_samples)
        metrics["bootstrap_seed"] = int(seed)
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (truth.size, len(label_list)):
            raise ValueError("probabilities must have shape (n_samples, n_labels)")
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        pred_indices = np.asarray([label_list.index(label) for label in predicted], dtype=int)
        true_indices = np.asarray([label_list.index(label) for label in truth], dtype=int)
        confidence = probs.max(axis=1)
        top_k = min(3, len(label_list))
        top_indices = np.argsort(probs, axis=1)[:, -top_k:]
        metrics["probability_diagnostics"] = {
            "mean_max_probability": float(confidence.mean()),
            "expected_calibration_error_10_bins": _expected_calibration_error(
                true_indices, pred_indices, confidence
            ),
            "log_loss": float(log_loss(truth, probs, labels=label_list)),
            "top_3_accuracy": float(np.mean((top_indices == true_indices[:, None]).any(axis=1))),
            "warning": "Scores are softmax outputs and are not calibrated probabilities.",
        }
    return metrics


def summarize_errors(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidence: Sequence[float],
    word_lengths: Sequence[int],
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    """Return aggregate confusion, length, and confidence patterns; never text examples."""
    truth = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    conf = np.asarray(confidence, dtype=float)
    lengths = np.asarray(word_lengths, dtype=int)
    if not (truth.size == predicted.size == conf.size == lengths.size):
        raise ValueError("All error-analysis inputs must have matching lengths")
    errors = truth != predicted
    confusion_counts = Counter(
        (str(gold), str(pred)) for gold, pred in zip(truth, predicted, strict=True) if gold != pred
    )
    top_confusions = [
        {"gold": gold, "predicted": pred, "count": int(count)}
        for (gold, pred), count in confusion_counts.most_common(12)
    ]
    quantiles = np.quantile(lengths, [0.25, 0.50, 0.75]).tolist() if lengths.size else [0, 0, 0]
    boundaries = [-1, *quantiles, float("inf")]
    length_bins = []
    names = ("shortest_quartile", "second_quartile", "third_quartile", "longest_quartile")
    for name, low, high in zip(names, boundaries[:-1], boundaries[1:], strict=True):
        mask = (lengths > low) & (lengths <= high)
        if mask.any():
            length_bins.append(
                {
                    "bin": name,
                    "min_words_exclusive": float(low),
                    "max_words_inclusive": None if np.isinf(high) else float(high),
                    "n": int(mask.sum()),
                    "error_rate": float(errors[mask].mean()),
                }
            )
    confidence_edges = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
    confidence_bins = []
    for lower, upper in zip(confidence_edges[:-1], confidence_edges[1:], strict=True):
        mask = (conf >= lower) & (conf < upper)
        if mask.any():
            confidence_bins.append(
                {
                    "range": f"[{lower:.2f}, {min(upper, 1.0):.2f}{']' if upper > 1 else ')'}",
                    "n": int(mask.sum()),
                    "error_rate": float(errors[mask].mean()),
                    "mean_confidence": float(conf[mask].mean()),
                }
            )
    return {
        "n": int(truth.size),
        "errors": int(errors.sum()),
        "error_rate": float(errors.mean()) if errors.size else 0.0,
        "top_confusions": top_confusions,
        "length_bins": length_bins,
        "confidence_bins": confidence_bins,
        "label_error_rates": {
            label: {
                "n": int((truth == label).sum()),
                "error_rate": float(errors[truth == label].mean())
                if np.any(truth == label)
                else None,
            }
            for label in labels
        },
        "privacy_note": "No article text or headline is included in this aggregate report.",
    }
