from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .baselines import select_baselines
from .data import LABEL_DESCRIPTIONS, LABEL_NAMES, MODEL_REPO, MODEL_REVISION, load_processed
from .metrics import classification_metrics

LABELS = sorted(set(LABEL_NAMES.values()))
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _encode_lengths(tokenizer: Any, texts: Sequence[str], batch_size: int = 256) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        lengths.extend(len(item) for item in batch)
    return lengths


def train_only_length_profile(
    tokenizer: Any, train_texts: Sequence[str], cap: int
) -> dict[str, Any]:
    lengths = np.asarray(_encode_lengths(tokenizer, train_texts), dtype=np.int32)
    p95 = int(math.ceil(float(np.percentile(lengths, 95))))
    selected = min(cap, max(64, int(math.ceil(p95 / 32) * 32)))
    return {
        "measured_on": "training split only",
        "examples": int(len(lengths)),
        "p50_subwords": float(np.percentile(lengths, 50)),
        "p95_subwords": float(np.percentile(lengths, 95)),
        "p99_subwords": float(np.percentile(lengths, 99)),
        "selected_max_length": int(selected),
        "cap": int(cap),
        "training_fraction_truncated_at_selected_length": float(np.mean(lengths > selected)),
    }


class EncodedNewsDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, texts: Sequence[str], labels: Sequence[str], tokenizer: Any, max_length: int
    ):
        self.labels = [LABEL_TO_ID[label] for label in labels]
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def _model_loader(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray]:
    model.eval()
    predictions: list[str] = []
    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels", None)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probability_batches.append(probs)
            predictions.extend(ID_TO_LABEL[int(i)] for i in probs.argmax(axis=1))
            if labels is not None:
                del labels
    return predictions, np.concatenate(probability_batches, axis=0)


def fit_transformer(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int = 42,
    batch_size: int = 8,
    grad_accumulation: int = 2,
    max_epochs: int = 4,
    patience: int = 2,
    learning_rate: float = 2e-5,
    max_length_cap: int = 256,
    num_workers: int = 0,
) -> dict[str, Any]:
    set_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION, use_fast=True)
    train_texts = train_frame["text"].tolist()
    profile = train_only_length_profile(tokenizer, train_texts, max_length_cap)
    max_length = profile["selected_max_length"]

    train_data = EncodedNewsDataset(
        train_texts, train_frame["label"].tolist(), tokenizer, max_length
    )
    validation_data = EncodedNewsDataset(
        validation_frame["text"].tolist(), validation_frame["label"].tolist(), tokenizer, max_length
    )
    test_data = EncodedNewsDataset(
        test_frame["text"].tolist(), test_frame["label"].tolist(), tokenizer, max_length
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    label_counts = train_frame["label"].value_counts().to_dict()
    weights = torch.tensor(
        [
            len(train_frame) / (len(LABELS) * max(1, int(label_counts.get(label, 0))))
            for label in LABELS
        ],
        dtype=torch.float32,
        device=device,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / grad_accumulation)
    total_updates = updates_per_epoch * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_updates * 0.10), num_training_steps=total_updates
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, labels, weight=weights)
            epoch_loss += float(loss.detach().cpu())
            (loss / grad_accumulation).backward()
            if step % grad_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 250 == 0:
                print(f"epoch={epoch}/{max_epochs} batch={step}/{len(train_loader)}", flush=True)

        val_predictions, _ = _model_loader(model, validation_loader, device)
        val_score = float(
            f1_score(
                validation_frame["label"].tolist(),
                val_predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )
        record = {
            "epoch": epoch,
            "training_loss_per_batch_mean": epoch_loss / max(1, len(train_loader)),
            "validation_macro_f1": val_score,
            "learning_rate_next": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Early stopping after epoch {epoch}; validation macro-F1 did not improve.",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Transformer training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_predictions, test_probabilities = _model_loader(model, test_loader, device)
    validation_predictions, validation_probabilities = _model_loader(
        model, validation_loader, device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.id2label = ID_TO_LABEL
    model.config.label2id = LABEL_TO_ID
    model.config.text_classification_max_length = max_length
    model.save_pretrained(output_dir / "transformer", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "transformer")

    test_metric = classification_metrics(
        test_frame["label"].tolist(),
        test_predictions,
        labels=LABELS,
        probabilities=test_probabilities,
        bootstrap_samples=2000,
        seed=seed,
    )
    val_metric = classification_metrics(
        validation_frame["label"].tolist(),
        validation_predictions,
        labels=LABELS,
        probabilities=validation_probabilities,
        bootstrap_samples=0,
    )
    prediction_frame = pd.DataFrame(
        {
            "row_id": test_frame["row_id"].tolist(),
            "y_true": test_frame["label"].tolist(),
            "y_pred": test_predictions,
            "max_softmax_score": test_probabilities.max(axis=1),
            "word_length": [len(text.split()) for text in test_frame["text"].tolist()],
        }
    )
    prediction_frame.to_csv(output_dir / "transformer_test_predictions.csv", index=False)
    config = {
        "base_model": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "seed": seed,
        "selection_metric": "validation macro-F1",
        "selected_epoch": best_epoch,
        "selected_validation_macro_f1": best_score,
        "training_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "class_weighting": "inverse class frequency on training split",
        "max_length_profile": profile,
        "device": str(device),
        "elapsed_seconds": float(time.time() - started),
        "history": history,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "label_descriptions": LABEL_DESCRIPTIONS,
        "test_metrics": test_metric,
        "validation_metrics": val_metric,
    }
    (output_dir / "transformer_training.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config


def run_baselines(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models, grid = select_baselines(
        train_frame["text"].tolist(),
        train_frame["label"].tolist(),
        validation_frame["text"].tolist(),
        validation_frame["label"].tolist(),
    )
    results: dict[str, Any] = {"validation_grid": grid, "models": {}}
    for name, candidate in models.items():
        validation_predictions = (
            candidate.estimator.predict(np.zeros((len(validation_frame), 1)))
            if name == "majority"
            else candidate.estimator.predict(validation_frame["text"].tolist())
        )
        test_predictions = (
            candidate.estimator.predict(np.zeros((len(test_frame), 1)))
            if name == "majority"
            else candidate.estimator.predict(test_frame["text"].tolist())
        )
        validation_metrics = classification_metrics(
            validation_frame["label"].tolist(),
            validation_predictions,
            labels=LABELS,
            bootstrap_samples=0,
        )
        test_metrics = classification_metrics(
            test_frame["label"].tolist(),
            test_predictions,
            labels=LABELS,
            bootstrap_samples=0,
            seed=seed,
        )
        results["models"][name] = {
            "selected_C": candidate.selected_c,
            "validation_macro_f1": candidate.validation_macro_f1,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }
        pd.DataFrame(
            {
                "row_id": test_frame["row_id"].tolist(),
                "y_true": test_frame["label"].tolist(),
                "y_pred": test_predictions,
            }
        ).to_csv(output_dir / f"{name}_test_predictions.csv", index=False)
        print(
            f"baseline {name}: val_macro_f1={candidate.validation_macro_f1:.4f}; "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}",
            flush=True,
        )
    selected_name = max(
        results["models"], key=lambda name: results["models"][name]["validation_macro_f1"]
    )
    results["selected_baseline_by_validation_macro_f1"] = selected_name
    return results


def _cap_training_rows(
    frame: pd.DataFrame, max_per_class: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = {str(label): int(count) for label, count in frame["label"].value_counts().items()}
    if max_per_class <= 0:
        sampled = frame.copy()
    else:
        parts = [
            group.sample(n=min(max_per_class, len(group)), random_state=seed)
            for _, group in frame.groupby("label", sort=True)
        ]
        sampled = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    sampled = sampled.reset_index(drop=True)
    after = {str(label): int(count) for label, count in sampled["label"].value_counts().items()}
    return sampled, {
        "rule": (
            "fixed seed; sample up to the configured maximum per category "
            "from cleaned training only"
        ),
        "seed": int(seed),
        "max_per_class": int(max_per_class),
        "full_clean_training_rows": int(len(frame)),
        "used_training_rows": int(len(sampled)),
        "full_label_counts": before,
        "used_label_counts": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare Amharic news classifiers")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length-cap", type=int, default=192)
    parser.add_argument("--max-train-per-class", type=int, default=1500)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    train_frame = load_processed(args.data_dir / "train.csv")
    validation_frame = load_processed(args.data_dir / "validation.csv")
    test_frame = load_processed(args.data_dir / "test.csv")
    train_frame, sample_info = _cap_training_rows(train_frame, args.max_train_per_class, args.seed)
    audit_path = args.data_dir / "data_audit.json"
    data_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    baseline_results = None
    if not args.skip_baselines:
        baseline_results = run_baselines(
            train_frame,
            validation_frame,
            test_frame,
            args.output_dir / "baselines",
            seed=args.seed,
        )
    transformer_results = fit_transformer(
        train_frame,
        validation_frame,
        test_frame,
        args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        max_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        max_length_cap=args.max_length_cap,
    )
    result = {
        "seed": args.seed,
        "dataset_audit": data_audit,
        "training_sample": sample_info,
        "label_names": LABEL_NAMES,
        "label_descriptions": LABEL_DESCRIPTIONS,
        "baselines": baseline_results,
        "transformer": transformer_results,
        "selected_overall_by_validation_macro_f1": max(
            [
                *(
                    [
                        ("baseline:" + name, item["validation_macro_f1"])
                        for name, item in baseline_results["models"].items()
                    ]
                    if baseline_results
                    else []
                ),
                ("transformer", transformer_results["selected_validation_macro_f1"]),
            ],
            key=lambda item: item[1],
        )[0],
        "test_set_policy": (
            "The full eligible official test split was not used for model selection, tuning, "
            "or checkpoint choice."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "benchmark_results.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Benchmark report: {report_path}")
    print(f"Selected overall: {result['selected_overall_by_validation_macro_f1']}")


if __name__ == "__main__":
    main()
