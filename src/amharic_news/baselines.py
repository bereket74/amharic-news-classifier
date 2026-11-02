from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


@dataclass
class BaselineCandidate:
    name: str
    estimator: Any
    validation_macro_f1: float
    selected_c: float | None = None


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        strip_accents=None,
        dtype=np.float32,
    )


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _build(name: str, c: float) -> Pipeline:
    if name == "word_tfidf_linear_svc":
        vectorizer: Any = _word_vectorizer()
    elif name == "char_tfidf_linear_svc":
        vectorizer = _char_vectorizer()
    elif name == "word_char_tfidf_linear_svc":
        vectorizer = FeatureUnion(
            [("word", _word_vectorizer()), ("char", _char_vectorizer())],
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", LinearSVC(C=c, class_weight="balanced", random_state=42)),
        ]
    )


def select_baselines(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    validation_texts: Sequence[str],
    validation_labels: Sequence[str],
    *,
    c_values: Sequence[float] = (1.0,),
) -> tuple[dict[str, BaselineCandidate], dict[str, Any]]:
    """Fit a fixed-C baseline set; compare models using validation macro-F1."""
    candidates: dict[str, BaselineCandidate] = {}
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train_labels), 1)), train_labels)
    majority_score = f1_score(
        validation_labels,
        majority.predict(np.zeros((len(validation_labels), 1))),
        average="macro",
        zero_division=0,
    )
    candidates["majority"] = BaselineCandidate("majority", majority, float(majority_score))

    validation_grid: dict[str, Any] = {}
    for name in (
        "word_tfidf_linear_svc",
        "char_tfidf_linear_svc",
        "word_char_tfidf_linear_svc",
    ):
        best: BaselineCandidate | None = None
        trials = []
        for c in c_values:
            estimator = _build(name, float(c))
            estimator.fit(train_texts, train_labels)
            prediction = estimator.predict(validation_texts)
            score = float(f1_score(validation_labels, prediction, average="macro", zero_division=0))
            trials.append({"C": float(c), "validation_macro_f1": score})
            if best is None or score > best.validation_macro_f1:
                best = BaselineCandidate(name, estimator, score, float(c))
        assert best is not None
        candidates[name] = best
        validation_grid[name] = {"trials": trials, "selected_C": best.selected_c}
    return candidates, validation_grid
