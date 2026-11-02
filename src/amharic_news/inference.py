from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import (
    ID_TO_LABEL,
    LABEL_AMHARIC,
    LABEL_DESCRIPTIONS,
    MODEL_REPO,
    MODEL_REVISION,
)
from .text import join_headline_and_article


class NewsClassifier:
    def __init__(self, model_path: str | Path = MODEL_REPO, *, revision: str | None = None):
        path = str(model_path)
        pinned_revision = revision or (MODEL_REVISION if path == MODEL_REPO else None)
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=pinned_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, revision=pinned_revision
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {
            int(key): value for key, value in (self.model.config.id2label or {}).items()
        }
        if len(self.id2label) != self.model.config.num_labels:
            self.id2label = ID_TO_LABEL
        self.max_length = int(getattr(self.model.config, "text_classification_max_length", 256))
        self.max_length = min(self.max_length, 256)

    @torch.inference_mode()
    def predict(self, headline: str, article: str = "", *, top_k: int = 3) -> list[dict[str, Any]]:
        text = join_headline_and_article(headline, article)
        if not text:
            raise ValueError("Enter an Amharic headline or article body.")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        probabilities = torch.softmax(self.model(**inputs).logits[0], dim=-1).cpu().tolist()
        ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)[
            : max(1, min(top_k, len(probabilities)))
        ]
        return [
            {
                "label": self.id2label[index],
                "amharic_label": LABEL_AMHARIC.get(self.id2label[index], ""),
                "description": LABEL_DESCRIPTIONS.get(self.id2label[index], self.id2label[index]),
                "score": float(score),
            }
            for index, score in ranked
        ]
