from __future__ import annotations

from amharic_news.labels import ID_TO_LABEL, LABEL_AMHARIC, LABEL_ORDER
from amharic_news.text import join_headline_and_article, normalize_inference_text


def test_inference_text_normalization_preserves_ethiopic_script() -> None:
    assert normalize_inference_text("\ufeff  ዜና\u200b  ነው  ") == "ዜና ነው"
    assert join_headline_and_article("ርዕስ", " ይዘት ") == "ርዕስ\nይዘት"


def test_bilingual_labels_have_stable_six_class_mapping() -> None:
    assert len(LABEL_ORDER) == 6
    assert set(ID_TO_LABEL.values()) == set(LABEL_AMHARIC)
    assert LABEL_AMHARIC["sports"] == "ስፖርት"
