---
title: Amharic News Classifier
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: cc-by-4.0
models:
  - klohdfhk/amharic-news-classifier
tags:
  - amharic
  - text-classification
  - news
  - afroxlmr
---

# Amharic News Classifier

Interactive demo for a six-category Amharic news-topic transformer. It uses synthetic examples only; no benchmark article text is bundled in the Space.

**Benchmark warning:** this AfroXLMR transformer achieved 0.581 test macro-F1, below the combined TF–IDF baseline (0.784). National-news error rate was 89.6%. It is an experimental model for research and learning, not the preferred classifier for this dataset or a newsroom decision system.

See the [model card](https://huggingface.co/klohdfhk/amharic-news-classifier) for dataset provenance, full evaluation, error analysis, intended use, and limitations.
