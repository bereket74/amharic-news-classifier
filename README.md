# Amharic News Classifier

A reproducible six-category Amharic news classification project: audited dataset processing, classical baselines, AfroXLMR-mini fine-tuning, held-out evaluation, aggregate error analysis, and a bilingual Gradio demo.

**Source:** This is a local-only Git repository; no GitHub remote or push was configured.  
**Hugging Face model:** [`klohdfhk/amharic-news-classifier`](https://huggingface.co/klohdfhk/amharic-news-classifier)  
**Hugging Face Space:** Not published: Hugging Face returned HTTP 402 for CPU Basic and free ZeroGPU creation. The local Gradio app is included; no paid hardware was attempted.

## What it predicts

The model ranks six news topics: business, entertainment, national news, politics, sports, and world news. The demo shows both English and Amharic category names. Its scores are raw softmax outputs, not calibrated probabilities.

[Pipeline diagram (PNG)](docs/architecture.png) · [Editable Mermaid source](docs/architecture.mmd)

## Dataset and preprocessing

The source is the public [`israel/Amharic-News-Text-classification-Dataset`](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset), pinned to revision `b646090ef0d09981da9c9765c4d376b407aa5955`. Its Hub card declares CC BY 4.0 and cites Israel Abebe Azime and Nebil Mohammed. The project uses headline and article body only; source URL/domain, date, views, and source row index are excluded.

Processing NFC-normalizes text, removes zero-width controls, collapses whitespace, screens for at least 10% Ethiopic letters, drops blank labels and conflicting exact-text duplicates, collapses same-label exact duplicates, and removes exact train/test overlaps from training. The 10% script cutoff is a heuristic, not a language detector. The full clean partitions contain **31,949 train**, **7,988 validation**, and **10,179 test** examples. For bounded CPU training, a fixed-seed sample of up to **1,500 training rows per class** was used (**7,899 total**); the validation and test partitions remain complete. Exact audit counts and per-class split sizes are in [`reports/data_audit.md`](reports/data_audit.md) and [`reports/data_audit.json`](reports/data_audit.json).

Raw news text and per-article predictions are excluded from Git and from the Hugging Face releases. The publisher-declared dataset license is recorded as supplied; this project has not independently cleared the underlying publishers' copyright chain. See [`docs/sources.md`](docs/sources.md).

## Models and evaluation

Classical baselines are a majority classifier, word TF–IDF with LinearSVC, character TF–IDF with LinearSVC, and combined word/character TF–IDF with LinearSVC. They use fixed `C=1.0`. The AfroXLMR-mini transformer uses class-weighted cross-entropy and a 192-token cap. Validation macro-F1 selects the best baseline and the transformer's checkpoint. The full eligible test split remains out of model selection; each candidate is evaluated once on that test split after validation-based selection.

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| Majority | 0.024 | 0.024 | 0.077 |
| Word TF–IDF + LinearSVC | 0.768 | 0.774 | 0.797 |
| Character TF–IDF + LinearSVC | 0.773 | 0.781 | 0.802 |
| Word + character TF–IDF + LinearSVC | **0.786** | **0.784** | **0.805** |
| AfroXLMR-mini | 0.566 | 0.581 (95% bootstrap CI 0.569–0.593) | 0.555 |

The combined TF–IDF baseline is the best overall model in this benchmark; the published Hugging Face transformer is included because transformer fine-tuning and comparison are project goals, not because it won. The transformer test balanced accuracy is 0.732 and weighted-F1 is 0.496. Its error analysis found an **89.6% error rate on national-news test items**, most often predicting politics (2,304 items) or business (875). **87.5% of sampled training texts exceeded the 192-token cap**, so most transformer inputs were truncated; this is a major limitation, though it does not by itself establish the cause of each error. Treat this checkpoint as a research artifact, not a production classifier.

The transformer report includes per-class precision/recall/F1, confusion counts, calibration summaries, and a fixed-seed bootstrap interval for test macro-F1. [`reports/error_analysis.md`](reports/error_analysis.md) aggregates errors by class, confidence, and text length without printing headlines or article excerpts.

## Reproduce

Use Python 3.12 and the pinned versions in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/prepare_data.py --script-share 0.10
python scripts/train.py \
  --seed 42 --epochs 2 --patience 1 --batch-size 8 \
  --grad-accumulation 2 --max-length-cap 192 --max-train-per-class 1500
python scripts/error_analysis.py
python scripts/build_release.py --output-dir release/project04-v1
```

`prepare_data.py` downloads only the pinned public dataset revision and stores raw/processed data in Git-ignored folders. Training writes its benchmark artifacts under `runs/benchmark/`. If a GPU is available the transformer uses it; otherwise it runs on CPU.

To serve a locally trained model with Gradio:

```bash
MODEL_ID=runs/benchmark/transformer python app.py
```

## Limitations and intended use

This is a research and learning demo for exploratory topic triage, not a replacement for editorial judgment or a tool for high-impact decisions or political targeting. News topics overlap; language, code-switching, publishers, and time periods shift; exact duplicates do not catch every syndicated or near-duplicate story; and a random split does not establish source- or time-generalization. The entertainment category is much smaller than national news. Long articles are truncated, and softmax scores are not calibrated confidence. The transformer substantially trails the fixed-C TF–IDF baselines on this split and should not be used as the preferred classifier without further development and evaluation.

## License and attribution

The root `LICENSE` applies to original project code only. The dataset and base-model terms remain separate. The fine-tuned model is labeled CC BY 4.0 with dataset-author attribution, consistent with the upstream dataset card's declaration; this project does not claim independent clearance of every source article. No corpus rows or text excerpts are redistributed.
