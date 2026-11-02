# Data and model provenance

## News corpus

This project uses the public Hugging Face dataset [`israel/Amharic-News-Text-classification-Dataset`](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset), pinned at revision `b646090ef0d09981da9c9765c4d376b407aa5955`. Its dataset card declares **CC BY 4.0** and attributes the corpus to Israel Abebe Azime and Nebil Mohammed. The associated paper introduces more than 50,000 news articles in six categories: [Azime & Mohammed (2021), “An Amharic News Text classification Dataset”](https://arxiv.org/abs/2103.05639).

At the pinned [Hub dataset revision](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset/tree/b646090ef0d09981da9c9765c4d376b407aa5955), the official LFS SHA-256 values are `train.csv` = `446bab6e0380dd3b131115186aa87168ebe17a07c92b8290543dfc6ee48fa529` and `test.csv` = `7deb14a8d01655b80e77d66ef495b1bf123b91ca450eacb3e470ed9ba220de73`. The downloader verifies both cached and newly downloaded files against those immutable values before processing.

The Hub release exposes `headline`, `article`, `category`, date, views, and a source URL. The model uses only the headline and article body. Date, views, URL, source domain, and row index are excluded. The raw corpus is downloaded from its pinned revision for local use and is not copied into this repository, the model package, evaluation reports, or demo. Empty labels and low-Ethiopic-share rows are removed; within-partition exact text duplicates are collapsed; exact text groups with conflicting labels are excluded; and exact train records that overlap retained test records are removed from training. The eligible test split is fixed before training and is not used for tuning or checkpoint selection.

The card's CC BY 4.0 declaration is recorded as supplied by the dataset publisher. This project has not independently established the copyright chain for every underlying publisher article; it therefore does not republish the source articles or claim that every source article is separately licensed. The 10% Ethiopic-letter threshold is a script heuristic, not proof of language identity. Near-duplicate and publisher-level leakage are not fully audited. `reports/data_audit.json` contains aggregate counts and source-file checksums without article text.

## Base transformer

The transformer is initialized from [`Davlan/afro-xlmr-mini`](https://huggingface.co/Davlan/afro-xlmr-mini), revision `bc04038b969667884bd83cdd37bed9559c2c3d9c`. The model card identifies AfroXLMR-mini as multilingual-adapted for 17 African languages including Amharic, and the Hub metadata declares **MIT** licensing. Its paper is [Alabi et al. (2022), “Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning”](https://aclanthology.org/2022.coling-1.382/).

## Rights and attribution

`LICENSE` applies only to original project code. The upstream dataset and base-model terms remain separate. The already-public fine-tuned model is labeled CC BY 4.0 with attribution to the dataset authors, based on the dataset repository's declared license; the underlying article-rights chain remains unverified. No dataset split, headline, article text, or copied article excerpt is included in public artifacts. This task did not publish additional model weights or a converted browser model while that rights question remains unresolved.

## Hugging Face demo hosting

Hugging Face's [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) says Static HTML Spaces are free for everyone, while Gradio and Docker Spaces require a paid plan except for eligible free ZeroGPU hosting. The [ZeroGPU guide](https://huggingface.co/docs/hub/en/spaces-zerogpu) describes eligibility and usage quotas. A Project 4 ZeroGPU creation attempt using the account's project-specific write credential was rejected with HTTP 402 and the message that the account is not eligible to host ZeroGPU Spaces for free. No paid hardware or credits were enabled. Static hosting is technically available without compute, but this Project 4 Space has not been published because a usable browser demo requires a redistributable model artifact and the underlying source-article rights have not been resolved.
