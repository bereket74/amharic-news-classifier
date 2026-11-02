# Dataset audit

## Source and scope

Training uses the public [Amharic News Text Classification dataset](https://huggingface.co/datasets/israel/Amharic-News-Text-classification-Dataset) at revision `b646090ef0d09981da9c9765c4d376b407aa5955`. Its Hub card declares CC BY 4.0 and credits Israel Abebe Azime and Nebil Mohammed. Only the headline and article body are model inputs; date, publisher URL/domain, view count, and source row index are excluded. Raw source text is not committed or published.

The source has six classes: business, entertainment, national news, politics, sports, and world news. The file-level audit and SHA-256 checksums are in `data_audit.json`; no headline or article text appears in that file.

## Cleaning and leakage checks

| Rule | Training rows | Test rows |
|---|---:|---:|
| Source rows | 41,186 | 10,297 |
| Missing/blank label removed | 1 | 0 |
| Ethiopic letters below the 10% threshold removed | 380 | 92 |
| Exact-text groups with conflicting labels removed | 185 | 8 |
| Repeated exact text with the same label collapsed | 425 | 18 |
| Exact train/test overlaps removed from training | 258 | — |

There were 47 exact train/test text overlaps whose labels disagreed; the training-side copy was removed, and the held-out test label was retained. The 10% Ethiopic-letter rule is a script heuristic, not a language identifier, so code-mixed and wrongly labelled examples may remain. Near-duplicate stories, publisher-level leakage, and the underlying articles' copyright chain were not fully adjudicated.

The cleaned full training partition has 39,937 rows before model-training sampling. It is split once, stratified by class, into 31,949 train and 7,988 validation rows (seed 42). The de-duplicated eligible test partition has 10,179 rows. Validation is used for model selection; the full test split is reserved for the final comparison.

## Final clean split counts

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| Business | 2,457 | 614 | 781 |
| Entertainment | 399 | 100 | 135 |
| National news | 13,143 | 3,286 | 4,083 |
| Politics | 5,925 | 1,481 | 1,848 |
| Sports | 6,328 | 1,582 | 2,024 |
| World news | 3,697 | 925 | 1,308 |
| **Total** | **31,949** | **7,988** | **10,179** |

Entertainment is the smallest class, so its per-class estimates are less precise than those for national news. The dataset license declaration is recorded as supplied by the Hub publisher; this project does not claim that every underlying publisher article was independently cleared. Only model weights, aggregate metrics, and synthetic demo examples are planned for public release.
