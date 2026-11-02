# Held-out error analysis

The transformer made **4,526 errors** in **10,179** eligible held-out articles (error rate 44.5%). The report is aggregate-only; it contains no headlines or article text.

## Most frequent confusions

| Gold category | Predicted category | Count |
|---|---|---:|
| `national_news` | `politics` | 2,304 |
| `national_news` | `business` | 875 |
| `national_news` | `world_news` | 351 |
| `politics` | `business` | 291 |
| `politics` | `world_news` | 171 |
| `world_news` | `politics` | 137 |
| `national_news` | `entertainment` | 87 |
| `business` | `politics` | 68 |
| `world_news` | `business` | 43 |
| `national_news` | `sports` | 41 |
| `world_news` | `sports` | 25 |
| `world_news` | `entertainment` | 18 |

## Error rate by gold category

| Category | Test examples | Error rate |
|---|---:|---:|
| `business` | 781 | 14.0% |
| `entertainment` | 135 | 11.9% |
| `national_news` | 4,083 | 89.6% |
| `politics` | 1,848 | 26.0% |
| `sports` | 2,024 | 1.3% |
| `world_news` | 1,308 | 18.0% |

## Error rate by article length

| Length cohort | Examples | Error rate |
|---|---:|---:|
| shortest quartile | 2,578 | 46.4% |
| second quartile | 2,517 | 42.0% |
| third quartile | 2,552 | 47.5% |
| longest quartile | 2,532 | 41.9% |

## Error rate by model score

| Maximum softmax score | Examples | Error rate |
|---|---:|---:|
| [0.00, 0.50) | 7,390 | 56.9% |
| [0.50, 0.60) | 746 | 35.7% |
| [0.60, 0.70) | 162 | 17.9% |
| [0.70, 0.80) | 1,881 | 1.5% |

## Interpretation

These are measurable error patterns, not explanations of individual mistakes. Category boundaries overlap in real reporting, and the small entertainment class has less evaluation support than the larger classes. The model's maximum softmax score is not calibrated confidence. Length cohorts use the processed text's whitespace-delimited word count; long articles are truncated using a limit selected from training data only.

Publisher/source-level separation, near-duplicate detection, and external-domain validation remain future work. No article examples are copied into this report because the source corpus contains publisher text.
