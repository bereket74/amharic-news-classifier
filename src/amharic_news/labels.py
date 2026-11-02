from __future__ import annotations

MODEL_REPO = "Davlan/afro-xlmr-mini"
MODEL_REVISION = "bc04038b969667884bd83cdd37bed9559c2c3d9c"
PUBLISHED_MODEL_ID = "klohdfhk/amharic-news-classifier"
PUBLISHED_SPACE_ID = "klohdfhk/amharic-news-classifier-demo"

LABEL_NAMES = {
    "ቢዝነስ": "business",
    "መዝናኛ": "entertainment",
    "ሀገር አቀፍ ዜና": "national_news",
    "ፖለቲካ": "politics",
    "ስፖርት": "sports",
    "ዓለም አቀፍ ዜና": "world_news",
}
LABEL_AMHARIC = {english: amharic for amharic, english in LABEL_NAMES.items()}
LABEL_ORDER = tuple(sorted(LABEL_NAMES.values()))
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LABEL_DESCRIPTIONS = {
    "business": "Business and economy",
    "entertainment": "Entertainment and culture",
    "national_news": "National and domestic news",
    "politics": "Politics and government",
    "sports": "Sports",
    "world_news": "International news",
}
