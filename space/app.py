from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from amharic_news.inference import NewsClassifier  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "klohdfhk/amharic-news-classifier")
classifier = NewsClassifier(MODEL_ID)


@spaces.GPU(duration=30)
def predict(headline: str, article: str) -> pd.DataFrame:
    rows = classifier.predict(headline, article, top_k=6)
    return pd.DataFrame(
        [
            {
                "Category / ምድብ": item["description"],
                "የአማርኛ ምድብ": item["amharic_label"],
                "Model label": item["label"],
                "Softmax score (not calibrated)": item["score"],
            }
            for item in rows
        ]
    )


with gr.Blocks(title="Amharic News Classifier") as demo:
    gr.Markdown(
        "# Amharic News Classifier\n"
        "A six-topic Amharic news classifier fine-tuned from AfroXLMR-mini. "
        "Scores are softmax outputs, not calibrated probabilities."
    )
    with gr.Row():
        headline = gr.Textbox(label="Headline / ርዕስ", lines=2, placeholder="የዜናውን ርዕስ ያስገቡ")
        article = gr.Textbox(label="Article body / የዜና ይዘት", lines=8, placeholder="የዜናውን ይዘት ያስገቡ")
    submit = gr.Button("Classify / መድብ", variant="primary")
    output = gr.Dataframe(label="Ranked topic predictions", interactive=False, wrap=True)
    submit.click(predict, inputs=[headline, article], outputs=output)
    gr.Examples(
        examples=[
            [
                "የኢትዮጵያ ቡድን በአህጉራዊ ውድድር ድል ተቀዳጀ",
                "በአዲስ አበባ በተካሄደው የውድድር ጨዋታ ቡድኑ ተጋጣሚውን በሶስት ለአንድ አሸንፏል።",
            ],
            [
                "የወጪ ንግድ ገቢ እድገት አሳየ",
                "የንግድ ሚኒስቴሩ ባወጣው ሪፖርት የወጪ ንግድ ገቢ በዚህ ዓመት መጨመሩን ገልጿል።",
            ],
        ],
        inputs=[headline, article],
        label="Synthetic examples (not dataset excerpts)",
    )
    gr.Markdown(
        "**Benchmark warning:** AfroXLMR test macro-F1 was 0.581, below the combined "
        "TF–IDF baseline (0.784); national-news error rate was 89.6%. This requested "
        "transformer demo is experimental, not the preferred classifier. "
        "Publisher/domain shifts, topic overlap, code-switching, and long-article "
        "truncation can also affect results."
    )

demo.launch()
