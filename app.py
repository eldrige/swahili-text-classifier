"""Swahili news topic classifier for local use and DigitalOcean App Platform.

Loads the strongest checkpoint that is present: AfroXLMR, then the BiGRU,
then the TF-IDF baseline. The text cleaner is the same one used in the notebooks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NONWORD = re.compile(r"[^a-zA-Z\u00C0-\u017F'\s]")
SPACES = re.compile(r"\s+")

GLOSS = {
    "uchumi": "economy",
    "kitaifa": "national",
    "michezo": "sports",
    "kimataifa": "international",
    "burudani": "entertainment",
    "afya": "health",
}

EXAMPLES = [
    "Timu ya taifa ya mpira wa miguu imeshinda mechi ya jana usiku",
    "Bei ya sukari na mafuta imepanda katika soko la dunia",
    "Wizara ya afya yatoa tahadhari kuhusu mlipuko wa ugonjwa",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NONWORD.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def display_name(label: str) -> str:
    gloss = GLOSS.get(label)
    return f"{label} ({gloss})" if gloss else label


class Predictor:
    def __init__(self, name: str, label_names: list[str], fn):
        self.name = name
        self.label_names = label_names
        self.fn = fn

    def __call__(self, text: str) -> dict[str, float]:
        cleaned = clean_text(text or "")
        if not cleaned:
            return {display_name(name): 0.0 for name in self.label_names}
        probs = self.fn(cleaned)
        return {display_name(name): float(probs[i]) for i, name in enumerate(self.label_names)}


def load_bigru(path: Path) -> Predictor:
    import torch
    import torch.nn as nn

    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    stoi = blob["stoi"]
    label_names = list(blob["label_names"])
    pad_idx = cfg["pad_idx"]
    unk_idx = cfg["unk_idx"]
    max_len = cfg["max_len"]
    embed_dim = cfg["embed_dim"]
    hidden_dim = cfg["hidden_dim"]
    dropout = cfg["dropout"]

    class BiGRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(stoi), embed_dim, padding_idx=pad_idx)
            self.embed_drop = nn.Dropout(dropout)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.fc_drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, len(label_names))

        def forward(self, x, lengths):
            emb = self.embed_drop(self.embedding(x))
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            return self.fc(self.fc_drop(hidden))

    model = BiGRUClassifier()
    model.load_state_dict(blob["state_dict"])
    model.eval()

    def predict(cleaned: str) -> np.ndarray:
        ids = [stoi.get(tok, unk_idx) for tok in cleaned.split()[:max_len]] or [unk_idx]
        x = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([len(ids)])
        with torch.no_grad():
            logits = model(x, lengths)
            return torch.softmax(logits, dim=-1)[0].numpy()

    return Predictor("fastText + BiGRU", label_names, predict)


def load_baseline(path: Path) -> Predictor:
    import joblib

    blob = joblib.load(path)
    pipe = blob["model"]
    label_names = list(blob["label_names"])

    def predict(cleaned: str) -> np.ndarray:
        return pipe.predict_proba([cleaned])[0]

    return Predictor("TF-IDF + logistic regression", label_names, predict)


def load_afroxlmr(path: Path) -> Predictor:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    id2label = model.config.id2label
    label_names = [id2label[i] for i in range(len(id2label))]
    max_len = int(os.environ.get("SWAHILI_MAX_LEN", "256"))

    def predict(cleaned: str) -> np.ndarray:
        enc = tokenizer(cleaned, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            return torch.softmax(logits, dim=-1)[0].numpy()

    return Predictor("AfroXLMR-base", label_names, predict)


def resolve_checkpoint() -> Predictor:
    forced = os.environ.get("SWAHILI_MODEL", "").strip().lower()
    afro = ROOT / "swahili_afroxlmr"
    bigru = ROOT / "swahili_bigru.pt"
    baseline = ROOT / "swahili_baseline.joblib"

    if forced in {"afroxlmr", "xlmr", "transformer"}:
        return load_afroxlmr(afro)
    if forced in {"bigru", "gru"}:
        return load_bigru(bigru)
    if forced in {"baseline", "tfidf"}:
        return load_baseline(baseline)

    if (afro / "config.json").exists():
        return load_afroxlmr(afro)
    if bigru.exists():
        return load_bigru(bigru)
    if baseline.exists():
        return load_baseline(baseline)
    raise FileNotFoundError(
        "No checkpoint found. Place swahili_bigru.pt, swahili_baseline.joblib, "
        "or the swahili_afroxlmr/ folder next to app.py."
    )


predictor = resolve_checkpoint()


def classify(text: str):
    return predictor(text)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=8,
        label="Swahili news text",
        placeholder="Andika au bandika habari kwa Kiswahili…",
    ),
    outputs=gr.Label(num_top_classes=6, label="Topic"),
    title="Swahili news topic classifier",
    description=(
        f"Loaded **{predictor.name}**. Six topics from the "
        "`community-datasets/swahili_news` collection: economy, national, sports, "
        "international, entertainment, and health. Cleaning matches the training notebooks."
    ),
    examples=EXAMPLES,
    allow_flagging="never",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
