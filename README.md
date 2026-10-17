---
title: Swahili News Classifier
emoji: 📰
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Swahili news topic classification

Six-way topic classification for Swahili news (`uchumi`, `kitaifa`, `michezo`, `kimataifa`, `burudani`, `afya`) on the Hugging Face `community-datasets/swahili_news` set.

Notebooks, in order:

1. `swahili_news_baseline.ipynb` — TF-IDF + logistic regression
2. `swahili_news_bigru.ipynb` — fastText Swahili vectors + BiGRU
3. `swahili_news_afroxlmr.ipynb` — AfroXLMR-base fine-tune (run on a Colab T4)
4. `swahili_news_comparison.ipynb` — accuracy, macro-F1, time, size

`app.py` loads the strongest checkpoint that is present: `swahili_afroxlmr/`, then `swahili_bigru.pt`, then `swahili_baseline.joblib`. Override with `SWAHILI_MODEL=bigru` (or `baseline` / `afroxlmr`).

## Local demo

```bash
python -m pip install -r requirements.txt
python app.py
```

The BiGRU checkpoint is gitignored because it is 57 MB. Copy `swahili_bigru.pt` next to `app.py` before launching. For Hugging Face Spaces, upload that file (Git LFS) with `app.py` and `requirements.txt`.
