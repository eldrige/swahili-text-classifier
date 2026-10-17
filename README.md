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

The app binds to `0.0.0.0` and uses `PORT` (default `7860`). Checkpoints (`swahili_baseline.joblib`, `swahili_bigru.pt`, `swahili_afroxlmr/`) are gitignored because of size.

## Deploy on DigitalOcean App Platform

This path uses the TF-IDF baseline so a 512 MB instance is enough (no PyTorch). Train or copy `swahili_baseline.joblib` next to `app.py` first.

1. Force-add the checkpoint (it is gitignored):

```bash
git add -f swahili_baseline.joblib Dockerfile requirements-do.txt app.py
git commit -m "Add DigitalOcean baseline deploy"
git push
```

2. In DigitalOcean: **Create** → **Apps** → GitHub repo → Dockerfile.
3. HTTP port `8080`, smallest instance (`apps-s-1vcpu-0.5gb`). Optional env: `SWAHILI_MODEL=baseline`.
4. Open the `*.ondigitalocean.app` URL after the build.

See `.do/app.yaml` for a sample app spec. BiGRU / AfroXLMR need PyTorch and a larger droplet or App Platform instance; they are not in the default `Dockerfile`.
