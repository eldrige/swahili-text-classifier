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

## Layout

```
app.py                  Gradio demo (loads a checkpoint from models/)
notebooks/
  01_baseline_tfidf.ipynb   TF-IDF + logistic regression
  02_bigru_fasttext.ipynb   fastText Swahili vectors + BiGRU
  03_afroxlmr.ipynb         AfroXLMR-base fine-tune (run on a Colab T4)
  04_comparison.ipynb       accuracy, macro-F1, time, size
models/
  swahili_baseline.joblib
  swahili_bigru.pt
  swahili_afroxlmr/         Hugging Face checkpoint (gitignored, ~1.1 GB)
results/
  baseline/ bigru/ afroxlmr/  metrics, training history, error CSVs per model
  comparison.csv
embeddings/             fastText cc.sw.300 vectors (gitignored, downloaded by notebook 02)
```

Run the notebooks in order from `notebooks/` (or upload one to Colab). They write checkpoints to `models/` and metrics to `results/<model>/`. The `*_errors.csv` files are gitignored.

`app.py` loads the strongest checkpoint that is present: `models/swahili_afroxlmr/`, then `models/swahili_bigru.pt`, then `models/swahili_baseline.joblib`. Override with `SWAHILI_MODEL=bigru` (or `baseline` / `afroxlmr`).

## Local demo

```bash
python -m pip install -r requirements.txt
python app.py
```

The app binds to `0.0.0.0` and uses `PORT` (default `7860`). The baseline and BiGRU checkpoints are committed; `models/swahili_afroxlmr/` is gitignored because of size. Copy the folder saved by notebook 03 there to use it.

## Deploy on DigitalOcean App Platform

This path uses the TF-IDF baseline so a 512 MB instance is enough. The `Dockerfile` copies only `app.py` and `models/swahili_baseline.joblib`.

1. Push the repo to GitHub.
2. In DigitalOcean: **Create** → **Apps** → GitHub repo → Dockerfile.
3. HTTP port `8080`, smallest instance (`apps-s-1vcpu-0.5gb`). Optional env: `SWAHILI_MODEL=baseline`.
4. Open the `*.ondigitalocean.app` URL after the build.

See `.do/app.yaml` for a sample app spec. BiGRU / AfroXLMR need PyTorch and a larger droplet or App Platform instance; they are not in the default `Dockerfile`.
