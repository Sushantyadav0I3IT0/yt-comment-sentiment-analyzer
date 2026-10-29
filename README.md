# YouTube Comment Sentiment Analysis

This project trains a three-class comment sentiment model, logs evaluation
results to MLflow, and exposes predictions and visualization endpoints through
a Flask API. Sentiment labels are `-1` (negative), `0` (neutral), and `1`
(positive).

## Run locally

Use Python 3.10, then install the project dependencies and NLTK data:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```

Train the model and create the local MLflow registry:

```bash
dvc repro
```

The ingestion stage downloads the configured CSV dataset. MLflow uses
`sqlite:///mlflow.db` by default; set `MLFLOW_TRACKING_URI` to use another
tracking server. To start the API after training:

```bash
PORT=5057 python flask_app/app.py
```

The API loads `lgbm_model.pkl` and `tfidf_vectorizer.pkl` from the repository
root. Override their locations with `MODEL_PATH` and `VECTORIZER_PATH`.
Predictions return HTTP 503 with setup guidance until both artifacts exist.
Check readiness at `GET /health`. Set `PORT` if port 5000 is already in use.

Example request:

```bash
curl -X POST http://localhost:5057/predict \
  -H 'Content-Type: application/json' \
  -d '{"comments":["This is great!","This was disappointing."]}'
```

The response contains the original comment and its predicted sentiment:

```json
[
  {"comment": "This is great!", "sentiment": "1"},
  {"comment": "This was disappointing.", "sentiment": "-1"}
]
```

Other endpoints are `POST /predict_with_timestamps`,
`POST /generate_chart`, `POST /generate_wordcloud`, and
`POST /generate_trend_graph`.

## Docker

Run `dvc repro` first so the two model artifacts exist, then build and start
the API:

```bash
docker build -t yt-comment-sentiment .
docker run --rm --detach --name yt-comment-api -p 5058:5000 yt-comment-sentiment
```

The first port is the port on your Mac; the second is the port inside the
container. If host port 5000 is free, you can use `-p 5000:5000` instead.
The detached container leaves your terminal available. Check the API at
`http://localhost:5058/health`, view logs with `docker logs yt-comment-api`,
and stop it with `docker stop yt-comment-api`.

## Install the Chrome extension

With the API container running:

1. Open `chrome://extensions` in Chrome and enable **Developer mode**.
2. Click **Load unpacked** and select the `chrome-extension` folder in this repository (not the repository root).
3. Open or reload a YouTube video and scroll until comments are loaded.
4. Click **Analyze loaded comments** in the floating panel. It will show positive/neutral/negative counts and percentages, and labels appear beside each comment. Scroll down to the comment section to see the individual labels.

The extension sends comment text only to your local API at
`http://127.0.0.1:5058`. It analyzes comments currently loaded on the page;
scroll to load more comments and click the button again. Predictions depend
on the model and can be incorrect.

## Tests

Run the API unit tests with:

```bash
python -m pytest tests
```
