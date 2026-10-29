FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY flask_app/requirements.txt /app/requirements.txt
COPY flask_app/ /app/flask_app/

COPY lgbm_model.pkl /app/lgbm_model.pkl
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

RUN apt-get purge -y --auto-remove build-essential

WORKDIR /app/flask_app

EXPOSE 5000

CMD ["waitress-serve", "--listen=0.0.0.0:5000", "app:app"]