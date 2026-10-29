import pytest

from flask_app import app as api


class FakeVectorizer:
    def transform(self, comments):
        return comments


class FakeModel:
    def predict(self, comments):
        return [1 if 'good' in comment else -1 for comment in comments]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api,
        'get_model_and_vectorizer',
        lambda: (FakeModel(), FakeVectorizer()),
    )
    monkeypatch.setattr(
        api, 'preprocess_comment', lambda comment: comment.lower()
    )
    return api.app.test_client()


def test_health(client):
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json == {'status': 'ok'}


def test_predict(client):
    response = client.post(
        '/predict',
        json={'comments': ['Good video', 'Bad video']},
    )

    assert response.status_code == 200
    assert response.json == [
        {'comment': 'Good video', 'sentiment': '1'},
        {'comment': 'Bad video', 'sentiment': '-1'},
    ]


def test_predict_with_timestamps(client):
    response = client.post(
        '/predict_with_timestamps',
        json={'comments': [
            {'text': 'Good video', 'timestamp': '2026-10-29T10:00:00Z'}
        ]},
    )

    assert response.status_code == 200
    assert response.json == [{
        'comment': 'Good video',
        'sentiment': '1',
        'timestamp': '2026-10-29T10:00:00Z',
    }]


@pytest.mark.parametrize(
    'payload',
    [
        {},
        {'comments': 'not a list'},
        {'comments': ['   ']},
        {'comments': [None]},
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    response = client.post('/predict', json=payload)

    assert response.status_code == 400
    assert 'error' in response.json


def test_prediction_without_artifacts_returns_service_unavailable(client, monkeypatch):
    def missing_artifacts():
        raise FileNotFoundError('model files missing')

    monkeypatch.setattr(api, 'get_model_and_vectorizer', missing_artifacts)
    response = client.post('/predict', json={'comments': ['A comment']})

    assert response.status_code == 503
    assert 'Model artifacts are unavailable' in response.json['error']


def test_health_without_artifacts_returns_service_unavailable(client, monkeypatch):
    def missing_artifacts():
        raise FileNotFoundError('model files missing')

    monkeypatch.setattr(api, 'get_model_and_vectorizer', missing_artifacts)
    response = client.get('/health')

    assert response.status_code == 503
    assert 'Model artifacts are unavailable' in response.json['error']


@pytest.mark.parametrize(
    'path,payload',
    [
        ('/generate_chart', {}),
        ('/generate_wordcloud', {'comments': []}),
        ('/generate_trend_graph', {'sentiment_data': [{}]}),
    ],
)
def test_visualization_endpoints_reject_invalid_input(client, path, payload):
    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert 'error' in response.json
