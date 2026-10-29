# app.py

import os
from pathlib import Path
import re
from io import BytesIO

os.environ.setdefault('MPLBACKEND', 'Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
ROOT_DIR = Path(__file__).resolve().parent.parent
model = None
vectorizer = None

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
    stop_words = set(stopwords.words('english')) - {
        'not', 'but', 'however', 'no', 'yet'
    }
    comment = ' '.join(
        word for word in comment.split() if word not in stop_words
    )
    lemmatizer = WordNetLemmatizer()
    return ' '.join(
        lemmatizer.lemmatize(word) for word in comment.split()
    )


def load_model_and_vectorizer():
    """Load model artifacts created by the local training pipeline."""
    model_path = Path(os.environ.get(
        'MODEL_PATH', ROOT_DIR / 'lgbm_model.pkl'
    ))
    vectorizer_path = Path(os.environ.get(
        'VECTORIZER_PATH', ROOT_DIR / 'tfidf_vectorizer.pkl'
    ))
    return joblib.load(model_path), joblib.load(vectorizer_path)


def get_model_and_vectorizer():
    """Load model files once, when the first prediction is requested."""
    global model, vectorizer
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()
    return model, vectorizer


def predict_comments(comments):
    """Preprocess comments and return model predictions."""
    loaded_model, loaded_vectorizer = get_model_and_vectorizer()
    processed_comments = [
        preprocess_comment(comment) for comment in comments
    ]
    predictions = loaded_model.predict(
        loaded_vectorizer.transform(processed_comments)
    )
    return list(predictions)


def model_unavailable(error):
    app.logger.error('Unable to load model artifacts: %s', error)
    return jsonify({
        'error': (
            'Model artifacts are unavailable. Run the training pipeline '
            'or set MODEL_PATH and VECTORIZER_PATH.'
        )
    }), 503

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/health')
def health():
    try:
        get_model_and_vectorizer()
    except FileNotFoundError as error:
        return model_unavailable(error)
    except Exception:
        app.logger.exception('Model health check failed')
        return jsonify({'status': 'error'}), 500
    return jsonify({'status': 'ok'})

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.get_json(silent=True)
    comments_data = data.get('comments') if isinstance(data, dict) else None
    if not isinstance(comments_data, list) or not comments_data:
        return jsonify({"error": "Provide a non-empty comments list"}), 400
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get('text'), str)
        or not item['text'].strip()
        or 'timestamp' not in item
        for item in comments_data
    ):
        return jsonify({
            "error": "Each comment must include non-empty text and a timestamp"
        }), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        predictions = [str(value) for value in predict_comments(comments)]
    except FileNotFoundError as error:
        return model_unavailable(error)
    except Exception:
        app.logger.exception('Prediction with timestamps failed')
        return jsonify({"error": "Prediction failed"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    comments = data.get('comments') if isinstance(data, dict) else None
    if (
        not isinstance(comments, list)
        or not comments
        or any(not isinstance(comment, str) or not comment.strip()
               for comment in comments)
    ):
        return jsonify({
            "error": "Provide a non-empty list of comment strings"
        }), 400

    try:
        predictions = [str(value) for value in predict_comments(comments)]
    except FileNotFoundError as error:
        return model_unavailable(error)
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({"error": "Prediction failed"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "A JSON object is required"}), 400
        sentiment_counts = data.get('sentiment_counts')
        
        if not isinstance(sentiment_counts, dict) or not sentiment_counts:
            return jsonify({"error": "Provide sentiment counts"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        try:
            sizes = [
                int(sentiment_counts.get('1', 0)),
                int(sentiment_counts.get('0', 0)),
                int(sentiment_counts.get('-1', 0))
            ]
        except (TypeError, ValueError):
            return jsonify({"error": "Sentiment counts must be integers"}), 400
        if any(size < 0 for size in sizes):
            return jsonify({"error": "Sentiment counts cannot be negative"}), 400
        if sum(sizes) == 0:
            return jsonify({"error": "At least one sentiment count is required"}), 400
        
        import matplotlib.pyplot as plt

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "A JSON object is required"}), 400
        comments = data.get('comments')

        if (
            not isinstance(comments, list)
            or not comments
            or any(not isinstance(comment, str) or not comment.strip()
                   for comment in comments)
        ):
            return jsonify({
                "error": "Provide a non-empty list of comment strings"
            }), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)
        if not text:
            return jsonify({"error": "Comments contain no usable words"}), 400

        from wordcloud import WordCloud
        from nltk.corpus import stopwords

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "A JSON object is required"}), 400
        sentiment_data = data.get('sentiment_data')

        if (
            not isinstance(sentiment_data, list)
            or not sentiment_data
            or any(
                not isinstance(item, dict)
                or 'timestamp' not in item
                or 'sentiment' not in item
                for item in sentiment_data
            )
        ):
            return jsonify({
                "error": "Provide timestamp and sentiment for each item"
            }), 400

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', '5000')),
        debug=False,
    )
