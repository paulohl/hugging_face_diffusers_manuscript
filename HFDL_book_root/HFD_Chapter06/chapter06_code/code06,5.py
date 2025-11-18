# Sentiment analysis pipeline using Flask, 

from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask application and NLP pipeline
app = Flask(__name__)
nlp_pipeline = pipeline("sentiment-analysis")
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    result = nlp_pipeline(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
