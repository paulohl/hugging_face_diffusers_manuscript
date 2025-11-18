# Sentiment analysis pipeline designed to process customer feedback.

from transformers import pipeline
# Initialize pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Analyze customer feedback
feedback = ["Great product!", "Terrible customer service."]
results = sentiment_analysis(feedback)
for result in results:
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}")
