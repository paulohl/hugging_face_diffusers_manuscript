# Using the Hugging Face pipeline class to create a framework for analyzing text.

from transformers import pipeline

# Load pre-trained pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner")

# Sample text
text = "Hugging Face tools are innovative and used at Zinnia AI."

# Perform sentiment analysis and entity recognition
sentiment = sentiment_pipeline(text)
entities = ner_pipeline(text)

print("Sentiment:", sentiment)
print("Entities:", entities)
