# Integration of multiple models.


from transformers import pipeline

# Load pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner")

# Example text
text = "Zinnia Health provides excellent AI-driven care solutions."

# Process text with both pipelines
sentiment = sentiment_pipeline(text)
entities = ner_pipeline(text)
print("Sentiment:", sentiment)
print("Entities:", entities)

