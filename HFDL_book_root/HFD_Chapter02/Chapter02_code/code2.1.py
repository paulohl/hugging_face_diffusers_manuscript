###### Hugging Face Transformers library to analyze customer reviews for sentiment analysis ######
##################################################################################################
from transformers import pipeline
# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
# Example customer reviews
reviews = [
 "I absolutely love this product! It works wonders for me.",
 "This is the worst product I have ever purchased.",
 "It's okay, not great but not terrible either."
]
# Analyze sentiment of each review
results = sentiment_pipeline(reviews)
# Print the results
for review, result in zip(reviews, results):
 print(f"Review: '{review}'")
 print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")
