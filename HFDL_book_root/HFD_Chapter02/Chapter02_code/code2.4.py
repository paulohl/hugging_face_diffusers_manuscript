###### Multilingual sentiment pipeline example using XLM-RoBERTa ######
#######################################################################
from transformers import pipeline
# Load multilingual sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis",
 model="cardiffnlp/twitter-xlm-roberta￾base-sentiment")
reviews = [
 "Este producto es fantástico, funciona perfectamente.", # Spanish
 "Ce service est terrible, je ne le recommande pas.", # French
 "Dieses Gerät ist in Ordnung, aber unauffällig." # German
]
results = sentiment_pipeline(reviews)
for r, res in zip(reviews, results):
 print(f"Review: {r}\nSentiment: {res['label']}, Confidence: 
 {res['score']:.2f}\n")
