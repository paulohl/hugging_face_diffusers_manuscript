# First comparison of model outputs for a single input.

from transformers import pipeline
# Load different models
models = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']
for model_name in models:
    sentiment_model = pipeline('sentiment-analysis', model=model_name)
    print(f"Results for {model_name}:")
    print(sentiment_model("The product is fantastic!"))
