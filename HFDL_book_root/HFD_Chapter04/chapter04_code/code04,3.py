# Implementing NER using the Hugging Face transformers library,
# using a pre-trained BERT model fine-tuned for NER tasks.

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load pre-trained model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Setup NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example text
text = "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975."

# Perform NER
results = ner_pipeline(text)

# Display results
print("Detected Entities:")
for entity in results:
    print(f"Entity: {entity['word']}, Type: {entity['entity']}")
