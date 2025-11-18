# Implementing POS tagging using the Hugging Face transformers 
# library with a pre-trained BERT model fine-tuned for the POS
# tagging task. 

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load tokenizer and model
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased-finetuned-pos")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-pos")

# Setup POS tagging pipeline
pos_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Perform POS tagging
pos_results = pos_pipeline(sentence)

# Display POS tags
print("POS Tags:")
for token in pos_results:
    print(f"Word: {token['word']}, POS Tag: {token['entity']}")
