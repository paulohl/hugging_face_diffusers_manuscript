# Loading a pre-trained model fine-tuned on NER tasks and 
# fine-tuned further for domain-specific use cases. This snippet uses
# a pre-trained BERT model fine-tuned on NER for general purposes:

# Load a pre-trained model 
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 

# Use Hugging Face pipeline for Named Entity Recognition
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english") 

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer) 

# Test the NER pipeline text = "Hugging Face is a company based in New York." 
print(ner_pipeline(text))

