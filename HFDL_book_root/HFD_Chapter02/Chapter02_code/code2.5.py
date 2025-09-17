###### Zero-shot topic classification to avoid labeled data ######
##################################################################
from transformers import pipeline
clf = pipeline("zero-shot-classification", model="facebook/bart-large￾mnli")
text = "Researchers introduce a new transformer architecture for 
protein folding."
labels = ["Sports", "Finance", "Scientific Research", "Politics", 
"Biology"]
print(clf(text, labels)) # scores per candidate label
