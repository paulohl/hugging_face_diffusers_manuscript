###### Fine-tuning a pre-trained DistilBERT model for sentiment analysis ######
###############################################################################
common text classification task. We will use a portion of the IMDb dataset and the Hugging Face 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, 
Trainer, TrainingArguments
import numpy as np.
from datasets import load_dataset
# Load dataset
dataset = load_dataset ("imdb", split='train[:5000]')
# Preprocess data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
 return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)
dataset = dataset.map (tokenize, batched=True, batch_size=len(dataset))
dataset.set_format ('torch', columns=['input_ids', 'attention_mask', 'label'])
# Load DistilBERT for sequence classification
model = DistilBertForSequenceClassification.from_pretrained ('distilbert-base￾uncased', num_labels=2)
# Define training arguments
training_args = TrainingArguments (
 output_dir=‘. /results',
 num_train_epochs=3,
 per_device_train_batch_size=16,
 warmup_steps=500,
 weight_decay=0.01,
 logging_dir=‘. /logs',
 load_best_model_at_end=True)
# Initialize Trainer
trainer = Trainer (
 model=model,
 args=training_args,
 train_dataset=dataset)
# Train the model
trainer. train()
# Save the model
model_path = “. /distilbert-finetuned-imdb"
model. save_pretrained(model_path)
tokenizer. save_pretrained(model_path)
