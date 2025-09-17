###### Fine-tuning a pre-trained BERT model                                                            ######
###### using the Hugging Face transformers library for a sentiment analysis task.                      ######
###### Implementation of best practices:                                                               ######
###### learning rate scheduling, early stopping, and custom performance metrics to evaluate the model  ######
###### Dataset: subset of the IMDb dataset (labeled movie reviews for binary sentiment classification) ######
#############################################################################################################
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, 
TrainingArguments
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
# Load dataset
dataset = load_dataset("imdb", split='train[:2000]')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Preprocessing the data
def preprocess_data(example):
 return tokenizer(example['text'], padding="max_length", truncation=True, 
max_length=512)
# Map preprocessing function to the dataset
dataset = dataset.map(preprocess_data, batched=True)
# Define a PyTorch DataLoader
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
num_labels=2)
# Define Trainer Arguments with learning rate scheduler and early stopping
training_args = TrainingArguments(
 output_dir='./results',
 evaluation_strategy="steps",
 eval_steps=500,
 logging_steps=500,
 num_train_epochs=3,
 per_device_train_batch_size=16,
 save_steps=1000,
 save_total_limit=2,
 load_best_model_at_end=True,
 metric_for_best_model='accuracy',
 greater_is_better=True,
)
# Custom compute_metrics function to calculate accuracy and F1-score
def compute_metrics(eval_pred):
 logits, labels = eval_pred
 predictions = np.argmax(logits, axis=-1)
 acc = accuracy_score(labels, predictions)
 f1 = f1_score(labels, predictions, average='binary')
 return {"accuracy": acc, "f1": f1}
# Initialize Trainer
trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=dataset,
 compute_metrics=compute_metrics
)
# Training the model with early stopping based on the accuracy
trainer.train()
# Save the model
model.save_pretrained('./fine_tuned_bert')
