# Case study: fine-tuning for text classification

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Prepare dataset
dataset = load_dataset('ag_news')
dataset = dataset.map(lambda e: {'labels': e['label'], **tokenizer(e['text'], padding='max_length', truncation=True)}, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

# Fine-tune the model
trainer.train()
