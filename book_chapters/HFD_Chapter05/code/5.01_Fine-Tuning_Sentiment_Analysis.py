# This example illustrates the fine-tuning process of a BERT model using a small custom dataset containing 
# three sample product reviews categorized as positive, negative, or neutral. Initially, 
# a custom dataset is created and converted into a format compatible with the Hugging Face library. 

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Prepare a custom dataset
data = {"text": ["Great product!", "Terrible service.", "Average experience."],
        "label": [0, 1, 2]}  # 0: Positive, 1: Negative, 2: Neutral
dataset = Dataset.from_dict(data)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize data
def tokenize_data(example):
    return tokenizer(example['text'], truncation=True, padding='max_length')
dataset = dataset.map(tokenize_data, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs'
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

