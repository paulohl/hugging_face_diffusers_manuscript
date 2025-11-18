# Adapting a DistilBERT model:

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Prepare a sample dataset
data = {"text": ["Deep learning advances.", "Ethical concerns in AI.", "Data preprocessing techniques.",
        "label": [0, 1, 2]}  # 0: Machine Learning, 1: AI Ethics, 2: Data Science
dataset = Dataset.from_dict(data)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Tokenize data
dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

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
