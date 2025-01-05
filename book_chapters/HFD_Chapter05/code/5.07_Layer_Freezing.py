# The following code demonstrates how to apply layer freezing to fine-tune a BERT model for sentiment classification. 

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Prepare a small dataset
data = {"text": ["Amazing experience.", "Horrible outcome.", "Decent results."],
        "label": [0, 1, 2]}  # 0: Positive, 1: Negative, 2: Neutral
dataset = Dataset.from_dict(data)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Freeze all layers except the classification head
for param in model.bert.parameters():
    param.requires_grad = False

# Tokenize data
dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8
)
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
# Fine-tune the model
trainer.train()
