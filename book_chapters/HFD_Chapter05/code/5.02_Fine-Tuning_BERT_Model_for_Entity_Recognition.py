# The following script demonstrates the fine-tuning process using the widely recognized CoNLL-2003 dataset. 
# This dataset includes annotated text for entities, making it ideal for training and evaluating NER models. 
# The Hugging Face transformers library provides robust tools to streamline the fine-tuning process

from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)

# Load and preprocess dataset
dataset = load_dataset("conll2003")
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [label[word_idx] if word_idx is not None else -100 for word_idx in word_ids]
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']

