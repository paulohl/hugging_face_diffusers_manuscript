###### Fine-tuning a pre-trained GPT-2 model to produce creative writing samples. ######
######################################################################################## 
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, 
DataCollatorForLanguageModeling, Trainer, TrainingArguments
# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
# Prepare dataset
train_path = 'path_to_training_data.txt'
train_dataset = TextDataset (
 tokenizer=tokenizer,
 file_path=train_path,
 block_size=128)
data_collator = DataCollatorForLanguageModeling (
 tokenizer=tokenizer, mlm=False)
# Define training arguments
training_args = TrainingArguments (
 output_dir=‘. /gpt2-finetuned',
 overwrite_output_dir=True,
 num_train_epochs=3,
 per_device_train_batch_size=4,
 save_steps=10_000,
 save_total_limit=2)
# Initialize Trainer
trainer = Trainer (
 model=model,
 args=training_args,
 data_collator=data_collator,
 train_dataset=train_dataset)
# Train the model
trainer. train()
# Save the model
model. save_pretrained('./gpt2-finetuned')
tokenizer. save_pretrained('./gpt2-finetuned')
