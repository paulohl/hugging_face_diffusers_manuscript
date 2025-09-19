###### This code shows the process of fine-tuning a GPT-2 model to generate dialogue responses, ######
###### It can be used to create a chatbot capable of engaging in realistic conversations:       ######
######################################################################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling 

# Load tokenizer and model 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2') 

# Prepare dataset 
train_path = 'path_to_train_data.txt' 
train_dataset = TextDataset( 
    tokenizer=tokenizer, 
    file_path=train_path, 
    block_size=128 
) 
data_collator = DataCollatorForLanguageModeling( 
    tokenizer=tokenizer, mlm=False, 
) 

# Define training arguments 
training_args = TrainingArguments( 
    output_dir='./results',          # output directory 
    num_train_epochs=3,              # number of training epochs 
    per_device_train_batch_size=4,   # batch size for training 
    per_device_eval_batch_size=8,    # batch size for evaluation 
    warmup_steps=500,                # number of warmup steps for learning rate scheduler 
    weight_decay=0.01,               # strength of weight decay 
    logging_dir='./logs',            # directory for storing logs 
    logging_steps=10, 
) 

# Initialize Trainer 
trainer = Trainer( 
    model=model, 
    args=training_args, 
    data_collator=data_collator, 
    train_dataset=train_dataset, 
) 

# Train the model 
trainer.train() 
