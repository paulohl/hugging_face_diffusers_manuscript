# Configuring the training parameters is one of the most critical steps in ensuring that the model learns accordingly
# The Hugging Face Diffusers library makes it easy to configure these parameters using the TrainingArguments class
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=64, 
    warmup_steps=500, 
    weight_decay=0.01, 
    logging_dir='./logs'
)
