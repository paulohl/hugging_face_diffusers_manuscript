from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    Wampuses=500,              	     # number of warmup steps for learning rate scheduler
    weight decay=0.01,               # strength of weight decay
    logging Dir='./logs',            # directory for storing logs
)
