# These arguments are then passed to the Trainer class, which manages the training loop and evaluation automatically
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()
