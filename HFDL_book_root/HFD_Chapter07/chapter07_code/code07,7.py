# Case study 1: text summarization.

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define model, optimizer, and data loader
model = SummarizationModel()
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Training loop with scheduler
for epoch in range(100):
    for batch in data_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    scheduler.step()

# Evaluate model performance
bleu_score = evaluate_model(model, test_data)
print(f"BLEU Score: {bleu_score:.2f}")
