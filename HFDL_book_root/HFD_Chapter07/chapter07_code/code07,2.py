# Example: Cosine annealing and exponential decay.

import torch.optim.lr_scheduler as lr_scheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(100):
    train(model, train_loader, optimizer)
    scheduler.step()
