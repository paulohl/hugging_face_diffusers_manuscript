# Cosine annealing scheduler implementation:

from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50)
for epoch in range(epochs):
    train(...)
    scheduler.step()
