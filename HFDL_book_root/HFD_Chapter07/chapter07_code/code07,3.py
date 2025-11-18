# Implementing a step decay scheduler using PyTorch.

from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(epochs):
    train(...)
    scheduler.step()
