# Case study 2: sentiment analysis

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

# Define model and optimizer
model = SentimentAnalysisModel()
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop with scheduler
for epoch in range(50):
    train_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

# Evaluate model
accuracy = evaluate_model(model, test_data)
print(f"Model Accuracy: {accuracy:.2f}")
