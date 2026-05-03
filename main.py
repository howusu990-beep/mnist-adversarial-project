import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train import Net, load_data, train_model, evaluate
from attack import attack_model, fgsm_attack
from defense import adversarial_training
python main.py
# Load data
train_loader, test_loader = load_data()

# Create model
model = Net()

# Train model
model = train_model(model, train_loader)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Baseline
baseline_acc = evaluate(model, test_loader)
print("Baseline Accuracy:", baseline_acc)

# Attack
attack_acc = attack_model(model, test_loader, criterion)
print("Accuracy under attack:", attack_acc)

# Defense
model = adversarial_training(model, train_loader, criterion, optimizer, fgsm_attack)

# Evaluate again after defense
defense_acc = attack_model(model, test_loader, criterion)
print("Accuracy after defense:", defense_acc)

# Graph
labels = ["Baseline", "Attack", "Defense"]
values = [baseline_acc, attack_acc, defense_acc]

plt.figure()
plt.bar(labels, values)
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy (%)")

plt.savefig("results.png")
print("Graph saved as results.png")

plt.show()
