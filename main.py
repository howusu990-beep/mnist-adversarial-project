import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # ✅ REQUIRED FOR GRAPH

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Build model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

# FGSM Attack Function
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training complete")

# Baseline Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

baseline_acc = 100 * correct / total
print("Baseline Accuracy:", baseline_acc)

# Attack
epsilon = 0.15
correct = 0

for images, labels in test_loader:
    images.requires_grad = True

    output = model(images)
    loss = criterion(output, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_data = fgsm_attack(images, epsilon, data_grad)

    output = model(perturbed_data)
    _, final_pred = torch.max(output.data, 1)

    if final_pred.item() == labels.item():
        correct += 1

attack_acc = 100 * correct / len(test_loader)
print("Accuracy under attack:", attack_acc)

# Defense (Adversarial Training)
for epoch in range(5):
    for images, labels in train_loader:
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        adv_images = fgsm_attack(images, epsilon, data_grad)

        outputs = model(adv_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Defense training complete")

# Evaluate AFTER defense
correct = 0

for images, labels in test_loader:
    images.requires_grad = True

    output = model(images)
    loss = criterion(output, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_data = fgsm_attack(images, epsilon, data_grad)

    output = model(perturbed_data)
    _, final_pred = torch.max(output.data, 1)

    if final_pred.item() == labels.item():
        correct += 1

defense_acc = 100 * correct / len(test_loader)
print("Accuracy after defense:", defense_acc)

# 📊 GRAPH (FINAL + SAVED)
labels = ["Baseline", "Attack", "Defense"]
values = [baseline_acc, attack_acc, defense_acc]

plt.figure()
plt.bar(labels, values)
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy (%)")

plt.savefig("results.png")  # ✅ THIS CREATES THE FILE
print("Graph saved as results.png")

plt.show()