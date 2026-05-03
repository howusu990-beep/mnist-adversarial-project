import torch

# FGSM function
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# Run attack
def attack_model(model, test_loader, criterion, epsilon=0.15):
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

    return 100 * correct / len(test_loader)
