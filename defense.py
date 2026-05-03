def adversarial_training(model, train_loader, criterion, optimizer, fgsm_attack, epsilon=0.15):
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

    print("Defense training complete")  # ✅ ADD THIS
    return model
