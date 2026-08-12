import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader: # iterating through the batches
        images = images.to(device) # moving data to gpu or cpu
        labels = labels.to(device)

        optimizer.zero_grad()  # clean up gradients from previous iterations because pytorch accumulates gradients

        outputs = model(images)
        loss = criterion(outputs, labels) # compute prediction error
        running_loss += loss.item() * labels.size(0)

        loss.backward()  # backpropagation
        optimizer.step()  # update weights

        # compute training accuracy
        _, predicted = outputs.max(1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = running_loss / total_samples if total_samples else 0.0
    accuracy = 100 * correct_predictions / total_samples if total_samples else 0.0

    return avg_loss, accuracy