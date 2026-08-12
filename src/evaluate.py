import torch


def evaluate(model, loader, criterion, device):
    model.eval() # switching to evaluation mode

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # disable gradient computation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            # compute validation accuracy
            _, predicted = outputs.max(1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    eval_avg_loss = running_loss / total_samples if total_samples else 0.0
    eval_accuracy = 100 * correct_predictions / total_samples if total_samples else 0.0

    return eval_avg_loss, eval_accuracy
