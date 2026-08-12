import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import (
    test_loader,
    classes,
    NUM_CLASSES,
)

from utils import load_model
from visualize import plot_confusion_matrix


MODEL_PATH = "models/resnet18_intel.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_model(model, loader, device):
    """
    Runs inference on the entire test dataset and returns
    the true and predicted labels.
    """
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim = 1) # get predicted class index

            all_predictions.extend(
                predicted.cpu().numpy()
            )
            all_labels.extend(
                labels.cpu().numpy()
            )

    return all_labels, all_predictions


def print_metrics(labels, predictions):
    # calculating accuracy
    accuracy = accuracy_score(
        labels,
        predictions
    )

    report = classification_report(
        labels,
        predictions,
        target_names = classes
    )

    # confusion matrix
    cm = confusion_matrix(
        labels,
        predictions
    )

    print(f"Test Accuracy: {accuracy:.4f}")
    print()
    print(report)
    print()
    print(cm)


def main():
    model = load_model(
        MODEL_PATH,
        NUM_CLASSES,
        device
    )

    labels, predictions = evaluate_model(
        model,
        test_loader,
        device
    )

    print_metrics(labels, predictions)

    plot_confusion_matrix(
        labels,
        predictions,
        classes
    )


if __name__ == "__main__":
    main()