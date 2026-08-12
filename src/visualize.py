import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils import load_history


HISTORY_PATH = "models/training_history.json"

def plot_training_history(history):
    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()

    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()

    plt.show()


def plot_confusion_matrix(labels, predictions, classes):
    """
    Plot the confusion matrix for the test set
    """

    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=classes,
        cmap="Blues"
    )

    display.ax_.set_title("Confusion Matrix")

    plt.show()


def main():
    # Load previously saved training history.
    history = load_history(HISTORY_PATH)

    # Generate training plots.
    plot_training_history(history)


if __name__ == "__main__":
    main()