import torch

from dataset import train_loader, eval_loader, NUM_CLASSES
from evaluate import evaluate
from train import train_one_epoch
from utils import create_model, create_optimizer, save_model, save_history

# training hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available, otherwise, cpu


def main():
    # build model and optimnizer
    model = create_model(NUM_CLASSES, device)
    optimizer = create_optimizer(model, LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss() # loss function for multiclass classification

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(
            model,
            eval_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print("-" * 50)

    save_model(model, "models/resnet18_intel.pth")
    print("Model is saved")

    save_history(history, "models/training_history.json")
    print("Training history is saved")
    
    return history


if __name__ == "__main__":
    main()
