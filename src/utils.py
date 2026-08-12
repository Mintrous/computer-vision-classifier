import os
import json

import torch
import torch.optim as optim

from model import get_model


def create_model(num_classes, device):
    model = get_model(num_classes)
    model.to(device)
    return model


def create_optimizer(model, learning_rate):
    # optimize only the parameters that require gradients (the last layer)
    params = [param for param in model.parameters() if param.requires_grad]
    return optim.Adam(params, lr=learning_rate)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) # creating output directory in case it does not exist
    torch.save(model.state_dict(), path) # saving learned weights


def load_model(path, num_classes, device):
    model = create_model(num_classes, device)

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    model.eval()

    return model


def save_history(history, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w") as file:
        json.dump(
            history,
            file,
            indent=4
        )


def load_history(path):

    with open(path, "r") as file:
        return json.load(file)