from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from dataset import train_dataset

def get_model(classes_num: int):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  # load the pre-trained ResNet-18 model

    # freezing first layers
    for param in model.parameters():
        param.requires_grad = False

    # replacing the classifier with one matching our dataset number of classes
    model.fc = nn.Linear(
        model.fc.in_features,
        classes_num
    )

    return model

