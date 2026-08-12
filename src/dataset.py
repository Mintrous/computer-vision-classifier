from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from transforms import train_transform, eval_transform
import torch

train_dataset = ImageFolder(
    "../data/seg_train/seg_train",
    transform = train_transform
)

eval_dataset = ImageFolder(
    "../data/seg_train/seg_train",
    transform = eval_transform
)

classes = train_dataset.classes
NUM_CLASSES = len(classes) # number of output classes

# 80% -> train, 20% -> validation
# spliting the dataset
train_size = int(0.8 * len(train_dataset))
eval_size = len(train_dataset) - train_size

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(len(train_dataset), generator=generator).tolist()

train_indices = indices[:train_size]
eval_indices = indices[train_size:]

# train and evaluation subsets
train_subset = Subset(train_dataset, train_indices)
eval_subset = Subset(eval_dataset, eval_indices)


# DataLoaders -> Loads batches during training
train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True
)

eval_loader = DataLoader(
    eval_subset,
    batch_size=32,
    shuffle=False
)

test_dataset = ImageFolder(
    "../data/seg_test/seg_test",
    transform=eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)