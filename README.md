# Computer Vision Image Classifier

Deep learning image classification project built with Python and PyTorch.

The project implements an end-to-end computer vision pipeline covering data preprocessing, data augmentation, transfer learning, model training, validation, final evaluation and inference on unseen images.

The model is based on a pretrained ResNet18 CNN and was trained to classify images into six scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

---

## Overview
The objective of this project was to develop a complete image classification pipeline using deep learning and to gain practical experience with:

- PyTorch
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Data Augmentation
- Model Training
- Validation
- Model Evaluation
- Inference
- Classification Metrics
- Confusion Matrix
- Model Serialization
- Docker

The project was developed with a focus on understanding the complete machine learning lifecycle rather than only training a model.

---

## Problem
Given an input RGB image, the model must determine which of six scene categories best describes the image.

Formally, the model learns a function:

image → class

where the possible classes are:
text, buildings, forest, glacier, mountain, sea, street

---

## Dataset
The project uses the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download).

The dataset contains images belonging those six different scene categories.

The training dataset was divided into: 80% -> training and 20% -> validation

The original test dataset was kept separate and was only used for the final evaluation. This separation prevents the test set from influencing model training or hyperparameter decisions.

---

## Project Pipeline
Dataset -> Image Processing -> Data Augmentation -> Train / Validation split -> Pretained ResNet18 -> Transfer Learning -> Training -> Validation -> Trained Model -> (Final Evaluation + Test Metrics) and (Inference + New images)

---

## Model
The model uses ResNet18 from torchvision.
Instead of training the CNN from scratch, a ResNet18 pretrained on ImageNet is used as the starting point.

```resnet18(weights=ResNet18_Weights.DEFAULT)```

The original classification layer was replaced with a new fully connected layer containing six output classes

Conventional layers -> Feature extraction -> Fully connected layer -> 6 classes

---

## Code structure

### dataset.py
- Loading the dataset
- Creating the train/validation split
- Creating DataLoaders
- Defining the available classes

### transforms.py

Contains image preprocessing and augmentation pipelines.

### model.py

Responsible for creating the ResNet18 architecture and configuring transfer learning.

### train.py

Contains the training loop for one epoch.

### evaluate.py

Evaluates the model on the validation dataset during training.

### test.py

Performs final evaluation on the independent test dataset.

### predict.py

Runs inference on individual images.

### visualize.py

Generates:
- Training loss curves
- Validation loss curves
- Training accuracy curves
- Validation accuracy curves
- Confusion matrix visualization

### utils.py

Contains reusable utilities such as:
- Model creation
- Optimizer creation
- Model saving/loading
- Training history saving/loading

### main.py

Orchestrates the complete training pipeline.

---

## Requirements
The requirements are listed on the requirements.txt file and can be installed by running: ```pip install -r requirements.txt```

## Training
```python src/main.py```

The model will be saved to: models/resnet18_intel.pth
The training history will be saved to: models/training_history.json

## Visualization
```python src/visualize.py```

This generates:
- Training vs. validation loss
- Training vs. validation accuracy

## Final evaluation
```python src/test.py```

This loads the trained model and evaluates it against the test dataset.

The output includes:
- Test accuracy
- Classification report
- Confusion matrix

## Individual image inference
```python src/predict.py```

The model predicts and returns:
- Predicted class
- Confidence