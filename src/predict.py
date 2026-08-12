import torch
from PIL import Image

from dataset import classes, NUM_CLASSES
from transforms import eval_transform
from utils import load_model


MODEL_PATH = "models/resnet18_intel.pth" # trained model path
IMAGE_PATH = "images/forest2.jpeg" # testing with a single image

# use GPU if CUDA is available; otherwise, use CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def load_image(image_path):
    """
    Load and preprocess a single image
    """

    # opening the image and convertindg to 3 channels
    image = Image.open(image_path).convert("RGB")

    # applying the same preprocessing used during validation/test:
    # resize -> tensor -> normalization
    image = eval_transform(image)

    # add a batch dimension
    # before:
    # image shape: [3, 224, 224]
    # after:
    # image shape: [1, 3, 224, 224]
    image = image.unsqueeze(0)

    # move the image to the same device as the model
    image = image.to(device)

    return image


def predict(model, image, classes):
    """
    Run inference and return the predicted class
    and its confidence.
    """

    with torch.no_grad():
        # forward pass
        # the image goes through the nn
        outputs = model(image)

        # converting the model's logits to probabilities using softmax
        probabilities = torch.softmax(outputs, dim = 1)

        # get the highest probability and its corresponding class index
        confidence, predicted_class_index = torch.max(probabilities, dim = 1)

    predicted_index = predicted_class_index.item()
    predicted_class = classes[predicted_index]
    confidence = confidence.item()

    return predicted_class, confidence


def main():
    # load the trained resnet18 and its weights
    model = load_model(
        MODEL_PATH,
        NUM_CLASSES,
        device
    )

    # evaluation mode
    model.eval()

    # load and preprocess the image
    image = load_image(
        IMAGE_PATH,
    )

    # run inference
    predicted_class, confidence = predict(
        model,
        image,
        classes
    )

    print(f"image path: {IMAGE_PATH}")
    print(f"predicted class: {predicted_class}")
    print(f"confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()