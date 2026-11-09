import os

import numpy as np
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
NUM_CLASSES = 22
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    0: "Abdomen",
    1: "Ankle",
    2: "Cervical Spine",
    3: "Chest",
    4: "Clavicles",
    5: "Elbow",
    6: "Feet",
    7: "Finger",
    8: "Forearm",
    9: "Hand",
    10: "Hip",
    11: "Knee",
    12: "Lower Leg",
    13: "Lumbar Spine",
    14: "Others",
    15: "Pelvis",
    16: "Shoulder",
    17: "Sinus",
    18: "Skull",
    19: "Thigh",
    20: "Thoracic Spine",
    21: "Wrist",
}

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def read_dicom(path):
    dcm = pydicom.dcmread(path)
    image = dcm.pixel_array.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def _load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


MODEL = _load_model()


def _predict_pil_image(image):
    rgb_image = image.convert("RGB")
    tensor = TRANSFORM(rgb_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = MODEL(tensor)
        idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][idx].item()
    return LABEL_MAP[idx], confidence


def predict(path):
    image = read_dicom(path)
    return _predict_pil_image(image)


def predict_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("predict_image expects a PIL.Image.Image instance")
    return _predict_pil_image(image)
