import json
import os

import numpy as np
import pydicom
from PIL import Image

from simpleCNN.user_functions import guess, load_model, transform

MODEL = load_model()
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "label_map.json")

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as label_map_file:
    LABEL_MAP = {int(key): value for key, value in json.load(label_map_file).items()}


def read_dicom(path):
    dcm = pydicom.dcmread(path)
    image = dcm.pixel_array.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image).convert("L")


def _predict_pil_image(image):
    gray_image = image.convert("L")
    image_tensor = transform(np.asarray(gray_image).copy())
    class_idx, confidence = guess(MODEL, image_tensor)
    return LABEL_MAP[class_idx], confidence


def predict(path):
    image = read_dicom(path)
    return _predict_pil_image(image)


def predict_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("predict_image expects a PIL.Image.Image instance")
    return _predict_pil_image(image)
