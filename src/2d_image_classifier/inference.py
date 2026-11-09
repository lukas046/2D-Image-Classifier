import os

import cv2
import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.keras")
IMG_SIZE = 224

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

from tensorflow.keras import layers, models

num_classes = 22

def _build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(22, activation='softmax'),
    ])
    model.load_weights(MODEL_PATH)
    return model

MODEL = _build_model()

# preprocesses gray scale images to be used for the input
# normalizes and resizes the image
def _preprocess(img_array):
    img = img_array.astype(float)
    if img.max() != 0:
        img = (img / img.max()) * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.stack((img,) * 3, axis=-1)
    return np.expand_dims(img, axis=0)

# reads the dicom file and covnerts it so that its displayable
def read_dicom(path):
    dcm = pydicom.dcmread(path)
    image = dcm.pixel_array.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")

# predicts what body part it is for dcm
def predict(path):
    ds = pydicom.dcmread(path)
    img_tensor = _preprocess(ds.pixel_array)
    y_prob = MODEL.predict(img_tensor, verbose=0)[0]
    pred_index = int(np.argmax(y_prob))
    label = LABEL_MAP[pred_index]
    confidence = float(y_prob[pred_index])
    return label, confidence

# predict for regular photos
def predict_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("predict_image expects a PIL.Image.Image instance")
    gray = np.array(image.convert("L"))
    img_tensor = _preprocess(gray)
    y_prob = MODEL.predict(img_tensor, verbose=0)[0]
    pred_index = int(np.argmax(y_prob))
    label = LABEL_MAP[pred_index]
    confidence = float(y_prob[pred_index])
    return label, confidence
