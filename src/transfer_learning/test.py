import glob
import os

import numpy as np
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
TEST_DIR = os.path.join(os.path.dirname(__file__), "../../data/test")
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

# ── Preprocessing ────────────────────────────────────────────────────────────
transform = transforms.Compose(
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


# ── Load Model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)
print("Model loaded\n")


# ── Predict Single Image ─────────────────────────────────────────────────────
def predict(path):
    image = read_dicom(path)
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][idx].item()
    return LABEL_MAP[idx], confidence


# ── Run on All Test Images ───────────────────────────────────────────────────
print("Scanning test folder...")
test_files = glob.glob(os.path.join(TEST_DIR, "**", "*.dcm"), recursive=True)
print(f"Found {len(test_files)} test images\n")

results = []
for i, path in enumerate(test_files):
    label, confidence = predict(path)
    results.append((path, label, confidence))
    print(
        f"[{i+1}/{len(test_files)}] {label} ({confidence*100:.1f}%) — {os.path.basename(path)}"
    )

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────")
from collections import Counter

counts = Counter(label for _, label, _ in results)
for label, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count} images")

avg_confidence = sum(c for _, _, c in results) / len(results)
print(f"\nAverage confidence: {avg_confidence*100:.1f}%")
print(f"Total images tested: {len(results)}")
