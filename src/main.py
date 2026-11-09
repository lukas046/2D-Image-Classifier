import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import tempfile
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Add project root to path so both model packages are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import importlib

import allModelsInference
from simpleCNN import inference as cnn_inference
from transfer_learning import inference as tl_inference
from transfer_learning.inference import read_dicom

td_inference = importlib.import_module("2d_image_classifier.inference")


MODEL_OPTIONS = {
    # Combined Accuracy selects the best result from running all models
    "Combined Accuracy": allModelsInference,
    "Transfer Learning": tl_inference,  # Jimmy's model
    "2D Image Classifier": td_inference,  # Ishani's model
    "Simple CNN (fastest)": cnn_inference,  # Mitchell's model
}

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FULL_BODY_IMAGE = os.path.join(ASSETS_DIR, "full_body_xray.jpg")


BODY_PART_BOXES = {
    "Skull": (0.38, 0.02, 0.62, 0.10),
    "Sinus": (0.445, 0.085, 0.555, 0.13),  # Good
    "Clavicles": (0.34, 0.17, 0.66, 0.22),
    "Shoulder": (0.22, 0.15, 0.78, 0.23),
    "Chest": (0.30, 0.20, 0.70, 0.38),  # Good
    "Thoracic Spine": (0.44, 0.20, 0.56, 0.38),
    "Elbow": (0.31, 0.29, 0.38, 0.39),  # Good
    "Forearm": (0.27, 0.39, 0.36, 0.49),  # Good
    "Abdomen": (0.33, 0.38, 0.67, 0.48),  # Good
    "Lumbar Spine": (0.44, 0.38, 0.56, 0.47),  # Good
    "Wrist": (0.25, 0.46, 0.35, 0.52),  # good
    "Hand": (0.22, 0.49, 0.39, 0.58),  # good
    "Finger": (0.24, 0.52, 0.32, 0.59),  # Good
    "Hip": (0.30, 0.46, 0.70, 0.55),  # good
    "Pelvis": (0.32, 0.44, 0.68, 0.55),  # good
    "Thigh": (0.32, 0.55, 0.68, 0.70),  # Good
    "Knee": (0.34, 0.60, 0.66, 0.76),  # good
    "Lower Leg": (0.34, 0.76, 0.66, 0.88),  # good
    "Ankle": (0.35, 0.88, 0.65, 0.93),  # Good
    "Feet": (0.32, 0.93, 0.68, 0.99),  # Good
}


def draw_body_part_box(label, confidence):
    if not os.path.exists(FULL_BODY_IMAGE):
        return None
    img = Image.open(FULL_BODY_IMAGE).convert("RGB")
    box_fracs = BODY_PART_BOXES.get(label)
    if box_fracs is None:
        return img
    w, h = img.size
    left = int(box_fracs[0] * w)
    top = int(box_fracs[1] * h)
    right = int(box_fracs[2] * w)
    bottom = int(box_fracs[3] * h)
    draw = ImageDraw.Draw(img)
    draw.rectangle([left, top, right, bottom], outline="red", width=3)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, h // 30))
    except OSError:
        font = ImageFont.load_default()
    text = f"{label} ({round(confidence * 100, 2)}%)"
    if isinstance(font, ImageFont.FreeTypeFont):
        text_box = draw.textbbox((0, 0), text, font=font)
        text_h = text_box[3] - text_box[1]
    else:
        text_h = 12
    draw.text((left + 5, top - text_h - 5), text, fill="red", font=font)
    return img


st.set_page_config(page_title="X-Ray Classification", layout="centered", page_icon="🩻")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
# X-Ray Body Part Classification
""")

selected_model = st.selectbox("Select a model:", list(MODEL_OPTIONS.keys()))
inference_module = MODEL_OPTIONS[selected_model]

uploaded_file = st.file_uploader(
    "Upload an image to be classified:",
    type=["jpg", "jpeg", "png", "dcm"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    with st.spinner("Analyzing image..."):
        selected_model_name = None
        suffix = uploaded_file.name.lower()
        if suffix.endswith((".dcm", ".dicom")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            tmpImage = read_dicom(temp_path)
            prediction = inference_module.predict(temp_path)
        else:
            tmpImage = Image.open(uploaded_file)
            prediction = inference_module.predict_image(tmpImage)

        if len(prediction) == 3:
            label, confidence, selected_model_name = prediction
        else:
            label, confidence = prediction

    body_img = draw_body_part_box(label, confidence)

    if body_img is not None:
        tab1, tab2 = st.tabs(["Original", "Body Map"])
    else:
        tab1 = st.tabs(["Original"])[0]

    tab1.markdown(
        f"##### Predicted: {label}\n ##### Confidence: {round(confidence * 100, 2)}%"
    )
    tab1.image(tmpImage, width="stretch")

    if body_img is not None:
        bodyMapText = []
        # Display the model that was selected
        if selected_model_name is not None:
            bodyMapText.append(f"##### Model Used: {selected_model_name}")
        # Label and display the image
        bodyMapText.append(f"##### Identified Region: {label}")
        tab2.markdown("\n".join(bodyMapText))
        tab2.image(body_img, width="stretch")
