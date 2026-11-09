import numpy as np
import tensorflow as tf
import pandas as pd
import json
import random
import os
import glob
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pydicom 
import cv2
from collections import Counter
import csv


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


labels_df = pd.read_csv("data/train.csv")

labels_df['filename'] = labels_df['SOPInstanceUID'] + "-c.dcm"


with open("shared/label_map.json") as f:
    label_map = json.load(f)


num_classes = 22
IMG_SIZE = 224


def load_data_train(labels_df, path_map):
    images = []
    labels = []


    for _, row in labels_df.iterrows():

        uid = row['SOPInstanceUID']
        
        if uid not in path_map:
            continue 
        
        file_path = path_map[uid]

        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array

        img = img.astype(float)
        img = (img / img.max()) * 255.0
        img = img.astype(np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.stack((img,)*3, axis=-1)

        images.append(img)
        labels.append(row['Target'])

    return np.array(images), np.array(labels)

def load_data_test(file_list, img_size=224):
    import pydicom
    import numpy as np
    import cv2
    images = []
    filenames = []

    for path in file_list:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(float)
        img = (img / img.max()) * 255.0
        img = img.astype(np.uint8)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = np.stack((img,)*3, axis=-1) 

        images.append(img)
        filenames.append(os.path.basename(path)) 

    return np.array(images), filenames

def build_path_map(file_list):
    path_map = {}
    
    for path in file_list:
        filename = os.path.basename(path) 
        
        uid = filename.replace(".dcm", "").replace("-c", "")
        
        path_map[uid] = path
    
    return path_map

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DIR = os.path.join(_REPO_ROOT, "data", "train")
TEST_DIR = os.path.join(_REPO_ROOT, "data", "test")

train_files = glob.glob(os.path.join(TRAIN_DIR, "**", "*.dcm"), recursive=True)
test_files = glob.glob(os.path.join(TEST_DIR, "**", "*.dcm"), recursive=True)

train_map = build_path_map(train_files)
test_map  = build_path_map(test_files)

train_df = labels_df[labels_df['SOPInstanceUID'].isin(train_map.keys())]
X_train, y_train = load_data_train(train_df, train_map)

X_test, test_filenames = load_data_test(test_files, img_size=IMG_SIZE)

def encode_labels(label_str):
    return int(str(label_str).split()[0])


y_train = np.array([encode_labels(lbl) for lbl in train_df['Target']])

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (IMG_SIZE,IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_classes, activation = 'softmax')


])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs=20)


def predict_single_image(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(float)
    img = (img / img.max()) * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.stack((img,)*3, axis=-1)
    
    img_tensor = np.expand_dims(img, axis=0)
    
    y_prob = model.predict(img_tensor, verbose=0)[0]
    
    pred_index = np.argmax(y_prob)
    pred_label = label_map[str(pred_index)]
    
    return pred_index, pred_label, y_prob


results = []

for i, path in enumerate(test_files):
    pred_index, pred_label, y_prob = predict_single_image(path)

    results.append({
        'filename': os.path.basename(path),
        'predicted_label': pred_label
    })
    
    if pred_label:
        labels_str = f"{pred_index}:{pred_label}"
    else:
        labels_str = "No prediction"
        
    print(f"[{i+1}/{len(test_files)}] {labels_str} — {os.path.basename(path)}")


MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "best_model.keras"))
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")

all_pred_indices = [np.argmax(model.predict(np.expand_dims(img,0))[0]) for img in X_test]

counts = Counter(all_pred_indices)
summary = {label_map[str(idx)]: count for idx, count in counts.items()}

print("\nSummary:")
for label, count in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"{label}: {count} images")


