import numpy as np
import pydicom
import user_functions
from PIL import Image

image = np.asarray(Image.open("sample.jpg").convert("L"))  # Read in grayscale image

# Transform to correct shape and to a tensor
image_tensor = user_functions.transform(image.copy())

model = user_functions.load_model()

class_label, confidence = user_functions.guess(model, image_tensor)

print(f"Guessed Label: {class_label}, Confidence: {confidence}")
