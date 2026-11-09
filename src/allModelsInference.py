import importlib
from collections import Counter

from simpleCNN import inference as cnn_inference

try:
    from transfer_learning import inference as tl_inference
except ModuleNotFoundError:
    from .transfer_learning import inference as tl_inference

td_inference = importlib.import_module("2d_image_classifier.inference")

INFERENCE_MODULES = (
    ("Transfer Learning", tl_inference),
    ("2D Image Classifier", td_inference),
    ("Simple CNN", cnn_inference),
)


def _pickBest(results):
    noOtherResults = [result for result in results if result[1] != "Others"]
    finalResults = noOtherResults or results
    counts = Counter(r[1] for r in finalResults)
    max_votes = max(counts.values())
    top_labels = {label for label, c in counts.items() if c == max_votes}
    contenders = [r for r in finalResults if r[1] in top_labels]
    name, label, confidence = max(contenders, key=lambda r: r[2])
    return label, confidence, name


def predict(path):
    results = [(name, *module.predict(path)) for name, module in INFERENCE_MODULES]
    return _pickBest(results)


def predict_image(image):
    results = [
        (name, *module.predict_image(image)) for name, module in INFERENCE_MODULES
    ]
    return _pickBest(results)
