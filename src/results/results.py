import csv
import importlib
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

resultsRoot = Path(__file__).resolve().parent
projectRoot = resultsRoot.parent.parent
srcRoot = resultsRoot.parent

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))
if str(srcRoot) not in sys.path:
    sys.path.insert(0, str(srcRoot))

trainCsvPath = projectRoot / "data" / "train.csv"
trainDir = projectRoot / "data" / "train"
testDir = projectRoot / "data" / "test"
accuracyPlotPath = resultsRoot / "accuracy_results.png"
speedPlotPath = resultsRoot / "speed_results.png"
testConfidencePlotPath = resultsRoot / "test_confidence_results.png"

modelOrder = (
    "Combined Accuracy",
    "Transfer Learning",
    "2D Image Classifier",
    "Simple CNN",
)

accuracyModels = (
    "Transfer Learning",
    "2D Image Classifier",
    "Simple CNN",
)

modelImports = {
    "Combined Accuracy": "allModelsInference",
    "Transfer Learning": "transfer_learning.inference",
    "2D Image Classifier": "2d_image_classifier.inference",
    "Simple CNN": "simpleCNN.inference",
}

barColors = ["#496ddb", "#d06b2c", "#2c9a69", "#8a5bd4"]


def findDicomFiles(root: Path):
    return sorted(path for path in root.rglob("*.dcm") if path.is_file())


def loadInferenceModules():
    modules = {}
    failures = {}

    for label, moduleName in modelImports.items():
        try:
            modules[label] = importlib.import_module(moduleName)
        except Exception as error:
            failures[label] = f"{type(error).__name__}: {error}"

    return modules, failures


def buildHoldoutSamples():
    uidToLabel = {}
    with trainCsvPath.open(newline="", encoding="utf-8") as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            labelText = str(row["Target"]).strip().split()[0]
            uidToLabel[row["SOPInstanceUID"]] = int(labelText)

    uidToPath = {}
    for path in findDicomFiles(trainDir):
        uidToPath[path.stem.removesuffix("-c")] = path

    samples = []
    for uid, label in uidToLabel.items():
        if uid in uidToPath:
            samples.append((uidToPath[uid], label))

    random.Random(42).shuffle(samples)
    return samples[: min(60, len(samples))]


def getReverseLabelMap():
    labelMap = getattr(
        importlib.import_module("transfer_learning.inference"), "LABEL_MAP"
    )
    return {label: index for index, label in labelMap.items()}


def getModuleMetrics(module, holdoutSamples, reverseLabelMap):
    if module is None:
        return None, None

    correctCount = 0
    totalCount = 0
    confidenceTotal = 0.0

    for path, expectedIndex in holdoutSamples:
        prediction = module.predict(str(path))
        predictedLabel = prediction[0]
        confidence = float(prediction[1])
        predictedIndex = reverseLabelMap.get(predictedLabel)

        if predictedIndex is None:
            continue

        totalCount += 1
        confidenceTotal += confidence
        if predictedIndex == expectedIndex:
            correctCount += 1

    if totalCount == 0:
        return None, None

    return correctCount / totalCount, confidenceTotal / totalCount


def getHoldoutMetrics(modules, holdoutSamples):
    emptyResults = {label: None for label in modelOrder}
    if not holdoutSamples:
        return emptyResults.copy(), emptyResults.copy()

    reverseLabelMap = getReverseLabelMap()
    accuracies = {}
    confidences = {}

    for label in modelOrder:
        accuracy, confidence = getModuleMetrics(
            modules.get(label), holdoutSamples, reverseLabelMap
        )
        accuracies[label] = accuracy
        confidences[label] = confidence

    return accuracies, confidences


def benchmarkTestSet(modules):
    testFiles = findDicomFiles(testDir)
    sampleFiles = testFiles[: min(25, len(testFiles))]
    speedResults = {}
    confidenceResults = {}

    for label in modelOrder:
        module = modules.get(label)
        if module is None:
            speedResults[label] = None
            confidenceResults[label] = None
            continue

        warmupFiles = sampleFiles[:1]
        benchmarkFiles = sampleFiles[1:] or sampleFiles

        try:
            for path in warmupFiles:
                module.predict(str(path))

            startTime = time.perf_counter()
            confidenceTotal = 0.0

            for path in benchmarkFiles:
                prediction = module.predict(str(path))
                confidenceTotal += float(prediction[1])

            elapsedTime = time.perf_counter() - startTime
            speedResults[label] = elapsedTime / len(benchmarkFiles)
            confidenceResults[label] = confidenceTotal / len(benchmarkFiles)
        except Exception:
            speedResults[label] = None
            confidenceResults[label] = None

    return speedResults, confidenceResults


def annotateBars(axis, bars, formatValue):
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            formatValue(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def saveBarChart(metricValues, outputPath: Path, title: str, yLabel: str, formatValue):
    availableValues = [
        (label, value) for label, value in metricValues if value is not None
    ]
    figure, axis = plt.subplots(figsize=(10, 6))

    if availableValues:
        labels = [label for label, _ in availableValues]
        values = [value for _, value in availableValues]
        bars = axis.bar(labels, values, color=barColors[: len(values)])
        annotateBars(axis, bars, formatValue)
        axis.set_ylabel(yLabel)
        axis.tick_params(axis="x", rotation=15)
    else:
        axis.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        axis.set_xticks([])
        axis.set_yticks([])

    axis.set_title(title)
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=200)
    plt.close(figure)


def saveAccuracyChart(accuracies, confidences, outputPath: Path):
    chartRows = []
    for label in accuracyModels:
        accuracy = accuracies.get(label)
        confidence = confidences.get(label)
        if accuracy is not None:
            chartRows.append((label, accuracy, confidence))

    figure, axis = plt.subplots(figsize=(10, 6))

    if chartRows:
        labels = [label for label, _, _ in chartRows]
        accuracyValues = [accuracy for _, accuracy, _ in chartRows]
        confidenceValues = [confidence for _, _, confidence in chartRows]
        xPositions = list(range(len(labels)))

        bars = axis.bar(
            xPositions,
            accuracyValues,
            color=barColors[: len(labels)],
            label="Accuracy",
        )
        annotateBars(axis, bars, lambda value: f"{value:.1%}")

        axis.plot(
            xPositions,
            confidenceValues,
            color="#111111",
            marker="o",
            linewidth=2,
            label="Average confidence",
        )

        for xPos, confidence in zip(xPositions, confidenceValues):
            axis.annotate(
                f"{confidence:.1%}",
                xy=(xPos, confidence),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111111",
            )

        axis.set_xticks(xPositions)
        axis.set_xticklabels(labels, rotation=15)
        axis.set_ylabel("Rate")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        axis.set_xticks([])
        axis.set_yticks([])

    axis.set_title("Model Accuracy & Average Confidence on Train Dataset")
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=200)
    plt.close(figure)


def main():
    modules, _ = loadInferenceModules()

    holdoutSamples = buildHoldoutSamples()
    accuracies, holdoutConfidences = getHoldoutMetrics(modules, holdoutSamples)
    speedResults, testConfidences = benchmarkTestSet(modules)

    saveAccuracyChart(accuracies, holdoutConfidences, accuracyPlotPath)
    saveBarChart(
        [(label, speedResults.get(label)) for label in modelOrder],
        speedPlotPath,
        "Speed of Models on Test Dataset",
        "Seconds per image",
        lambda value: f"{value:.3f}s",
    )
    saveBarChart(
        [(label, testConfidences.get(label)) for label in modelOrder],
        testConfidencePlotPath,
        "Average Confidence on Test Dataset",
        "Average confidence",
        lambda value: f"{value:.1%}",
    )


if __name__ == "__main__":
    main()
