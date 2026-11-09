# X-ray Classification (XC)

This project classifies X-ray images by body part by combining different computer vision models. This includes a simple CNN, a transfer learning model, and a 2D image classifier. It also includes a web app that lets users upload images to predict and visualize the detected body region on a body map.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/IQCA0x7CT41FR7lRRU1pfcvfAQnRP-SR_MOWR_PrySA9x1s?e=TtFeHg) | [Slack channel](https://app.slack.com/client/T0A6UH0JBNH/C0AA31T1CJZ) | [Project report](https://www.overleaf.com/project/69c456b114adfcde19a468e0) | [Overleaf](https://www.overleaf.com/9572655145qdqsjmdwfpcp#6bc792)
|-----------|---------------|-------------------------|----------------------------|


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

[Video](https://www.youtube.com/watch?v=6VpR6Wxg3do)

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

```bash
cd src
streamlit run main.py
```

### What to find where

```bash
repository
├── data                         ## train/test image splits (+ train.csv)
│   ├── test
│   ├── train
│   └── train.csv
├── shared                       ## shared config across models
│   └── label_map.json
├── src                          ## app and all model code
│   ├── .streamlit               ## Streamlit UI config
│   ├── assets                   ## stock image for the UI
│   ├── 2d_image_classifier      ## Keras model — one of three
│   ├── simpleCNN                ## PyTorch from-scratch CNN — one of three
│   ├── transfer_learning        ## PyTorch transfer learning — one of three
│   ├── existingarchitectures    ## PyTorch predefined architectures trained from scratch
│   ├── allModelsInference.py    ## runs all three; label by majority vote
│   ├── results                  ## Script to generate performance graphs
│   ├── get_data.py              ## Script to download the original dataset from kaggle
│   ├── preprocess_data.py       ## Script to preprocess the original dataset
│   ├── preprocessing.ipynb      ## Notebook showing preprocessing steps of the original dataset
│   └── main.py                  ## Streamlit frontend
├── LICENSE
├── README.md                    ## You are here
├── requirements.txt             ## Python dependencies
└── .gitignore
```

<a name="installation"></a>

## 2. Installation

Clone the repository and install Python dependencies:

```bash
git clone $THISREPO
cd $THISREPO
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r .\requirements.txt
```

### Data and Keras model weights

Download **`data.zip`** and **`best_model.zip`** from the shared Google Drive folder:  
[data and models](https://drive.google.com/drive/folders/1ptwOH3FbOxwhTLPVAKhpawDIsNJs5ZqO?usp=drive_link).

For training and tests for the SimpleCNN model and existing architecture models, please use the following dataset:
[dataset2](https://drive.google.com/drive/folders/1io2yKOQI6JfDI5ORq1CGx1tBddf5tOyO?usp=sharing)

1. Unzip **`data.zip`** at the **repository root** so you have `data/train/`, `data/test/`, and `data/train.csv` (if the archive contains a single top-level folder, move its contents into `data/`).
2. Unzip **`best_model.zip`** and place **`best_model.keras`** in `src/2d_image_classifier/` (next to `inference.py`).

<a name="repro"></a>
## 3. Reproduction
To generate the keras model for 2d_image_classifier:
```bash
python .\src\2d_image_classifier\single_classifier.py
```

After [installing](#installation) dependencies and any required model weights, make sure the venv is running and start the app from `src`:

```bash

cd src
streamlit run main.py
```

Streamlit opens the app in your browser. Load a chest X-ray as a **DICOM** file (`.dcm`) or **JPG** file (`.jpg`) through the file picker. Once loaded, the app runs classification and shows the predictions.

**Main UI**

![Streamlit main view](https://github.com/user-attachments/assets/2c0e651a-4883-4b39-85be-cd995a90e4bf)

**Classification after a `.dcm` is loaded**

![Classification results with a loaded DICOM](https://github.com/user-attachments/assets/572061de-613f-4752-94d4-5acd9f4211b4)

**Reproducing the results graph** \
Make sure the venv is running
```bash
cd .\src\results\
python results.py
```
will generate `accuracy_results.png`, `speed_results.png`, and `test_confidence_resutls.png`

*SimpleCNN and Existing Architectures* - Run the associated ipynb files within their respective directories, ensuring [dataset2](https://drive.google.com/file/d/1mqkyOVI-2a6uFk7P6KYBVlCw2BRFFCbz/view?usp=sharing) is downloaded and in a folder named `/dataset/`.

**Reproduction of training** \
*SimpleCNN and Existing Architectures* - Run the associated ipynb files within their respective directories, ensuring [dataset2](https://drive.google.com/file/d/1mqkyOVI-2a6uFk7P6KYBVlCw2BRFFCbz/view?usp=sharing) is downloaded and in a folder named `/dataset/`.

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)
