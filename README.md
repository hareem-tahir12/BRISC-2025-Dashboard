# BRISC Brain Tumor MRI Classification

Classification of brain tumor MRI images using the BRISC dataset, comparing two approaches:
a traditional machine learning pipeline and a deep learning pipeline.

## Overview

Accurate brain tumor classification from MRI supports faster, more consistent diagnosis.
This project implements and compares two pipelines end to end:

| | Pipeline A: Traditional ML | Pipeline B: Deep Learning |
|---|---|---|
| Preprocessing | Resize, grayscale, normalization | Resize, normalization, augmentation |
| Feature extraction | HOG (Histogram of Oriented Gradients) | Learned automatically by EfficientNetB7 |
| Dimensionality reduction | PCA | Not required (handled by the network) |
| Classifier | <e.g. SVM / Random Forest> | Fully connected head + softmax |

The two pipelines are mapped stage by stage (see Figure 2) to show how manual feature
engineering in Pipeline A corresponds to learned representations in Pipeline B.

## Dataset

BRISC brain tumor MRI dataset. Classes: <glioma, meningioma, pituitary, no tumor>.
Split: <train / validation / test>.

## Methodology

**Pipeline A: HOG + PCA + classifier**
1. Preprocess images
2. Extract HOG features
3. Reduce dimensionality with PCA
4. Train and evaluate the classifier

**Pipeline B: EfficientNetB7**
1. Preprocess and augment images
2. Fine-tune pretrained EfficientNetB7 (transfer learning)
3. Train the classification head
4. Evaluate on the test set

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| HOG + PCA + <classifier> | | | | |
| EfficientNetB7 | | | | |

## Project Structure

```
├── data/
├── notebooks/
├── src/
├── figures/
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Tech Stack

Python, scikit-learn, scikit-image, TensorFlow/Keras, NumPy, Matplotlib

## Author

Hareem Tahir
