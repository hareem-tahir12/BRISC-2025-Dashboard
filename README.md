# 🧠 BRISC 2025 Brain Tumor Classification Dashboard

A machine learning project for **brain tumor classification from MRI images** using both classical machine learning algorithms and a deep learning model based on **EfficientNetB7**.

The project includes data preprocessing, feature extraction, dimensionality reduction, model training, evaluation, and an interactive dashboard for brain tumor classification.

---

## 📌 Project Overview

Brain tumor classification from MRI images is an important medical imaging task. This project explores different machine learning approaches for classifying brain MRI images into four categories:

* **Glioma**
* **Meningioma**
* **No Tumor**
* **Pituitary**

The project compares classical machine learning models with a deep learning approach using **EfficientNetB7**.

---

## 🎯 Objectives

The main objectives of this project are:

* Preprocess and organize brain MRI image data.
* Extract meaningful image features.
* Apply dimensionality reduction using PCA.
* Train multiple classical machine learning models.
* Train an EfficientNetB7 deep learning model.
* Evaluate model performance using standard classification metrics.
* Develop a dashboard for model-based brain tumor classification.
* Provide a structured and reproducible machine learning project.

---

## 📂 Dataset

The project uses the **BRISC 2025** brain MRI classification dataset.

### Classes

| Class      | Description                |
| ---------- | -------------------------- |
| Glioma     | Glioma brain tumor         |
| Meningioma | Meningioma brain tumor     |
| No Tumor   | MRI with no detected tumor |
| Pituitary  | Pituitary tumor            |

### Dataset Distribution

| Class      | Training Images |
| ---------- | --------------: |
| Glioma     |           1,147 |
| Meningioma |           1,329 |
| No Tumor   |           1,067 |
| Pituitary  |           1,457 |
| **Total**  |       **5,000** |

The test set contains **1,000 MRI images**.

During preprocessing, one corrupted training image was removed. Therefore, **4,999 training images remained**, while all **1,000 test images were retained**.

---

## 🔄 Project Workflow

```text
Brain MRI Dataset
        ↓
Data Preprocessing
        ↓
Feature Extraction
        ↓
PCA Dimensionality Reduction
        ↓
Classical ML Models
        ↓
EfficientNetB7 Deep Learning
        ↓
Model Evaluation
        ↓
Interactive Dashboard
```

---

## 🤖 Machine Learning Models

The following classical machine learning algorithms were trained and evaluated:

* Decision Tree
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Random Forest

### Classical Model Results

| Model               | Accuracy | Macro F1-Score |
| ------------------- | -------: | -------------: |
| Decision Tree       |   66.70% |         67.14% |
| KNN                 |   89.10% |         89.86% |
| Logistic Regression |   90.90% |         91.35% |
| Random Forest       |   88.70% |         89.65% |

---

## 🧠 EfficientNetB7

A transfer-learning based **EfficientNetB7** model was trained for brain tumor classification.

### Configuration

* **Model:** EfficientNetB7
* **Image Size:** 600 × 600
* **Batch Size:** 4
* **Number of Classes:** 4
* **Training Images:** 4,999
* **Test Images:** 1,000
* **Training Epochs:** 15

The best validation accuracy reached approximately **99.50%** during training.

### Final Test Performance

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **92.30%** |
| Macro Precision | **93.50%** |
| Macro Recall    | **93.06%** |
| Macro F1-Score  | **93.19%** |

The final model was evaluated on the untouched test set.

---

## 📊 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

These metrics provide a detailed view of classification performance across all four tumor classes.

---

## 📁 Project Structure

```text
BRISC-2025-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── classification_task/
│   ├── train/
│   └── test/
│
├── features/
│   ├── X_test_pca.npy
│   ├── X_train_pca.npy
│   ├── y_test.npy
│   ├── y_train.npy
│   ├── label_map.json
│   ├── pca_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│
└── results/
```

> The large EfficientNetB7 `.keras` model file is hosted separately because GitHub has file-size limitations.

---

## 💾 Trained EfficientNetB7 Model

The trained EfficientNetB7 model is available through Google Drive:

**[Download EfficientNetB7 Model](https://drive.google.com/file/d/1cGeoEtKkX_A9ytis9J-b6Xg9Qm6C5qey/view?usp=drive_link)**

Model file:

```text
efficientnetb7_best.keras
```

---

## 🖥️ Dashboard

The project includes an interactive dashboard designed to make the trained models easier to use.

The dashboard provides a user-friendly interface for:

* Uploading brain MRI images
* Processing input images
* Running model predictions
* Displaying the predicted tumor class
* Presenting classification results

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/hareem-tahir12/BRISC-2025-Dashboard.git
```

Navigate to the project directory:

```bash
cd BRISC-2025-Dashboard
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Streamlit
* PCA
* EfficientNetB7
* Git & GitHub

---

## 📈 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Image data preprocessing
* Feature extraction
* PCA dimensionality reduction
* Classical machine learning
* Transfer learning
* EfficientNetB7
* Model evaluation
* Classification metrics
* Model serialization
* Dashboard development
* Git and GitHub project management

---

## 👩‍💻 Team

**Manahil Shabbir**
**Hareem Tahir**

**Semester:** 7th Semester
**Instructor:** Tariq Mehmood

---

## 📌 Project Status

The project includes:

* ✅ Dataset preparation
* ✅ Data preprocessing
* ✅ Feature extraction
* ✅ PCA
* ✅ Classical ML models
* ✅ EfficientNetB7 training
* ✅ Model evaluation
* ✅ Results
* ✅ Interactive dashboard
* ✅ GitHub repository
* ✅ Trained model hosted externally

---

## 📜 Disclaimer

This project is developed for **academic and educational purposes**. The predictions produced by the system should not be considered a medical diagnosis or a replacement for professional medical evaluation.
