# Breast-cancer-classification

## 📁 Dataset

The dataset used is the **Breast Histopathology Images** from Kaggle:
🔗 [Click here to access it](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

This dataset contains:
- Microscopic images of breast tissue.
- Labeled as:
  - `0`: Non-cancerous
  - `1`: IDC positive (cancerous)

---

## 🧠 Project Overview

### 📌 Step 1: Feature Extraction (Deep Learning)
- Images are processed using **MobileNetV2** (pre-trained on ImageNet).
- The top classification layer is removed.
- Features are extracted using **Global Average Pooling**.
- Extracted features are saved into a CSV file for further processing.

### 📌 Step 2: Classification (Machine Learning)
- Features from the CSV are fed into an **XGBoost Classifier**.
- Handles class imbalance using `scale_pos_weight`.
- Uses `logloss` as the evaluation metric.
- Performance is evaluated using accuracy, F1-score, and ROC AUC.

---

## 🔧 Technologies Used

- Python 🐍
- TensorFlow / Keras 🧠
- OpenCV 📷
- XGBoost 🌲
- Pandas & NumPy 📊
- tqdm ⏳
