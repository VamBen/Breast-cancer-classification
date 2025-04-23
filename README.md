# Breast-cancer-classification

##  Dataset

The dataset used is the **Breast Histopathology Images** from Kaggle:
🔗 [Click here to access it](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

This IDC dataset contains:
- Microscopic images of breast tissue, labeled as:
  - `0`: Non-cancerous
  - `1`: Cancerous

---

## Project Overview

### 1: Feature Extraction (Deep Learning)
- Images are processed using **MobileNetV2** (pre-trained on ImageNet).
- The top classification layer is removed.
- Features are extracted using **Global Average Pooling**.
- Extracted features are saved into a CSV file for further processing.

###  2: Classification (Machine Learning)
- Features from the CSV are fed into **decision Tree Classifier** initially, but a more robust approach is taken later via **XGBoost Classifier**.
- Handles class imbalance using SMOTE
- Performance is evaluated using accuracy and F1-score.


