import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Base path to 'idc' folder
base_dir = r"C:\Users\hp\Desktop\jupyy\breast-cancer-classification\breast-cancer-classification\datasets\idc"
splits = ['training', 'validation', 'testing']
img_size = 224

# Load MobileNetV2 without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

data = []

for split in splits:
    for label in ['0', '1']:  # 0 = non-cancer, 1 = cancer
        folder = os.path.join(base_dir, split, label)
        for img_name in tqdm(os.listdir(folder), desc=f"Processing {split}/{label}"):
            try:
                img_path = os.path.join(folder, img_name)

                # Read and preprocess image
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))

                if img.shape[-1] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)

                # Extract features
                features = model.predict(img, verbose=0).flatten()
                row = [img_name, split, int(label)] + features.tolist()
                data.append(row)
            except Exception as e:
                print(f"⚠️ Skipping {img_name} due to error: {e}")

 
# Create dataframe
feature_count = len(data[0]) - 3  # exclude filename, split, label
columns = ['filename', 'split', 'label'] + [f'feature{i}' for i in range(feature_count)]
df = pd.DataFrame(data, columns=columns)

# Save CSV
csv_path = os.path.join(base_dir, 'mobilenetv2_features.csv')
df.to_csv(csv_path, index=False)

print(f"✅ CSV saved to: {csv_path}")
