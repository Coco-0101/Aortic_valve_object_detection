import cv2 as cv
import numpy as np
import os
from tqdm import tqdm

input_dir = "./datasets/training_image"
process_dir = "./datasets/preprocessed_image"

os.makedirs(process_dir, exist_ok=True)

target_brightness = 40
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for patient in sorted(os.listdir(input_dir)):
    patient_dir = os.path.join(input_dir, patient)

    if not os.path.isdir(patient_dir):
        continue  # 跳過不是資料夾的東西

    os.makedirs(os.path.join(process_dir, patient), exist_ok=True)

    for file in tqdm(os.listdir(patient_dir), desc=f"Processing {patient}"):
        
        if not file.lower().endswith(valid_ext):
            continue
        
        img_path = os.path.join(patient_dir, file)
        img = cv.imread(img_path)

        if img is None:
            print(f"[SKIP] Cannot read: {img_path}")
            continue

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        original_brightness = img_gray.mean()

        if original_brightness < 1:
            original_brightness = 1

        gamma = np.log(target_brightness / 255.0) / np.log(original_brightness / 255.0)

        darken_img = np.power(img / 255.0, gamma) * 255
        darken_img = np.clip(darken_img, 0, 255).astype(np.uint8)

        out_dir = os.path.join(process_dir, patient)
        os.makedirs(out_dir, exist_ok=True)
        cv.imwrite(os.path.join(out_dir, file), darken_img)
