import os
import random
from glob import glob

img_src = "./datasets/training_image"          # original image
# img_src = "./datasets/preprocessed_image"    # preprocess image
label_src = "./datasets/training_label"
test_src = "./datasets/testing_image"

split_dir = "./datasets"
os.makedirs(split_dir, exist_ok=True)

# YOLO 預設會找 "images" 與 "labels" 這兩層
train_img_dir = "./datasets/yolo_train/images"
train_lbl_dir = "./datasets/yolo_train/labels"
val_img_dir   = "./datasets/yolo_val/images"
val_lbl_dir   = "./datasets/yolo_val/labels"
test_img_dir  = "./datasets/yolo_test/images"

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir]:
    os.makedirs(d, exist_ok=True)


# split
patients = sorted(os.listdir(img_src))
split = int(len(patients) * 0.9)
# random.seed(42) 
# random.shuffle(patients)
train_patients = patients[:split]
val_patients   = patients[split:]

print(f"共 {len(patients)} 個病人，訓練 {len(train_patients)}，驗證 {len(val_patients)}")


# 修正 label 格式 (0.000 → 0)
def fix_label_format(lbl_path):
    if not os.path.exists(lbl_path):
        return False

    changed = False
    new_lines = []

    with open(lbl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls_raw = parts[0]

        try:
            cls_int = int(float(cls_raw))  # 0.000 → 0
        except:
            cls_int = 0

        new_line = f"{cls_int} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
        new_lines.append(new_line)

        if cls_raw != str(cls_int):
            changed = True

    with open(lbl_path, "w") as f:
        f.writelines(new_lines)

    return changed


# Train/Val set symbolic links
def collect_pairs(patient_list, dst_img_dir, dst_lbl_dir):
    pairs = []
    for p in patient_list:
        imgs = glob(os.path.join(img_src, p, "*.png"))
        for img in imgs:
            lbl = img.replace("training_image", "training_label").replace(".png", ".txt")
            # 若是 preprocess 影像請改：
            # lbl = img.replace("preprocessed_image", "training_label").replace(".png", ".txt")

            if os.path.exists(lbl):
                fix_label_format(lbl)

                dst_img = os.path.join(dst_img_dir, os.path.basename(img))
                dst_lbl = os.path.join(dst_lbl_dir, os.path.basename(lbl))

                if not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(img), dst_img)
                if not os.path.exists(dst_lbl):
                    os.symlink(os.path.abspath(lbl), dst_lbl)

                pairs.append(dst_img)
    return pairs


train_imgs = collect_pairs(train_patients, train_img_dir, train_lbl_dir)
val_imgs   = collect_pairs(val_patients, val_img_dir, val_lbl_dir)

print(f"Training 共: {len(train_imgs)}張影像, Valliation 共: {len(val_imgs)}張影像")


# Testing set symbolic links
count = 0
patients_test = sorted(os.listdir(test_src))
for pid in patients_test:
    patient_dir = os.path.join(test_src, pid)
    if not os.path.isdir(patient_dir):
        continue

    imgs = sorted(glob(os.path.join(patient_dir, "*.png")))
    for img in imgs:
        basename = os.path.basename(img)
        dst = os.path.join(test_img_dir, basename)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(img), dst)
            count += 1

print(f"Testing 共 {count} 張影像")
print(f"📂  輸出目錄：{os.path.abspath(test_img_dir)}")
