import os
from glob import glob

img_src = "./datasets/training_image" # original img
# img_src = "./datasets/preprocessed_image" # preprocessd img
label_src = "./datasets/training_label"
split_dir = "./datasets"
os.makedirs(split_dir, exist_ok=True)


# YOLO 預設會找 "images" 與 "labels" 這兩層
train_img_dir = "./datasets/yolo_train/images"
train_lbl_dir = "./datasets/yolo_train/labels"
val_img_dir   = "./datasets/yolo_val/images"
val_lbl_dir   = "./datasets/yolo_val/labels"

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# split
patients = sorted(os.listdir(img_src))
split = int(len(patients) * 0.6)
train_patients = patients[:split]
val_patients   = patients[split:]
print(f"共 {len(patients)} 個病人，訓練 {len(train_patients)}，驗證 {len(val_patients)}")

# 收集影像與標籤
def collect_pairs(patient_list, dst_img_dir, dst_lbl_dir):
    pairs = []
    for p in patient_list:
        imgs = glob(os.path.join(img_src, p, "*.png"))
        for img in imgs:
            lbl = img.replace("training_image", "training_label").replace(".png", ".txt") # original img
            # lbl = img.replace("preprocessed_image", "training_label").replace(".png", ".txt") # preprocessd img
            if os.path.exists(lbl):
                # 用 symbolic link 比較省空間，也可改成 shutil.copy
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

print(f"✅ YOLO 結構建立完成")
print(f"Train 影像: {len(train_imgs)}, Val 影像: {len(val_imgs)}")