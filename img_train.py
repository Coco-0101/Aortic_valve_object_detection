import os
import pandas as pd
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()


train = "./datasets/yolo_train"
val = "./datasets/yolo_val"

yaml_path = "./configs/aortic_valve.yaml"
os.makedirs("./configs", exist_ok=True)

with open(yaml_path, "w") as f:
    f.write(f"""
train: {os.path.abspath(train)}
val: {os.path.abspath(val)}

nc: 1
names: ['aortic_valve']
""")


model = YOLO("yolo11l.pt")  # 直接使用預訓練權重，不用自訂結構

# === 訓練 ===
model.train(
    device="0",
    project="./runs",
    name="preprocessed_train",
    workers=1,
    data=yaml_path,
    epochs=100,
    imgsz=512,
    batch=16,
    patience=30,
    mosaic=0.0,
    fliplr=0.0,
    scale=0.0,
)

# === 抓取最新 results.csv 並輸出最終結果 ===
runs_path = "./runs"
candidate_paths = []

for root, dirs, files in os.walk(runs_path):
    if 'results.csv' in files:
        candidate_paths.append(os.path.join(root, 'results.csv'))

if not candidate_paths:
    raise FileNotFoundError("⚠️ 沒找到任何 results.csv，請確認訓練已完成。")

latest_csv = max(candidate_paths, key=os.path.getmtime)
print(f"📊 最新訓練結果檔案: {latest_csv}")

df = pd.read_csv(latest_csv)
last = df.iloc[-1]

print("\n📈 最終訓練結果：")
print(f"Precision: {last['metrics/precision(B)']:.3f}")
print(f"Recall: {last['metrics/recall(B)']:.3f}")
print(f"mAP@0.5: {last['metrics/mAP50(B)']:.3f}")
print(f"mAP@0.5:0.95: {last['metrics/mAP50-95(B)']:.3f}")

summary_path = os.path.join(os.path.dirname(latest_csv), 'final_metrics.txt')
with open(summary_path, 'w') as f:
    f.write(f"Precision: {last['metrics/precision(B)']:.3f}\n")
    f.write(f"Recall: {last['metrics/recall(B)']:.3f}\n")
    f.write(f"mAP@0.5: {last['metrics/mAP50(B)']:.3f}\n")
    f.write(f"mAP@0.5:0.95: {last['metrics/mAP50-95(B)']:.3f}\n")

print(f"✅ 已將最終結果儲存到: {summary_path}")