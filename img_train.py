import sys, os, builtins
import pandas as pd
import torch
from ultralytics import YOLO
import ultralytics.nn.modules
import ultralytics.nn.tasks
from fightingcv_attention.attention.SEAttention import SEAttention

# 註冊 SEAttention
sys.modules['SEAttention'] = SEAttention
builtins.SEAttention = SEAttention
ultralytics.nn.modules.SEAttention = SEAttention
ultralytics.nn.tasks.SEAttention = SEAttention
globals()['SEAttention'] = SEAttention

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
    
RUN_NAME = "yolo11l_preprocessed"

model_cfg = "./configs/aortic_valve_yolo11L_SEAttention.yaml"
# model = YOLO("yolo12l.pt")  # 直接使用預訓練權重，不用自訂結構
model = YOLO(model_cfg).load("yolo11l.pt")  # 在YOLO11l預訓練權重上微調

# 訓練 
model.train(
    device="0",
    project="./runs",
    name=RUN_NAME,
    workers=0,
    data=yaml_path,
    epochs=150,
    imgsz=512,
    batch=4,
    patience=50,
    auto_augment=None,
    mosaic=0.0,
    fliplr=0.0,
    scale=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    lr0=0.0005,
    cos_lr=True,
    weight_decay=0.0002,
    box=10, cls=0.2, kobj=2.5, dfl=1.5,
    dropout=0.05,
)

# 取result.csv最後結果
target_folder = os.path.join("./runs", RUN_NAME)
results_csv = os.path.join(target_folder, "results.csv")

df = pd.read_csv(results_csv)
last = df.iloc[-1]

print("\n最終訓練結果：")
print(f"Precision: {last['metrics/precision(B)']:.3f}")
print(f"Recall: {last['metrics/recall(B)']:.3f}")
print(f"mAP@0.5: {last['metrics/mAP50(B)']:.3f}")
print(f"mAP@0.5:0.95: {last['metrics/mAP50-95(B)']:.3f}\n")

summary_path = os.path.join(target_folder, 'final_metrics.txt')
with open(summary_path, 'w') as f:
    f.write(f"Precision: {last['metrics/precision(B)']:.3f}\n")
    f.write(f"Recall: {last['metrics/recall(B)']:.3f}\n")
    f.write(f"mAP@0.5: {last['metrics/mAP50(B)']:.3f}\n")
    f.write(f"mAP@0.5:0.95: {last['metrics/mAP50-95(B)']:.3f}\n")

print(f"已將最終結果儲存到: {summary_path}")
