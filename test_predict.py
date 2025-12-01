import os
import glob
from ultralytics import YOLO

WEIGHTS = "./runs/yolo11l_preprocessed/weights/best.pt"
SOURCE = "./datasets/yolo_test/images"
DEVICE = "0"
IMGSZ = 512
CONF = 0.7
OUTDIR = "./runs"

# ============================================================
# Utility: Convert YOLO normalized xywh -> original xyxy
# ============================================================
def yolo_to_xyxy(xc, yc, w, h, img_size=512):
    x_min = round((xc - w / 2) * img_size)
    x_max = round((xc + w / 2) * img_size)
    y_min = round((yc - h / 2) * img_size)
    y_max = round((yc + h / 2) * img_size)
    return x_min, y_min, x_max, y_max


# ============================================================
# Main routine
# ============================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"🚀 Loading model: {WEIGHTS}")
    model = YOLO(WEIGHTS)

    print(f"📂 Running inference on: {SOURCE}")
    results = model.predict(
        source=SOURCE,
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF,
        save_txt=True,
        save_conf=True,
        save=True,           # whether save img or not
        project=OUTDIR,
        name="",
        exist_ok=True,
        workers=0
    )

    # ============================================================
    # Convert YOLO txt -> merged.txt for AICUP submission
    # ============================================================
    label_dir = os.path.join(OUTDIR, "predict/labels")
    output_txt = os.path.join(OUTDIR, "predict/merged.txt")
    img_size = IMGSZ

    print("🧾 Converting YOLO predictions to AICUP submission format...")
    count = 0
    with open(output_txt, "w") as out_f:
        for txt_file in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
            img_name = os.path.splitext(os.path.basename(txt_file))[0]
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls, xc, yc, w, h, conf = parts
                    xc, yc, w, h, conf = map(float, [xc, yc, w, h, conf])
                    x_min, y_min, x_max, y_max = yolo_to_xyxy(xc, yc, w, h, img_size)
                    out_f.write(f"{img_name} {int(cls)} {conf:.4f} {x_min} {y_min} {x_max} {y_max}\n")
                    count += 1

    print(f"✅ Finished! Total {count} detections saved.")
    print(f"📄 Merged result file: {output_txt}")


if __name__ == "__main__":
    main()
