"""
End-to-end inference: YOLOv8 detection + angle regression.
Writes a predictions.csv with image, cx, cy, angle_deg, ...
"""
import argparse, math
import cv2, numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import config as C

cnn_tf = transforms.Compose([
    transforms.Resize((C.CROP_SIZE, C.CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_cnn(path, device):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(1280, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval().to(device)


def crop_at(img, cx, cy):
    H, W = img.shape[:2]
    half = C.CROP_SIZE // 2
    x1, y1 = int(cx) - half, int(cy) - half
    x2, y2 = int(cx) + half, int(cy) + half
    pad_l = max(0, -x1); pad_t = max(0, -y1)
    pad_r = max(0, x2 - W); pad_b = max(0, y2 - H)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    crop = img[y1c:y2c, x1c:x2c]
    if any([pad_l, pad_r, pad_t, pad_b]):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
    return cv2.resize(crop, (C.CROP_SIZE, C.CROP_SIZE))


def predict_angle(model, crop_bgr, device):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = cnn_tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    sin_t, cos_t = float(pred[0]), float(pred[1])
    norm = math.hypot(sin_t, cos_t) + 1e-8
    sin_t, cos_t = sin_t / norm, cos_t / norm
    angle = math.degrees(math.atan2(sin_t, cos_t)) % 360
    return float(angle), float(norm)


def run(yolo_w, cnn_w, images_dir, out_dir, conf=0.25, iou=0.4, vis=True):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    detector = YOLO(yolo_w)
    angle_model = load_cnn(cnn_w, device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    vis_d = out / "overlays"
    if vis:
        vis_d.mkdir(parents=True, exist_ok=True)

    rows = []
    paths = sorted(Path(images_dir).glob("*.png"))
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = detector.predict(source=str(p), conf=conf, iou=iou,
                               imgsz=640, verbose=False)
        vis_img = img.copy() if vis else None
        n = 0
        for r in res:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), det_conf in zip(xyxy, confs):
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = x2 - x1, y2 - y1
                crop = crop_at(img, cx, cy)
                angle, ang_conf = predict_angle(angle_model, crop, device)
                rows.append({
                    "image": p.name,
                    "cx": round(float(cx), 2),
                    "cy": round(float(cy), 2),
                    "radius": round(float((bw + bh) / 4), 2),
                    "angle_deg": round(angle, 2),
                    "det_conf": round(float(det_conf), 4),
                    "angle_conf": round(ang_conf, 4),
                    "bbox_w": round(float(bw), 2),
                    "bbox_h": round(float(bh), 2),
                })
                n += 1
                if vis_img is not None:
                    r_px = max(8, int((bw + bh) / 4))
                    cxi, cyi = int(cx), int(cy)
                    cv2.circle(vis_img, (cxi, cyi), r_px, (0, 140, 255), 2)
                    rad = math.radians(angle)
                    ex = int(cx + r_px * 1.4 * math.cos(rad))
                    ey = int(cy - r_px * 1.4 * math.sin(rad))
                    cv2.arrowedLine(vis_img, (cxi, cyi), (ex, ey),
                                    (0, 140, 255), 2, tipLength=0.3)
        if vis_img is not None:
            cv2.imwrite(str(vis_d / p.name), vis_img)
        print(f"  [{i:3d}/{len(paths)}] {p.name}: {n} detected")

    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions.csv", index=False)
    print(f"Saved {len(df)} predictions → {out/'predictions.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--cnn_weights",  required=True)
    p.add_argument("--images",       required=True)
    p.add_argument("--out_dir",      default="results")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.4)
    p.add_argument("--no_vis",       action="store_true")
    args = p.parse_args()
    run(args.yolo_weights, args.cnn_weights, args.images, args.out_dir,
        conf=args.conf, iou=args.iou, vis=not args.no_vis)