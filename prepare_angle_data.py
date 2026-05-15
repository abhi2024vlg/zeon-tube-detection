"""
Creates fixed-size crops centered on each GT tube for angle regression.
Reuses the train/val/test splits produced by prepare_data.py.
"""
import argparse, cv2
import pandas as pd
from pathlib import Path
import config as C


def crop_at(img, cx, cy):
    H, W = img.shape[:2]
    half = C.CROP_SIZE // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    pad_l = max(0, -x1); pad_t = max(0, -y1)
    pad_r = max(0, x2 - W); pad_b = max(0, y2 - H)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    crop = img[y1c:y2c, x1c:x2c]
    if any([pad_l, pad_r, pad_t, pad_b]):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
    return cv2.resize(crop, (C.CROP_SIZE, C.CROP_SIZE))


def prepare(images_dir, annots_csv, out_dir="angle_dataset"):
    gt = pd.read_csv(annots_csv)
    val_imgs  = set(p.name for p in Path("yolo_dataset/images/val").glob("*.png"))
    test_imgs = set(Path("yolo_dataset/test_images.txt").read_text().strip().splitlines())

    out = Path(out_dir)
    for s in ["train", "val", "test"]:
        (out / s).mkdir(parents=True, exist_ok=True)

    rows = []
    for img_name in sorted(gt["image"].unique()):
        split = "test" if img_name in test_imgs else ("val" if img_name in val_imgs else "train")
        img = cv2.imread(str(Path(images_dir) / img_name))
        if img is None:
            continue
        for idx, r in gt[gt["image"] == img_name].iterrows():
            cx = int(round(float(r["center_x"])))
            cy = int(round(float(r["center_y"])))
            crop = crop_at(img, cx, cy)
            fname = f"{Path(img_name).stem}_{idx}.png"
            cv2.imwrite(str(out / split / fname), crop)
            rows.append({"crop": fname, "split": split,
                         "image": img_name, "angle_deg": float(r["angle_deg"])})

    df = pd.DataFrame(rows)
    df.to_csv(out / "angles.csv", index=False)
    print(f"Crops → {out}/")
    print(df.groupby("split").size())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--annots", required=True)
    p.add_argument("--out", default="angle_dataset")
    args = p.parse_args()
    prepare(args.images, args.annots, args.out)