"""
Builds YOLOv8 detection dataset with train/val/test splits.
Labels are axis-aligned bounding boxes around each tube lid.
"""
import argparse, shutil, random
from pathlib import Path
import pandas as pd
import config as C


def bbox_to_yolo(row):
    cx = float(row["center_x"]) / C.IMG_W
    cy = float(row["center_y"]) / C.IMG_H
    w  = float(row["bbox_w"])   / C.IMG_W
    h  = float(row["bbox_h"])   / C.IMG_H
    return cx, cy, w, h


def write_label(path, rows):
    lines = []
    for _, row in rows.iterrows():
        cx, cy, w, h = bbox_to_yolo(row)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))


def prepare(images_dir, annots_csv, out_dir="yolo_dataset"):
    random.seed(C.SEED)
    gt = pd.read_csv(annots_csv)
    img_dir = Path(images_dir)
    out = Path(out_dir)

    all_imgs = sorted(gt["image"].unique())
    random.shuffle(all_imgs)
    n_val  = max(1, int(len(all_imgs) * C.VAL_RATIO))
    n_test = max(1, int(len(all_imgs) * C.TEST_RATIO))
    val_set  = set(all_imgs[:n_val])
    test_set = set(all_imgs[n_val:n_val + n_test])
    trn_set  = set(all_imgs[n_val + n_test:])
    print(f"Train: {len(trn_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_name in all_imgs:
        split = "train" if img_name in trn_set else ("val" if img_name in val_set else "test")
        src = img_dir / img_name
        if src.exists():
            shutil.copy2(src, out / "images" / split / img_name)
        rows = gt[gt["image"] == img_name]
        write_label(out / "labels" / split / (Path(img_name).stem + ".txt"), rows)

    yaml_text = f"""path: {out.resolve()}
train: images/train
val: images/val
nc: 1
names: ['tube_lid']
"""
    (out / "dataset.yaml").write_text(yaml_text)
    (out / "test_images.txt").write_text("\n".join(sorted(test_set)))
    print(f"Dataset ready → {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--annots", required=True)
    p.add_argument("--out", default="yolo_dataset")
    args = p.parse_args()
    prepare(args.images, args.annots, args.out)