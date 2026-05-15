"""
Evaluates predictions against ground truth.
Use --test to evaluate on the held-out test split, otherwise val.
"""
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import config as C


def angular_error(pred, gt):
    d = abs(pred - gt) % 360
    return min(d, 360 - d)


def match_image(pred_rows, gt_rows, th):
    n_pred, n_gt = len(pred_rows), len(gt_rows)
    if n_pred == 0 and n_gt == 0: return dict(tp=0, fp=0, fn=0, dist=[], ang=[])
    if n_pred == 0:                return dict(tp=0, fp=0, fn=n_gt, dist=[], ang=[])
    if n_gt == 0:                  return dict(tp=0, fp=n_pred, fn=0, dist=[], ang=[])

    pred_xy = pred_rows[["cx", "cy"]].values.astype(float)
    gt_xy   = gt_rows[["center_x", "center_y"]].values.astype(float)
    dists = np.linalg.norm(pred_xy[:, None] - gt_xy[None, :], axis=2)
    cost = dists.copy()
    cost[cost > th] = 1e9
    row_ind, col_ind = linear_sum_assignment(cost)
    tp_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if dists[r, c] <= th]
    tp = len(tp_pairs)

    dist_errs, ang_errs = [], []
    for r, c in tp_pairs:
        dist_errs.append(float(dists[r, c]))
        pa = pred_rows.iloc[r]["angle_deg"]
        ga = gt_rows.iloc[c]["angle_deg"]
        if pd.notna(pa) and pd.notna(ga):
            ang_errs.append(angular_error(float(pa), float(ga)))
    return dict(tp=tp, fp=n_pred - tp, fn=n_gt - tp, dist=dist_errs, ang=ang_errs)


def evaluate(pred_df, gt_df, th=C.EVAL_DIST_THRESH):
    all_imgs = gt_df["image"].unique()
    per_img = []
    all_dist, all_ang = [], []

    for img_name in all_imgs:
        gt_rows  = gt_df[gt_df["image"] == img_name]
        pred_rows = pred_df[pred_df["image"] == img_name] if len(pred_df) > 0 else pd.DataFrame()
        m = match_image(pred_rows, gt_rows, th)
        n_gt = len(gt_rows); n_pred = len(pred_rows)
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        ae = [x for x in m["ang"] if not np.isnan(x)]
        per_img.append(dict(
            image=img_name, n_gt=n_gt, n_pred=n_pred, tp=tp, fp=fp, fn=fn,
            precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
            dist_mae=round(np.mean(m["dist"]), 2) if m["dist"] else None,
            angle_mae=round(np.mean(ae), 2) if ae else None,
        ))
        all_dist += m["dist"]
        all_ang  += ae

    per_img_df = pd.DataFrame(per_img)
    total_tp = int(per_img_df["tp"].sum())
    total_fp = int(per_img_df["fp"].sum())
    total_fn = int(per_img_df["fn"].sum())
    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    rec  = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    def s(fn, arr): return round(float(fn(arr)), 3) if arr else None
    summary = dict(
        total_gt=int(gt_df.shape[0]), total_pred=int(len(pred_df)),
        total_tp=total_tp, total_fp=total_fp, total_fn=total_fn,
        dist_thresh_px=th,
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        dist_mae_px=s(np.mean, all_dist),
        dist_rmse_px=s(lambda x: np.sqrt(np.mean(np.array(x) ** 2)), all_dist),
        angle_mae_deg=s(np.mean, all_ang),
        angle_rmse_deg=s(lambda x: np.sqrt(np.mean(np.array(x) ** 2)), all_ang),
        angle_within_10deg=s(lambda x: np.mean(np.array(x) <= 10) * 100, all_ang),
        angle_within_22deg=s(lambda x: np.mean(np.array(x) <= 22.5) * 100, all_ang),
        angle_within_45deg=s(lambda x: np.mean(np.array(x) <= 45) * 100, all_ang),
        angle_coverage_pct=round(len(all_ang) / total_tp * 100, 1) if total_tp > 0 else 0,
    )
    return summary, per_img_df


def print_summary(s, label=""):
    print(f"\n{'=' * 58}")
    print(f" RESULTS {label}".strip())
    print(f"{'=' * 58}")
    print(f"  Ground-truth tubes : {s['total_gt']}")
    print(f"  Predicted tubes    : {s['total_pred']}")
    print(f"  True Positives     : {s['total_tp']}")
    print(f"  False Positives    : {s['total_fp']}")
    print(f"  False Negatives    : {s['total_fn']}")
    print(f"  Match threshold    : {s['dist_thresh_px']} px")
    print(f"  Precision          : {s['precision']:.4f}")
    print(f"  Recall             : {s['recall']:.4f}")
    print(f"  F1                 : {s['f1']:.4f}")
    print(f"  Centre MAE         : {s['dist_mae_px']} px")
    print(f"  Centre RMSE        : {s['dist_rmse_px']} px")
    print(f"  Angle MAE          : {s['angle_mae_deg']}°")
    print(f"  Angle RMSE         : {s['angle_rmse_deg']}°")
    print(f"  Angle ≤10°         : {s['angle_within_10deg']}%")
    print(f"  Angle ≤22.5°       : {s['angle_within_22deg']}%")
    print(f"  Angle ≤45°         : {s['angle_within_45deg']}%")
    print(f"  Angle coverage     : {s['angle_coverage_pct']}% of TPs")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds",  required=True)
    p.add_argument("--annots", required=True)
    p.add_argument("--out",    default="results")
    p.add_argument("--dist",   type=int, default=C.EVAL_DIST_THRESH)
    p.add_argument("--test",   action="store_true")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    preds = pd.read_csv(args.preds)
    gt    = pd.read_csv(args.annots)

    if args.test:
        test_imgs = set(Path("yolo_dataset/test_images.txt").read_text().strip().splitlines())
        preds = preds[preds["image"].isin(test_imgs)]
        gt    = gt[gt["image"].isin(test_imgs)]
        label = "test"
    else:
        val_imgs = set(p.name for p in Path("yolo_dataset/images/val").glob("*.png"))
        preds = preds[preds["image"].isin(val_imgs)]
        gt    = gt[gt["image"].isin(val_imgs)]
        label = "val"

    print(f"Evaluating on {len(gt['image'].unique())} {label} images")
    summary, per_img = evaluate(preds, gt, args.dist)
    print_summary(summary, label=f"YOLOv8 detection + angle regression ({label})")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    per_img.to_csv(out / "per_image.csv", index=False)
    print(f"Results saved → {out}/")