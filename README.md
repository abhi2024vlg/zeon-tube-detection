# Microcentrifuge Tube Detection & Rotation

Detects tube lids in overhead RGB images and estimates their full 0-360° orientation. Two-stage pipeline: YOLOv8n for axis-aligned detection, MobileNetV2 for orientation regression on the unit circle.

## Results

Evaluated on a 15% held-out test split (10 images / 50 tubes), seed 42. Predictions matched to ground truth via Hungarian assignment on centre distance with a 15 px threshold.

| Metric | Test | Val |
|---|---|---|
| Precision | **1.0000** | **1.0000** |
| Recall | **1.0000** | **1.0000** |
| F1 | **1.0000** | **1.0000** |
| Centre MAE | **0.80 px** | 0.98 px |
| Centre RMSE | 0.93 px | 1.12 px |
| **Angle MAE** | **4.98°** | **6.67°** |
| Angle RMSE | 6.55° | 10.77° |
| Angle ≤ 10° | **90 %** | 82 % |
| Angle ≤ 22.5° | **100 %** | 92 % |
| Angle ≤ 45° | 100 % | 98 % |

Tight centre localisation (~1 px MAE) means the matching result is robust to threshold choice - P=R=F1=1.0 holds for any threshold ≥ 5 px.

## Approach

Detection and orientation are decoupled into two stages.

**Stage 1 — Detection.** YOLOv8n is fine-tuned on the train split to localise tube lids. The model predicts axis-aligned bounding boxes; downstream we use only the box centre `(cx, cy)`.

**Stage 2 — Orientation.** A MobileNetV2 (ImageNet-initialised) takes a 128×128 crop centred on each detected tube and regresses two outputs interpreted as `(sin θ, cos θ)`. Outputs are L2-normalised to the unit circle before loss and inference; the final angle is `atan2(sin θ, cos θ)` wrapped to `[0, 360°)`.

This formulation handles the full circular range continuously and removes the need for a separate head/tail classifier or any convention conversion between detector output and the annotation frame.

Training augmentations expose the regressor to all orientations from a small (271-tube) training set: random rotation in `[-180°, 180°]` with corresponding label correction, horizontal flip (θ → 180-θ), vertical flip (θ → -θ), and colour jitter. Loss is the squared error between predicted and target unit vectors. Optimiser: AdamW, lr 1e-3 with cosine annealing over 60 epochs. Best checkpoint by validation angular MAE.

## Reproduction

```bash
# 1. Install
pip install -r requirements.txt

# 2. Build detection dataset and train YOLOv8n
python3 prepare_data.py --images images --annots annotations.csv --out yolo_dataset
python3 train_yolo.py

# 3. Build crop dataset and train angle regressor
python3 prepare_angle_data.py --images images --annots annotations.csv --out angle_dataset
python3 train_angle.py --data angle_dataset --epochs 60

# 4. Inference on test and val splits
python3 infer.py \
    --yolo_weights runs/detect/tube_detector/weights/best.pt \
    --cnn_weights  angle_best.pt \
    --images       yolo_dataset/images/test \
    --out_dir      results_test

python3 infer.py \
    --yolo_weights runs/detect/tube_detector/weights/best.pt \
    --cnn_weights  angle_best.pt \
    --images       yolo_dataset/images/val \
    --out_dir      results_val

# 5. Evaluate
python3 evaluate.py --test \
    --preds  results_test/predictions.csv \
    --annots annotations.csv \
    --out    results_test_metrics

python3 evaluate.py \
    --preds  results_val/predictions.csv \
    --annots annotations.csv \
    --out    results_val_metrics
```



## Files

| File | Purpose |
|---|---|
| `config.py` | Shared constants (image size, crop size, match threshold, split ratios) |
| `prepare_data.py` | Builds YOLO detection dataset with train/val/test split |
| `train_yolo.py` | Fine-tunes YOLOv8n |
| `prepare_angle_data.py` | Builds 128×128 crop dataset for orientation regression |
| `train_angle.py` | Trains MobileNetV2 `(sin θ, cos θ)` regressor |
| `infer.py` | End-to-end inference (detector + angle CNN → `predictions.csv`) |
| `evaluate.py` | Hungarian-matched precision/recall + angle MAE/RMSE/threshold metrics |
| `requirements.txt` | Python dependencies |

## Limitations & next steps

- **Small evaluation set.** 50 tubes per split is statistically thin; stratified k-fold cross-validation would tighten the confidence interval on the angle MAE.
- **Single domain.** All images share similar lighting and overhead viewpoint. Performance on other lab setups (different lighting, surfaces, tube types, camera angles) is untested.

- **Keypoint head.** Replacing `(sin θ, cos θ)` regression with explicit detection of two keypoints (joint and tab) would yield a geometrically interpretable angle and per-pixel error visualisations.
- **Unified model.** A single multi-task head (detection + orientation) on a shared backbone would be faster at inference and might benefit from shared features.
- **Confidence calibration.** The pre-normalisation magnitude of the regressor output is a proxy for prediction certainty; calibrating it against angular error would let downstream consumers abstain on uncertain tubes.
