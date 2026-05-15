"""
Trains a MobileNetV2 to regress tube orientation as (sin θ, cos θ).
Final angle = atan2(sin θ, cos θ), recovers full [0, 360°) range.
"""
import argparse, math, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import config as C


class AngleDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        df = pd.read_csv(self.root / "angles.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.augment = augment
        self.tf = transforms.Compose([
            transforms.Resize((C.CROP_SIZE, C.CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.color_tf = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / self.split / row["crop"]).convert("RGB")
        angle = float(row["angle_deg"])

        if self.augment:
            img = self.color_tf(img)
            if random.random() < 0.8:
                rot = random.uniform(-180, 180)
                img = img.rotate(rot)                  # PIL: CCW positive
                angle = (angle + rot) % 360
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                angle = (180 - angle) % 360
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                angle = (-angle) % 360

        x = self.tf(img)
        rad = math.radians(angle)
        target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        return x, target


def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(1280, 2)
    return m


def ang_err_deg(pred_sc, target_sc):
    pred_ang = torch.atan2(pred_sc[:, 0], pred_sc[:, 1])
    targ_ang = torch.atan2(target_sc[:, 0], target_sc[:, 1])
    d = torch.abs(pred_ang - targ_ang) * 180.0 / math.pi
    d = torch.minimum(d, 360 - d)
    return d


def train(data, epochs=60, batch=16, lr=1e-3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_ds = AngleDataset(data, "train", augment=True)
    val_ds   = AngleDataset(data, "val",   augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    print(f"Device: {device} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_mae = 1e9
    for epoch in range(1, epochs + 1):
        model.train()
        for x, t in train_dl:
            x, t = x.to(device), t.to(device)
            pred = model(x)
            pred_n = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
            loss = ((pred_n - t) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        errs = []
        with torch.no_grad():
            for x, t in val_dl:
                x, t = x.to(device), t.to(device)
                pred = model(x)
                pred_n = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
                errs.append(ang_err_deg(pred_n, t).cpu().numpy())
        mae = float(np.concatenate(errs).mean())
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}  val angular MAE: {mae:.2f}°")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "angle_best.pt")

    print(f"Best val angular MAE: {best_mae:.2f}° → angle_best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="angle_dataset")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    train(args.data, args.epochs)