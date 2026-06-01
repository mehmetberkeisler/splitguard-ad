#!/usr/bin/env python3
"""
Inflation Gap Experiment
========================
Runs the same ResNet-18 under two protocols and reports the inflation gap:

  Protocol A — LEAKY:  random image-level split (reproduces published 99% papers)
  Protocol B — SAFE:   SplitGuard component-safe split (what we built)

This is the core result of the paper.
Usage:
  python scripts/run_inflation_gap.py --epochs 30 --device auto
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
IMAGE_DIR      = PROJECT_ROOT / "Alzheimer_MRI_4_classes_dataset"
RESULTS_DIR    = PROJECT_ROOT / "reports" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_TARGET = {"NonDemented": 0, "Demented": 1}


# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob <= right)
        if not np.any(mask): continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
    return ece

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    try:    auc = float(roc_auc_score(y_true, y_prob))
    except: auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":              round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy":     round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "auroc":                 round(auc, 4),
        "f1_demented":           round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "sensitivity":           round(tp / (tp + fn) if (tp + fn) else 0.0, 4),
        "specificity":           round(tn / (tn + fp) if (tn + fp) else 0.0, 4),
        "brier_score":           round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece":                   round(float(expected_calibration_error(y_true, y_prob)), 4),
        "n_test":                int(len(y_true)),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────
class RowDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        label = torch.tensor(LABEL_TO_TARGET[row["binary_label"]], dtype=torch.float32)
        return img, label


def make_transforms(size=224):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    return train_tf, eval_tf


# ── Protocol A: LEAKY split ────────────────────────────────────────────────────
def build_leaky_split(manifest_rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """
    Leaky: randomly shuffle all 6400 images and split 70/15/15 by image count.
    Subjects WILL cross splits — this is what most published papers do.
    """
    rng = random.Random(seed)
    rows = manifest_rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    return {
        "train": rows[:n_train],
        "val":   rows[n_train:n_train + n_val],
        "test":  rows[n_train + n_val:],
    }


def check_leaky_subject_overlap(splits: dict[str, list[dict]]) -> dict:
    """Count how many subjects appear in both train and test (the leakage)."""
    train_subjects = {r["subject_id"] for r in splits["train"] if r.get("subject_id")}
    test_subjects  = {r["subject_id"] for r in splits["test"]  if r.get("subject_id")}
    overlap = train_subjects & test_subjects
    return {
        "train_subjects": len(train_subjects),
        "test_subjects":  len(test_subjects),
        "overlapping_subjects": len(overlap),
        "overlap_pct_of_test": round(100 * len(overlap) / max(1, len(test_subjects)), 1),
    }


# ── Protocol B: SAFE split (SplitGuard) ───────────────────────────────────────
def load_splitguard_split(manifest_path: Path) -> dict[str, list[dict]]:
    """Load the pre-built component-safe SplitGuard split from CSV."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        s = row.get("split", "")
        if s in splits:
            splits[s].append(row)
    return splits


# ── Model + Training ──────────────────────────────────────────────────────────
def make_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def make_loader(rows, transform, batch_size, shuffle, device):
    ds = RowDataset(rows, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(device.type == "cuda"))


def train_and_eval(splits: dict, device: torch.device, epochs: int,
                   batch_size: int, lr: float, seed: int,
                   label: str) -> dict:
    set_seed(seed)
    train_tf, eval_tf = make_transforms()

    train_loader = make_loader(splits["train"], train_tf, batch_size, True, device)
    val_loader   = make_loader(splits["val"],   eval_tf, batch_size, False, device)
    test_loader  = make_loader(splits["test"],  eval_tf, batch_size, False, device)

    model = make_model(pretrained=True).to(device)

    # Weighted loss (balanced classes in safe split, may be unbalanced in leaky)
    n_pos = sum(LABEL_TO_TARGET[r["binary_label"]] for r in splits["train"])
    n_neg = len(splits["train"]) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_auc = -1.0
    best_state = None
    history = []

    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"  Device: {device} | Epochs: {epochs}")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type if device.type != "mps" else "cpu",
                                enabled=use_amp):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            ep_loss += loss.item() * imgs.size(0)
            ep_n    += imgs.size(0)
        scheduler.step()

        # Val eval
        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                p = torch.sigmoid(model(imgs.to(device)).squeeze(1)).cpu()
                probs.append(p); targets.append(lbls)
        yp = torch.cat(probs).numpy(); yt = torch.cat(targets).numpy().astype(int)
        try:    val_auc = roc_auc_score(yt, yp)
        except: val_auc = 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "loss": round(ep_loss/ep_n, 4),
                         "val_auc": round(val_auc, 4)})
        print(f"  ep={epoch:03d}  loss={ep_loss/ep_n:.4f}  val_auc={val_auc:.4f}")

    # Load best, eval test
    model.load_state_dict(best_state)
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            p = torch.sigmoid(model(imgs.to(device)).squeeze(1)).cpu()
            probs.append(p); targets.append(lbls)
    yp = torch.cat(probs).numpy(); yt = torch.cat(targets).numpy().astype(int)

    test_metrics = compute_metrics(yt, yp)
    elapsed = round(time.time() - t0, 1)

    print(f"\n  ✅ {label} DONE in {elapsed}s")
    print(f"     Test AUROC:    {test_metrics['auroc']:.4f}")
    print(f"     Test Bal.Acc:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"     Test F1:       {test_metrics['f1_demented']:.4f}")
    print(f"     Sensitivity:   {test_metrics['sensitivity']:.4f}")
    print(f"     Specificity:   {test_metrics['specificity']:.4f}")

    return {"label": label, "test_metrics": test_metrics,
            "history": history, "elapsed_seconds": elapsed,
            "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "n_test": len(splits["test"])}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    set_seed(args.seed)

    # ── Load the SplitGuard manifest (has path, binary_label, subject_id, split)
    print(f"\nLoading split manifest: {SPLIT_MANIFEST}")
    safe_splits = load_splitguard_split(SPLIT_MANIFEST)

    all_rows = safe_splits["train"] + safe_splits["val"] + safe_splits["test"]
    print(f"Total images: {len(all_rows)}")

    # ── Build leaky split from same images
    leaky_splits = build_leaky_split(all_rows, seed=args.seed)
    overlap_info = check_leaky_subject_overlap(leaky_splits)
    print(f"\nLeaky split subject overlap (train→test): {overlap_info}")

    # ── Run both protocols
    result_a = train_and_eval(
        leaky_splits, device, args.epochs, args.batch_size, args.lr, args.seed,
        label="A_LEAKY_random_image_split"
    )
    result_b = train_and_eval(
        safe_splits, device, args.epochs, args.batch_size, args.lr, args.seed,
        label="B_SAFE_splitguard_component"
    )

    # ── Inflation gap table
    a = result_a["test_metrics"]
    b = result_b["test_metrics"]

    print(f"\n{'='*60}")
    print("  INFLATION GAP TABLE (Protocol A - Protocol B)")
    print(f"{'='*60}")
    metrics = ["auroc", "balanced_accuracy", "f1_demented", "sensitivity", "specificity"]
    for m in metrics:
        gap = a[m] - b[m]
        sign = "+" if gap > 0 else ""
        print(f"  {m:<22}  A={a[m]:.4f}  B={b[m]:.4f}  gap={sign}{gap:.4f}")

    inflation_gap = {
        m: {"leaky": a[m], "safe": b[m], "inflation": round(a[m] - b[m], 4)}
        for m in metrics
    }

    # ── Save results
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "leaky_subject_overlap": overlap_info,
        "protocol_A_leaky": result_a,
        "protocol_B_safe": result_b,
        "inflation_gap": inflation_gap,
    }
    out_path = RESULTS_DIR / "inflation_gap_experiment.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n✅ Full results saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
