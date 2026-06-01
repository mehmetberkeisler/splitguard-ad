#!/usr/bin/env python3
"""Train a baseline model from the frozen SplitGuard split manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "checkpoints"
DEFAULT_LOG_DIR = PROJECT_ROOT / "runs" / "logs"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

LABEL_TO_TARGET = {"NonDemented": 0, "Demented": 1}
TARGET_TO_LABEL = {value: key for key, value in LABEL_TO_TARGET.items()}


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform=None) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(LABEL_TO_TARGET[row["binary_label"]], dtype=torch.float32)
        return image, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "split", "binary_label", "component_id", "subject_id"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {sorted(missing)}")
    unknown_labels = sorted({row["binary_label"] for row in rows}.difference(LABEL_TO_TARGET))
    if unknown_labels:
        raise ValueError(f"Unsupported binary labels in split manifest: {unknown_labels}")
    return rows


def stratified_limit(rows: list[dict[str, str]], max_samples: int | None, seed: int) -> list[dict[str, str]]:
    if max_samples is None or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = {label: [] for label in LABEL_TO_TARGET}
    for row in rows:
        by_label[row["binary_label"]].append(row)

    selected: list[dict[str, str]] = []
    remaining_slots = max_samples
    labels = list(LABEL_TO_TARGET)
    for label_index, label in enumerate(labels):
        label_rows = by_label[label][:]
        rng.shuffle(label_rows)
        slots_left_after_this_label = len(labels) - label_index - 1
        take = min(len(label_rows), max(0, remaining_slots // (slots_left_after_this_label + 1)))
        selected.extend(label_rows[:take])
        remaining_slots -= take

    if len(selected) < max_samples:
        already = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in already]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    selected.sort(key=lambda row: row["relative_path"])
    return selected


def split_rows(
    rows: list[dict[str, str]],
    seed: int,
    smoke_test: bool,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
) -> dict[str, list[dict[str, str]]]:
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ["train", "val", "test"]
    }
    if smoke_test:
        max_train_samples = max_train_samples or 64
        max_val_samples = max_val_samples or 32
        max_test_samples = max_test_samples or 32

    return {
        "train": stratified_limit(by_split["train"], max_train_samples, seed),
        "val": stratified_limit(by_split["val"], max_val_samples, seed + 1),
        "test": stratified_limit(by_split["test"], max_test_samples, seed + 2),
    }


def make_transforms(image_size: int, train_rotation_degrees: float) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(train_rotation_degrees),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def make_model(model_name: str, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def make_loader(
    rows: list[dict[str, str]],
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = ManifestImageDataset(rows, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return ece


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_demented_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc_demented_positive": auc,
        "sensitivity_demented": float(sensitivity),
        "specificity_non_demented": float(specificity),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": float(expected_calibration_error(y_true, y_prob)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(1).detach().cpu()
            probs.append(torch.sigmoid(logits))
            targets.append(labels.cpu())

    y_prob = torch.cat(probs).numpy()
    y_true = torch.cat(targets).numpy().astype(int)
    return compute_metrics(y_true, y_prob), y_true, y_prob


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def count_labels(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["binary_label"] for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-rotation-degrees", type=float, default=7.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda"

    run_name = args.run_name or f"current_jpeg_splitguard_seed{args.seed}_{args.model}"
    if args.smoke_test and "smoke" not in run_name:
        run_name = f"{run_name}_smoke"

    all_rows = read_split_rows(args.split)
    rows_by_split = split_rows(
        all_rows,
        seed=args.seed,
        smoke_test=args.smoke_test,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    train_tf, eval_tf = make_transforms(args.image_size, args.train_rotation_degrees)
    train_loader = make_loader(rows_by_split["train"], train_tf, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(rows_by_split["val"], eval_tf, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(rows_by_split["test"], eval_tf, args.batch_size, False, args.num_workers, device)

    model = make_model(args.model, pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone)
    model.to(device)

    train_targets = np.array([LABEL_TO_TARGET[row["binary_label"]] for row in rows_by_split["train"]])
    positive = int(np.sum(train_targets == 1))
    negative = int(np.sum(train_targets == 0))
    pos_weight = torch.tensor([negative / max(1, positive)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = args.checkpoint_dir / f"{run_name}_best.pt"
    history_path = args.log_dir / f"{run_name}_history.csv"
    metrics_path = args.table_dir / f"{run_name}_metrics.json"
    predictions_path = args.table_dir / f"{run_name}_test_predictions.csv"

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Split manifest: {args.split}")
    print(f"Label mapping: {LABEL_TO_TARGET}")
    print(f"Train labels: {count_labels(rows_by_split['train'])}")
    print(f"Val labels: {count_labels(rows_by_split['val'])}")
    print(f"Test labels: {count_labels(rows_by_split['test'])}")

    best_val_auc = -1.0
    best_epoch = 0
    history: list[dict] = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_auc = float(val_metrics["auroc_demented_positive"])
        if np.isnan(val_auc):
            val_auc = -1.0

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_auroc": round(float(val_metrics["auroc_demented_positive"]), 6),
            "val_balanced_accuracy": round(float(val_metrics["balanced_accuracy"]), 6),
            "val_f1_demented_positive": round(float(val_metrics["f1_demented_positive"]), 6),
            "val_sensitivity_demented": round(float(val_metrics["sensitivity_demented"]), 6),
            "val_specificity_non_demented": round(float(val_metrics["specificity_non_demented"]), 6),
            "seconds": round(time.time() - epoch_started, 2),
        }
        history.append(epoch_row)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "label_to_target": LABEL_TO_TARGET,
                    "split_manifest": str(args.split),
                    "run_name": run_name,
                },
                checkpoint_path,
            )

        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"val_auc={val_metrics['auroc_demented_positive']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_f1={val_metrics['f1_demented_positive']:.4f}"
        )

    write_history(history_path, history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _, _ = evaluate(model, val_loader, device)
    test_metrics, test_true, test_prob = evaluate(model, test_loader, device)

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image_id",
            "relative_path",
            "subject_id",
            "component_id",
            "raw_class_label",
            "binary_label",
            "target",
            "prob_demented",
            "predicted_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, target, prob in zip(rows_by_split["test"], test_true, test_prob):
            predicted = int(prob >= 0.5)
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "relative_path": row["relative_path"],
                    "subject_id": row["subject_id"],
                    "component_id": row["component_id"],
                    "raw_class_label": row["raw_class_label"],
                    "binary_label": row["binary_label"],
                    "target": int(target),
                    "prob_demented": float(prob),
                    "predicted_label": TARGET_TO_LABEL[predicted],
                }
            )

    result = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_manifest": str(args.split),
        "model": args.model,
        "pretrained": not args.no_pretrained,
        "freeze_backbone": args.freeze_backbone,
        "seed": args.seed,
        "device": str(device),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "label_to_target": LABEL_TO_TARGET,
        "clinical_positive_label": "Demented",
        "smoke_test": args.smoke_test,
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "label_counts": {split: count_labels(rows) for split, rows in rows_by_split.items()},
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "test_predictions_path": str(predictions_path),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Best epoch: {best_epoch}")
    print(f"Test AUROC: {test_metrics['auroc_demented_positive']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test F1 dementia-positive: {test_metrics['f1_demented_positive']:.4f}")
    print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote history: {history_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote test predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
