#!/usr/bin/env python3
"""Run the OASIS-1 leaky-vs-SplitGuard inflation gap experiment."""

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
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "splits" / "oasis1_splitguard_seed42.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "tables"
CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "checkpoints" / "oasis1"
LOG_DIR = PROJECT_ROOT / "runs" / "logs"

LABEL_TO_TARGET = {"NonDemented": 0, "Demented": 1}
TARGET_TO_LABEL = {value: key for key, value in LABEL_TO_TARGET.items()}


class RowDataset(Dataset):
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


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "auroc": round(auc, 4),
        "f1_demented": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece": round(float(expected_calibration_error(y_true, y_prob)), 4),
        "n_test": int(len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("binary_label") in LABEL_TO_TARGET]
    required = {"path", "split", "binary_label", "subject_id", "component_id", "relative_path"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {sorted(missing)}")
    return rows


def load_splitguard_split(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_split_rows(path)
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        if row["split"] in splits:
            splits[row["split"]].append(row)
    return splits


def build_leaky_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def leaky_subject_overlap(splits: dict[str, list[dict[str, str]]]) -> dict[str, float | int]:
    train_subjects = {row["subject_id"] for row in splits["train"]}
    test_subjects = {row["subject_id"] for row in splits["test"]}
    overlap = train_subjects & test_subjects
    return {
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "overlapping_subjects": len(overlap),
        "overlap_pct_of_test": round(100 * len(overlap) / max(1, len(test_subjects)), 1),
    }


def stratified_limit(rows: list[dict[str, str]], max_samples: int | None, seed: int) -> list[dict[str, str]]:
    if max_samples is None or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = {label: [] for label in LABEL_TO_TARGET}
    for row in rows:
        by_label[row["binary_label"]].append(row)
    selected: list[dict[str, str]] = []
    remaining = max_samples
    labels = list(LABEL_TO_TARGET)
    for label_index, label in enumerate(labels):
        label_rows = by_label[label][:]
        rng.shuffle(label_rows)
        slots_left = len(labels) - label_index
        take = min(len(label_rows), max(1, remaining // slots_left))
        selected.extend(label_rows[:take])
        remaining -= take
    if len(selected) < max_samples:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])
    return selected[:max_samples]


def maybe_limit_splits(
    splits: dict[str, list[dict[str, str]]],
    seed: int,
    smoke_test: bool,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
) -> dict[str, list[dict[str, str]]]:
    if smoke_test:
        max_train_samples = max_train_samples or 64
        max_val_samples = max_val_samples or 32
        max_test_samples = max_test_samples or 32
    return {
        "train": stratified_limit(splits["train"], max_train_samples, seed),
        "val": stratified_limit(splits["val"], max_val_samples, seed + 1),
        "test": stratified_limit(splits["test"], max_test_samples, seed + 2),
    }


def make_transforms(size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def make_loader(
    rows: list[dict[str, str]],
    transform,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        RowDataset(rows, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_model(pretrained: bool, arch: str = "resnet18") -> nn.Module:
    """Build the binary classifier head.

    Supported architectures (all ImageNet-pretrained):
      - resnet18    (default; primary manuscript results)
      - densenet121 (architecture-breadth sensitivity arm)
    """
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        raise ValueError(
            f"Unsupported --arch {arch!r}; expected 'resnet18' or 'densenet121'."
        )
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1).detach().cpu()
            probs.append(torch.sigmoid(logits))
            targets.append(labels.cpu())
    y_prob = torch.cat(probs).numpy()
    y_true = torch.cat(targets).numpy().astype(int)
    return compute_metrics(y_true, y_prob), y_true, y_prob


def train_and_eval(
    splits: dict[str, list[dict[str, str]]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    label: str,
    image_size: int,
    pretrained: bool,
    checkpoint_path: Path,
    arch: str = "resnet18",
) -> dict:
    set_seed(seed)
    train_tf, eval_tf = make_transforms(image_size)
    train_loader = make_loader(splits["train"], train_tf, batch_size, True, device)
    val_loader = make_loader(splits["val"], eval_tf, batch_size, False, device)
    test_loader = make_loader(splits["test"], eval_tf, batch_size, False, device)

    model = make_model(pretrained=pretrained, arch=arch).to(device)
    train_targets = np.array([LABEL_TO_TARGET[row["binary_label"]] for row in splits["train"]])
    positive = int(np.sum(train_targets == 1))
    negative = int(np.sum(train_targets == 0))
    pos_weight = torch.tensor([negative / max(1, positive)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None
    history: list[dict[str, float | int]] = []
    started_at = time.time()
    print(f"\nTraining {label}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size_actual = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size_actual
            total_samples += batch_size_actual
        scheduler.step()
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_auc = float(val_metrics["auroc"])
        if np.isnan(val_auc):
            val_auc = -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "loss": round(total_loss / max(1, total_samples), 4),
                "val_auc": round(float(val_metrics["auroc"]), 4),
                "val_balanced_accuracy": round(float(val_metrics["balanced_accuracy"]), 4),
            }
        )
        print(
            f"  ep={epoch:03d} loss={total_loss / max(1, total_samples):.4f} "
            f"val_auc={val_metrics['auroc']:.4f}"
        )

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint state produced for {label}")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "label": label, "seed": seed}, checkpoint_path)
    test_metrics, test_true, test_prob = evaluate(model, test_loader, device)
    elapsed = round(time.time() - started_at, 1)
    return {
        "label": label,
        "test_metrics": test_metrics,
        "history": history,
        "elapsed_seconds": elapsed,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "checkpoint_path": str(checkpoint_path),
        "label_counts": {split: dict(Counter(row["binary_label"] for row in rows)) for split, rows in splits.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "oasis1_inflation_gap_experiment.json")
    parser.add_argument(
        "--arch", default="resnet18",
        choices=["resnet18", "densenet121"],
        help="Backbone architecture. Default resnet18 matches the primary "
             "manuscript results; use densenet121 for the cross-cohort "
             "architecture-breadth sensitivity arm.",
    )
    args = parser.parse_args()
    # Namespace output filename by arch so DenseNet runs don't clobber the
    # primary ResNet-18 results.
    if args.arch != "resnet18" and args.output == RESULTS_DIR / "oasis1_inflation_gap_experiment.json":
        args.output = RESULTS_DIR / f"oasis1_inflation_gap_experiment__{args.arch}__seed{args.seed}.json"

    set_seed(args.seed)
    device = choose_device(args.device)
    safe_splits_full = load_splitguard_split(args.split)
    all_rows = safe_splits_full["train"] + safe_splits_full["val"] + safe_splits_full["test"]
    leaky_splits_full = build_leaky_split(all_rows, seed=args.seed)
    overlap = leaky_subject_overlap(leaky_splits_full)

    safe_splits = maybe_limit_splits(
        safe_splits_full,
        args.seed,
        args.smoke_test,
        args.max_train_samples,
        args.max_val_samples,
        args.max_test_samples,
    )
    leaky_splits = maybe_limit_splits(
        leaky_splits_full,
        args.seed,
        args.smoke_test,
        args.max_train_samples,
        args.max_val_samples,
        args.max_test_samples,
    )

    run_suffix = "smoke" if args.smoke_test else f"seed{args.seed}"
    arch_suffix = "" if args.arch == "resnet18" else f"_{args.arch}"
    result_a = train_and_eval(
        leaky_splits,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        "A_LEAKY_random_slice_split",
        args.image_size,
        pretrained=not args.no_pretrained,
        checkpoint_path=CHECKPOINT_DIR / f"oasis1_leaky_{run_suffix}{arch_suffix}.pt",
        arch=args.arch,
    )
    result_b = train_and_eval(
        safe_splits,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        "B_SAFE_splitguard_subject_component",
        args.image_size,
        pretrained=not args.no_pretrained,
        checkpoint_path=CHECKPOINT_DIR / f"oasis1_safe_{run_suffix}{arch_suffix}.pt",
        arch=args.arch,
    )

    metrics = ["auroc", "balanced_accuracy", "f1_demented", "sensitivity", "specificity"]
    inflation_gap = {
        metric: {
            "leaky": result_a["test_metrics"][metric],
            "safe": result_b["test_metrics"][metric],
            "inflation": round(result_a["test_metrics"][metric] - result_b["test_metrics"][metric], 4),
        }
        for metric in metrics
    }

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_manifest": str(args.split),
        "arch":   args.arch,
        "epochs": args.epochs,
        "seed":   args.seed,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "pretrained": not args.no_pretrained,
        "leaky_subject_overlap": overlap,
        "protocol_A_leaky": result_a,
        "protocol_B_safe": result_b,
        "inflation_gap": inflation_gap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote OASIS-1 inflation gap result: {args.output}")
    print(json.dumps({"leaky_subject_overlap": overlap, "inflation_gap": inflation_gap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
