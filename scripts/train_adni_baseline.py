#!/usr/bin/env python3
"""Train the ResNet-18 baseline on the ADNI SplitGuard split.

This module mirrors the OASIS-1 training protocol so the inflation-gap
comparison across cohorts is apples-to-apples. Differences from
``run_oasis1_inflation_gap.py``:

* Label column is ``diagnosis_group`` (CN/MCI/AD), not ``binary_label``.
  Only CN/AD rows participate in the primary binary task; MCI is excluded.
* Image rows are 3-D volumes (.nii / .nii.gz / .img+.hdr). The loader picks
  a single coronal slice at the volume center for each example, normalizes
  to 0-255, and feeds it to the same pretrained ResNet-18.

The script is structured so ``run_adni_inflation_gap.py`` can import and
reuse its core training/evaluation function for the three split modes
(random / subject-only / component-safe).

Inputs
------
* ``--manifest``: ADNI manifest CSV with the canonical columns and a
  ``diagnosis_group`` column.
* ``--split``: a SplitGuard split CSV under
  ``data/splits/adni/adni_splitguard_seed{N}.csv``.

Outputs
-------
* ``runs/adni/baseline_seed{N}/metrics.json``
* ``runs/adni/baseline_seed{N}/training_log.csv``

This script does not run unless the ADNI manifest and split exist. It
loads ``nibabel`` lazily because the runtime is only needed once ADNI
data is actually present.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "adni"

# ADNI CN/AD → binary targets. MCI is filtered out before training.
LABEL_TO_TARGET = {"CN": 0, "AD": 1}


def lazy_imports():
    """Import heavy deps lazily so this script can be linted/--help-ed in
    an environment without torch/nibabel installed."""
    import numpy as np  # noqa: F401
    import torch
    import torch.nn as nn  # noqa: F401
    from sklearn.metrics import (  # noqa: F401
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from torch.utils.data import DataLoader, Dataset  # noqa: F401
    from torchvision import models, transforms  # noqa: F401
    import nibabel as nib  # noqa: F401
    return {
        "np": __import__("numpy"),
        "torch": __import__("torch"),
        "nn": __import__("torch").nn,
        "DataLoader": __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader,
        "Dataset": __import__("torch.utils.data", fromlist=["Dataset"]).Dataset,
        "models": __import__("torchvision.models", fromlist=["resnet18"]),
        "transforms": __import__("torchvision.transforms", fromlist=["Compose"]),
        "nib": __import__("nibabel"),
        "sklearn_metrics": __import__("sklearn.metrics", fromlist=["roc_auc_score"]),
        "Image": __import__("PIL.Image", fromlist=["Image"]),
    }


def set_seed(seed: int) -> None:
    deps = lazy_imports()
    np = deps["np"]
    torch = deps["torch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str):
    deps = lazy_imports()
    torch = deps["torch"]
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "subject_id", "component_id", "diagnosis_group", "split"}
    missing = required - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")
    return [row for row in rows if row.get("diagnosis_group") in LABEL_TO_TARGET]


def load_volume_center_slice(path: Path):
    """Load a NIfTI / Analyze volume and return the coronal center slice as a
    uint8 array of shape (H, W). The same convention as OASIS-1."""
    deps = lazy_imports()
    nib = deps["nib"]
    np = deps["np"]
    image = nib.load(str(path))
    data = np.squeeze(image.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume after squeeze, got shape {data.shape} for {path}")
    # Coronal axis = 1 to match OASIS-1.
    center = data.shape[1] // 2
    slice_2d = np.take(data, center, axis=1)
    slice_2d = np.nan_to_num(slice_2d, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = float(np.min(slice_2d)), float(np.max(slice_2d))
    if high <= low:
        high = low + 1.0
    slice_2d = np.clip((slice_2d - low) / (high - low), 0.0, 1.0)
    slice_2d = (slice_2d * 255).astype(np.uint8)
    return np.rot90(slice_2d)


class VolumeSliceDataset:
    def __init__(self, rows: list[dict[str, str]], transform=None) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        deps = lazy_imports()
        torch = deps["torch"]
        Image = deps["Image"]
        row = self.rows[index]
        path = Path(row["image_path"])
        # Preprocessed slices live as single-channel PNGs (or .npy). For .nii
        # / .nii.gz we fall back to on-the-fly center-slice extraction via
        # nibabel so the script stays backwards-compatible with the volume
        # layout used before preprocess_adni_volumes_to_slices.py.
        suffix = path.suffix.lower()
        if suffix == ".png":
            image = Image.open(path).convert("RGB")
        elif suffix == ".npy":
            import numpy as np
            slice_uint8 = np.load(path).astype("uint8")
            image = Image.fromarray(slice_uint8, mode="L").convert("RGB")
        else:
            slice_uint8 = load_volume_center_slice(path)
            image = Image.fromarray(slice_uint8, mode="L").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(LABEL_TO_TARGET[row["diagnosis_group"]], dtype=torch.float32)
        return image, target


def make_transforms(size: int):
    deps = lazy_imports()
    transforms = deps["transforms"]
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def make_loader(rows, transform, batch_size, shuffle, device):
    deps = lazy_imports()
    DataLoader = deps["DataLoader"]
    return DataLoader(
        VolumeSliceDataset(rows, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_model(pretrained: bool, arch: str = "resnet18"):
    """Build the binary classifier head.

    Supported architectures (all ImageNet-pretrained):
      - resnet18 (default; primary results)
      - densenet121 (architecture-breadth sensitivity arm)
      - efficientnet_b0 (additional architecture, optional)
    Each architecture's classification head is replaced with a single
    logit output for BCE-with-logits training.
    """
    deps = lazy_imports()
    models = deps["models"]
    nn = deps["nn"]
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        raise ValueError(
            f"Unsupported --arch {arch!r}; expected one of "
            "'resnet18', 'densenet121', 'efficientnet_b0'."
        )
    return model


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    deps = lazy_imports()
    np = deps["np"]
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


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float | int]:
    deps = lazy_imports()
    np = deps["np"]
    skm = deps["sklearn_metrics"]
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = float(skm.roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(float(skm.accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(skm.balanced_accuracy_score(y_true, y_pred)), 4),
        "auroc": round(auc, 4),
        "f1_ad": round(float(skm.f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "brier_score": round(float(skm.brier_score_loss(y_true, y_prob)), 4),
        "ece": round(float(expected_calibration_error(y_true, y_prob)), 4),
        "n_test": int(len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(model, loader, device):
    deps = lazy_imports()
    torch = deps["torch"]
    np = deps["np"]
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1).detach().cpu()
            probs.append(torch.sigmoid(logits))
            targets.append(labels.cpu())
    y_prob = torch.cat(probs).numpy()
    y_true = torch.cat(targets).numpy().astype(int)
    return compute_metrics(y_true, y_prob), y_true, y_prob


def split_by_phase(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        if row["split"] in out:
            out[row["split"]].append(row)
    return out


def train_and_eval(
    splits: dict[str, list[dict[str, str]]],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    pretrained: bool,
    label: str,
    output_dir: Path,
    device_str: str,
    arch: str = "resnet18",
) -> dict:
    deps = lazy_imports()
    torch = deps["torch"]
    nn = deps["nn"]
    np = deps["np"]
    set_seed(seed)
    device = choose_device(device_str)
    train_tf, eval_tf = make_transforms(image_size)
    train_loader = make_loader(splits["train"], train_tf, batch_size, True, device)
    val_loader = make_loader(splits["val"], eval_tf, batch_size, False, device)
    test_loader = make_loader(splits["test"], eval_tf, batch_size, False, device)

    model = make_model(pretrained=pretrained, arch=arch).to(device)
    train_targets = np.array([LABEL_TO_TARGET[row["diagnosis_group"]] for row in splits["train"]])
    positive = int(np.sum(train_targets == 1))
    negative = int(np.sum(train_targets == 0))
    pos_weight = torch.tensor([negative / max(1, positive)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_state = None
    history = []
    started_at = time.time()
    print(
        f"\nTraining ADNI [{label}] seed={seed}: "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels_t in train_loader:
            images = images.to(device)
            labels_t = labels_t.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()
            n = images.shape[0]
            total_loss += float(loss.detach().cpu()) * n
            total_samples += n
        scheduler.step()
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_auc = float(val_metrics["auroc"])
        if np.isnan(val_auc):
            val_auc = -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        history.append({
            "epoch": epoch,
            "loss": round(total_loss / max(1, total_samples), 4),
            "val_auroc": round(float(val_metrics["auroc"]), 4),
            "val_balanced_accuracy": round(float(val_metrics["balanced_accuracy"]), 4),
        })

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, y_true, y_prob = evaluate(model, test_loader, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    log_path = output_dir / "training_log.csv"
    predictions_path = output_dir / "test_predictions.csv"
    checkpoint_path = output_dir / "best_state.pt"

    # Persist per-image predictions for downstream subgroup and probe
    # analyses (TRIPOD+AI 17 / CLAIM 42 follow-ups).
    test_rows = splits["test"]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "subject_id", "diagnosis_group",
                        "y_true", "y_prob"],
        )
        writer.writeheader()
        for row, y_t, y_p in zip(test_rows, y_true.tolist(), y_prob.tolist()):
            writer.writerow({
                "image_id": row.get("image_id", ""),
                "subject_id": row.get("subject_id", ""),
                "diagnosis_group": row.get("diagnosis_group", ""),
                "y_true": int(y_t),
                "y_prob": round(float(y_p), 6),
            })

    # Persist best model state for the WP6 biometric-persistence probe.
    torch.save({"state": model.state_dict(), "seed": seed, "label": label},
               checkpoint_path)
    metrics_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "label": label,
        "test_metrics": test_metrics,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size,
        "pretrained": pretrained,
        "best_val_auroc": round(best_val_auc, 4),
        "wall_time_sec": round(time.time() - started_at, 1),
        "label_distribution": dict(Counter(row["diagnosis_group"] for row in splits["train"])),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "val_auroc", "val_balanced_accuracy"])
        writer.writeheader()
        writer.writerows(history)
    print(f"  test_metrics={test_metrics}")
    return metrics_payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, required=False, help="SplitGuard split CSV under data/splits/adni/")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "densenet121", "efficientnet_b0"],
        help="Backbone architecture for the binary classifier head.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.split is None:
        args.split = PROJECT_ROOT / "data" / "splits" / "adni" / f"adni_splitguard_seed{args.seed}.csv"
    if not args.split.exists():
        raise FileNotFoundError(
            f"Split file does not exist: {args.split}. "
            "Run scripts/make_adni_splitguard_split.py first."
        )

    rows = read_split_rows(args.split)
    splits = split_by_phase(rows)
    output_dir = args.output_root / f"baseline_seed{args.seed}"
    train_and_eval(
        splits,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        label="component_safe",
        output_dir=output_dir,
        device_str=args.device,
        arch=args.arch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
