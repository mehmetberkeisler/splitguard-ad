#!/usr/bin/env python3
"""Train a 3D ResNet-18 on the ADNI SplitGuard split (P1-1 volumetric arm).

Reviewer §5.2 asked for a 3D volumetric replication of the 2D headline
result. This script mirrors ``train_adni_baseline.py`` (the 2D pipeline)
but consumes 128^3 volumes preprocessed by ``preprocess_adni1_to_3d.py``
and trains a 3D ResNet-18 (from-scratch or MedicalNet-initialised).

Design choices:

* **Backbone**: MONAI's ``resnet18`` at spatial_dims=3, keeps parameter
  count comparable to 2D ResNet-18 for apples-to-apples comparison. If
  MedicalNet weights are available locally, ``--pretrained`` loads them.
* **Input**: single-channel 128×128×128 float32 volume from .npz cache
  (mean-0 var-1 within brain mask by preprocessing).
* **Loss & optimiser**: same class-weighted BCE, AdamW, cosine schedule
  as the 2D pipeline — the only thing that changes is dimensionality.
* **Splits**: same Protocol A/B/C split CSVs — the file-format is
  identical to 2D, only ``image_path`` points to .npz.
* **Determinism**: same PYTHONHASHSEED / cudnn.deterministic pattern as
  the 2D pipeline for byte-reproducibility of predictions.

Timeline expectation on Apple Silicon MPS: ~30–60 min per protocol epoch,
15 epochs, 3 protocols, 5 seeds → ~20–40 hr per protocol × seed. Prefer
RunPod A100 (~12–24 hr per full seed × 3 protocols).

Inputs
------
--manifest : ADNI manifest CSV with subject_id / image_uid / diagnosis_group
--split    : SplitGuard split CSV (same format as 2D)
--protocol : {random, subject_only, component_safe}
--seed     : integer seed
--output-dir : run directory (checkpoints + predictions saved here)
--preprocessed-root : path to .npz cache
             (default data/adni3d_128, where the GPU bundle extracts)

Outputs
-------
<output_dir>/best_state.pt              — best-val-AUROC checkpoint
<output_dir>/metrics.json               — final test metrics
<output_dir>/test_predictions.csv       — per-image y_true, y_prob
<output_dir>/val_predictions.csv        — per-image val y_true, y_prob
<output_dir>/training_log.csv           — per-epoch loss / metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# ── Determinism pinning (must precede torch import) ─────────────────────
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print(f"error: torch/numpy required: {e}", file=sys.stderr)
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED_ROOT = Path(__file__).resolve().parents[1] / "data" / "adni3d_128"

LABEL_TO_TARGET = {"CN": 0, "AD": 1}
BINARY_LABELS = set(LABEL_TO_TARGET)


# ── Dataset ──────────────────────────────────────────────────────────────
class ADNIVolumeDataset(Dataset):
    """Loads a preprocessed 128^3 volume from .npz per row."""
    def __init__(self, rows: list[dict], preprocessed_root: Path):
        self.rows = rows
        self.root = preprocessed_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        npz_path = self.root / row["subject_id"] / f"{row['image_uid']}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"missing preprocessed volume: {npz_path}")
        d = np.load(npz_path)
        vol = d["volume"].astype(np.float32)   # (128, 128, 128), z-scored
        x = torch.from_numpy(vol).unsqueeze(0)  # → (1, 128, 128, 128)
        y = torch.tensor(LABEL_TO_TARGET[row["diagnosis_group"]], dtype=torch.float32)
        return x, y, row["image_uid"], row["subject_id"]


# ── Backbone (deferred MONAI import so file inspection is dep-free) ─────
def build_3d_resnet18(pretrained: bool = False, pretrained_path: str | None = None):
    try:
        from monai.networks.nets import resnet18
    except ImportError:
        raise ImportError(
            "MONAI required for 3D ResNet-18. Install: pip install monai"
        )
    model = resnet18(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1,
        widen_factor=1.0,
    )
    if pretrained and pretrained_path:
        # A MONAI pretrained bundle: tensors and primitives only.
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=False)
    return model


# ── Metrics ──────────────────────────────────────────────────────────────
def auroc(y_true, y_score) -> float:
    """Rank-based AUROC, nan if one class is absent.

    Ranks ascend with the score and ties share their mid-rank, as in
    ``scripts/gpu_postprocess.py``. Ranking the scores descending instead
    returns 1 - AUROC, which inverts every epoch's validation metric and makes
    the best-val checkpoint the least-trained one.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(y_score, kind="stable")
    ranks = np.empty(y_score.size, dtype=float)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=float)
    scores_sorted = y_score[order]
    tie_start = 0
    for i in range(1, scores_sorted.size + 1):
        if i == scores_sorted.size or scores_sorted[i] != scores_sorted[tie_start]:
            if i - tie_start > 1:
                ranks[order[tie_start:i]] = (tie_start + i + 1) / 2
            tie_start = i
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ── Data loading helper ──────────────────────────────────────────────────
def load_split(split_csv: Path, protocol: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (train_rows, val_rows, test_rows) filtered to binary CN/AD."""
    train, val, test = [], [], []
    proto_col = {"random": "split_random", "subject_only": "split_subject_only",
                 "component_safe": "split"}[protocol]
    with split_csv.open() as f:
        reader = csv.DictReader(f)
        if proto_col not in reader.fieldnames:
            # Fall back: single-protocol split file has just a "split" column.
            proto_col = "split"
        for row in reader:
            if row.get("diagnosis_group") not in BINARY_LABELS:
                continue
            s = row.get(proto_col)
            if s == "train":   train.append(row)
            elif s == "val":   val.append(row)
            elif s == "test":  test.append(row)
    return train, val, test


# ── Train loop ───────────────────────────────────────────────────────────
def choose_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def epoch_run(model, loader, device, criterion, optim=None) -> tuple[float, list, list]:
    training = optim is not None
    model.train() if training else model.eval()
    total_loss = 0.0; y_true_all, y_score_all = [], []
    for x, y, _uid, _sid in loader:
        x = x.to(device); y = y.to(device)
        if training: optim.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
        if training:
            loss.backward(); optim.step()
        total_loss += float(loss.item()) * x.shape[0]
        y_true_all.extend(y.detach().cpu().tolist())
        y_score_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return total_loss / max(1, len(y_true_all)), y_true_all, y_score_all


def save_predictions(rows: list[dict], y_true: list, y_score: list, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_uid", "subject_id",
                                          "diagnosis_group", "y_true", "y_prob"])
        w.writeheader()
        for row, t, s in zip(rows, y_true, y_score):
            w.writerow({
                "image_uid": row["image_uid"], "subject_id": row["subject_id"],
                "diagnosis_group": row["diagnosis_group"],
                "y_true": int(t), "y_prob": round(float(s), 6),
            })


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--protocol", choices=["random", "subject_only", "component_safe"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers; volume decoding is the bottleneck.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--pretrained-path", type=str, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    device = choose_device()
    print(f"device={device} seed={args.seed} protocol={args.protocol}")

    train_rows, val_rows, test_rows = load_split(args.split, args.protocol)
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train_ds = ADNIVolumeDataset(train_rows, args.preprocessed_root)
    val_ds = ADNIVolumeDataset(val_rows, args.preprocessed_root)
    test_ds = ADNIVolumeDataset(test_rows, args.preprocessed_root)

    # Each sample is a zlib-compressed 128^3 volume, so decoding is the
    # bottleneck: with num_workers=0 the GPU waits on the main thread.
    loader_args = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                       persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, **loader_args)

    model = build_3d_resnet18(pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)

    # Class-weighted BCE
    n_pos = sum(1 for r in train_rows if r["diagnosis_group"] == "AD")
    n_neg = sum(1 for r in train_rows if r["diagnosis_group"] == "CN")
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_auroc", "lr"])
        writer.writeheader()

        best_val_auroc = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = epoch_run(model, train_loader, device, criterion, optim)
            val_loss, val_y_true, val_y_score = epoch_run(model, val_loader, device, criterion, None)
            val_auroc = auroc(val_y_true, val_y_score)
            writer.writerow({"epoch": epoch, "train_loss": round(train_loss, 4),
                             "val_loss": round(val_loss, 4),
                             "val_auroc": round(val_auroc, 4),
                             "lr": round(optim.param_groups[0]["lr"], 6)})
            f.flush()
            print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_auroc={val_auroc:.4f}")
            if val_auroc != val_auroc:      # NaN when a partition holds one class
                val_auroc = -1.0
            if val_auroc >= best_val_auroc or not (args.output_dir / "best_state.pt").exists():
                best_val_auroc = val_auroc
                torch.save({"state_dict": model.state_dict(),
                            "epoch": epoch, "val_auroc": val_auroc},
                           args.output_dir / "best_state.pt")
                # Cache val predictions at best-val checkpoint
                save_predictions(val_rows, val_y_true, val_y_score,
                                 args.output_dir / "val_predictions.csv")
            sched.step()

    # Load best checkpoint and evaluate on test set
    # Written a few lines above: state_dict, epoch, val_auroc.
    ckpt = torch.load(args.output_dir / "best_state.pt", map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_y_true, test_y_score = epoch_run(model, test_loader, device, criterion, None)
    test_auroc = auroc(test_y_true, test_y_score)
    print(f"TEST AUROC = {test_auroc:.4f}")

    save_predictions(test_rows, test_y_true, test_y_score,
                     args.output_dir / "test_predictions.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "seed": args.seed, "protocol": args.protocol,
        "best_val_auroc": round(best_val_auroc, 4),
        "test_metrics": {"auroc": round(test_auroc, 4)},
        "test_auroc": round(test_auroc, 4),
        "test_loss": round(test_loss, 4),
        "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
