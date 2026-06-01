#!/usr/bin/env python3
"""Run the ADNI three-point inflation-gap experiment.

For each seed, this script trains the ResNet-18 baseline under three
split protocols on the same ADNI manifest:

1. **random** — image-level random 70/15/15 split (the leaky protocol).
2. **subject_only** — train/val/test partition by subject only, ignoring
   session, series, and longitudinal links.
3. **component_safe** — the frozen SplitGuard split (whole leakage
   components stay in one partition).

The point is to isolate how much of the apparent ADNI AUROC is driven by
subject-identity leakage, mirroring the OASIS-1 and JPEG protocols.

Inputs
------
* ``data/splits/adni/adni_splitguard_seed{N}.csv`` — for the
  ``component_safe`` mode. The script derives the ``random`` and
  ``subject_only`` splits from the same rows so all three protocols see
  the identical CN/AD universe.

Outputs
-------
* ``runs/adni/inflation_gap_seed{N}.json`` per seed.
* ``reports/tables/adni/adni_inflation_gap.csv`` aggregated across seeds.

The script imports its training loop from ``train_adni_baseline``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adni_baseline import (  # noqa: E402 — sys.path tweak above
    LABEL_TO_TARGET,
    read_split_rows,
    split_by_phase,
    train_and_eval,
)

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "adni"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "adni"
DEFAULT_TABLE = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_inflation_gap.csv"


def build_random_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    """Image-level random 70/15/15."""
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": [{**row, "split": "train"} for row in shuffled[:n_train]],
        "val": [{**row, "split": "val"} for row in shuffled[n_train : n_train + n_val]],
        "test": [{**row, "split": "test"} for row in shuffled[n_train + n_val :]],
    }


def build_subject_only_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    """Subject-level 70/15/15. Ignores sessions / series / longitudinal."""
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)
    subjects = list(by_subject)
    random.Random(seed).shuffle(subjects)
    n = len(subjects)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])

    out: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for subject, subject_rows in by_subject.items():
        if subject in train_subjects:
            split = "train"
        elif subject in val_subjects:
            split = "val"
        else:
            split = "test"
        for row in subject_rows:
            out[split].append({**row, "split": split})
    return out


def overlap_stats(splits: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    train_subjects = {row["subject_id"] for row in splits["train"]}
    test_subjects = {row["subject_id"] for row in splits["test"]}
    train_components = {row["component_id"] for row in splits["train"]}
    test_components = {row["component_id"] for row in splits["test"]}
    return {
        "train_test_subject_overlap": len(train_subjects & test_subjects),
        "train_test_component_overlap": len(train_components & test_components),
        "test_subject_contamination_pct": round(
            100 * len(train_subjects & test_subjects) / max(1, len(test_subjects)), 2
        ),
    }


def run_one_seed(
    seed: int,
    split_path: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    pretrained: bool,
    device: str,
    resume: bool = False,
    arch: str = "resnet18",
) -> dict:
    rows = read_split_rows(split_path)
    if not rows:
        raise SystemExit(
            f"No CN/AD rows found in {split_path}. Confirm the label ontology "
            "is finalized and the manifest carries diagnosis_group ∈ {CN, AD}."
        )
    component_safe = split_by_phase(rows)
    random_split = build_random_split(rows, seed)
    subject_only = build_subject_only_split(rows, seed)

    configurations = {
        "random": random_split,
        "subject_only": subject_only,
        "component_safe": component_safe,
    }

    seed_record: dict[str, dict] = {}
    for label, splits in configurations.items():
        if not splits["train"] or not splits["test"]:
            print(f"Skipping {label}: empty train/test under this split.")
            continue
        output_dir = output_root / f"inflation_gap_seed{seed}" / label
        existing_metrics = output_dir / "metrics.json"
        if resume and existing_metrics.exists():
            metrics_payload = json.loads(existing_metrics.read_text(encoding="utf-8"))
            metrics_payload["overlap"] = overlap_stats(splits)
            metrics_payload["resumed_from_disk"] = True
            seed_record[label] = metrics_payload
            print(f"  [resume] seed={seed} {label}: loaded existing metrics.json (skipping training)")
            continue
        metrics_payload = train_and_eval(
            splits,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            image_size=image_size,
            pretrained=pretrained,
            label=label,
            output_dir=output_dir,
            device_str=device,
            arch=arch,
        )
        metrics_payload["overlap"] = overlap_stats(splits)
        seed_record[label] = metrics_payload

    out_path = output_root / f"inflation_gap_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resolve to absolute before relative_to so callers can pass either
    # absolute or relative --split-dir without tripping over relative_to's
    # subpath requirement.
    try:
        split_rel = str(split_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        split_rel = str(split_path)
    out_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": seed,
                "split_file": split_rel,
                "results": seed_record,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return seed_record


def aggregate_table(per_seed: dict[int, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "protocol",
        "n_train",
        "n_test",
        "auroc",
        "balanced_accuracy",
        "f1_ad",
        "sensitivity",
        "specificity",
        "brier_score",
        "ece",
        "train_test_subject_overlap",
        "test_subject_contamination_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed, record in per_seed.items():
            for protocol, payload in record.items():
                metrics = payload["test_metrics"]
                writer.writerow({
                    "seed": seed,
                    "protocol": protocol,
                    "n_train": payload["n_train"],
                    "n_test": payload["n_test"],
                    "auroc": metrics["auroc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1_ad": metrics["f1_ad"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "brier_score": metrics["brier_score"],
                    "ece": metrics["ece"],
                    "train_test_subject_overlap": payload["overlap"]["train_test_subject_overlap"],
                    "test_subject_contamination_pct": payload["overlap"]["test_subject_contamination_pct"],
                })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip protocols whose runs/adni/inflation_gap_seed{N}/{label}/metrics.json "
        "already exists. Lets a killed run pick up where it stopped.",
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "densenet121", "efficientnet_b0"],
        help="Backbone architecture. Default resnet18 matches the primary "
        "manuscript results; use densenet121 or efficientnet_b0 for the "
        "architecture-breadth sensitivity arm.",
    )
    args = parser.parse_args()

    per_seed: dict[int, dict] = {}
    for seed in args.seeds:
        split_path = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {split_path}. "
                "Run scripts/make_adni_splitguard_split.py with all seeds first."
            )
        per_seed[seed] = run_one_seed(
            seed=seed,
            split_path=split_path,
            output_root=args.output_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            pretrained=not args.no_pretrained,
            device=args.device,
            resume=args.resume,
            arch=args.arch,
        )

    aggregate_table(per_seed, args.table)
    try:
        table_rel = str(args.table.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        table_rel = str(args.table)
    print(f"Wrote {table_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
