#!/usr/bin/env python3
"""Regenerate validation predictions from saved training checkpoints.

Why this exists
---------------
``train_adni_baseline.py`` originally persisted only ``test_predictions.csv``.
Downstream analyses that need a decision *threshold* — the fixed-specificity
anchors and the Youden-J optimum in
``operating_point_sensitivity_table.py`` — therefore had no choice but to
scan the test ROC and take the best-scoring threshold. That is test-set
selection: it reports the sensitivity a model *could* reach at a
retrospectively chosen operating point, not one it *would* reach at a
prospectively chosen one. In a manuscript whose entire subject is
evaluation integrity, that is the one class of error we cannot ship.

The trainer now writes ``val_predictions.csv`` alongside the test file. This
script backfills that artefact for runs that predate the change, so existing
results can be recomputed **without retraining**: every run directory already
carries ``best_state.pt``, so we only need forward passes over the validation
partition.

Usage
-----
    python3 scripts/regenerate_val_predictions.py \
        --runs-root runs/adni_with_converters \
        --split-dir data/splits/adni_with_converters

For each ``inflation_gap_seed<S>/<protocol>/`` directory holding a
``best_state.pt`` but no ``val_predictions.csv``, the corresponding split
manifest is re-read, the checkpoint is loaded, and the validation partition
is scored.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "adni_with_converters"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "adni_with_converters"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv"

def _load_module(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def val_rows_for(runner, split_csv: Path, protocol: str, seed: int) -> list[dict[str, str]]:
    """Validation rows for one protocol.

    Only ``component_safe`` lives in the manifest's ``split`` column; the
    ``random`` and ``subject_only`` partitions are *derived* at training time
    from the same rows and the same seed. We therefore call the exact builders
    the training runner uses, rather than reading a column that does not exist
    — otherwise the "validation" set used for threshold selection would not be
    the one the model was validated on.
    """
    rows = runner.read_split_rows(split_csv)
    if not rows:
        return []
    if protocol == "component_safe":
        splits = runner.split_by_phase(rows)
    elif protocol == "random":
        splits = runner.build_random_split(rows, seed)
    elif protocol == "subject_only":
        splits = runner.build_subject_only_split(rows, seed)
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")
    return splits.get("val", [])


def regenerate_one(
    trainer,
    runner,
    run_dir: Path,
    split_csv: Path,
    protocol: str,
    seed: int,
    image_size: int,
    batch_size: int,
    device_str: str,
) -> int:
    checkpoint = run_dir / "best_state.pt"
    out_path = run_dir / "val_predictions.csv"
    if not checkpoint.exists():
        print(f"  - {run_dir.name}: no best_state.pt, skipping")
        return 0

    rows = val_rows_for(runner, split_csv, protocol, seed)
    if not rows:
        print(f"  - {run_dir.name}: no validation rows found in {split_csv.name}")
        return 0

    deps = trainer.lazy_imports()
    torch = deps["torch"]
    device = trainer.choose_device(device_str)
    _train_tf, eval_tf = trainer.make_transforms(image_size)
    loader = trainer.make_loader(rows, eval_tf, batch_size, False, device, seed=0)

    model = trainer.make_model(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("state", payload)
    model.load_state_dict(state)

    _metrics, y_true, y_prob = trainer.evaluate(model, loader, device)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "subject_id", "diagnosis_group",
                        "y_true", "y_prob"],
        )
        writer.writeheader()
        for row, y_t, y_p in zip(rows, y_true.tolist(), y_prob.tolist()):
            writer.writerow({
                "image_id": row.get("image_id", ""),
                "subject_id": row.get("subject_id", ""),
                "diagnosis_group": row.get("diagnosis_group", ""),
                "y_true": int(y_t),
                "y_prob": round(float(y_p), 6),
            })
    print(f"  + {run_dir.parent.name}/{run_dir.name}: wrote {len(rows)} val predictions")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing val_predictions.csv.")
    args = parser.parse_args()

    if not args.runs_root.exists():
        raise SystemExit(f"Runs root not found: {args.runs_root}")

    trainer = _load_module("train_adni_baseline")
    runner = _load_module("run_adni_inflation_gap")
    total = 0
    for seed in args.seeds:
        seed_dir = args.runs_root / f"inflation_gap_seed{seed}"
        if not seed_dir.exists():
            print(f"  - seed {seed}: {seed_dir} missing, skipping")
            continue
        split_csv = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        if not split_csv.exists():
            print(f"  - seed {seed}: split manifest {split_csv} missing, skipping")
            continue
        for protocol_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            if (protocol_dir / "val_predictions.csv").exists() and not args.force:
                print(f"  = {seed_dir.name}/{protocol_dir.name}: already present")
                continue
            total += regenerate_one(
                trainer, runner, protocol_dir, split_csv, protocol_dir.name,
                seed, args.image_size, args.batch_size, args.device,
            )

    print(f"\nWrote {total} validation predictions in total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
