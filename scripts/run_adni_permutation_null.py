#!/usr/bin/env python3
"""Label-permutation null control for the ADNI inflation-gap experiment.

What this answers
-----------------
Two questions the main experiment cannot answer on its own.

1. **Is Protocol C's AUROC a degraded signal, or is it noise?** Protocol C is
   estimated on roughly 38 independent patients per seed. Five seeds tell you
   how much the estimate moves between seeds; they do not tell you how much it
   would move if there were no signal at all. Permuting the labels gives that
   reference directly: the spread of permuted Protocol C AUROCs is the
   empirical noise floor for a test partition of this size. Every marginal the
   paper reports --- the +0.019 component layer, the -0.043 converter-arm
   inversion, the -0.018 MT1 inversion --- can then be compared against a
   measured floor instead of argued about at n=5.

2. **Can leakage alone manufacture the gap?** This is the stronger result and
   it falls out of the same run. Labels are permuted *per subject*, not per
   image: every scan of a patient receives the same permuted label, so
   within-subject label consistency survives while the association between
   anatomy and diagnosis is destroyed. Under Protocol A a model can still
   memorise "this patient carries label 1" from training and recognise the
   same patient in the test partition, so its AUROC should exceed 0.5 despite
   there being no disease signal to learn. Under Protocol C the patient is
   never seen twice, so it should sit at 0.5. A gap that reappears under
   permuted labels is a gap that leakage produced by itself.

Permuting per image instead would break the second question by construction:
five scans of one patient would carry five different random labels, so
identity memorisation could not pay off under Protocol A, and the null would
collapse to 0.5 everywhere for reasons that have nothing to do with the split
protocol.

The permutation is applied independently within each partition so that the
class balance of every partition is preserved exactly; only the assignment of
labels to patients moves.

Usage
-----
    python3 scripts/run_adni_permutation_null.py --device cuda
    python3 scripts/run_adni_permutation_null.py --seeds 0 1 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adni_inflation_gap import (  # noqa: E402 — sys.path tweak above
    build_random_split,
    build_subject_only_split,
    overlap_stats,
)
from train_adni_baseline import (  # noqa: E402
    read_split_rows,
    split_by_phase,
    train_and_eval,
)

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "adni"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "adni_permutation_null"
DEFAULT_TABLE = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_permutation_null.json"


def permute_labels_by_subject(
    rows: list[dict[str, str]], rng: random.Random
) -> list[dict[str, str]]:
    """Reassign diagnosis labels across subjects, keeping each subject internally consistent.

    The multiset of subject-level labels is shuffled and dealt back out, so the
    number of CN and AD *subjects* is unchanged. Scan counts differ between
    subjects, so the number of CN and AD *scans* can shift slightly; that is
    the honest consequence of permuting at the unit the leakage operates on,
    and it is recorded in the run manifest.
    """
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)

    subjects = sorted(by_subject)
    labels = [by_subject[s][0]["diagnosis_group"] for s in subjects]
    rng.shuffle(labels)

    permuted: list[dict[str, str]] = []
    for subject, label in zip(subjects, labels):
        for row in by_subject[subject]:
            new_row = dict(row)
            new_row["diagnosis_group"] = label
            permuted.append(new_row)
    return permuted


def permute_splits(
    splits: dict[str, list[dict[str, str]]], rng: random.Random
) -> dict[str, list[dict[str, str]]]:
    """Permute within each partition independently, preserving per-partition balance."""
    return {phase: permute_labels_by_subject(rows, rng) for phase, rows in splits.items()}


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
    arch: str,
) -> dict:
    rows = read_split_rows(split_path)
    if not rows:
        raise SystemExit(f"No CN/AD rows found in {split_path}.")

    configurations = {
        "random": build_random_split(rows, seed),
        "subject_only": build_subject_only_split(rows, seed),
        "component_safe": split_by_phase(rows),
    }

    seed_record: dict[str, dict] = {}
    for label, splits in configurations.items():
        if not splits["train"] or not splits["test"]:
            print(f"Skipping {label}: empty train/test under this split.")
            continue
        # A permutation RNG separate from the training seed, so the label
        # shuffle is reproducible without perturbing the training trajectory.
        rng = random.Random(10_000 + seed)
        permuted = permute_splits(splits, rng)

        output_dir = output_root / f"permutation_null_seed{seed}" / label
        finished = output_dir / "metrics.json"
        if finished.exists():
            # Resume: this cell was trained by an earlier, interrupted run.
            metrics_payload = json.loads(finished.read_text())
            metrics_payload["resumed_from_disk"] = True
        else:
            metrics_payload = train_and_eval(
                permuted,
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
        metrics_payload["overlap"] = overlap_stats(permuted)
        metrics_payload["labels_permuted"] = "per_subject_within_partition"
        seed_record[label] = metrics_payload
        auroc = metrics_payload.get("test_metrics", {}).get("auroc")
        print(f"  seed={seed} {label:15s} permuted-label AUROC = {auroc}")

    out_path = output_root / f"permutation_null_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": seed,
                "split_file": str(split_path),
                "permutation": "per_subject_within_partition",
                "results": seed_record,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return seed_record


def summarise(per_seed: dict[int, dict], output_path: Path) -> None:
    """Emit the noise floor per protocol: mean, SD and range of permuted AUROC."""
    protocols = ("random", "subject_only", "component_safe")
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "permutation": "per_subject_within_partition",
        "n_seeds": len(per_seed),
        "by_protocol": {},
    }
    for protocol in protocols:
        aurocs = [
            record[protocol]["test_metrics"]["auroc"]
            for record in per_seed.values()
            if protocol in record
        ]
        if not aurocs:
            continue
        mean = sum(aurocs) / len(aurocs)
        var = sum((a - mean) ** 2 for a in aurocs) / (len(aurocs) - 1) if len(aurocs) > 1 else 0.0
        summary["by_protocol"][protocol] = {
            "per_seed_auroc": [round(a, 4) for a in aurocs],
            "mean": round(mean, 4),
            "sd": round(var**0.5, 4),
            "min": round(min(aurocs), 4),
            "max": round(max(aurocs), 4),
        }

    by = summary["by_protocol"]
    if "random" in by and "component_safe" in by:
        summary["permuted_inflation_gap"] = round(
            by["random"]["mean"] - by["component_safe"]["mean"], 4
        )
        summary["interpretation"] = (
            "component_safe under permuted labels is the empirical noise floor for a "
            "test partition of this size: any protocol-to-protocol difference smaller "
            "than its spread is not distinguishable from noise at this n. "
            "permuted_inflation_gap is the gap produced by patient-identity leakage "
            "alone, with no disease signal available to learn; a positive value is "
            "direct evidence that leakage manufactures inflation by itself."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}")
    for protocol, stats in by.items():
        print(f"  {protocol:15s} mean {stats['mean']:.4f}  sd {stats['sd']:.4f}  "
              f"range [{stats['min']:.4f}, {stats['max']:.4f}]")
    if "permuted_inflation_gap" in summary:
        print(f"\n  gap under permuted labels: {summary['permuted_inflation_gap']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "densenet121", "efficientnet_b0", "vit_b_16"])
    args = ap.parse_args()

    per_seed: dict[int, dict] = {}
    for seed in args.seeds:
        split_path = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        print(f"\n=== seed {seed} (labels permuted per subject) ===")
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
            arch=args.arch,
        )

    summarise(per_seed, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
