#!/usr/bin/env python3
"""Freeze ADNI SplitGuard train / val / test splits.

This script reads the ADNI leakage-components manifest and assigns whole
connected components to a single partition, so that no subject (or session,
or series) crosses partitions.

Inputs
------
* ``data/manifests/adni/adni_leakage_components.csv`` (from
  ``scripts/build_adni_leakage_graph.py``).

Outputs
-------
* ``data/splits/adni/adni_splitguard_seed{SEED}.csv`` — one file per seed,
  with the same columns as the components manifest plus a ``split`` column
  in ``{train, val, test}``.
* ``reports/audits/adni/adni_splitguard_seed{SEED}_summary.json`` — split
  counts and zero-overlap assertions.

Allocation
----------
For the primary CN vs AD task (``diagnosis_group`` ∈ {CN, AD}):

* Components with ``component_label`` ∉ {CN, AD} are excluded from this
  split file (they are kept in a separate ``data/splits/adni/excluded_*.csv``
  for traceability).
* Each label is split independently into 70 / 15 / 15 by image count, using
  a greedy subset-sum approach to keep label balance tight.
* Stratification by ``scanner_field_strength`` is best-effort: components
  are bucketed by majority field strength within the label, and each bucket
  is split independently before merging.

The script enforces ``contamination == 0`` and aborts with a non-zero exit
code if any leakage component crosses partitions.

Run::

    python3 scripts/make_adni_splitguard_split.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import display_path  # noqa: E402

DEFAULT_COMPONENTS = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_leakage_components.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "adni"
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "reports" / "audits" / "adni"

SPLIT_ORDER = ["train", "val", "test"]
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
PRIMARY_LABELS = {"CN", "AD"}


def read_components(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "subject_id", "component_id", "component_label", "diagnosis_group"}
    missing = required - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Components manifest missing columns: {sorted(missing)}")
    return rows


def class_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = round(total * ratios["train"])
    val = round(total * ratios["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def choose_subset_by_size(
    components: list[dict[str, Any]],
    target: int,
) -> set[str]:
    """Greedy subset-sum: pick component IDs whose total image count is
    closest to ``target`` from below, breaking ties toward smaller totals."""
    if target <= 0:
        return set()
    # Sort descending by size, breaking ties by the caller's seeded shuffle
    # position (``_shuffle_rank``, assigned in assign_components_for_label).
    #
    # This tie-break is load-bearing for BOTH contracts at once:
    #   * byte-reproducibility — the rank is an explicit integer key, so the
    #     ordering does not depend on sort stability, dict iteration order or
    #     the platform;
    #   * seed sensitivity — the rank comes from ``random.Random(seed)``, so
    #     different seeds genuinely produce different val/test subsets.
    #
    # Do NOT replace the rank with ``component_id``. A total order on a
    # seed-independent key erases the shuffle entirely and silently makes the
    # seed inert: every "per-seed" manifest then regenerates byte-identically
    # and the reported seed-to-seed variability collapses to training noise
    # alone. That regression shipped once (commit 877ca72) and is what
    # ``tests/test_manifest_reproducibility.py`` now guards against.
    sorted_components = sorted(
        components,
        key=lambda component: (
            -int(component["n_images"]),
            int(component["_shuffle_rank"]),
        ),
    )
    chosen: set[str] = set()
    remaining = target
    for component in sorted_components:
        size = int(component["n_images"])
        if size <= remaining:
            chosen.add(str(component["component_id"]))
            remaining -= size
            if remaining <= 0:
                break
    return chosen


def choose_subset_shuffled(
    components: list[dict[str, Any]],
    target: int,
) -> set[str]:
    """Fill ``target`` by visiting components in seeded-shuffle order.

    Control for ``choose_subset_by_size``. Visiting the largest components
    first means val, filled first, takes the largest participants and train
    keeps the smallest, so partitions differ in how many scans a participant
    contributes. Visiting in shuffle order removes that ordering while keeping
    both contracts the default relies on: the rank is an explicit integer key
    (byte-reproducible) and it comes from the seed (seed-sensitive).
    """
    if target <= 0:
        return set()
    chosen: set[str] = set()
    remaining = target
    for component in sorted(components, key=lambda c: int(c["_shuffle_rank"])):
        size = int(component["n_images"])
        if size <= remaining:
            chosen.add(str(component["component_id"]))
            remaining -= size
            if remaining <= 0:
                break
    return chosen


CHOOSERS = {"size_descending": choose_subset_by_size, "shuffled": choose_subset_shuffled}


def bucket_field_strength(rows: list[dict[str, str]]) -> str:
    counts = Counter(row.get("scanner_field_strength") or "missing" for row in rows)
    return counts.most_common(1)[0][0]


def assign_components_for_label(
    components: list[dict[str, Any]],
    seed: int,
    chooser=choose_subset_by_size,
) -> dict[str, str]:
    """Return component_id → split assignment for one label group."""
    rng = random.Random(seed)
    rng.shuffle(components)
    # Freeze the seeded order into an explicit key so downstream sorts can
    # break ties on it without depending on Python's sort stability.
    for rank, component in enumerate(components):
        component["_shuffle_rank"] = rank
    # Group by majority field strength, then split each bucket independently.
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        by_bucket[component["bucket"]].append(component)

    assignments: dict[str, str] = {}
    for bucket, bucket_components in by_bucket.items():
        total = sum(int(c["n_images"]) for c in bucket_components)
        targets = class_targets(total, DEFAULT_RATIOS)
        val_ids = chooser(bucket_components, targets["val"])
        remaining = [c for c in bucket_components if c["component_id"] not in val_ids]
        test_ids = chooser(remaining, targets["test"])
        for component in bucket_components:
            cid = str(component["component_id"])
            if cid in val_ids:
                assignments[cid] = "val"
            elif cid in test_ids:
                assignments[cid] = "test"
            else:
                assignments[cid] = "train"
    return assignments


def build_assignments(
    components_by_label: dict[str, list[dict[str, Any]]],
    seed: int,
    chooser=choose_subset_by_size,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for label in sorted(components_by_label):
        per_label = assign_components_for_label(components_by_label[label], seed, chooser)
        assignments.update(per_label)
    return assignments


def write_split(
    rows: list[dict[str, str]],
    assignments: dict[str, str],
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["split"] if rows else ["split"]
    out_rows: list[dict[str, str]] = []
    label_counts: dict[str, Counter] = defaultdict(Counter)
    split_counts: Counter = Counter()
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    components_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        component_id = row["component_id"]
        if component_id not in assignments:
            continue  # excluded label / mixed component
        split = assignments[component_id]
        out_rows.append({**row, "split": split})
        label_counts[split][row["diagnosis_group"]] += 1
        split_counts[split] += 1
        subjects_by_split[split].add(row["subject_id"])
        components_by_split[split].add(component_id)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    overlap = {
        "train_test_subject_overlap": len(subjects_by_split["train"] & subjects_by_split["test"]),
        "train_val_subject_overlap": len(subjects_by_split["train"] & subjects_by_split["val"]),
        "val_test_subject_overlap": len(subjects_by_split["val"] & subjects_by_split["test"]),
        "train_test_component_overlap": len(components_by_split["train"] & components_by_split["test"]),
    }
    return {
        "split_counts": dict(split_counts),
        "label_counts": {split: dict(counter) for split, counter in label_counts.items()},
        "subject_counts": {split: len(subjects) for split, subjects in subjects_by_split.items()},
        "component_counts": {split: len(components) for split, components in components_by_split.items()},
        "overlap": overlap,
    }


def assert_zero_contamination(stats: dict[str, Any]) -> None:
    overlap = stats["overlap"]
    failed = {key: value for key, value in overlap.items() if value > 0}
    if failed:
        raise SystemExit(f"Non-zero contamination in split: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--assignment", choices=sorted(CHOOSERS), default="size_descending",
        help="Order in which components fill val and test. size_descending is "
             "the frozen protocol; shuffled is the composition control and "
             "must be written to its own --split-dir.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=sorted(PRIMARY_LABELS),
        help="Diagnosis labels included in the primary task (default: CN AD).",
    )
    parser.add_argument(
        "--include-mixed-by-majority",
        action="store_true",
        help="Reclassify mixed_label components (longitudinal converters whose "
             "visits span CN/MCI/AD) to CN or AD by majority vote of their "
             "CN+AD visits, and include them in the split. Used for the "
             "289-scan exclusion sensitivity analysis. The component-safe "
             "guarantee is preserved: each converter component is still "
             "assigned to a single partition.",
    )
    args = parser.parse_args()
    if args.assignment != "size_descending" and args.split_dir == DEFAULT_SPLIT_DIR:
        raise SystemExit("--assignment shuffled would overwrite the frozen splits; pass --split-dir and --summary-dir.")

    if not args.components.exists():
        raise FileNotFoundError(
            f"Components manifest does not exist: {args.components}. "
            "Run scripts/build_adni_leakage_graph.py first."
        )

    rows = read_components(args.components)
    primary_labels = set(args.labels)

    # Pre-aggregate: one record per component (size, majority bucket, label).
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_component[row["component_id"]].append(row)

    components_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, str]] = []
    for component_id, component_rows in by_component.items():
        label = component_rows[0]["component_label"]
        if label == "mixed_label" and args.include_mixed_by_majority:
            # Sensitivity arm: reclassify the component to whichever of
            # {CN, AD} carries the majority of CN+AD visits within it.
            # MCI visits in the component remain MCI (and will be excluded
            # from the binary CN-vs-AD task downstream).
            votes = Counter(
                r["diagnosis_group"] for r in component_rows
                if r["diagnosis_group"] in primary_labels
            )
            if votes:
                label = votes.most_common(1)[0][0]
                for r in component_rows:
                    r["component_label"] = label
            else:
                excluded.extend(component_rows)
                continue
        if label not in primary_labels:
            excluded.extend(component_rows)
            continue
        components_by_label[label].append(
            {
                "component_id": component_id,
                "n_images": len(component_rows),
                "label": label,
                "bucket": bucket_field_strength(component_rows),
            }
        )

    if not components_by_label:
        raise SystemExit(
            "No components matched the primary label set "
            f"{sorted(primary_labels)}. Either lower the threshold or finalize "
            "the label ontology so the manifest carries CN/AD diagnoses."
        )

    args.split_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    if excluded:
        excluded_path = args.split_dir / "excluded_components.csv"
        with excluded_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(excluded[0].keys()))
            writer.writeheader()
            writer.writerows(excluded)
        try:
            excluded_rel = display_path(excluded_path)
        except ValueError:
            excluded_rel = str(excluded_path)
        print(f"Wrote {excluded_rel} ({len(excluded)} excluded rows)")

    overall: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        assignments = build_assignments(components_by_label, seed, CHOOSERS[args.assignment])
        out_path = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        stats = write_split(rows, assignments, out_path)
        assert_zero_contamination(stats)
        summary_path = args.summary_dir / f"adni_splitguard_seed{seed}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "seed": seed,
                    "labels": sorted(primary_labels),
                    **stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        overall[seed] = stats
        try:
            out_rel = display_path(out_path)
        except ValueError:
            out_rel = str(out_path)
        print(f"Wrote {out_rel} (split_counts={stats['split_counts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
