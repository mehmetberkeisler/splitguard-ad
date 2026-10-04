#!/usr/bin/env python3
"""Create the first component-safe SplitGuard train/val/test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splitguard_ad.splitting import (  # noqa: E402
    assign_splits, build_components, choose_subset_by_size, class_order,
    class_targets, compositional_warnings)
from _paths import display_path  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_manifest.csv"
DEFAULT_COMPONENTS = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_leakage_components.csv"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_splitguard_seed42_audit.md"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_splitguard_seed42_summary.json"

# Preferred ordering for the Tier-1 benchmark's own four classes. It fixes the
# order in which classes are stratified, which is what keeps a given seed
# byte-reproducible. It is NOT the set of permitted classes: any class present
# in the manifest but absent here is stratified after these, in sorted order.
# Treating it as a permitted set is what it used to do, and a component whose
# label fell outside it was silently skipped during assignment and then raised
# KeyError on lookup -- so the splitter worked on this benchmark and crashed on
# anybody else's cohort.
RAW_CLASS_ORDER = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


SPLIT_ORDER = ["train", "val", "test"]
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)










def build_split_rows(
    manifest_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
    assignments: dict[str, str],
    seed: int,
) -> list[dict[str, object]]:
    # Fail on the manifest as a whole rather than on the first row that happens
    # to be missing something. A bare KeyError raised inside this loop names one
    # column and gives no indication that the caller's manifest is simply a
    # different shape, which is what a third party supplying their own cohort
    # actually hits. Descriptive columns are optional and default to empty.
    required = {"image_id"}
    if manifest_rows:
        missing = sorted(required - set(manifest_rows[0]))
        if missing:
            raise SystemExit(
                f"Manifest is missing required column(s): {missing}. "
                f"Present columns: {sorted(manifest_rows[0])}"
            )

    manifest_by_id = {row["image_id"]: row for row in manifest_rows}

    output_rows = []
    for component_row in sorted(component_rows, key=lambda row: row["relative_path"]):
        image_id = component_row["image_id"]
        manifest_row = manifest_by_id[image_id]
        component_id = component_row["component_id"]
        split = assignments[component_id]
        output_rows.append(
            {
                "image_id": image_id,
                "split": split,
                "component_id": component_id,
                "component_size": component_row["component_size"],
                "component_primary_reason": component_row["component_primary_reason"],
                "split_policy": "splitguard_component_safe_raw_class_stratified_v1",
                "split_seed": seed,
                "path": manifest_row.get("path", ""),
                "relative_path": component_row["relative_path"],
                "source_dataset": manifest_row.get("source_dataset", ""),
                "raw_class_label": component_row["raw_class_label"],
                "binary_label": component_row["binary_label"],
                "clinical_label": manifest_row.get("clinical_label", ""),
                "label_confidence": manifest_row.get("label_confidence", ""),
                "subject_id": component_row["subject_id"],
                "subject_id_confidence": component_row["subject_id_confidence"],
                "subject_parse_status": component_row["subject_parse_status"],
                "preprocessing_version": manifest_row.get("preprocessing_version", ""),
            }
        )
    return output_rows


def summarize(split_rows: list[dict[str, object]], target_rows: list[dict]) -> dict:
    images_by_split = Counter(row["split"] for row in split_rows)
    binary_by_split: dict[str, Counter] = {split: Counter() for split in SPLIT_ORDER}
    raw_by_split: dict[str, Counter] = {split: Counter() for split in SPLIT_ORDER}
    components_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_ORDER}
    rows_by_component: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in split_rows:
        split = str(row["split"])
        binary_by_split[split][str(row["binary_label"])] += 1
        raw_by_split[split][str(row["raw_class_label"])] += 1
        components_by_split[split].add(str(row["component_id"]))
        rows_by_component[str(row["component_id"])].append(row)

    leaking_components = []
    for component_id, rows in rows_by_component.items():
        splits = sorted({str(row["split"]) for row in rows})
        if len(splits) > 1:
            leaking_components.append({"component_id": component_id, "splits": splits})

    component_overlap = {}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlap = sorted(components_by_split[left].intersection(components_by_split[right]))
            component_overlap[f"{left}_vs_{right}"] = overlap

    return {
        "total_images": len(split_rows),
        "images_by_split": dict(images_by_split),
        "binary_by_split": {
            split: dict(binary_by_split[split]) for split in SPLIT_ORDER
        },
        "raw_class_by_split": {
            split: dict(raw_by_split[split]) for split in SPLIT_ORDER
        },
        "components_by_split": {
            split: len(components_by_split[split]) for split in SPLIT_ORDER
        },
        "target_rows": target_rows,
        "leaking_components": leaking_components,
        "component_overlap": component_overlap,
        "overlap_check_passed": not leaking_components
        and all(len(overlap) == 0 for overlap in component_overlap.values()),
    }




def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def write_audit(summary: dict, split_path: Path, audit_path: Path, seed: int) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    images_by_split = markdown_table(
        ["Split", "Images", "Components"],
        [
            [
                split,
                summary["images_by_split"].get(split, 0),
                summary["components_by_split"].get(split, 0),
            ]
            for split in SPLIT_ORDER
        ],
    )
    # Column headers come from the labels the split actually contains. Hardcoding
    # this benchmark's own label names produced a table of zeros for any cohort
    # that names its classes differently, which reads as a finding rather than
    # as a mismatch and is worse than failing outright.
    binary_labels = sorted(
        {label for counts in summary["binary_by_split"].values() for label in counts}
    )
    binary_by_split = markdown_table(
        ["Split", *binary_labels],
        [
            [
                split,
                *[
                    summary["binary_by_split"].get(split, {}).get(label, 0)
                    for label in binary_labels
                ],
            ]
            for split in SPLIT_ORDER
        ],
    )
    present_raw = {
        label for counts in summary["raw_class_by_split"].values() for label in counts
    }
    raw_labels = [c for c in RAW_CLASS_ORDER if c in present_raw] + sorted(
        present_raw - set(RAW_CLASS_ORDER)
    )
    raw_by_split = markdown_table(
        ["Split", *raw_labels],
        [
            [
                split,
                *[
                    summary["raw_class_by_split"].get(split, {}).get(raw_class, 0)
                    for raw_class in raw_labels
                ],
            ]
            for split in SPLIT_ORDER
        ],
    )
    target_table = markdown_table(
        ["Raw Class", "Split", "Target Images", "Achieved Images", "Components"],
        [
            [
                row["raw_class_label"],
                row["split"],
                row["target_images"],
                row["achieved_images"],
                row["component_count"],
            ]
            for row in summary["target_rows"]
        ],
    )

    # A single undifferentiated GO conflates two different things. Partition
    # disjointness is a hard constraint: violate it and the evaluation is
    # invalid. Composition balance is not: a split can be provably
    # leakage-free and still route long-record participants away from
    # training, which is what the frozen ADNI manifests do and what the audit
    # as originally specified could not say. Blocking and advisory findings
    # are therefore reported separately, and a warning never downgrades a GO.
    warnings = compositional_warnings(summary)
    if not summary["overlap_check_passed"]:
        go_no_go = ("NO-GO: component overlap was detected and must be fixed "
                    "before training.")
    elif warnings:
        go_no_go = (
            "GO for Step 4, with " + str(len(warnings)) + " compositional "
            "warning(s): the split is component-safe and preserves exact "
            "binary balance, so it is valid for training. The warnings below "
            "concern how the partitions are composed, not whether they are "
            "disjoint; they do not block training and are not errors.")
    else:
        go_no_go = ("GO for Step 4: baseline training can use this split "
                    "manifest. The split is component-safe, preserves exact "
                    "binary balance, and raises no compositional warning.")
    warning_block = ("\n".join(f"- {w}" for w in warnings)
                     if warnings else "- None.")

    content = f"""# Current JPEG SplitGuard Seed {seed} Split Audit

## Summary

- Split manifest: `{display_path(split_path)}`
- Split policy: `splitguard_component_safe_raw_class_stratified_v1`
- Seed: **{seed}**
- Total images: **{summary["total_images"]}**
- Overlap check passed: **{summary["overlap_check_passed"]}**

## Images And Components By Split

{images_by_split}

## Binary Label Distribution By Split

{binary_by_split}

## Raw Class Distribution By Split

{raw_by_split}

## Raw-Class Stratification Targets

{target_table}

## Leakage Safety Checks

- Components appearing in multiple splits: **{len(summary["leaking_components"])}**
- `train` vs `val` overlap: **{len(summary["component_overlap"].get("train_vs_val", []))}**
- `train` vs `test` overlap: **{len(summary["component_overlap"].get("train_vs_test", []))}**
- `val` vs `test` overlap: **{len(summary["component_overlap"].get("val_vs_test", []))}**

## QC Decision

**{go_no_go}**

### Blocking checks

These determine the verdict. A failure here invalidates the evaluation.

- Component overlap across partitions: **{len(summary["leaking_components"])}**

### Compositional warnings

These do not block training. They describe how the partitions are composed
rather than whether they are disjoint, and a split can be provably
leakage-free while raising every one of them.

{warning_block}

## Next Step

Train the first baseline model using only this frozen split manifest. Training code should not create, shuffle, or rediscover splits from folders.
"""
    audit_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest)
    component_rows = read_csv(args.components)
    components = build_components(component_rows)
    assignments, target_rows = assign_splits(components, seed=args.seed, ratios=DEFAULT_RATIOS)
    split_rows = build_split_rows(manifest_rows, component_rows, assignments, seed=args.seed)
    summary = summarize(split_rows, target_rows)

    fieldnames = [
        "image_id",
        "split",
        "component_id",
        "component_size",
        "component_primary_reason",
        "split_policy",
        "split_seed",
        "path",
        "relative_path",
        "source_dataset",
        "raw_class_label",
        "binary_label",
        "clinical_label",
        "label_confidence",
        "subject_id",
        "subject_id_confidence",
        "subject_parse_status",
        "preprocessing_version",
    ]
    write_csv(args.output, split_rows, fieldnames)
    write_audit(summary, args.output, args.audit, seed=args.seed)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote split manifest: {args.output}")
    print(f"Wrote audit: {args.audit}")
    print(f"Wrote summary JSON: {args.summary_json}")
    print(f"Images by split: {summary['images_by_split']}")
    print(f"Binary by split: {summary['binary_by_split']}")
    print(f"Overlap check passed: {summary['overlap_check_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
