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


def class_order(components: list[dict]) -> list[str]:
    """Stratification order: the known classes first, then whatever else is present."""
    present = {component["raw_class_label"] for component in components}
    known = [label for label in RAW_CLASS_ORDER if label in present]
    return known + sorted(present - set(RAW_CLASS_ORDER))
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


def class_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = round(total * ratios["train"])
    val = round(total * ratios["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def choose_subset_by_size(components: list[dict], target: int) -> set[str]:
    """Return component IDs with total image count closest to target.

    The DP iteration order depends on the input list order, so callers
    that pass a differently-ordered list can end up with different
    subsets on ties. We therefore start by sorting the components on
    ``(-n_images, component_id)`` so this function is a pure function of
    the component set (identities + sizes), independent of the caller's
    ordering. This guarantee is a prerequisite for byte-reproducible
    manifests.
    """
    if target <= 0:
        return set()

    sorted_components = sorted(
        components,
        key=lambda component: (
            -int(component["n_images"]),
            # Seeded rank, not component_id — see assign_splits() for why.
            int(component.get("_shuffle_rank", 0)),
        ),
    )

    # dp[sum] = tuple(component_ids)
    dp: dict[int, tuple[str, ...]] = {0: ()}
    for component in sorted_components:
        component_id = component["component_id"]
        size = component["n_images"]
        additions: dict[int, tuple[str, ...]] = {}
        for current_sum, chosen in dp.items():
            new_sum = current_sum + size
            if new_sum not in dp and new_sum not in additions:
                additions[new_sum] = (*chosen, component_id)
        dp.update(additions)

    best_sum = min(dp, key=lambda value: (abs(value - target), value > target, value))
    return set(dp[best_sum])


def build_components(component_rows: list[dict[str, str]]) -> list[dict]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in component_rows:
        groups[row["component_id"]].append(row)

    components = []
    for component_id, rows in groups.items():
        raw_labels = sorted({row["raw_class_label"] for row in rows})
        binary_labels = sorted({row["binary_label"] for row in rows})
        if len(raw_labels) != 1 or len(binary_labels) != 1:
            raise ValueError(
                "SplitGuard v0 requires pure-label components. "
                f"Component {component_id} has raw={raw_labels}, binary={binary_labels}."
            )
        components.append(
            {
                "component_id": component_id,
                "n_images": len(rows),
                "raw_class_label": raw_labels[0],
                "binary_label": binary_labels[0],
                "subject_ids": sorted({row["subject_id"] for row in rows}),
                "primary_reason": rows[0]["component_primary_reason"],
            }
        )
    return components


def assign_splits(components: list[dict], seed: int, ratios: dict[str, float]) -> tuple[dict[str, str], list[dict]]:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    target_rows = []

    components_by_raw_class: dict[str, list[dict]] = defaultdict(list)
    for component in components:
        components_by_raw_class[component["raw_class_label"]].append(component)

    for raw_class in class_order(components):
        class_components = components_by_raw_class.get(raw_class, [])
        if not class_components:
            continue

        shuffled = class_components[:]
        rng.shuffle(shuffled)
        # Freeze the seeded order into an explicit key, then sort by size with
        # that rank as the tie-break. This keeps the ordering byte-identical
        # for a given seed WITHOUT making it seed-independent: tie-breaking on
        # ``component_id`` instead would impose a total order that erases the
        # shuffle, leaving the seed inert and every per-seed manifest identical.
        for rank, item in enumerate(shuffled):
            item["_shuffle_rank"] = rank
        shuffled.sort(
            key=lambda item: (-int(item["n_images"]), int(item["_shuffle_rank"])),
        )

        total_images = sum(component["n_images"] for component in shuffled)
        targets = class_targets(total_images, ratios)

        val_components = choose_subset_by_size(shuffled, targets["val"])
        remaining_after_val = [
            component for component in shuffled if component["component_id"] not in val_components
        ]
        test_components = choose_subset_by_size(remaining_after_val, targets["test"])

        for component in shuffled:
            component_id = component["component_id"]
            if component_id in val_components:
                split = "val"
            elif component_id in test_components:
                split = "test"
            else:
                split = "train"
            assignments[component_id] = split

        achieved = Counter()
        component_counts = Counter()
        for component in shuffled:
            split = assignments[component["component_id"]]
            achieved[split] += component["n_images"]
            component_counts[split] += 1

        for split in SPLIT_ORDER:
            target_rows.append(
                {
                    "raw_class_label": raw_class,
                    "split": split,
                    "target_images": targets[split],
                    "achieved_images": achieved[split],
                    "component_count": component_counts[split],
                }
            )

    return assignments, target_rows


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

    go_no_go = (
        "GO for Step 4: baseline training can use this split manifest. "
        "The split is component-safe and preserves exact binary balance."
        if summary["overlap_check_passed"]
        else "NO-GO: component overlap was detected and must be fixed before training."
    )

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
