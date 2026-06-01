#!/usr/bin/env python3
"""Build SplitGuard-AD leakage components for the current JPEG manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_manifest.csv"
DEFAULT_COMPONENTS = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_leakage_components.csv"
DEFAULT_NEAR_DUPES = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_near_duplicate_candidates.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_leakage_graph_audit.md"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_leakage_graph_summary.json"


@dataclass(frozen=True)
class Record:
    image_id: str
    path: str
    relative_path: str
    raw_class_label: str
    binary_label: str
    subject_id: str
    subject_id_confidence: str
    subject_parse_status: str
    file_sha256: str


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}
        self.reasons: dict[str, set[str]] = defaultdict(set)

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str, reason: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            self.reasons[root_left].add(reason)
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.reasons[root_left].update(self.reasons[root_right])
        self.reasons[root_left].add(reason)
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def stable_token(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def read_manifest(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "image_id",
        "path",
        "relative_path",
        "raw_class_label",
        "binary_label",
        "subject_id",
        "subject_id_confidence",
        "subject_parse_status",
        "file_sha256",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    return [
        Record(
            image_id=row["image_id"],
            path=row["path"],
            relative_path=row["relative_path"],
            raw_class_label=row["raw_class_label"],
            binary_label=row["binary_label"],
            subject_id=row["subject_id"],
            subject_id_confidence=row["subject_id_confidence"],
            subject_parse_status=row["subject_parse_status"],
            file_sha256=row["file_sha256"],
        )
        for row in rows
    ]


def dhash(path: Path, hash_size: int = 16) -> int:
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())

    bits = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_blocking_components(records: list[Record]) -> tuple[UnionFind, dict]:
    uf = UnionFind([record.image_id for record in records])

    subject_groups: dict[str, list[Record]] = defaultdict(list)
    sha_groups: dict[str, list[Record]] = defaultdict(list)

    for record in records:
        subject_groups[record.subject_id].append(record)
        sha_groups[record.file_sha256].append(record)

    subject_edges = 0
    for group in subject_groups.values():
        if len(group) <= 1:
            continue
        first = group[0].image_id
        for record in group[1:]:
            uf.union(first, record.image_id, "same_subject")
            subject_edges += 1

    sha_edges = 0
    duplicate_sha_groups = 0
    for group in sha_groups.values():
        if len(group) <= 1:
            continue
        duplicate_sha_groups += 1
        first = group[0].image_id
        for record in group[1:]:
            uf.union(first, record.image_id, "exact_sha256_duplicate")
            sha_edges += 1

    return uf, {
        "same_subject_edges": subject_edges,
        "exact_sha256_edges": sha_edges,
        "exact_sha256_duplicate_groups": duplicate_sha_groups,
    }


def component_maps(records: list[Record], uf: UnionFind) -> tuple[dict[str, str], dict[str, list[Record]], dict[str, str]]:
    root_to_records: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        root_to_records[uf.find(record.image_id)].append(record)

    root_to_component_id: dict[str, str] = {}
    root_to_reason: dict[str, str] = {}
    for root, group in root_to_records.items():
        image_ids = sorted(record.image_id for record in group)
        component_id = f"comp_{stable_token(';'.join(image_ids))}"
        root_to_component_id[root] = component_id

        reasons = sorted(uf.reasons.get(root, set()))
        if not reasons and len(group) == 1:
            reason = "singleton"
        elif reasons:
            reason = "+".join(reasons)
        else:
            reason = "implicit_group"
        root_to_reason[root] = reason

    image_to_component_id = {
        record.image_id: root_to_component_id[uf.find(record.image_id)]
        for record in records
    }
    component_id_to_records = {
        root_to_component_id[root]: group for root, group in root_to_records.items()
    }
    component_id_to_reason = {
        root_to_component_id[root]: reason for root, reason in root_to_reason.items()
    }
    return image_to_component_id, component_id_to_records, component_id_to_reason


def detect_near_duplicates(
    records: list[Record],
    threshold: int,
    max_rows: int,
) -> tuple[list[dict], dict]:
    hashes = []
    failed = []
    for record in records:
        try:
            hashes.append((record, dhash(Path(record.path))))
        except Exception as exc:  # pragma: no cover - data dependent
            failed.append({"image_id": record.image_id, "error": str(exc)})

    candidates: list[dict] = []
    total_candidates = 0
    cross_subject_candidates = 0
    same_subject_candidates = 0
    distance_counts: Counter[int] = Counter()

    for idx, (left_record, left_hash) in enumerate(hashes):
        for right_record, right_hash in hashes[idx + 1 :]:
            distance = hamming(left_hash, right_hash)
            if distance > threshold:
                continue

            total_candidates += 1
            distance_counts[distance] += 1
            same_subject = left_record.subject_id == right_record.subject_id
            if same_subject:
                same_subject_candidates += 1
            else:
                cross_subject_candidates += 1

            if len(candidates) < max_rows:
                candidates.append(
                    {
                        "image_id_a": left_record.image_id,
                        "image_id_b": right_record.image_id,
                        "relative_path_a": left_record.relative_path,
                        "relative_path_b": right_record.relative_path,
                        "raw_class_label_a": left_record.raw_class_label,
                        "raw_class_label_b": right_record.raw_class_label,
                        "binary_label_a": left_record.binary_label,
                        "binary_label_b": right_record.binary_label,
                        "subject_id_a": left_record.subject_id,
                        "subject_id_b": right_record.subject_id,
                        "same_subject": same_subject,
                        "same_raw_class": left_record.raw_class_label == right_record.raw_class_label,
                        "same_binary_label": left_record.binary_label == right_record.binary_label,
                        "dhash_hamming_distance": distance,
                        "policy": "review_only_not_split_blocking",
                    }
                )

    return candidates, {
        "dhash_threshold": threshold,
        "dhash_hash_size": 16,
        "dhash_failed_images": failed,
        "near_duplicate_candidates_total": total_candidates,
        "near_duplicate_candidates_written": len(candidates),
        "near_duplicate_same_subject_candidates": same_subject_candidates,
        "near_duplicate_cross_subject_candidates": cross_subject_candidates,
        "near_duplicate_distance_counts": dict(sorted(distance_counts.items())),
        "near_duplicate_policy": "review_only_not_split_blocking",
    }


def write_components(
    records: list[Record],
    image_to_component_id: dict[str, str],
    component_id_to_records: dict[str, list[Record]],
    component_id_to_reason: dict[str, str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "component_id",
        "component_size",
        "component_primary_reason",
        "split_blocking",
        "relative_path",
        "raw_class_label",
        "binary_label",
        "subject_id",
        "subject_id_confidence",
        "subject_parse_status",
        "file_sha256",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.relative_path):
            component_id = image_to_component_id[record.image_id]
            component_size = len(component_id_to_records[component_id])
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "component_id": component_id,
                    "component_size": component_size,
                    "component_primary_reason": component_id_to_reason[component_id],
                    "split_blocking": "yes",
                    "relative_path": record.relative_path,
                    "raw_class_label": record.raw_class_label,
                    "binary_label": record.binary_label,
                    "subject_id": record.subject_id,
                    "subject_id_confidence": record.subject_id_confidence,
                    "subject_parse_status": record.subject_parse_status,
                    "file_sha256": record.file_sha256,
                }
            )


def write_near_duplicate_candidates(candidates: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id_a",
        "image_id_b",
        "relative_path_a",
        "relative_path_b",
        "raw_class_label_a",
        "raw_class_label_b",
        "binary_label_a",
        "binary_label_b",
        "subject_id_a",
        "subject_id_b",
        "same_subject",
        "same_raw_class",
        "same_binary_label",
        "dhash_hamming_distance",
        "policy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow(row)


def summarize(
    records: list[Record],
    component_id_to_records: dict[str, list[Record]],
    component_id_to_reason: dict[str, str],
    edge_summary: dict,
    near_duplicate_summary: dict,
) -> dict:
    component_sizes = Counter(len(group) for group in component_id_to_records.values())
    reason_counts = Counter(component_id_to_reason.values())
    mixed_raw_label_components = []
    mixed_binary_label_components = []

    components_by_class: dict[str, set[str]] = defaultdict(set)
    for component_id, group in component_id_to_records.items():
        raw_labels = sorted({record.raw_class_label for record in group})
        binary_labels = sorted({record.binary_label for record in group})
        for raw_label in raw_labels:
            components_by_class[raw_label].add(component_id)
        if len(raw_labels) > 1:
            mixed_raw_label_components.append(
                {
                    "component_id": component_id,
                    "size": len(group),
                    "raw_class_labels": raw_labels,
                    "image_ids": sorted(record.image_id for record in group),
                }
            )
        if len(binary_labels) > 1:
            mixed_binary_label_components.append(
                {
                    "component_id": component_id,
                    "size": len(group),
                    "binary_labels": binary_labels,
                    "image_ids": sorted(record.image_id for record in group),
                }
            )

    largest_components = sorted(
        (
            {
                "component_id": component_id,
                "size": len(group),
                "reason": component_id_to_reason[component_id],
                "raw_class_labels": sorted({record.raw_class_label for record in group}),
                "subject_ids": sorted({record.subject_id for record in group})[:5],
            }
            for component_id, group in component_id_to_records.items()
        ),
        key=lambda item: item["size"],
        reverse=True,
    )[:12]

    return {
        "total_images": len(records),
        "total_components": len(component_id_to_records),
        "singleton_components": component_sizes.get(1, 0),
        "multi_image_components": len(component_id_to_records) - component_sizes.get(1, 0),
        "component_size_distribution": dict(sorted(component_sizes.items())),
        "component_reason_counts": dict(reason_counts),
        "components_by_raw_class": {
            key: len(value) for key, value in sorted(components_by_class.items())
        },
        "mixed_raw_label_components": mixed_raw_label_components,
        "mixed_binary_label_components": mixed_binary_label_components,
        "largest_components": largest_components,
        **edge_summary,
        **near_duplicate_summary,
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def write_audit(
    summary: dict,
    components_path: Path,
    near_dupes_path: Path,
    audit_path: Path,
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    component_sizes = markdown_table(
        ["Component Size", "Number of Components"],
        [[size, count] for size, count in summary["component_size_distribution"].items()],
    )
    component_reasons = markdown_table(
        ["Reason", "Components"],
        [[reason, count] for reason, count in summary["component_reason_counts"].items()],
    )
    components_by_class = markdown_table(
        ["Raw Class", "Components"],
        [[label, count] for label, count in summary["components_by_raw_class"].items()],
    )
    largest_components = markdown_table(
        ["Component ID", "Size", "Reason", "Raw Labels", "Subject IDs"],
        [
            [
                row["component_id"],
                row["size"],
                row["reason"],
                ", ".join(row["raw_class_labels"]),
                ", ".join(row["subject_ids"]),
            ]
            for row in summary["largest_components"]
        ],
    )

    mixed_warning = "No mixed raw-label or binary-label blocking components were found."
    if summary["mixed_raw_label_components"] or summary["mixed_binary_label_components"]:
        mixed_warning = (
            "Mixed-label blocking components were found and must be inspected before "
            "any split generation."
        )

    content = f"""# Current JPEG Leakage Graph Audit

## Summary

- Component file: `{components_path.relative_to(PROJECT_ROOT)}`
- Near-duplicate candidate file: `{near_dupes_path.relative_to(PROJECT_ROOT)}`
- Total images: **{summary["total_images"]}**
- Total split-blocking components: **{summary["total_components"]}**
- Singleton components: **{summary["singleton_components"]}**
- Multi-image components: **{summary["multi_image_components"]}**
- Same-subject blocking edges: **{summary["same_subject_edges"]}**
- Exact SHA-256 duplicate edges: **{summary["exact_sha256_edges"]}**
- Exact SHA-256 duplicate groups: **{summary["exact_sha256_duplicate_groups"]}**

## Component Size Distribution

{component_sizes}

## Component Reasons

{component_reasons}

## Components By Raw Class

{components_by_class}

## Largest Components

{largest_components}

## Mixed-Label Component Check

{mixed_warning}

- Mixed raw-label components: **{len(summary["mixed_raw_label_components"])}**
- Mixed binary-label components: **{len(summary["mixed_binary_label_components"])}**

## Perceptual Near-Duplicate Review

- dHash size: **{summary["dhash_hash_size"]}**
- Hamming threshold: **{summary["dhash_threshold"]}**
- Policy: **{summary["near_duplicate_policy"]}**
- Total near-duplicate candidates: **{summary["near_duplicate_candidates_total"]}**
- Candidates written to CSV: **{summary["near_duplicate_candidates_written"]}**
- Same-subject candidates: **{summary["near_duplicate_same_subject_candidates"]}**
- Cross-subject candidates: **{summary["near_duplicate_cross_subject_candidates"]}**

Distance counts:

```json
{json.dumps(summary["near_duplicate_distance_counts"], indent=2)}
```

## QC Decision

**GO for Step 3 if the split generator uses `component_id` as the grouping unit.**

The split-blocking leakage graph currently uses identity and exact duplicate evidence. Perceptual near-duplicates are reported for review, not used as automatic split-blocking edges, because low-distance MRI hashes can represent anatomically similar but unrelated subjects. This policy avoids over-merging the dataset while still making suspicious candidates visible.

## Next Step

Generate the first SplitGuard train/validation/test manifest from these components:

1. Assign entire components, never individual images, to a split.
2. Preserve binary class balance as much as possible.
3. Export a frozen split CSV with seed and split policy.
4. Run an overlap check proving no component crosses splits.
"""
    audit_path.write_text(content, encoding="utf-8")


def write_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--near-dupes", type=Path, default=DEFAULT_NEAR_DUPES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--max-near-dupe-rows", type=int, default=5000)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    uf, edge_summary = build_blocking_components(records)
    image_to_component_id, component_id_to_records, component_id_to_reason = component_maps(records, uf)
    near_duplicate_candidates, near_duplicate_summary = detect_near_duplicates(
        records=records,
        threshold=args.near_threshold,
        max_rows=args.max_near_dupe_rows,
    )

    write_components(
        records=records,
        image_to_component_id=image_to_component_id,
        component_id_to_records=component_id_to_records,
        component_id_to_reason=component_id_to_reason,
        output_path=args.components,
    )
    write_near_duplicate_candidates(near_duplicate_candidates, args.near_dupes)

    summary = summarize(
        records=records,
        component_id_to_records=component_id_to_records,
        component_id_to_reason=component_id_to_reason,
        edge_summary=edge_summary,
        near_duplicate_summary=near_duplicate_summary,
    )
    write_audit(summary, args.components, args.near_dupes, args.audit)
    write_summary_json(summary, args.summary_json)

    print(f"Wrote components: {args.components}")
    print(f"Wrote near-duplicate candidates: {args.near_dupes}")
    print(f"Wrote audit: {args.audit}")
    print(f"Wrote summary JSON: {args.summary_json}")
    print(f"Images: {summary['total_images']}")
    print(f"Components: {summary['total_components']}")
    print(f"Multi-image components: {summary['multi_image_components']}")
    print(f"Near-duplicate candidates: {summary['near_duplicate_candidates_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
