#!/usr/bin/env python3
"""Build the first SplitGuard-AD manifest and audit for the current JPEG dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled in runtime output
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "Alzheimer_MRI_4_classes_dataset"
    / "Alzheimer_MRI_4_classes_dataset"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_manifest.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_manifest_audit.md"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_manifest_summary.json"

SOURCE_DATASET = "current_kaggle_oasis_jpeg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NON_DEMENTED_CLASSES = {"NonDemented"}
DEMENTED_CLASSES = {"VeryMildDemented", "MildDemented", "ModerateDemented"}
CLASS_ORDER = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    relative_path: str
    source_dataset: str
    raw_class_label: str
    binary_label: str
    clinical_label: str
    label_confidence: str
    subject_id: str
    subject_id_confidence: str
    subject_parse_status: str
    patient_number: str
    slice_number: str
    session_id: str
    scan_id: str
    derived_from: str
    image_width: str
    image_height: str
    file_size_bytes: int
    file_sha256: str
    preprocessing_version: str


def stable_token(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filename(path: Path, class_label: str) -> dict[str, str]:
    base = path.stem
    base_no_space = base.replace(" ", "")
    parenthesized = re.search(r"\((\d+)\)", base_no_space)
    leading_number = re.match(r"^(\d+)", base_no_space)

    if parenthesized:
        patient_number = parenthesized.group(1)
        subject_id = f"{class_label}_patient{patient_number}"
        confidence = "high_filename_parentheses"
        status = "parsed_parentheses"
    else:
        patient_number = ""
        subject_id = f"{class_label}_unique_{base_no_space}_{stable_token(str(path))}"
        confidence = "low_unparseable_unique"
        status = "unparseable_unique"

    return {
        "subject_id": subject_id,
        "subject_id_confidence": confidence,
        "subject_parse_status": status,
        "patient_number": patient_number,
        "slice_number": leading_number.group(1) if leading_number else "",
    }


def binary_label_for(raw_class_label: str) -> str:
    if raw_class_label in NON_DEMENTED_CLASSES:
        return "NonDemented"
    if raw_class_label in DEMENTED_CLASSES:
        return "Demented"
    return "Unknown"


def image_size(path: Path) -> tuple[str, str]:
    if Image is None:
        return "", ""
    try:
        with Image.open(path) as img:
            return str(img.width), str(img.height)
    except Exception:
        return "", ""


def iter_images(source_dir: Path) -> Iterable[Path]:
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path


def build_records(source_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for image_path in iter_images(source_dir):
        raw_class_label = image_path.parent.name
        rel_path = image_path.relative_to(PROJECT_ROOT)
        parse = parse_filename(image_path, raw_class_label)
        width, height = image_size(image_path)
        sha256 = file_sha256(image_path)
        image_id = f"img_{stable_token(str(rel_path))}"
        binary_label = binary_label_for(raw_class_label)

        records.append(
            ImageRecord(
                image_id=image_id,
                path=str(image_path.resolve()),
                relative_path=str(rel_path),
                source_dataset=SOURCE_DATASET,
                raw_class_label=raw_class_label,
                binary_label=binary_label,
                clinical_label=raw_class_label,
                label_confidence="weak_folder_derived_label",
                subject_id=parse["subject_id"],
                subject_id_confidence=parse["subject_id_confidence"],
                subject_parse_status=parse["subject_parse_status"],
                patient_number=parse["patient_number"],
                slice_number=parse["slice_number"],
                session_id="",
                scan_id="",
                derived_from=str(rel_path),
                image_width=width,
                image_height=height,
                file_size_bytes=image_path.stat().st_size,
                file_sha256=sha256,
                preprocessing_version="raw_jpeg_v0",
            )
        )

    return records


def write_manifest(records: list[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ImageRecord.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def summarize(records: list[ImageRecord]) -> dict:
    by_class = Counter(record.raw_class_label for record in records)
    by_binary_label = Counter(record.binary_label for record in records)
    by_parse_status = Counter(record.subject_parse_status for record in records)
    by_dimensions = Counter(
        f"{record.image_width}x{record.image_height}"
        for record in records
        if record.image_width and record.image_height
    )
    subject_to_records: dict[str, list[ImageRecord]] = defaultdict(list)
    sha_to_records: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        subject_to_records[record.subject_id].append(record)
        sha_to_records[record.file_sha256].append(record)

    subjects_by_class: dict[str, set[str]] = defaultdict(set)
    unparseable_by_class: Counter[str] = Counter()
    for record in records:
        subjects_by_class[record.raw_class_label].add(record.subject_id)
        if record.subject_parse_status == "unparseable_unique":
            unparseable_by_class[record.raw_class_label] += 1

    exact_duplicate_groups = [
        group for group in sha_to_records.values() if len(group) > 1
    ]
    largest_subjects = sorted(
        (
            {
                "subject_id": subject_id,
                "n_images": len(group),
                "class": group[0].raw_class_label,
                "parse_status": group[0].subject_parse_status,
            }
            for subject_id, group in subject_to_records.items()
        ),
        key=lambda item: item["n_images"],
        reverse=True,
    )[:12]

    class_rows = []
    for class_label in CLASS_ORDER:
        if class_label not in by_class:
            continue
        total = by_class[class_label]
        unparseable = unparseable_by_class[class_label]
        class_rows.append(
            {
                "class": class_label,
                "images": total,
                "subjects": len(subjects_by_class[class_label]),
                "unparseable_images": unparseable,
                "unparseable_pct": round(100 * unparseable / max(1, total), 2),
            }
        )

    return {
        "source_dataset": SOURCE_DATASET,
        "total_images": len(records),
        "total_subjects_inferred": len(subject_to_records),
        "class_distribution": dict(by_class),
        "binary_distribution": dict(by_binary_label),
        "parse_status_distribution": dict(by_parse_status),
        "dimension_distribution": dict(by_dimensions),
        "class_rows": class_rows,
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "exact_duplicate_images": sum(len(group) for group in exact_duplicate_groups),
        "largest_subjects": largest_subjects,
        "low_confidence_subject_images": by_parse_status.get("unparseable_unique", 0),
        "high_confidence_subject_images": by_parse_status.get("parsed_parentheses", 0),
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def write_audit(summary: dict, manifest_path: Path, audit_path: Path) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    class_table = markdown_table(
        ["Class", "Images", "Inferred Subjects", "Unparseable Images", "Unparseable %"],
        [
            [
                row["class"],
                row["images"],
                row["subjects"],
                row["unparseable_images"],
                row["unparseable_pct"],
            ]
            for row in summary["class_rows"]
        ],
    )

    largest_subjects = markdown_table(
        ["Subject ID", "Class", "Images", "Parse Status"],
        [
            [
                row["subject_id"],
                row["class"],
                row["n_images"],
                row["parse_status"],
            ]
            for row in summary["largest_subjects"]
        ],
    )

    dimension_rows = sorted(
        summary["dimension_distribution"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    dimensions_table = markdown_table(
        ["Image Size", "Images"],
        [[dimension, count] for dimension, count in dimension_rows],
    )

    go_no_go = (
        "GO for Step 2: build leakage graph v0. "
        "This dataset is usable as a public benchmark stress-test, but labels remain "
        "weak folder-derived labels and low-confidence IDs must be treated cautiously."
    )

    content = f"""# Current JPEG Dataset Manifest Audit

## Summary

- Source dataset: `{summary["source_dataset"]}`
- Manifest: `{manifest_path.relative_to(PROJECT_ROOT)}`
- Total images: **{summary["total_images"]}**
- Total inferred subjects: **{summary["total_subjects_inferred"]}**
- High-confidence subject images: **{summary["high_confidence_subject_images"]}**
- Low-confidence/unparseable subject images: **{summary["low_confidence_subject_images"]}**
- Exact duplicate groups by SHA-256: **{summary["exact_duplicate_groups"]}**
- Exact duplicate images by SHA-256: **{summary["exact_duplicate_images"]}**

## Clinical Claim Level

This dataset should be treated as a **public benchmark stress-test**, not a clinically definitive Alzheimer diagnosis dataset. The labels are folder-derived and should be described as weak labels unless linked back to original clinical metadata.

## Class And Subject Distribution

{class_table}

## Binary Label Distribution

```json
{json.dumps(summary["binary_distribution"], indent=2)}
```

## Subject-ID Parse Status

```json
{json.dumps(summary["parse_status_distribution"], indent=2)}
```

## Image Dimensions

{dimensions_table}

## Largest Inferred Subject Groups

{largest_subjects}

## Leakage-Relevant Findings

- Parenthesized filenames support high-confidence pseudo-subject grouping.
- Non-parenthesized filenames cannot be reliably assigned to a shared subject and are marked `unparseable_unique`.
- Subject-wise splitting is necessary but not sufficient for this dataset because original session, scan, scanner, and site metadata are absent.
- Exact duplicate SHA-256 groups are reported here; perceptual near-duplicate and leakage-component analysis should happen in Step 2.

## QC Decision

**{go_no_go}**

## Next Step

Build the leakage graph v0:

1. Connect all images with the same inferred subject ID.
2. Add exact duplicate edges by SHA-256.
3. Add perceptual near-duplicate edges.
4. Generate leakage components that cannot cross train/validation/test splits.
"""
    audit_path.write_text(content, encoding="utf-8")


def write_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {args.source_dir}")

    records = build_records(args.source_dir)
    if not records:
        raise RuntimeError(f"No images found under {args.source_dir}")

    write_manifest(records, args.manifest)
    summary = summarize(records)
    write_audit(summary, args.manifest, args.audit)
    write_summary_json(summary, args.summary_json)

    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote audit: {args.audit}")
    print(f"Wrote summary JSON: {args.summary_json}")
    print(f"Images: {summary['total_images']}")
    print(f"Inferred subjects: {summary['total_subjects_inferred']}")
    print(f"Low-confidence images: {summary['low_confidence_subject_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
