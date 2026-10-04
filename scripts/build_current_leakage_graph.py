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



PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splitguard_ad.graph import (  # noqa: E402
    MISSING_TOKENS, Record, UnionFind, component_maps, dhash, hamming,
    normalise_identifier, stable_token)
from _paths import display_path, resolve_data_path  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_manifest.csv"
DEFAULT_COMPONENTS = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_leakage_components.csv"
DEFAULT_NEAR_DUPES = PROJECT_ROOT / "data" / "manifests" / "current_jpeg_near_duplicate_candidates.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_leakage_graph_audit.md"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "reports" / "audits" / "current_jpeg_leakage_graph_summary.json"






# Values that mean "this field is absent", not "this is the identifier".






def read_manifest(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Only the row key is required, which is what the README promises for a
    # bring-your-own-cohort manifest. Every other column is optional and
    # defaults to empty, and an absent column simply switches its edge family
    # off: no subject_id means no same_subject edges, no file_sha256 means no
    # exact-duplicate edges. Demanding ten columns while advertising one was a
    # contract the implementation did not honour, and a user whose manifest
    # carries a participant key under a different name should get a clear
    # error about image_id, not a list of columns the README never mentioned.
    missing = {"image_id"}.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(
            "Manifest is missing the required column 'image_id'. Every other "
            "column is optional; see the column table in the README for which "
            "edge family each one enables.")

    return [
        Record(
            image_id=row["image_id"],
            path=resolve_data_path(row) if row.get("relative_path") or row.get("path") else "",
            relative_path=row.get("relative_path", ""),
            raw_class_label=row.get("raw_class_label", ""),
            binary_label=row.get("binary_label", ""),
            subject_id=row.get("subject_id", ""),
            subject_id_confidence=row.get("subject_id_confidence", ""),
            subject_parse_status=row.get("subject_parse_status", ""),
            file_sha256=row.get("file_sha256", ""),
        )
        for row in rows
    ]






def build_blocking_components(
    records: list[Record],
    near_duplicate_pairs: list[tuple[str, str]] | None = None,
) -> tuple[UnionFind, dict]:
    """Union images that must not be separated, and report which rule joined them.

    Two edge families are always blocking: ``same_subject`` and
    ``exact_sha256_duplicate``.

    Near-duplicate (perceptual-hash) pairs are blocking only when
    ``near_duplicate_pairs`` is supplied, i.e. under
    ``--near-duplicate-policy blocking``. The default remains ``review``, which
    detects and reports the pairs without merging their components, and that is
    the policy under which the released manifests and every published number
    were produced. The distinction is load-bearing and we make it switchable
    rather than implicit: promoting these pairs to blocking edges changes the
    connected components, hence the frozen splits, hence every downstream
    result, so it cannot be turned on silently.
    """
    uf = UnionFind([record.image_id for record in records])

    subject_groups: dict[str, list[Record]] = defaultdict(list)
    sha_groups: dict[str, list[Record]] = defaultdict(list)

    # Group only on identifiers that are actually present. Two records whose
    # subject_id is "" are not two scans of one patient; they are two scans
    # whose patient is unknown, and joining them invents a participant. The
    # same applies to a missing hash. This changes no published number -- the
    # Tier-1 manifest has no missing identifiers -- but the README invites
    # other cohorts, and on one with partial metadata the unguarded version
    # would merge every unlabelled scan into a single component and then
    # report, correctly and uselessly, that the split is safe.
    for record in records:
        subject = normalise_identifier(record.subject_id)
        if subject is not None:
            subject_groups[subject].append(record)
        digest = normalise_identifier(record.file_sha256)
        if digest is not None:
            sha_groups[digest].append(record)

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

    near_dup_edges = 0
    for left_id, right_id in (near_duplicate_pairs or []):
        uf.union(left_id, right_id, "near_duplicate")
        near_dup_edges += 1

    return uf, {
        "same_subject_edges": subject_edges,
        "exact_sha256_edges": sha_edges,
        "exact_sha256_duplicate_groups": duplicate_sha_groups,
        "near_duplicate_blocking_edges": near_dup_edges,
        "near_duplicate_policy": "blocking" if near_duplicate_pairs else "review",
    }




def detect_near_duplicates(
    records: list[Record],
    threshold: int,
    max_rows: int,
    policy: str = "review",
) -> tuple[list[dict], dict]:
    # Recorded verbatim in both the per-pair rows and the summary, so an
    # artefact always states whether these pairs merged components or were
    # only surfaced for inspection.
    policy_label = (
        "blocking_split_edge" if policy == "blocking"
        else "review_only_not_split_blocking"
    )
    hashes = []
    failed = []
    for record in records:
        try:
            hashes.append((record, dhash(Path(record.path))))
        except (OSError, ValueError) as exc:  # pragma: no cover - data dependent
            # Unreadable or undecodable image. Recorded and reported in the
            # audit rather than suppressed, so the count of images that got no
            # perceptual hash is visible. Narrowed from Exception so that a
            # defect in dhash itself propagates instead of being logged as a
            # data problem.
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
                        "policy": policy_label,
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
        "near_duplicate_policy": policy_label,
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


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise.

    ``Path.relative_to`` raises for any output directory outside the repo, so
    using it unguarded made the script crash when a caller passed a custom
    ``--components``/``--audit`` path — which is exactly what someone
    reproducing the pipeline into a scratch directory would do.
    """
    try:
        return display_path(path)
    except ValueError:
        return str(path)


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

- Component file: `{_display_path(components_path)}`
- Near-duplicate candidate file: `{_display_path(near_dupes_path)}`
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
    parser.add_argument(
        "--near-duplicate-policy", choices=["review", "blocking"], default="review",
        help=(
            "review (default): detect near-duplicate pairs and report them, "
            "without merging their components — this is the policy under which "
            "the released manifests and all published results were produced. "
            "blocking: additionally union each detected pair, which changes the "
            "connected components and therefore the frozen splits, so any "
            "downstream result must be recomputed."
        ),
    )
    args = parser.parse_args()

    records = read_manifest(args.manifest)

    # Detection runs first so its output can optionally feed the graph.
    near_duplicate_candidates, near_duplicate_summary = detect_near_duplicates(
        records=records,
        threshold=args.near_threshold,
        max_rows=args.max_near_dupe_rows,
        policy=args.near_duplicate_policy,
    )

    blocking_pairs = None
    if args.near_duplicate_policy == "blocking":
        blocking_pairs = [
            (row["image_id_a"], row["image_id_b"])
            for row in near_duplicate_candidates
            if "image_id_a" in row and "image_id_b" in row
        ]
        print(
            f"near-duplicate policy=blocking: unioning {len(blocking_pairs)} pairs "
            "— components and splits will differ from the released manifests."
        )

    uf, edge_summary = build_blocking_components(records, blocking_pairs)
    image_to_component_id, component_id_to_records, component_id_to_reason = component_maps(records, uf)

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
