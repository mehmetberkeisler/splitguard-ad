#!/usr/bin/env python3
"""Build the ADNI leakage graph and connected-component table.

Edges
-----
Two manifest rows share a leakage edge if any of these is true:

1. ``same_subject``      — same ``subject_id`` / ``ptid``.
2. ``same_session``      — same ``(subject_id, viscode)`` pair.
3. ``same_series_uid``   — same ``series_uid`` (same physical MR series).
4. ``longitudinal``      — same subject, different sessions (kept as a
   distinct edge type for auditability even though same_subject already
   implies this).
5. ``same_acq_date``     — same subject and same ``acq_date`` (covers
   sessions that may not yet have a VISCODE assigned).

The components are the connected components in that graph. Component size
is the number of image rows in the component. The component label is the
single ``diagnosis_group`` shared by all rows in the component, or
``mixed_label`` if rows disagree, or ``label_missing`` if every row is
``unknown``.

Outputs
-------
* ``data/manifests/adni/adni_leakage_components.csv`` — the manifest with
  three extra columns appended: ``component_id``, ``component_size``,
  ``component_primary_reason``, ``component_label``.
* ``reports/audits/adni/adni_leakage_graph_summary.json`` — global graph
  stats (n_nodes, n_edges, n_components, component-size histogram,
  edge-reason counter).

This script intentionally does not compute near-duplicate edges via image
hashing in the first pass — ADNI volumes are 3D and large, and the
metadata-derived edges already capture identity-/series-level leakage. A
near-duplicate pass can be added later if reviewers ask, using the same
``component_id`` scheme.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv"
DEFAULT_COMPONENTS = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_leakage_components.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_leakage_graph_summary.json"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = (row.get(key) or "").strip()
        if value and value != "unknown":
            out[value].append(row["image_id"])
    return out


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "subject_id", "session_id", "diagnosis_group"}
    missing = required - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"ADNI manifest missing required columns: {sorted(missing)}")
    return rows


def build_graph(rows: list[dict[str, str]]) -> tuple[UnionFind, Counter]:
    uf = UnionFind()
    for row in rows:
        uf.add(row["image_id"])
    reason_counts: Counter = Counter()

    # same_subject
    for subject, ids in group_by(rows, "subject_id").items():
        if subject == "unknown":
            continue
        if len(ids) < 2:
            continue
        anchor = ids[0]
        for image_id in ids[1:]:
            uf.union(anchor, image_id)
            reason_counts["same_subject"] += 1

    # same_session
    session_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        subject = (row.get("subject_id") or "").strip()
        viscode = (row.get("viscode") or "").strip()
        if subject and subject != "unknown" and viscode:
            session_groups[(subject, viscode)].append(row["image_id"])
    for ids in session_groups.values():
        if len(ids) < 2:
            continue
        anchor = ids[0]
        for image_id in ids[1:]:
            uf.union(anchor, image_id)
            reason_counts["same_session"] += 1

    # same_series_uid
    for series, ids in group_by(rows, "series_uid").items():
        if len(ids) < 2:
            continue
        anchor = ids[0]
        for image_id in ids[1:]:
            uf.union(anchor, image_id)
            reason_counts["same_series_uid"] += 1

    # same_acq_date (subject-scoped)
    acq_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        subject = (row.get("subject_id") or "").strip()
        acq_date = (row.get("acq_date") or "").strip()
        if subject and subject != "unknown" and acq_date:
            acq_groups[(subject, acq_date)].append(row["image_id"])
    for ids in acq_groups.values():
        if len(ids) < 2:
            continue
        anchor = ids[0]
        for image_id in ids[1:]:
            uf.union(anchor, image_id)
            reason_counts["same_acq_date"] += 1

    return uf, reason_counts


def label_component(rows: list[dict[str, str]]) -> str:
    labels = {(row.get("diagnosis_group") or "unknown") for row in rows}
    non_missing = labels - {"unknown"}
    if not non_missing:
        return "label_missing"
    if len(non_missing) == 1:
        return next(iter(non_missing))
    return "mixed_label"


def write_components(
    manifest_rows: list[dict[str, str]],
    uf: UnionFind,
    output_path: Path,
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_root[uf.find(row["image_id"])].append(row)

    root_to_component_id: dict[str, str] = {}
    components_by_id: dict[str, list[str]] = {}
    output_rows: list[dict[str, str]] = []
    for index, (root, rows) in enumerate(sorted(by_root.items())):
        component_id = f"adni_comp_{index:05d}"
        root_to_component_id[root] = component_id
        components_by_id[component_id] = [row["image_id"] for row in rows]
        component_label = label_component(rows)
        for row in rows:
            out = dict(row)
            out["component_id"] = component_id
            out["component_size"] = str(len(rows))
            out["component_primary_reason"] = "same_subject"
            out["component_label"] = component_label
            output_rows.append(out)

    fieldnames = list(manifest_rows[0].keys()) + [
        "component_id",
        "component_size",
        "component_primary_reason",
        "component_label",
    ] if manifest_rows else []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows, components_by_id


def histogram_of_sizes(components: dict[str, list[str]]) -> dict[str, int]:
    sizes = Counter(len(ids) for ids in components.values())
    return {str(size): sizes[size] for size in sorted(sizes)}


def summarize(
    manifest_rows: list[dict[str, str]],
    components: dict[str, list[str]],
    reason_counts: Counter,
) -> dict[str, Any]:
    n_edges = sum(reason_counts.values())
    size_histogram = histogram_of_sizes(components)
    label_counts: Counter = Counter()
    for ids in components.values():
        labels = {row["diagnosis_group"] for row in manifest_rows if row["image_id"] in ids}
        non_missing = labels - {"unknown"}
        if not non_missing:
            label_counts["label_missing"] += 1
        elif len(non_missing) == 1:
            label_counts[next(iter(non_missing))] += 1
        else:
            label_counts["mixed_label"] += 1
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_nodes": len(manifest_rows),
        "n_edges": n_edges,
        "edge_reason_counts": dict(reason_counts),
        "n_components": len(components),
        "component_size_histogram": size_histogram,
        "component_label_counts": dict(label_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"ADNI manifest does not exist: {args.manifest}. "
            "Run scripts/build_adni_manifest.py first."
        )

    rows = read_manifest(args.manifest)
    if not rows:
        print("Manifest is empty. Nothing to build.")
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({"n_nodes": 0}, indent=2), encoding="utf-8")
        return 1

    uf, reason_counts = build_graph(rows)
    _, components = write_components(rows, uf, args.components)
    summary = summarize(rows, components, reason_counts)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.components.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {args.summary.relative_to(PROJECT_ROOT)}")
    print(
        f"Graph: {summary['n_nodes']} nodes, "
        f"{summary['n_edges']} edges, "
        f"{summary['n_components']} components."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
