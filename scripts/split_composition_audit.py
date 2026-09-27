#!/usr/bin/env python3
"""Report how component size, class and scan count distribute across partitions.

Why this exists
---------------
The pre-training audit checks that no subject, session or component appears in
more than one partition. That is a disjointness guarantee, and disjointness is
not sufficiency: a split can satisfy it exactly while being badly unbalanced in
ways that change what the test partition measures.

Auditing the frozen ADNI manifests turned up such a case in this project's own
output. ``choose_subset_by_size`` fills the validation and test bins first and
takes the largest components that still fit, so the partitions come out nearly
disjoint in component size -- training components span 4-6 scans, validation
7-9. Because scan count tracks retention and retention tracks disease course
(AD participants here contribute fewer scans than CN participants), the test
partition ends up enriched for longer-followed participants within each class.
Class balance itself survives, because components are stratified by label
before bin-packing, but the compositional imbalance does not show up anywhere
in the audit report.

This script measures that directly so the number is an artefact rather than a
remark, and so ``verify_paper_numbers.py`` can hold the manuscript to it.

Usage
-----
    python3 scripts/split_composition_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits" / "adni"
DEFAULT_OUTPUT = ROOT / "reports" / "tables" / "adni" / "adni_split_composition.json"
PHASES = ("train", "val", "test")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    sizes: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, Counter] = defaultdict(Counter)
    size_by_label: dict[str, list[int]] = defaultdict(list)

    for seed in args.seeds:
        path = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        members: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            members[row["component_id"]].append(row)

        for component_rows in members.values():
            phase = component_rows[0]["split"]
            sizes[phase].append(len(component_rows))
            label = component_rows[0].get("diagnosis_group", "unknown")
            if seed == args.seeds[0] and label in ("CN", "AD"):
                size_by_label[label].append(len(component_rows))
        for row in rows:
            labels[row["split"]][row.get("diagnosis_group", "unknown")] += 1

    summary: dict[str, object] = {
        "seeds": args.seeds,
        "unit": "leakage component (one participant on this cohort)",
        "by_partition": {},
        "component_size_by_diagnosis": {},
    }
    for phase in PHASES:
        s = sizes[phase]
        counts = labels[phase]
        total = sum(counts.values())
        summary["by_partition"][phase] = {
            "n_components": len(s),
            "n_images": sum(s),
            "component_size_mean": round(st.mean(s), 2),
            "component_size_median": round(st.median(s), 1),
            "component_size_min": min(s),
            "component_size_max": max(s),
            "ad_share_by_image": round(counts.get("AD", 0) / total, 3) if total else None,
        }
    for label, s in size_by_label.items():
        summary["component_size_by_diagnosis"][label] = {
            "n_participants": len(s),
            "scans_mean": round(st.mean(s), 2),
            "scans_sd": round(st.stdev(s), 2) if len(s) > 1 else 0.0,
        }

    summary["interpretation"] = (
        "Partitions that are disjoint by construction can still differ "
        "systematically in composition. Here the size-descending bin-packer "
        "routes larger components to validation and test, so the partitions "
        "are nearly disjoint in component size while remaining balanced by "
        "class. Because scan count tracks retention, the test partition is "
        "enriched for longer-followed participants within each class. The "
        "released audit gate does not check this and would pass the split."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for phase in PHASES:
        b = summary["by_partition"][phase]
        print(f"  {phase:>5}: {b['n_components']:>4} components, "
              f"size {b['component_size_mean']:.2f} "
              f"[{b['component_size_min']}-{b['component_size_max']}], "
              f"AD share {b['ad_share_by_image']}")
    for label, v in summary["component_size_by_diagnosis"].items():
        print(f"  {label}: {v['scans_mean']} +/- {v['scans_sd']} scans per participant")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
