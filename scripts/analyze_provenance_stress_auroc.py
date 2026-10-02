#!/usr/bin/env python3
"""Merge the stress test's AUROC half into its contamination artefact.

``run_provenance_stress_test.py`` measures what each protocol lets through
when provenance is corrupted. That half needs no model: contamination is a
property of the split. This script supplies the other half, the AUROC the
corrupted split then produces, by reading the per-cell predictions the GPU
programme wrote under ``runs/adni_stress/`` and folding them into the same
JSON.

The pairing matters more than either number alone. The contamination curve
says the graph prevents 90% of the leakage that participant fragmentation
admits; this says whether that prevented leakage was worth any optimism. A
protocol that admits contamination the model cannot exploit would show the
first without the second, and the paper should not claim a benefit it cannot
see in the metric.

Usage
-----
    python3 scripts/analyze_provenance_stress_auroc.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("gp", ROOT / "scripts" / "gpu_postprocess.py")
_gp = importlib.util.module_from_spec(_spec)
sys.modules["gp"] = _gp
_spec.loader.exec_module(_gp)

DEFAULT_RUNS = ROOT / "runs" / "adni_stress"
DEFAULT_JSON = ROOT / "reports" / "tables" / "adni" / "adni_provenance_stress_test.json"


def auroc_of(path: Path) -> float | None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    t = "y_true" if "y_true" in rows[0] else "label"
    s = "y_prob" if "y_prob" in rows[0] else "score"
    return _gp.auroc([(int(float(r[t])), float(r[s])) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--artefact", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    if not args.runs_root.is_dir():
        raise SystemExit(f"no stress runs at {args.runs_root}; run the adni_stress stage first")

    # Cell directories are named <mechanism>_lambda<level>_<protocol>_seed<seed>.
    by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    missing = []
    for run_dir in sorted(args.runs_root.iterdir()):
        pred = run_dir / "baseline_seed0" / "test_predictions.csv"
        if not pred.is_file():
            missing.append(run_dir.name)
            continue
        stem = run_dir.name
        mechanism, rest = stem.split("_lambda", 1)
        level, rest = rest.split("_", 1)
        protocol, _seed = rest.rsplit("_seed", 1)
        value = auroc_of(pred)
        if value is not None:
            by_cell[(mechanism, level, protocol)].append(value)

    payload = json.loads(args.artefact.read_text(encoding="utf-8"))
    summary = payload["summary"]
    for (mechanism, level, protocol), values in sorted(by_cell.items()):
        cell = summary.get(mechanism, {}).get(level)
        if cell is None:
            continue
        cell[protocol]["auroc_mean"] = round(st.mean(values), 4)
        cell[protocol]["auroc_sd"] = round(st.stdev(values), 4) if len(values) > 1 else 0.0
        cell[protocol]["n_seeds_trained"] = len(values)
    for mechanism, levels in summary.items():
        for level, cell in levels.items():
            a = cell.get("subject_only", {}).get("auroc_mean")
            b = cell.get("component_safe", {}).get("auroc_mean")
            cell["auroc_optimism_prevented"] = round(a - b, 4) if a is not None and b is not None else None
    payload["trained"] = True
    payload["auroc_note"] = (
        "Per-cell AUROC is the mean over seeds of the recomputed test AUROC, using the "
        "same rank statistic as every other arm. A positive auroc_optimism_prevented "
        "means the component-safe grouping scored lower, i.e. it declined optimism the "
        "subject-only grouping accepted."
    )
    args.artefact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{'mechanism':<9}{'lambda':>8}{'subj AUROC':>12}{'graph AUROC':>13}{'optimism prevented':>20}")
    for mechanism, levels in summary.items():
        for level, cell in levels.items():
            a = cell.get("subject_only", {}).get("auroc_mean")
            b = cell.get("component_safe", {}).get("auroc_mean")
            if a is None:
                continue
            print(f"{mechanism:<9}{level:>8}{a:>12.4f}{b:>13.4f}"
                  f"{cell['auroc_optimism_prevented']:>+20.4f}")
    if missing:
        print(f"\n{len(missing)} cells had no predictions yet (e.g. {missing[:2]})")
    print(f"\nUpdated {args.artefact.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
