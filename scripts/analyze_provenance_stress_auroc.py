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
import itertools
import json
import math
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


# t_{0.975} with four degrees of freedom: five paired seeds, no more.
T95_DF4 = 2.776


def paired_stats(diffs: list[float]) -> dict:
    """Mean, SD and paired-t interval of the per-seed differences.

    Averaging each protocol separately and subtracting the means throws away
    the pairing: seed s trains both protocols on the *same* corrupted
    manifest, so the seed-to-seed variance is common to both arms and cancels
    in the difference. The unpaired SDs here reach 0.066, larger than several
    of the effects, which would make every cell look hopeless; the paired SDs
    are what the comparison actually rests on. Five seeds still buy a wide
    interval, and the honest thing is to print it rather than the mean alone.
    """
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    half = T95_DF4 * sd / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return {"mean": round(mean, 4), "sd": round(sd, 4),
            "ci95_lo": round(mean - half, 4), "ci95_hi": round(mean + half, 4),
            "n_positive": sum(1 for d in diffs if d > 0), "n_seeds": len(diffs)}


def sign_test_p(n_positive: int, n_nonzero: int) -> float:
    """Exact one-sided binomial tail under a fair coin."""
    tail = sum(math.comb(n_nonzero, k) for k in range(n_positive, n_nonzero + 1))
    return tail / 2 ** n_nonzero


def wilcoxon_p(values: list[float]) -> tuple[int, float]:
    """Exact one-sided signed-rank test, enumerated over sign assignments.

    Eight cells, so 2^8 assignments: exact is cheaper than approximating.
    """
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    rank = {i: r + 1 for r, i in enumerate(order)}
    w_plus = sum(rank[i] for i in range(len(values)) if values[i] > 0)
    ge = sum(1 for signs in itertools.product((0, 1), repeat=len(values))
             if sum(rank[i] for i in range(len(values)) if signs[i]) >= w_plus)
    return w_plus, ge / 2 ** len(values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--artefact", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    if not args.runs_root.is_dir():
        raise SystemExit(f"no stress runs at {args.runs_root}; run the adni_stress stage first")

    # Cell directories are named <mechanism>_lambda<level>_<protocol>_seed<seed>.
    # Keyed by seed, not appended: the per-cell comparison is paired, and a
    # list loses which protocol run shares a corrupted manifest with which.
    by_cell: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    missing = []
    for run_dir in sorted(args.runs_root.iterdir()):
        pred = run_dir / "baseline_seed0" / "test_predictions.csv"
        if not pred.is_file():
            missing.append(run_dir.name)
            continue
        stem = run_dir.name
        mechanism, rest = stem.split("_lambda", 1)
        level, rest = rest.split("_", 1)
        protocol, seed = rest.rsplit("_seed", 1)
        value = auroc_of(pred)
        if value is not None:
            by_cell[(mechanism, level, protocol)][int(seed)] = value

    payload = json.loads(args.artefact.read_text(encoding="utf-8"))
    summary = payload["summary"]
    for (mechanism, level, protocol), by_seed in sorted(by_cell.items()):
        cell = summary.get(mechanism, {}).get(level)
        if cell is None:
            continue
        values = [by_seed[s] for s in sorted(by_seed)]
        cell[protocol]["auroc_mean"] = round(st.mean(values), 4)
        cell[protocol]["auroc_sd"] = round(st.stdev(values), 4) if len(values) > 1 else 0.0
        cell[protocol]["n_seeds_trained"] = len(values)
    informative: list[tuple[str, str, float]] = []   # mechanism, level, paired mean
    for mechanism, levels in summary.items():
        for level, cell in levels.items():
            a = cell.get("subject_only", {}).get("auroc_mean")
            b = cell.get("component_safe", {}).get("auroc_mean")
            cell["auroc_optimism_prevented"] = round(a - b, 4) if a is not None and b is not None else None
            subj = by_cell.get((mechanism, level, "subject_only"), {})
            graph = by_cell.get((mechanism, level, "component_safe"), {})
            shared = sorted(set(subj) & set(graph))
            if not shared:
                continue
            diffs = [subj[s] - graph[s] for s in shared]
            cell["auroc_paired"] = paired_stats(diffs)
            # merge is an identity: both protocols produce the same partition,
            # so its difference is exactly zero and carries no information
            # about the graph. A sign test discards ties in any case; saying so
            # is better than letting four zeros silently set the denominator.
            if mechanism != "merge":
                informative.append((mechanism, level, st.mean(diffs)))
    if informative:
        means = [m for _, _, m in informative]
        n_pos = sum(1 for m in means if m > 0)
        w_plus, w_p = wilcoxon_p(means)
        payload["auroc_aggregate"] = {
            "cells": len(means),
            "cells_positive": n_pos,
            "cell_labels": [f"{mech} {lvl}" for mech, lvl, _ in informative],
            "mean_of_cell_means": round(st.mean(means), 4),
            "sd_of_cell_means": round(st.stdev(means), 4) if len(means) > 1 else 0.0,
            "sign_test_p_one_sided": round(sign_test_p(n_pos, len(means)), 4),
            "wilcoxon_w_plus": w_plus,
            "wilcoxon_p_one_sided": round(w_p, 4),
            "cells_ci_excluding_zero": [
                f"{mech} {lvl}" for mech, lvl, _ in informative
                if (summary[mech][lvl].get("auroc_paired") or {}).get("ci95_lo", -1) > 0],
            "note": (
                "Cells share seeds and a base manifest, so these aggregates are "
                "descriptive of a consistent direction, not an independent test. "
                "merge is excluded: its difference is zero by construction."
            ),
        }
    payload["trained"] = True
    payload["auroc_note"] = (
        "Per-cell AUROC is the mean over seeds of the recomputed test AUROC, using the "
        "same rank statistic as every other arm. A positive auroc_optimism_prevented "
        "means the component-safe grouping scored lower, i.e. it declined optimism the "
        "subject-only grouping accepted."
    )
    args.artefact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{'mechanism':<9}{'lambda':>8}{'subj AUROC':>12}{'graph AUROC':>13}"
          f"{'paired diff':>13}{'95% CI':>22}{'pos':>6}")
    for mechanism, levels in summary.items():
        for level, cell in levels.items():
            a = cell.get("subject_only", {}).get("auroc_mean")
            b = cell.get("component_safe", {}).get("auroc_mean")
            if a is None:
                continue
            pr = cell.get("auroc_paired") or {}
            ci = (f"[{pr['ci95_lo']:+.3f}, {pr['ci95_hi']:+.3f}]" if pr else "")
            pos = f"{pr.get('n_positive', 0)}/{pr.get('n_seeds', 0)}" if pr else ""
            print(f"{mechanism:<9}{level:>8}{a:>12.4f}{b:>13.4f}"
                  f"{cell['auroc_optimism_prevented']:>+13.4f}{ci:>22}{pos:>6}")
    agg = payload.get("auroc_aggregate")
    if agg:
        print(f"\nAcross the {agg['cells']} cells where the partitions differ: "
              f"{agg['cells_positive']} favour the component-safe grouping, "
              f"mean {agg['mean_of_cell_means']:+.4f}")
        print(f"  sign test p = {agg['sign_test_p_one_sided']:.4f}, "
              f"Wilcoxon W+ = {agg['wilcoxon_w_plus']}, "
              f"p = {agg['wilcoxon_p_one_sided']:.4f} (one-sided, descriptive)")
        print(f"  cells whose own interval excludes zero: "
              f"{agg['cells_ci_excluding_zero'] or 'none'}")
    if missing:
        print(f"\n{len(missing)} cells had no predictions yet (e.g. {missing[:2]})")
    print(f"\nUpdated {args.artefact.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
