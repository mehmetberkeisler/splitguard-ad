#!/usr/bin/env python3
"""Multi-operating-point sensitivity table for the ADNI cost-of-leakage arm.

Reviewer §5.6 asked for the cost-of-leakage translation to be reported at
several operating points instead of a single fixed-specificity anchor, so
that the choice of anchor does not look cherry-picked. This script consumes
the same persisted per-image test predictions used by
:mod:`clinical_cost_of_leakage_adni` and, for each protocol × seed, reports:

    - Sensitivity at fixed specificity in {0.80, 0.85, 0.90, 0.95}
    - Specificity at fixed sensitivity in {0.85, 0.90, 0.95}
    - Youden's J optimum with matched (sensitivity, specificity)

The output is a compact JSON that the paper's Table 14 (report card) can
either reference in text or inline as a supplementary panel.

Reads
-----
runs/adni_with_converters/inflation_gap_seed<S>/<PROTO>/test_predictions.csv

Writes
------
reports/tables/adni/adni_operating_point_sensitivity.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT    = PROJECT_ROOT / "runs" / "adni_with_converters"
OUT_PATH     = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_operating_point_sensitivity.json"


# ── Shared helpers (kept local for zero-runtime-dep reproducibility) ─────
def roc_points(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), reverse=True)
    P = sum(1 for t in y_true if t == 1)
    N = sum(1 for t in y_true if t == 0)
    if P == 0 or N == 0:
        return [(1.0, 0.0, 0.0, float("-inf"))]
    tp = fp = 0
    pts = [(0.0, 0.0, 1.0, float("inf"))]  # (fpr, tpr, threshold, score)
    prev_score = None
    for s, t in pairs:
        if prev_score is not None and s != prev_score:
            pts.append((fp / N, tp / P, prev_score, prev_score))
        if t == 1: tp += 1
        else:      fp += 1
        prev_score = s
    pts.append((fp / N, tp / P, prev_score if prev_score is not None else 0.0,
                prev_score if prev_score is not None else 0.0))
    return pts


def auroc_from_pts(pts):
    a = 0.0
    for (fpr0, tpr0, _, _), (fpr1, tpr1, _, _) in zip(pts, pts[1:]):
        a += (fpr1 - fpr0) * (tpr0 + tpr1) * 0.5
    return a


# ── Threshold SELECTION (validation only) ───────────────────────────────
#
# Every function below returns a *threshold*, chosen on the validation ROC.
# None of them may be given test predictions. Scanning the test ROC for the
# threshold that maximises sensitivity at a target specificity — which this
# module previously did — reports the best value achievable in hindsight
# rather than one obtainable prospectively, and is precisely the class of
# optimistic evaluation this manuscript is about. Selection and reporting are
# therefore split: choose on validation, measure on test.

def threshold_at_fixed_spec(val_pts, target_spec):
    """Threshold achieving the highest validation sensitivity at spec >= target."""
    best_tpr, best_thr = -1.0, float("inf")
    for fpr, tpr, thr, _ in val_pts:
        if (1.0 - fpr) >= target_spec and tpr > best_tpr:
            best_tpr, best_thr = tpr, thr
    return best_thr


def threshold_at_fixed_sens(val_pts, target_sens):
    """Threshold achieving the highest validation specificity at sens >= target."""
    best_spec, best_thr = -1.0, float("-inf")
    for fpr, tpr, thr, _ in val_pts:
        spec = 1.0 - fpr
        if tpr >= target_sens and spec > best_spec:
            best_spec, best_thr = spec, thr
    return best_thr


def threshold_at_youden(val_pts):
    """Youden-J-optimal threshold on validation."""
    best_j, best_thr = -1.0, float("inf")
    for fpr, tpr, thr, _ in val_pts:
        j = tpr + (1.0 - fpr) - 1.0
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr


# ── Measurement (test only, at a threshold fixed beforehand) ────────────

def sens_spec_at_threshold(y_true, y_prob, threshold):
    """(sensitivity, specificity) on the given data at a FIXED threshold."""
    tp = fn = fp = tn = 0
    for t, s in zip(y_true, y_prob):
        pred = 1 if s >= threshold else 0
        if t == 1 and pred == 1: tp += 1
        elif t == 1:             fn += 1
        elif pred == 1:          fp += 1
        else:                    tn += 1
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x == x]
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ── Main ─────────────────────────────────────────────────────────────────
FIXED_SPEC_TARGETS  = [0.80, 0.85, 0.90, 0.95]
FIXED_SENS_TARGETS  = [0.85, 0.90, 0.95]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--protocols", nargs="+",
                   default=["random", "subject_only", "component_safe"])
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    proto_results = {}
    for proto in args.protocols:
        per_seed = []
        for seed in args.seeds:
            run_dir = args.runs_root / f"inflation_gap_seed{seed}" / proto
            test_path = run_dir / "test_predictions.csv"
            val_path = run_dir / "val_predictions.csv"
            if not test_path.exists():
                print(f"  warn: missing test predictions: {test_path}", file=sys.stderr)
                continue
            if not val_path.exists():
                print(
                    f"  ERROR: missing {val_path}.\n"
                    "  Thresholds must be selected on validation, not test. Run\n"
                    "  scripts/regenerate_val_predictions.py to backfill this artefact\n"
                    "  from the run's saved checkpoint (inference only, no retraining).",
                    file=sys.stderr,
                )
                continue

            test_rows = list(csv.DictReader(test_path.open()))
            y_true = [int(r["y_true"]) for r in test_rows]
            y_prob = [float(r["y_prob"]) for r in test_rows]
            pts = roc_points(y_true, y_prob)

            val_rows = list(csv.DictReader(val_path.open()))
            v_true = [int(r["y_true"]) for r in val_rows]
            v_prob = [float(r["y_prob"]) for r in val_rows]
            val_pts = roc_points(v_true, v_prob)

            entry = {
                "seed": seed,
                "n_val": len(val_rows),
                "n_test": len(test_rows),
                "auroc": round(auroc_from_pts(pts), 4),
                "threshold_selection": "validation",
            }
            # Fixed-specificity anchors: threshold chosen on val, measured on test.
            for tsp in FIXED_SPEC_TARGETS:
                thr = threshold_at_fixed_spec(val_pts, tsp)
                sens, spec = sens_spec_at_threshold(y_true, y_prob, thr)
                entry[f"sens_at_spec_{tsp:.2f}"] = round(sens, 4)
                entry[f"realised_spec_at_spec_{tsp:.2f}"] = round(spec, 4)
                entry[f"threshold_at_spec_{tsp:.2f}"] = round(float(thr), 6)
            # Fixed-sensitivity anchors, same discipline.
            for tsn in FIXED_SENS_TARGETS:
                thr = threshold_at_fixed_sens(val_pts, tsn)
                sens, spec = sens_spec_at_threshold(y_true, y_prob, thr)
                entry[f"spec_at_sens_{tsn:.2f}"] = round(spec, 4)
                entry[f"realised_sens_at_sens_{tsn:.2f}"] = round(sens, 4)
            # Youden-J optimum, also selected on validation.
            thr_j = threshold_at_youden(val_pts)
            j_sens, j_spec = sens_spec_at_threshold(y_true, y_prob, thr_j)
            entry["youden_sens"] = round(j_sens, 4)
            entry["youden_spec"] = round(j_spec, 4)
            entry["youden_j"] = round(j_sens + j_spec - 1.0, 4)
            entry["youden_threshold"] = round(float(thr_j), 6)
            per_seed.append(entry)

        if not per_seed:
            continue

        proto_results[proto] = {"per_seed": per_seed}
        for tsp in FIXED_SPEC_TARGETS:
            k = f"sens_at_spec_{tsp:.2f}"
            proto_results[proto][k + "_mean"] = round(mean([e[k] for e in per_seed]), 4)
            proto_results[proto][k + "_sd"]   = round(sd  ([e[k] for e in per_seed]), 4)
        for tsn in FIXED_SENS_TARGETS:
            k = f"spec_at_sens_{tsn:.2f}"
            proto_results[proto][k + "_mean"] = round(mean([e[k] for e in per_seed]), 4)
            proto_results[proto][k + "_sd"]   = round(sd  ([e[k] for e in per_seed]), 4)
        proto_results[proto]["mean_youden_j"]    = round(mean([e["youden_j"]    for e in per_seed]), 4)
        proto_results[proto]["mean_youden_sens"] = round(mean([e["youden_sens"] for e in per_seed]), 4)
        proto_results[proto]["mean_youden_spec"] = round(mean([e["youden_spec"] for e in per_seed]), 4)

    # ── Deltas between leaky and honest protocols at every operating pt ──
    if "random" in proto_results and "component_safe" in proto_results:
        deltas = {}
        for tsp in FIXED_SPEC_TARGETS:
            k = f"sens_at_spec_{tsp:.2f}"
            deltas[k] = round(
                proto_results["random"][k + "_mean"]
                - proto_results["component_safe"][k + "_mean"], 4
            )
        for tsn in FIXED_SENS_TARGETS:
            k = f"spec_at_sens_{tsn:.2f}"
            deltas[k] = round(
                proto_results["random"][k + "_mean"]
                - proto_results["component_safe"][k + "_mean"], 4
            )
        deltas["youden_sens"] = round(
            proto_results["random"]["mean_youden_sens"]
            - proto_results["component_safe"]["mean_youden_sens"], 4
        )
        deltas["youden_spec"] = round(
            proto_results["random"]["mean_youden_spec"]
            - proto_results["component_safe"]["mean_youden_spec"], 4
        )
        deltas["youden_j"] = round(
            proto_results["random"]["mean_youden_j"]
            - proto_results["component_safe"]["mean_youden_j"], 4
        )
        proto_results["_leaky_minus_honest"] = deltas

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proto_results, indent=2))
    print(f"wrote {args.output}")

    # ── Human-readable summary ──────────────────────────────────────────
    for proto in ["random", "subject_only", "component_safe"]:
        if proto not in proto_results: continue
        pr = proto_results[proto]
        print(f"\n[{proto}]")
        print(f"  AUROC (mean across seeds) = "
              f"{mean([e['auroc'] for e in pr['per_seed']]):.4f}")
        for tsp in FIXED_SPEC_TARGETS:
            k = f"sens_at_spec_{tsp:.2f}"
            print(f"  sens @ spec {tsp:.2f}: "
                  f"{pr[k + '_mean']:.3f} ± {pr[k + '_sd']:.3f}")
        for tsn in FIXED_SENS_TARGETS:
            k = f"spec_at_sens_{tsn:.2f}"
            print(f"  spec @ sens {tsn:.2f}: "
                  f"{pr[k + '_mean']:.3f} ± {pr[k + '_sd']:.3f}")
        print(f"  Youden J = {pr['mean_youden_j']:.3f} (sens "
              f"{pr['mean_youden_sens']:.3f}, spec "
              f"{pr['mean_youden_spec']:.3f})")

    if "_leaky_minus_honest" in proto_results:
        print("\n[leaky − honest deltas]")
        for k, v in proto_results["_leaky_minus_honest"].items():
            sign = "+" if v > 0 else ""
            print(f"  {k}: {sign}{v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
