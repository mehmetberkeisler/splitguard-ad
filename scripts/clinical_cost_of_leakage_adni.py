#!/usr/bin/env python3
"""Translate ADNI inflation-gap AUROC into clinical screening operating points.

The inflation-gap experiment reports a ~0.13 AUROC drop from random-split
(leaky) to component-safe (honest) evaluation. Without a clinical framing,
that number is hard for a reviewer to weigh. This script translates the
honest-vs-leaky gap into the concrete metric a clinical decision-maker
cares about: how many true Alzheimer's cases per 1000 screened patients
would the model miss at a clinically meaningful operating point?

For each (seed, protocol) we compute:
  * Sensitivity at a fixed-specificity screening anchor (default 0.90)
  * Youden's J optimum (sensitivity + specificity - 1, supplementary)
  * Full ROC curve (also written to JSON for the supplementary figure)

Then for two pre-specified prevalence anchors --- a population-screening
prevalence and a memory-clinic referral prevalence, both with literature
citations supplied at the call site --- we translate the protocol's mean
sens@spec=0.9 into expected counts per 1000 screened patients:
  TP, FN ("missed diagnoses"), TN, FP.

The headline cost-of-leakage number is the *additional* false negatives a
clinician would experience if they trusted the leaky benchmark's apparent
sensitivity, vs. the honest deployment's actual sensitivity, at the same
operating point. That quantity is reported as
``additional_missed_if_trusting_leaky`` per 1000 patients.

Reads
-----
runs/<runs-root>/inflation_gap_seed{S}/{protocol}/test_predictions.csv
  columns: image_id, subject_id, diagnosis_group, y_true, y_prob

Writes
------
reports/tables/adni/adni_cost_of_leakage.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── ROC primitives, no sklearn dep ────────────────────────────────────────
def roc_points(y_true: list[int], y_prob: list[float]):
    """Sort by descending probability and walk through every threshold,
    yielding (threshold, sensitivity, specificity) at each cut."""
    P = sum(y_true)
    N = len(y_true) - P
    pairs = sorted(zip(y_prob, y_true), reverse=True)
    tp = fp = 0
    pts = [(float("inf"), 0.0, 1.0)]  # threshold above max prob: sens=0, spec=1
    for prob, label in pairs:
        if label == 1: tp += 1
        else:          fp += 1
        sens = tp / P if P > 0 else 0.0
        spec = 1.0 - fp / N if N > 0 else 0.0
        pts.append((prob, sens, spec))
    return pts


def sens_at_fixed_spec(pts, target_spec: float) -> float:
    """Highest sensitivity achievable while keeping specificity >= target_spec.
    Standard screening operating-point definition."""
    best = 0.0
    for _, sens, spec in pts:
        if spec >= target_spec and sens > best:
            best = sens
    return best


def youden_optimum(pts):
    """Threshold maximising Youden's J = sens + spec - 1."""
    best = (-1.0, 0.0, 0.0, None)
    for thr, sens, spec in pts:
        j = sens + spec - 1
        if j > best[0]:
            best = (j, sens, spec, thr)
    return best  # (J, sens, spec, threshold)


def auroc_from_pts(pts) -> float:
    """Trapezoidal AUROC over (FPR, TPR)."""
    xy = sorted({(1 - s, sn) for _, sn, s in pts})
    area = 0.0
    for i in range(1, len(xy)):
        x0, y0 = xy[i-1]; x1, y1 = xy[i]
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    m = mean(xs)
    if len(xs) < 2: return 0.0
    return (sum((x - m)**2 for x in xs) / (len(xs) - 1)) ** 0.5


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path,
                   default=PROJECT_ROOT / "runs" / "adni_with_converters")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--protocols", nargs="+",
                   default=["random", "subject_only", "component_safe"])
    p.add_argument("--target-spec", type=float, default=0.90,
                   help="Fixed-specificity screening anchor (default 0.90)")
    p.add_argument("--prev-pop", type=float, required=True,
                   help="Population-screening prevalence (e.g. 0.10).")
    p.add_argument("--prev-clinic", type=float, required=True,
                   help="Memory-clinic referral prevalence (e.g. 0.30).")
    p.add_argument("--prev-pop-citation", type=str, default="",
                   help="Free-text citation for the population prevalence anchor")
    p.add_argument("--prev-clinic-citation", type=str, default="",
                   help="Free-text citation for the clinic prevalence anchor")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "reports" / "tables" / "adni" /
                           "adni_cost_of_leakage.json")
    args = p.parse_args()

    proto_results = {}
    for proto in args.protocols:
        per_seed = []
        for seed in args.seeds:
            path = args.runs_root / f"inflation_gap_seed{seed}" / proto / "test_predictions.csv"
            if not path.exists():
                raise SystemExit(f"Missing predictions: {path}")
            rows = list(csv.DictReader(path.open()))
            y_true = [int(r["y_true"])  for r in rows]
            y_prob = [float(r["y_prob"]) for r in rows]
            pts = roc_points(y_true, y_prob)
            j, j_sens, j_spec, j_thr = youden_optimum(pts)
            per_seed.append({
                "seed": seed,
                "n_test": len(rows),
                "auroc":               round(auroc_from_pts(pts), 4),
                "sens_at_fixed_spec":  round(sens_at_fixed_spec(pts, args.target_spec), 4),
                "youden_j":            round(j, 4),
                "youden_sens":         round(j_sens, 4),
                "youden_spec":         round(j_spec, 4),
                "youden_threshold":    round(j_thr, 4) if j_thr != float("inf") else None,
            })
        proto_results[proto] = {
            "per_seed":                per_seed,
            "mean_auroc":              round(mean([s["auroc"] for s in per_seed]), 4),
            "mean_sens_at_fixed_spec": round(mean([s["sens_at_fixed_spec"] for s in per_seed]), 4),
            "sd_sens_at_fixed_spec":   round(sd([s["sens_at_fixed_spec"] for s in per_seed]), 4),
            "mean_youden_j":           round(mean([s["youden_j"] for s in per_seed]), 4),
            "mean_youden_sens":        round(mean([s["youden_sens"] for s in per_seed]), 4),
            "mean_youden_spec":        round(mean([s["youden_spec"] for s in per_seed]), 4),
        }

    # ── Counts per 1000 screened, at the fixed-specificity operating point ──
    for proto in args.protocols:
        sens = proto_results[proto]["mean_sens_at_fixed_spec"]
        spec = args.target_spec
        for label, P in [("population", args.prev_pop), ("clinic", args.prev_clinic)]:
            n_pos = 1000 * P
            n_neg = 1000 * (1 - P)
            proto_results[proto][f"counts_per_1000_at_prev_{label}"] = {
                "prevalence":      P,
                "true_positives":  round(n_pos * sens,           1),
                "false_negatives": round(n_pos * (1 - sens),      1),
                "true_negatives":  round(n_neg * spec,            1),
                "false_positives": round(n_neg * (1 - spec),      1),
            }

    # ── Cost-of-leakage: same fixed-specificity point, leaky vs honest ────
    leaky_sens  = proto_results["random"]["mean_sens_at_fixed_spec"]
    honest_sens = proto_results["component_safe"]["mean_sens_at_fixed_spec"]
    cost = {}
    for label, P, cite in [
        ("population", args.prev_pop,    args.prev_pop_citation),
        ("clinic",     args.prev_clinic, args.prev_clinic_citation),
    ]:
        leaky_fn  = 1000 * P * (1 - leaky_sens)
        honest_fn = 1000 * P * (1 - honest_sens)
        cost[f"prev_{label}"] = {
            "prevalence":                         P,
            "prevalence_citation":                cite,
            "leaky_apparent_missed_per_1000":     round(leaky_fn,  1),
            "honest_actual_missed_per_1000":      round(honest_fn, 1),
            "additional_missed_if_trusting_leaky": round(honest_fn - leaky_fn, 1),
        }

    out = {
        "target_spec":              args.target_spec,
        "seeds":                    args.seeds,
        "protocols":                args.protocols,
        "runs_root":                str(args.runs_root.relative_to(PROJECT_ROOT)
                                       if str(args.runs_root).startswith(str(PROJECT_ROOT))
                                       else args.runs_root),
        "by_protocol":              proto_results,
        "cost_of_leakage":          cost,
        "notes": (
            "Sens@Spec=target is the highest sensitivity achievable at "
            "specificity >= target_spec on each protocol's held-out test set, "
            "averaged across seeds. Counts per 1000 at each prevalence anchor "
            "assume the fixed-specificity operating point; "
            "additional_missed_if_trusting_leaky is the per-1000-patient "
            "shortfall a clinician would experience by trusting the leaky "
            "benchmark's apparent sensitivity over the honest one."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Compact human-readable summary
    print(f"=== ADNI cost-of-leakage (sens at fixed spec={args.target_spec:.0%}) ===")
    for proto in args.protocols:
        r = proto_results[proto]
        print(f"  {proto:>15s}: AUROC {r['mean_auroc']:.4f}   "
              f"Sens@Spec={args.target_spec:.0%} = {r['mean_sens_at_fixed_spec']:.4f} "
              f"+/- {r['sd_sens_at_fixed_spec']:.4f}   "
              f"Youden J = {r['mean_youden_j']:.4f}")
    for label, P, cite in [
        ("population", args.prev_pop,    args.prev_pop_citation),
        ("clinic",     args.prev_clinic, args.prev_clinic_citation),
    ]:
        c = cost[f"prev_{label}"]
        print(f"\n  At prevalence={P:.0%} ({label}) [{cite}]:")
        print(f"    Leaky benchmark predicts:           {c['leaky_apparent_missed_per_1000']:>6.1f} missed AD diagnoses / 1000")
        print(f"    Honest deployment actually misses:  {c['honest_actual_missed_per_1000']:>6.1f} missed AD diagnoses / 1000")
        print(f"    Additional missed if trusting leaky: {c['additional_missed_if_trusting_leaky']:>6.1f} per 1000")
    try:
        rel = str(args.output.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(args.output)
    print(f"\n  Wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
