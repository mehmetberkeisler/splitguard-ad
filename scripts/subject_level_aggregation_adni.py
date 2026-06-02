#!/usr/bin/env python3
"""Subject-level AUROC for ADNI: image-level vs subject-aggregated comparison.

Closes the methodological vulnerability where image-level AUROC implicitly
treats each scan as i.i.d. — but ADNI's test set has ~5.5 scans per subject
on average (range 1–7), so a few subjects with many scans can dominate the
metric.  This script aggregates each subject's predictions (mean y_prob)
before computing AUROC, and compares the inflation gap at both levels.

For each (seed × protocol) on the converter-inclusive ADNI sensitivity arm:
  * Image-level AUROC (original convention, ~207 predictions per fold)
  * Subject-level AUROC (~38 subjects per fold, one prediction each)

Then per protocol, aggregate across 5 seeds:
  * Mean ± SD at each level
  * Paired-seed bootstrap (B=10,000) 95% CI on the level
  * Same for the inflation gap (Protocol A − Protocol C) at each level

Reads
-----
runs/adni_with_converters/inflation_gap_seed{S}/{protocol}/test_predictions.csv

Writes
------
reports/tables/adni/adni_subject_level_auroc.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "adni_with_converters"
OUT_PATH = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_subject_level_auroc.json"


# ── AUROC, percentile, mean, sd (no sklearn) ─────────────────────────────
def auroc(y_true: list[int], y_score: list[float]) -> float:
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0; wins = 0.0
    for p in pos:
        for n in neg:
            pairs += 1
            if p > n: wins += 1.0
            elif p == n: wins += 0.5
    return wins / pairs


def percentile(values, p):
    sv = sorted(v for v in values if v == v)
    if not sv: return float("nan")
    rank = (len(sv) - 1) * p / 100
    lo = int(rank); hi = min(lo + 1, len(sv) - 1); frac = rank - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x == x]
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ── Aggregate predictions to subject level ────────────────────────────────
def subject_level_predictions(rows):
    """Group rows by subject_id, average y_prob, take majority y_true.

    Returns (y_true_subj, y_prob_subj) lists, one entry per unique subject.
    Each subject's binary label is the rounded mean of its image-level
    labels (in practice unanimous because subject_safe / random both
    preserve the per-subject label of a single CN-vs-AD universe).
    """
    by_subj: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_subj[r["subject_id"]].append((int(r["y_true"]), float(r["y_prob"])))
    y_t, y_p = [], []
    for sid, pts in by_subj.items():
        labels = [t for t, _ in pts]
        probs  = [p for _, p in pts]
        # Take majority label.  In practice all labels of a subject agree
        # in the CN-vs-AD binary universe; defensive aggregation handles
        # any future edge cases.
        label = 1 if sum(labels) > len(labels) / 2 else 0
        y_t.append(label)
        y_p.append(mean(probs))
    return y_t, y_p


def image_level_predictions(rows):
    y_t = [int(r["y_true"]) for r in rows]
    y_p = [float(r["y_prob"]) for r in rows]
    return y_t, y_p


# ── Main aggregation + bootstrap ─────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--protocols", nargs="+",
                   default=["random", "subject_only", "component_safe"])
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    rng = random.Random(args.seed)
    lo_q = (100.0 - args.ci) / 2; hi_q = 100.0 - lo_q

    # ── Per (seed, protocol): compute both levels of AUROC ────────────────
    per_seed: dict[str, dict[str, dict[int, float]]] = {
        proto: {"image": {}, "subject": {}, "n_subjects": {}, "n_images": {}}
        for proto in args.protocols
    }
    for proto in args.protocols:
        for seed in args.seeds:
            path = RUNS_ROOT / f"inflation_gap_seed{seed}" / proto / "test_predictions.csv"
            if not path.exists():
                raise SystemExit(f"Missing predictions: {path}")
            rows = list(csv.DictReader(path.open()))
            yi, pi = image_level_predictions(rows)
            ys, ps = subject_level_predictions(rows)
            per_seed[proto]["image"][seed]      = auroc(yi, pi)
            per_seed[proto]["subject"][seed]    = auroc(ys, ps)
            per_seed[proto]["n_subjects"][seed] = len(ys)
            per_seed[proto]["n_images"][seed]   = len(yi)

    # ── Aggregate per protocol with paired-seed bootstrap ────────────────
    def boot_ci(values):
        """Paired-seed bootstrap mean + CI (B resamples of the seed array)."""
        if not values:
            return float("nan"), float("nan"), float("nan")
        boot_means = []
        for _ in range(args.n_boot):
            idx = [rng.randrange(len(values)) for _ in range(len(values))]
            boot_means.append(mean([values[i] for i in idx]))
        return (mean(values),
                percentile(boot_means, lo_q),
                percentile(boot_means, hi_q))

    by_proto = {}
    for proto in args.protocols:
        img_vals = [per_seed[proto]["image"][s]   for s in args.seeds]
        sub_vals = [per_seed[proto]["subject"][s] for s in args.seeds]
        img_mean, img_lo, img_hi = boot_ci(img_vals)
        sub_mean, sub_lo, sub_hi = boot_ci(sub_vals)
        by_proto[proto] = {
            "n_seeds": len(args.seeds),
            "n_subjects_per_seed_mean": round(mean(list(per_seed[proto]["n_subjects"].values())), 1),
            "n_images_per_seed_mean":   round(mean(list(per_seed[proto]["n_images"].values())), 1),
            "image_level": {
                "per_seed": [round(v, 4) for v in img_vals],
                "mean":     round(img_mean, 4),
                "sd":       round(sd(img_vals), 4),
                "ci_lo":    round(img_lo, 4),
                "ci_hi":    round(img_hi, 4),
            },
            "subject_level": {
                "per_seed": [round(v, 4) for v in sub_vals],
                "mean":     round(sub_mean, 4),
                "sd":       round(sd(sub_vals), 4),
                "ci_lo":    round(sub_lo, 4),
                "ci_hi":    round(sub_hi, 4),
            },
        }

    # ── Inflation gap at both levels: Protocol A − Protocol C, paired ────
    def paired_gap_ci(a_vals, c_vals):
        deltas = [a - c for a, c in zip(a_vals, c_vals)]
        boot_deltas = []
        for _ in range(args.n_boot):
            idx = [rng.randrange(len(deltas)) for _ in range(len(deltas))]
            boot_deltas.append(mean([deltas[i] for i in idx]))
        same_sign = sum(1 for d in boot_deltas if (d > 0) == (mean(deltas) > 0))
        return {
            "delta_per_seed":       [round(d, 4) for d in deltas],
            "delta_mean":           round(mean(deltas), 4),
            "delta_ci_lo":          round(percentile(boot_deltas, lo_q), 4),
            "delta_ci_hi":          round(percentile(boot_deltas, hi_q), 4),
            "direction_preserved":  f"{same_sign}/{args.n_boot}",
        }

    img_A = [per_seed["random"]["image"][s]         for s in args.seeds]
    img_C = [per_seed["component_safe"]["image"][s] for s in args.seeds]
    img_B = [per_seed["subject_only"]["image"][s]   for s in args.seeds]
    sub_A = [per_seed["random"]["subject"][s]         for s in args.seeds]
    sub_C = [per_seed["component_safe"]["subject"][s] for s in args.seeds]
    sub_B = [per_seed["subject_only"]["subject"][s]   for s in args.seeds]

    inflation_gap = {
        "image_level": {
            "total_gap_A_minus_C": paired_gap_ci(img_A, img_C),
            "subject_identity_A_minus_B": paired_gap_ci(img_A, img_B),
            "component_marginal_B_minus_C": paired_gap_ci(img_B, img_C),
        },
        "subject_level": {
            "total_gap_A_minus_C": paired_gap_ci(sub_A, sub_C),
            "subject_identity_A_minus_B": paired_gap_ci(sub_A, sub_B),
            "component_marginal_B_minus_C": paired_gap_ci(sub_B, sub_C),
        },
    }

    out = {
        "seeds":       args.seeds,
        "protocols":   args.protocols,
        "n_boot":      args.n_boot,
        "ci_pct":      args.ci,
        "by_protocol": by_proto,
        "inflation_gap": inflation_gap,
        "notes": (
            "Image-level AUROC is computed across every per-scan prediction "
            "(test set ~207 scans across ~38 subjects per seed, mean 5.45 scans/subject). "
            "Subject-level AUROC averages y_prob per subject_id then computes AUROC "
            "across ~38 subjects (one prediction each). "
            "The inflation gap is reported at both levels for transparency; the "
            "subject-level gap is the more conservative quantity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ── Compact human summary ────────────────────────────────────────────
    print(f"=== ADNI image-level vs subject-level AUROC (n=5 seeds, B={args.n_boot}) ===\n")
    print(f"  {'Protocol':>15s} {'n_img':>7s} {'n_subj':>7s} "
          f"{'Image AUROC':>18s} {'Subject AUROC':>20s}")
    for proto in args.protocols:
        d = by_proto[proto]
        i = d["image_level"]; s = d["subject_level"]
        print(f"  {proto:>15s} {d['n_images_per_seed_mean']:>7.0f} "
              f"{d['n_subjects_per_seed_mean']:>7.0f}  "
              f"{i['mean']:.4f} [{i['ci_lo']:.3f},{i['ci_hi']:.3f}]  "
              f"{s['mean']:.4f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")
    print()
    print("=== Inflation gap (Protocol A − Protocol C) ===\n")
    for level in ("image_level", "subject_level"):
        g = inflation_gap[level]["total_gap_A_minus_C"]
        print(f"  {level:>15s}: total gap = {g['delta_mean']:+.4f} "
              f"[{g['delta_ci_lo']:+.4f}, {g['delta_ci_hi']:+.4f}]  "
              f"direction {g['direction_preserved']}")
    print()
    print(f"  Wrote {args.output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
