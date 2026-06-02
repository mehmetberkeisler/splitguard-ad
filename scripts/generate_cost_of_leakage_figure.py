#!/usr/bin/env python3
"""Generate the clinical cost-of-leakage figure for §6.8.

Two panels:
  (a) Per-seed ROC curves for Protocol A (leaky) and Protocol C
      (component-safe \\SGA{}) on the converter-inclusive ADNI sensitivity
      arm, with the sens@spec=0.90 screening operating point marked on
      each protocol's mean curve.
  (b) Expected missed-diagnoses per 1000 screened patients at two
      literature-cited prevalence anchors (Rajan 2021 75-84 stratum,
      Thomas 2025 PROMPT tertiary clinic), showing what the leaky
      benchmark predicts vs.\\ what an honest-protocol model actually
      delivers at the same operating point.

Reads
-----
runs/adni_with_converters/inflation_gap_seed{S}/{protocol}/test_predictions.csv
reports/tables/adni/adni_cost_of_leakage.json  (for the per-1000 anchors)

Writes
------
paper/fig9_cost_of_leakage.{pdf,png}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs" / "adni_with_converters"
COST_JSON = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_cost_of_leakage.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig9_cost_of_leakage"

SEEDS = [0, 1, 2, 3, 4]
TARGET_SPEC = 0.90

LEAKY = "#c44e52"
SPLIT = "#4c72b0"
NEUTRAL = "#7f8c8d"


def roc_points(y_true, y_prob):
    P = sum(y_true); N = len(y_true) - P
    pairs = sorted(zip(y_prob, y_true), reverse=True)
    tp = fp = 0
    pts = [(0.0, 1.0)]  # (FPR=0, TPR=0) start at origin
    for _, label in pairs:
        if label == 1: tp += 1
        else:          fp += 1
        sens = tp / P; spec = 1 - fp / N
        pts.append((1 - spec, sens))
    return pts  # [(FPR, TPR), ...]


def interp_roc(xs, ys, n=200):
    """Linearly interpolate a ROC curve onto a fixed FPR grid for averaging."""
    import bisect
    # sort by FPR then dedupe
    seen = {}
    for x, y in zip(xs, ys):
        if x not in seen or y > seen[x]:
            seen[x] = y
    xs_s = sorted(seen)
    ys_s = [seen[x] for x in xs_s]
    grid = [i / (n - 1) for i in range(n)]
    out = []
    for g in grid:
        i = bisect.bisect_left(xs_s, g)
        if i == 0:
            out.append(ys_s[0])
        elif i >= len(xs_s):
            out.append(ys_s[-1])
        else:
            x0, x1 = xs_s[i-1], xs_s[i]
            y0, y1 = ys_s[i-1], ys_s[i]
            t = (g - x0) / (x1 - x0) if x1 > x0 else 0
            out.append(y0 + t * (y1 - y0))
    return grid, out


def sens_at_spec(pts, target_spec):
    best = 0.0
    for fpr, tpr in pts:
        spec = 1 - fpr
        if spec >= target_spec and tpr > best:
            best = tpr
    return best


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 180,
    })

    # ── Collect ROC points per (seed, protocol) ────────────────────────────
    rocs = {"random": [], "component_safe": []}
    for proto in rocs:
        for s in SEEDS:
            path = RUNS / f"inflation_gap_seed{s}" / proto / "test_predictions.csv"
            rows = list(csv.DictReader(path.open()))
            y_t = [int(r["y_true"])  for r in rows]
            y_p = [float(r["y_prob"]) for r in rows]
            rocs[proto].append(roc_points(y_t, y_p))

    # Mean curves on a common FPR grid
    mean_curves = {}
    for proto, runs in rocs.items():
        grids = []
        for pts in runs:
            xs, ys = zip(*pts)
            g, y = interp_roc(list(xs), list(ys), n=200)
            grids.append(y)
        grid = [i / 199 for i in range(200)]
        mean_y = [sum(g[i] for g in grids) / len(grids) for i in range(200)]
        mean_curves[proto] = (grid, mean_y)

    cost = json.loads(COST_JSON.read_text())

    # ── Build figure ────────────────────────────────────────────────────────
    # Two-row layout so panel titles never overlap.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                   gridspec_kw={"wspace": 0.28})

    # Panel A: ROC curves
    for pts in rocs["random"]:
        xs, ys = zip(*pts)
        axa.plot(xs, ys, color=LEAKY, alpha=0.18, lw=0.9, zorder=2)
    for pts in rocs["component_safe"]:
        xs, ys = zip(*pts)
        axa.plot(xs, ys, color=SPLIT, alpha=0.18, lw=0.9, zorder=2)
    gx, gy = mean_curves["random"]
    axa.plot(gx, gy, color=LEAKY, lw=2.0, zorder=3,
             label="Protocol A — Leaky (mean)")
    gx, gy = mean_curves["component_safe"]
    axa.plot(gx, gy, color=SPLIT, lw=2.0, zorder=3,
             label="Protocol C — SplitGuard (mean)")
    # Operating-point markers at spec=0.90 ⇒ FPR=0.10.
    # Use the *per-seed mean* sens@spec we report in the text, NOT a value
    # re-computed on the averaged ROC curve (which saturates at 1.0 because
    # individual seeds vary in where their step-curves jump).
    op_fpr = 1 - TARGET_SPEC
    sens_l = cost["by_protocol"]["random"]["mean_sens_at_fixed_spec"]
    sens_h = cost["by_protocol"]["component_safe"]["mean_sens_at_fixed_spec"]
    axa.axvline(op_fpr, color=NEUTRAL, lw=0.7, linestyle=":", zorder=1)
    axa.scatter([op_fpr], [sens_l], s=70, color=LEAKY,
                edgecolor="black", linewidth=0.8, zorder=4)
    axa.scatter([op_fpr], [sens_h], s=70, color=SPLIT,
                edgecolor="black", linewidth=0.8, zorder=4)
    axa.annotate(f"  sens = {sens_l:.3f}", xy=(op_fpr, sens_l),
                 xytext=(op_fpr + 0.04, sens_l + 0.02), va="center",
                 color=LEAKY, fontsize=8.5, weight="bold")
    axa.annotate(f"  sens = {sens_h:.3f}", xy=(op_fpr, sens_h),
                 xytext=(op_fpr + 0.04, sens_h - 0.04), va="center",
                 color=SPLIT, fontsize=8.5, weight="bold")
    axa.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.7, linestyle=":", zorder=1)
    axa.set_xlabel("False positive rate (1 − specificity)")
    axa.set_ylabel("True positive rate (sensitivity)")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1.02)
    axa.set_title("(a) ROC, operating point @ spec = 0.90")
    axa.legend(loc="lower right", ncol=1)

    # Panel B: per-1000 missed-diagnoses bar chart at both prevalences
    labels = ["Population\n13.8 % (Rajan 2021)", "Tertiary clinic\n58.9 % (Thomas 2025)"]
    leaky_miss = [
        cost["cost_of_leakage"]["prev_population"]["leaky_apparent_missed_per_1000"],
        cost["cost_of_leakage"]["prev_clinic"]["leaky_apparent_missed_per_1000"],
    ]
    honest_miss = [
        cost["cost_of_leakage"]["prev_population"]["honest_actual_missed_per_1000"],
        cost["cost_of_leakage"]["prev_clinic"]["honest_actual_missed_per_1000"],
    ]
    x = np.arange(len(labels))
    w = 0.36
    axb.bar(x - w/2, leaky_miss, w, color=LEAKY, edgecolor="black",
            linewidth=0.6, label="Leaky benchmark predicts", zorder=3)
    axb.bar(x + w/2, honest_miss, w, color=SPLIT, edgecolor="black",
            linewidth=0.6, label="Honest deployment yields", zorder=3)
    for i, v in enumerate(leaky_miss):
        axb.text(x[i] - w/2, v + 4, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=9, color=LEAKY, weight="bold")
    for i, v in enumerate(honest_miss):
        axb.text(x[i] + w/2, v + 4, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=9, color=SPLIT, weight="bold")
    # Gap annotation
    for i, (lm, hm) in enumerate(zip(leaky_miss, honest_miss)):
        axb.annotate(f"+{hm - lm:.0f}", xy=(x[i], max(lm, hm) + 22),
                     ha="center", fontsize=10, weight="bold", color=NEUTRAL)
    axb.set_xticks(x)
    axb.set_xticklabels(labels, fontsize=9)
    axb.set_ylabel("Missed AD diagnoses per 1000 screened")
    axb.set_ylim(0, max(honest_miss) * 1.30)
    axb.grid(axis="y", linestyle=":", alpha=0.5)
    axb.set_title("(b) Missed AD diagnoses per 1000 screened")
    axb.legend(loc="upper left", ncol=1)

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_STEM}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
