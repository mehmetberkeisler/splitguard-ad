#!/usr/bin/env python3
"""Generate Figure 9: the clinical cost-of-leakage figure.

Two-panel publication-quality figure:
  (a) Per-seed ROC curves for Protocol A (leaky) and Protocol C
      (component-safe SplitGuard-AD) on the converter-inclusive ADNI
      sensitivity arm, with the screening operating point (specificity
      = 0.90) marked on each protocol's mean curve.
  (b) Expected missed AD diagnoses per 1,000 screened patients at two
      literature-cited prevalence anchors (Rajan 2021 75-84 stratum,
      Thomas 2025 PROMPT tertiary memory clinic).

Style: shared publication-quality from scripts/_publication_style.py.

Reads:
  runs/adni_with_converters/inflation_gap_seed{S}/{protocol}/test_predictions.csv
  reports/tables/adni/adni_cost_of_leakage.json

Writes:
  paper/fig09_cost_of_leakage.{pdf,png}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid, reference_line, protocol_handles,
    INK, MUTED, NEUTRAL, PROTOCOL_COLOR, PROTOCOL_MARKER, MARKER_SIZE,
    REF_LW, REF_STYLE, SERIES_LW, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs" / "adni_with_converters"
COST_JSON = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_cost_of_leakage.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig09_cost_of_leakage"

SEEDS = [0, 1, 2, 3, 4]
TARGET_SPEC = 0.90


# ── ROC primitives ──────────────────────────────────────────────────────
def roc_points(y_true, y_prob):
    P = sum(y_true); N = len(y_true) - P
    pairs = sorted(zip(y_prob, y_true), reverse=True)
    tp = fp = 0
    pts = [(0.0, 0.0)]
    for _, label in pairs:
        if label == 1: tp += 1
        else:          fp += 1
        sens = tp / P; spec = 1 - fp / N
        pts.append((1 - spec, sens))
    return pts


def interp_roc(xs, ys, n=200):
    import bisect
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


def main() -> int:
    # ── Collect ROC points per (seed, protocol) ────────────────────────────
    rocs = {"random": [], "component_safe": []}
    for proto in rocs:
        for s in SEEDS:
            path = RUNS / f"inflation_gap_seed{s}" / proto / "test_predictions.csv"
            rows = list(csv.DictReader(path.open()))
            y_t = [int(r["y_true"])  for r in rows]
            y_p = [float(r["y_prob"]) for r in rows]
            rocs[proto].append(roc_points(y_t, y_p))

    # Mean curves on common FPR grid
    mean_curves = {}
    for proto, runs in rocs.items():
        grids = []
        for pts in runs:
            xs, ys = zip(*pts)
            _, y = interp_roc(list(xs), list(ys), n=200)
            grids.append(y)
        grid = [i / 199 for i in range(200)]
        mean_y = [sum(g[i] for g in grids) / len(grids) for i in range(200)]
        mean_curves[proto] = (grid, mean_y)

    cost = json.loads(COST_JSON.read_text())

    # ── Build figure ────────────────────────────────────────────────────────
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(TWO_COL_W, 3.3), gridspec_kw={"wspace": 0.35}
    )

    # ── Panel A: ROC curves ────────────────────────────────────────────────
    for proto, key in (("random", "A"), ("component_safe", "C")):
        for pts in rocs[proto]:
            xs, ys = zip(*pts)
            axa.plot(xs, ys, color=PROTOCOL_COLOR[key], alpha=0.14, lw=0.6,
                     zorder=2)
        gx, gy = mean_curves[proto]
        axa.plot(gx, gy, color=PROTOCOL_COLOR[key], lw=SERIES_LW, zorder=3)

    # Operating-point markers at spec=0.90
    op_fpr = 1 - TARGET_SPEC
    sens_l = cost["by_protocol"]["random"]["mean_sens_at_fixed_spec"]
    sens_h = cost["by_protocol"]["component_safe"]["mean_sens_at_fixed_spec"]
    reference_line(axa, op_fpr, orientation="v")
    for key, sens in (("A", sens_l), ("C", sens_h)):
        axa.plot([op_fpr], [sens], linestyle="none",
                 marker=PROTOCOL_MARKER[key], markersize=MARKER_SIZE + 1,
                 color=PROTOCOL_COLOR[key], markeredgecolor="white",
                 markeredgewidth=0.8, zorder=4)

    # Diagonal chance line
    axa.plot([0, 1], [0, 1], color=NEUTRAL, lw=REF_LW, linestyle=REF_STYLE,
             zorder=1)

    # Operating-point sensitivities in the lower-right white space. Serif ink,
    # like every other annotation; the markers on the curves carry identity.
    axa.text(0.97, 0.13, f"Protocol A sensitivity {sens_l:.3f}",
             color=INK, fontsize=8, ha="right", va="bottom")
    axa.text(0.97, 0.05, f"Protocol C sensitivity {sens_h:.3f}",
             color=INK, fontsize=8, ha="right", va="bottom")

    axa.set_xlabel("False positive rate (1 $-$ specificity)")
    axa.set_ylabel("True positive rate (sensitivity)")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1.02)
    axa.set_title("(a) ROC, operating point @ spec $= 0.90$", loc="left")
    thin_y_grid(axa)

    # ── Panel B: per-1000 missed-diagnoses bar chart ──────────────────────
    # Short tick labels; full prevalence citations move to the caption.
    labels = ["Population\n(13.8% prev.)",
              "Tertiary clinic\n(58.9% prev.)"]
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
    axb.bar(x - w/2, leaky_miss, w, color=PROTOCOL_COLOR["A"],
            edgecolor="none", zorder=3)
    axb.bar(x + w/2, honest_miss, w, color=PROTOCOL_COLOR["C"],
            edgecolor="none", zorder=3)

    ymax = max(honest_miss) * 1.40
    pad = ymax * 0.012
    for i, v in enumerate(leaky_miss):
        axb.text(x[i] - w/2, v + pad, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=7.5, color=INK)
    for i, v in enumerate(honest_miss):
        axb.text(x[i] + w/2, v + pad, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=7.5, color=INK)
    # Gap annotation anchored above the taller (honest) bar.
    for i, (lm, hm) in enumerate(zip(leaky_miss, honest_miss)):
        axb.annotate(
            f"+{hm - lm:.0f} missed",
            xy=(x[i] + w/2, hm),
            xytext=(0, 18), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color=MUTED,
        )
    axb.set_xticks(x)
    axb.set_xticklabels(labels)
    axb.set_ylabel("Missed AD diagnoses per 1,000 screened")
    axb.set_ylim(0, ymax)
    axb.set_title("(b) Per-1,000 missed-diagnosis shortfall", loc="left")
    thin_y_grid(axb)

    # One legend for both panels: the curves and the bars are the same two
    # protocols, so four entries would restate two.
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.23, wspace=0.35)
    fig.legend(handles=protocol_handles(["A", "C"]),
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=False, fontsize=8,
               columnspacing=2.2, handlelength=1.8)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
