#!/usr/bin/env python3
"""Generate Figure 7: cross-cohort inflation-gap comparison.

Two-panel publication-quality figure:
  (a) Per-protocol AUROC (leaky vs SplitGuard-AD) for each of the three
      tiers — JPEG, OASIS-1, ADNI — with 95% CIs where multi-seed.
  (b) Inflation gap (ΔAUROC) per tier with 95% CIs; all three intervals
      should be clear of zero.

Inputs (already on disk):
  reports/tables/jpeg_inflation_gap_bootstrap.json
  reports/tables/oasis1_inflation_gap_bootstrap.json
  reports/tables/adni/adni_inflation_gap_bootstrap.json

Style: shared publication-quality from scripts/_publication_style.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid,
    LEAKY, SPLIT, INTER, NEUTRAL, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_STEM = PROJECT_ROOT / "paper" / "fig5_cross_cohort_inflation"


def load_oasis():
    p = PROJECT_ROOT / "reports" / "tables" / "oasis1_inflation_gap_bootstrap.json"
    d = json.loads(p.read_text())
    return {
        "leaky": (d["auroc"]["leaky"]["point_mean"],
                  d["auroc"]["leaky"]["ci_lo"],
                  d["auroc"]["leaky"]["ci_hi"]),
        "splitguard": (d["auroc"]["splitguard"]["point_mean"],
                       d["auroc"]["splitguard"]["ci_lo"],
                       d["auroc"]["splitguard"]["ci_hi"]),
        "gap": (d["inflation_gap_leaky_minus_splitguard"]["point"],
                d["inflation_gap_leaky_minus_splitguard"]["ci_lo"],
                d["inflation_gap_leaky_minus_splitguard"]["ci_hi"]),
    }


def load_adni():
    p = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_inflation_gap_bootstrap.json"
    d = json.loads(p.read_text())
    return {
        "leaky": (d["auroc"]["random"]["point_mean"],
                  d["auroc"]["random"]["ci_lo"],
                  d["auroc"]["random"]["ci_hi"]),
        "splitguard": (d["auroc"]["component_safe"]["point_mean"],
                       d["auroc"]["component_safe"]["ci_lo"],
                       d["auroc"]["component_safe"]["ci_hi"]),
        "gap": (d["inflation_gap"]["total_random_minus_component_safe"]["point_estimate"],
                d["inflation_gap"]["total_random_minus_component_safe"]["ci_lo"],
                d["inflation_gap"]["total_random_minus_component_safe"]["ci_hi"]),
    }


def load_jpeg():
    p = PROJECT_ROOT / "reports" / "tables" / "jpeg_inflation_gap_bootstrap.json"
    d = json.loads(p.read_text())
    return {
        "leaky": (d["auroc"]["leaky"]["point_mean"],
                  d["auroc"]["leaky"]["ci_lo"],
                  d["auroc"]["leaky"]["ci_hi"]),
        "splitguard": (d["auroc"]["splitguard"]["point_mean"],
                       d["auroc"]["splitguard"]["ci_lo"],
                       d["auroc"]["splitguard"]["ci_hi"]),
        "gap": (d["inflation_gap_leaky_minus_splitguard"]["point"],
                d["inflation_gap_leaky_minus_splitguard"]["ci_lo"],
                d["inflation_gap_leaky_minus_splitguard"]["ci_hi"]),
    }


def main() -> int:
    cohorts = [
        ("Public 2D JPEG", load_jpeg()),
        ("OASIS-1",        load_oasis()),
        ("ADNI1",          load_adni()),
    ]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(TWO_COL_W, 3.1),
        gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.32},
    )

    x = np.arange(len(cohorts))
    offset = 0.18
    leaky_xs = x - offset
    sgd_xs   = x + offset
    leaky_y  = [c[1]["leaky"][0] for c in cohorts]
    sgd_y    = [c[1]["splitguard"][0] for c in cohorts]

    def err(y, lo, hi):
        return np.array([
            [(p - L) if L < p else 0.0 for p, L in zip(y, lo)],
            [(H - p) if H > p else 0.0 for p, H in zip(y, hi)],
        ])

    leaky_err = err(leaky_y,
                    [c[1]["leaky"][1] for c in cohorts],
                    [c[1]["leaky"][2] for c in cohorts])
    sgd_err   = err(sgd_y,
                    [c[1]["splitguard"][1] for c in cohorts],
                    [c[1]["splitguard"][2] for c in cohorts])

    ax_a.errorbar(leaky_xs, leaky_y, yerr=leaky_err, fmt="o", color=LEAKY,
                  capsize=2.0, lw=1.0, markersize=5,
                  label="Random / leaky split")
    ax_a.errorbar(sgd_xs, sgd_y, yerr=sgd_err, fmt="s", color=SPLIT,
                  capsize=2.0, lw=1.0, markersize=5,
                  label="SplitGuard-AD component-safe")
    for i in range(len(cohorts)):
        ax_a.plot([leaky_xs[i], sgd_xs[i]], [leaky_y[i], sgd_y[i]],
                  color=NEUTRAL, lw=0.5, alpha=0.5, zorder=0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([c[0] for c in cohorts])
    ax_a.set_ylabel("Test AUROC")
    ax_a.set_ylim(0.78, 1.005)
    ax_a.set_title("(a) Per-protocol AUROC, 95% CI", loc="left")
    thin_y_grid(ax_a)

    # Right panel: inflation gap per cohort with 95% CI
    gap_y  = [c[1]["gap"][0] for c in cohorts]
    gap_lo = [c[1]["gap"][1] for c in cohorts]
    gap_hi = [c[1]["gap"][2] for c in cohorts]
    gap_err = err(gap_y, gap_lo, gap_hi)
    ax_b.errorbar(x, gap_y, yerr=gap_err, fmt="D", color=INTER,
                  capsize=2.0, lw=1.0, markersize=5)
    ax_b.axhline(0, color=NEUTRAL, lw=0.4, linestyle=(0, (1, 2)))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["JPEG", "OASIS-1", "ADNI1"])
    ax_b.set_ylabel(r"Inflation gap ($\Delta$AUROC)")
    ax_b.set_ylim(-0.01, 0.20)
    ax_b.set_title("(b) Inflation gap, 95% CI", loc="left")
    thin_y_grid(ax_b)
    # Annotate gap values just above each marker, not to its right (avoids
    # clipping the right edge of the axes for the ADNI bar).
    for i, p in enumerate(gap_y):
        ax_b.annotate(f"+{p:.3f}",
                      xy=(x[i], gap_hi[i]),
                      xytext=(0, 4), textcoords="offset points",
                      ha="center", va="bottom",
                      fontsize=7.5, color=INTER)

    # Shared legend below both panels
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.22, wspace=0.32)
    fig.legend(handles_a, labels_a, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
