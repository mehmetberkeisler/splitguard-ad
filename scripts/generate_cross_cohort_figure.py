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
    apply_publication_style, thin_y_grid, reference_line, protocol_handles,
    INK, NEUTRAL, PROTOCOL_COLOR, PROTOCOL_LABEL, PROTOCOL_MARKER,
    CAPSIZE, ERROR_LW, MARKER_SIZE, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_STEM = PROJECT_ROOT / "paper" / "fig06_cross_cohort_inflation"


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
        ("Tier 1 (redistributed)", load_jpeg()),
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

    for key, xs_, ys_, err_ in (("A", leaky_xs, leaky_y, leaky_err),
                                ("C", sgd_xs, sgd_y, sgd_err)):
        ax_a.errorbar(xs_, ys_, yerr=err_, fmt=PROTOCOL_MARKER[key],
                      color=PROTOCOL_COLOR[key], capsize=CAPSIZE,
                      lw=ERROR_LW, markersize=MARKER_SIZE,
                      markeredgecolor="white", markeredgewidth=0.6,
                      label=PROTOCOL_LABEL[key])
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
    # The gap is A minus C, a derived quantity rather than a protocol, so it
    # is drawn in ink; green would read as Protocol B everywhere else.
    ax_b.errorbar(x, gap_y, yerr=gap_err, fmt="D", color=INK,
                  capsize=CAPSIZE, lw=ERROR_LW, markersize=MARKER_SIZE,
                  markeredgecolor="white", markeredgewidth=0.6)
    reference_line(ax_b, 0)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["JPEG", "OASIS-1", "ADNI1"])
    ax_b.set_ylabel(r"Inflation gap ($\Delta$AUROC)")
    ax_b.set_ylim(-0.01, 0.20)
    ax_b.set_title("(b) Inflation gap, 95% CI", loc="left")
    thin_y_grid(ax_b)
    # Annotate gap values just above each marker, not to its right (avoids
    # clipping the right edge of the axes for the ADNI bar). The end markers
    # are aligned inwards: centred on the first one, half the label fell
    # outside the axes and crossed the y-axis spine into the tick labels.
    for i, p in enumerate(gap_y):
        if i == 0:
            ha, dx = "left", -2
        elif i == len(gap_y) - 1:
            ha, dx = "right", 2
        else:
            ha, dx = "center", 0
        ax_b.annotate(f"+{p:.3f}",
                      xy=(x[i], gap_hi[i]),
                      xytext=(dx, 4), textcoords="offset points",
                      ha=ha, va="bottom",
                      fontsize=7.5, color=INK)

    # Shared legend below both panels. Built from the shared protocol handles
    # rather than from the errorbar containers, whose swatches are a vertical
    # bar and so differ from the legend in every other figure.
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.22, wspace=0.32)
    fig.legend(handles=protocol_handles(["A", "C"]), loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
