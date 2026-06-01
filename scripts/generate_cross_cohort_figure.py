#!/usr/bin/env python3
"""Generate Figure 5: cross-cohort inflation-gap comparison.

Pulls the leaky-vs-SplitGuard AUROC and inflation gap for the three
SplitGuard-AD tiers (JPEG, OASIS-1, ADNI) and renders a single
publication-grade dot plot with 95% confidence intervals where
available (parametric SD for JPEG, paired bootstrap CIs for OASIS-1
and ADNI).

Inputs (all already on disk)
----------------------------
* JPEG: hard-coded numbers from
  reports/tables/inflation_gap_experiment.json (single seed,
  parametric reporting per the v1 paper).
* OASIS-1: reports/tables/oasis1_inflation_gap_bootstrap.json
  (5 seeds, paired bootstrap, B=10000).
* ADNI: reports/tables/adni/adni_inflation_gap_bootstrap.json
  (5 seeds, paired bootstrap, B=10000).

Output
------
* paper/fig5_cross_cohort_inflation.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_STEM = PROJECT_ROOT / "paper" / "fig5_cross_cohort_inflation"


def load_oasis():
    p = PROJECT_ROOT / "reports" / "tables" / "oasis1_inflation_gap_bootstrap.json"
    d = json.loads(p.read_text())
    leaky = d["auroc"]["leaky"]
    splitguard = d["auroc"]["splitguard"]
    gap = d["inflation_gap_leaky_minus_splitguard"]
    return {
        "leaky": (leaky["point_mean"], leaky["ci_lo"], leaky["ci_hi"]),
        "splitguard": (splitguard["point_mean"], splitguard["ci_lo"], splitguard["ci_hi"]),
        "gap": (gap["point"], gap["ci_lo"], gap["ci_hi"]),
    }


def load_adni():
    p = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_inflation_gap_bootstrap.json"
    d = json.loads(p.read_text())
    leaky = d["auroc"]["random"]
    splitguard = d["auroc"]["component_safe"]
    gap = d["inflation_gap"]["total_random_minus_component_safe"]
    return {
        "leaky": (leaky["point_mean"], leaky["ci_lo"], leaky["ci_hi"]),
        "splitguard": (splitguard["point_mean"], splitguard["ci_lo"], splitguard["ci_hi"]),
        "gap": (gap["point_estimate"], gap["ci_lo"], gap["ci_hi"]),
    }


def load_jpeg():
    p = PROJECT_ROOT / "reports" / "tables" / "jpeg_inflation_gap_bootstrap.json"
    d = json.loads(p.read_text())
    leaky = d["auroc"]["leaky"]
    splitguard = d["auroc"]["splitguard"]
    gap = d["inflation_gap_leaky_minus_splitguard"]
    return {
        "leaky": (leaky["point_mean"], leaky["ci_lo"], leaky["ci_hi"]),
        "splitguard": (splitguard["point_mean"], splitguard["ci_lo"], splitguard["ci_hi"]),
        "gap": (gap["point"], gap["ci_lo"], gap["ci_hi"]),
    }


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cohorts = [
        ("Public 2D JPEG\n(5 seeds)", load_jpeg()),
        ("OASIS-1\n(5 seeds)", load_oasis()),
        ("ADNI1: Complete 3Yr 1.5T\n(5 seeds)", load_adni()),
    ]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(8.4, 4.0), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    # Left panel: per-protocol AUROCs (paired dots per cohort)
    x = np.arange(len(cohorts))
    offset = 0.18
    leaky_xs = x - offset
    sgd_xs = x + offset
    leaky_y = [c[1]["leaky"][0] for c in cohorts]
    sgd_y = [c[1]["splitguard"][0] for c in cohorts]
    leaky_lo = [c[1]["leaky"][1] for c in cohorts]
    leaky_hi = [c[1]["leaky"][2] for c in cohorts]
    sgd_lo = [c[1]["splitguard"][1] for c in cohorts]
    sgd_hi = [c[1]["splitguard"][2] for c in cohorts]

    # Use CI bars where lo < hi (multi-seed); otherwise no bar.
    leaky_err = np.array([
        [(p - lo) if lo < p else 0.0 for p, lo in zip(leaky_y, leaky_lo)],
        [(hi - p) if hi > p else 0.0 for p, hi in zip(leaky_y, leaky_hi)],
    ])
    sgd_err = np.array([
        [(p - lo) if lo < p else 0.0 for p, lo in zip(sgd_y, sgd_lo)],
        [(hi - p) if hi > p else 0.0 for p, hi in zip(sgd_y, sgd_hi)],
    ])

    leaky_color = "#c44e52"
    sgd_color = "#4c72b0"
    ax_a.errorbar(leaky_xs, leaky_y, yerr=leaky_err, fmt="o", color=leaky_color,
                  capsize=4, lw=1.4, markersize=8, label="Random / leaky split")
    ax_a.errorbar(sgd_xs, sgd_y, yerr=sgd_err, fmt="s", color=sgd_color,
                  capsize=4, lw=1.4, markersize=8, label="SplitGuard component-safe")
    for i in range(len(cohorts)):
        ax_a.plot([leaky_xs[i], sgd_xs[i]], [leaky_y[i], sgd_y[i]],
                  color="0.7", lw=0.8, zorder=0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([c[0] for c in cohorts], fontsize=9)
    ax_a.set_ylabel("Test AUROC", fontsize=11)
    ax_a.set_ylim(0.78, 1.005)
    ax_a.grid(axis="y", linestyle=":", alpha=0.5)
    ax_a.legend(loc="lower left", frameon=False, fontsize=9)
    ax_a.set_title("(a) Per-protocol AUROC", fontsize=11, loc="left")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Right panel: inflation gap per cohort with 95% CI
    gap_y = [c[1]["gap"][0] for c in cohorts]
    gap_lo = [c[1]["gap"][1] for c in cohorts]
    gap_hi = [c[1]["gap"][2] for c in cohorts]
    gap_err = np.array([
        [p - lo for p, lo in zip(gap_y, gap_lo)],
        [hi - p for p, hi in zip(gap_y, gap_hi)],
    ])
    ax_b.errorbar(x, gap_y, yerr=gap_err, fmt="D", color="#5b8a55",
                  capsize=4, lw=1.4, markersize=8)
    ax_b.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["JPEG", "OASIS-1", "ADNI"], fontsize=9)
    ax_b.set_ylabel("Inflation gap (ΔAUROC)", fontsize=11)
    ax_b.set_ylim(-0.01, 0.20)
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)
    ax_b.set_title("(b) Inflation gap, 95% CI", fontsize=11, loc="left")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Annotate each gap with its value
    for i, (p, lo, hi) in enumerate(zip(gap_y, gap_lo, gap_hi)):
        ax_b.annotate(f"+{p:.3f}", (x[i], p), xytext=(8, 0),
                      textcoords="offset points", fontsize=8, va="center")

    fig.suptitle(
        "Inflation gap is direction-preserved across all three tiers",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()

    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_STEM}.pdf")
    print(f"Wrote {OUT_STEM}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
