#!/usr/bin/env python3
"""Generate Figure 10: the leakage dose-response curve.

Two-panel figure:
  (a) Per-seed scatter + mean line with 95% paired-bootstrap CI band for
      each architecture (ResNet-18, DenseNet-121).  The x-axis is target
      test-subject overlap fraction, the y-axis is test AUROC.
  (b) Same two architectures overlaid with the OLS linear fits, showing
      the dose-response slope.  Annotation reports the headline finding:
      AUROC ≈ a + b × overlap.

Reads: reports/tables/adni/adni_dose_response.json (from analyze_dose_response.py)
Writes: paper/fig10_dose_response.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_dose_response.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig10_dose_response"


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

    d = json.loads(DATA.read_text())

    RESNET_COLOR    = "#4c72b0"   # SplitGuard blue from existing figs
    DENSENET_COLOR  = "#c44e52"   # leaky red from existing figs
    NEUTRAL         = "#7f8c8d"

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                   gridspec_kw={"wspace": 0.26})

    # ── Panel A: per-seed scatter + mean with CI band ─────────────────────
    for arch, color, label in [
        ("resnet18",    RESNET_COLOR,   "ResNet-18"),
        ("densenet121", DENSENET_COLOR, "DenseNet-121"),
    ]:
        agg = d["by_arch"][arch]
        overlaps = sorted(float(k) for k in agg)
        means    = [agg[f"{o:.1f}" if f"{o:.1f}" in agg else f"{o}"]["mean"]
                    if (f"{o:.1f}" in agg or f"{o}" in agg)
                    else agg[str(o)]["mean"]
                    for o in overlaps]
        # JSON keys come back as strings; align robustly:
        def get_d(o):
            for k in agg:
                if float(k) == o: return agg[k]
            raise KeyError(o)
        means = [get_d(o)["mean"] for o in overlaps]
        ci_lo = [get_d(o)["ci_lo"] for o in overlaps]
        ci_hi = [get_d(o)["ci_hi"] for o in overlaps]

        # Per-seed scatter (small markers, semitransparent)
        for o in overlaps:
            for v in get_d(o)["per_seed"]:
                axa.scatter([o], [v], s=18, color=color, alpha=0.45,
                            edgecolor="none", zorder=2)

        # Mean line + CI band
        axa.fill_between(overlaps, ci_lo, ci_hi, color=color, alpha=0.18,
                         linewidth=0, zorder=1)
        axa.plot(overlaps, means, color=color, lw=2.0, marker="o",
                 markersize=7, markeredgecolor="black", markeredgewidth=0.6,
                 label=label, zorder=3)

    axa.set_xlabel("Target test-subject overlap fraction")
    axa.set_ylabel("Test AUROC")
    axa.set_xlim(-0.05, 1.05)
    axa.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axa.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axa.set_ylim(0.79, 0.97)
    axa.grid(axis="y", linestyle=":", alpha=0.5)
    axa.set_title("(a) Per-seed AUROC, mean ± 95% CI")
    axa.legend(loc="lower right")

    # Anchor annotations: Protocol C and Protocol A reference levels
    axa.axhline(0.819, color=NEUTRAL, lw=0.7, linestyle=":", zorder=0)
    axa.text(0.02, 0.821, "Protocol C baseline (0.819)", fontsize=7.5,
             color=NEUTRAL, va="bottom")
    axa.axhline(0.949, color=NEUTRAL, lw=0.7, linestyle=":", zorder=0)
    axa.text(0.02, 0.951, "Protocol A leaky (0.949)", fontsize=7.5,
             color=NEUTRAL, va="bottom")

    # ── Panel B: linear fits ──────────────────────────────────────────────
    grid = np.linspace(0, 1, 100)
    for arch, color, label in [
        ("resnet18",    RESNET_COLOR,   "ResNet-18"),
        ("densenet121", DENSENET_COLOR, "DenseNet-121"),
    ]:
        f = d["linear_fits"][arch]
        intercept = f["intercept"]; slope = f["slope"]; r2 = f["r2"]
        # Scatter the per-seed points
        agg = d["by_arch"][arch]
        for k, dd in agg.items():
            o = float(k)
            for v in dd["per_seed"]:
                axb.scatter([o], [v], s=18, color=color, alpha=0.40,
                            edgecolor="none", zorder=2)
        # Fit line
        axb.plot(grid, intercept + slope * grid, color=color, lw=2.0,
                 zorder=3,
                 label=(f"{label}: AUROC = {intercept:.3f} + "
                        f"{slope:.3f}·overlap (R²={r2:.2f})"))

    axb.set_xlabel("Target test-subject overlap fraction")
    axb.set_ylabel("Test AUROC")
    axb.set_xlim(-0.05, 1.05)
    axb.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axb.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axb.set_ylim(0.79, 0.97)
    axb.grid(axis="y", linestyle=":", alpha=0.5)
    axb.set_title("(b) Linear dose-response fits")
    axb.legend(loc="lower right", fontsize=8)

    # Headline annotation in panel B inset
    rn = d["linear_fits"]["resnet18"]
    axb.text(0.02, 0.97,
             f"ResNet-18 slope = +{rn['slope']:.3f} AUROC per unit overlap\n"
             f"(i.e. +{rn['slope']*0.1:.3f} AUROC per 10pp of overlap)",
             transform=axb.transAxes, ha="left", va="top",
             fontsize=8.5, color="black", weight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="0.7", linewidth=0.6, alpha=0.92))

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_STEM}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
