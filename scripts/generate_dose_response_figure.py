#!/usr/bin/env python3
"""Generate Figure 10: the leakage dose-response curve.

Two-panel publication-quality figure:
  (a) Per-seed scatter + mean line with 95% paired-bootstrap CI band for
      each architecture (ResNet-18, DenseNet-121). x-axis is target
      test-subject overlap fraction; y-axis is test AUROC.
  (b) Same two architectures overlaid with OLS linear fits showing
      the dose-response slope.

Style: shared publication-quality from scripts/_publication_style.py
(STIX serif, Wong colorblind-safe palette, minimal grid, legend below
axes, 600 dpi PDF, Type-42 embedded fonts).

Reads:  reports/tables/adni/adni_dose_response.json
Writes: paper/fig10_dose_response.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid,
    SPLIT, DENSENET, NEUTRAL, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_dose_response.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig10_dose_response"


def main() -> int:
    d = json.loads(DATA.read_text())

    RESNET_COLOR   = SPLIT
    DENSENET_COLOR = DENSENET

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(TWO_COL_W, 3.2), gridspec_kw={"wspace": 0.30}
    )

    # ── Panel A: per-seed scatter + mean with CI band ─────────────────────
    for arch, color, label in [
        ("resnet18",    RESNET_COLOR,   "ResNet-18"),
        ("densenet121", DENSENET_COLOR, "DenseNet-121"),
    ]:
        agg = d["by_arch"][arch]
        overlaps = sorted(float(k) for k in agg)

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
                axa.scatter([o], [v], s=12, color=color, alpha=0.40,
                            edgecolor="none", zorder=2)

        # Mean line + CI band
        axa.fill_between(overlaps, ci_lo, ci_hi, color=color, alpha=0.15,
                         linewidth=0, zorder=1)
        axa.plot(overlaps, means, color=color, lw=1.4, marker="o",
                 markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                 label=label, zorder=3)

    axa.set_xlabel("Target test-subject overlap fraction")
    axa.set_ylabel("Test AUROC")
    axa.set_xlim(-0.05, 1.05)
    axa.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axa.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axa.set_ylim(0.79, 0.97)
    axa.set_title("(a) Per-seed AUROC, mean $\\pm$ 95% CI", loc="left")
    thin_y_grid(axa)

    # Anchor lines — labels placed INSIDE Panel A (top-left / lower-left),
    # not in the right margin where they spill into Panel B.
    axa.axhline(0.819, color=NEUTRAL, lw=0.4, linestyle=(0, (1, 2)), zorder=0)
    axa.axhline(0.949, color=NEUTRAL, lw=0.4, linestyle=(0, (1, 2)), zorder=0)
    axa.text(0.02, 0.952, "Protocol A — leaky (0.949)",
             fontsize=7, color=NEUTRAL, va="bottom", ha="left")
    axa.text(0.02, 0.812, "Protocol C — baseline (0.819)",
             fontsize=7, color=NEUTRAL, va="top", ha="left")

    # ── Panel B: linear fits ──────────────────────────────────────────────
    grid = np.linspace(0, 1, 100)
    for arch, color, label_short in [
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
                axb.scatter([o], [v], s=12, color=color, alpha=0.35,
                            edgecolor="none", zorder=2)
        # Fit line — short legend (slope + R² only); full intercept-slope
        # equation is fine for the caption, not the figure.
        axb.plot(grid, intercept + slope * grid, color=color, lw=1.4, zorder=3,
                 label=f"{label_short} (slope $= +{slope:.3f}$, $R^2 = {r2:.2f}$)")

    axb.set_xlabel("Target test-subject overlap fraction")
    axb.set_ylabel("Test AUROC")
    axb.set_xlim(-0.05, 1.05)
    axb.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axb.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axb.set_ylim(0.79, 0.97)
    axb.set_title("(b) Linear dose-response fits", loc="left")
    thin_y_grid(axb)

    # Headline (per-10pp slope) as quiet in-axes text, top-left.
    rn = d["linear_fits"]["resnet18"]
    axb.text(0.02, 0.97,
             f"ResNet-18: $+{rn['slope']*0.1:.3f}$ AUROC per 10pp overlap",
             transform=axb.transAxes, ha="left", va="top",
             fontsize=7.5, color=NEUTRAL)

    # Combined legend below both panels (Panel A: architecture lines;
    # Panel B: same architectures with linear-fit slope + R²).
    handles_a, labels_a = axa.get_legend_handles_labels()
    handles_b, labels_b = axb.get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.26, wspace=0.30)
    fig.legend(handles_b, labels_b, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8,
               columnspacing=2.0, handlelength=1.8)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
