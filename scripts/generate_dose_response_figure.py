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
Writes: paper/fig11_dose_response.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid,
    INK, MUTED, NEUTRAL, PROTOCOL_COLOR, PROTOCOL_LABEL,
    BAND_ALPHA, MARKER_SIZE, REF_LW, REF_STYLE, SERIES_LW, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_dose_response.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig11_dose_response"


def main() -> int:
    d = json.loads(DATA.read_text())

    # Colour is reserved for protocol identity. These series are two backbones
    # trained on the same Protocol C splits, so they are told apart by ink
    # weight, line style and marker rather than by hue.
    ARCH = (
        ("resnet18",    "ResNet-18",
         dict(color=INK, marker="o", linestyle="-")),
        ("densenet121", "DenseNet-121",
         dict(color=NEUTRAL, marker="s", linestyle=(0, (4, 2)))),
    )

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(TWO_COL_W, 3.2), gridspec_kw={"wspace": 0.30}
    )

    # ── Panel A: per-seed scatter + mean with CI band ─────────────────────
    for arch, label, style in ARCH:
        color = style["color"]
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
                axa.scatter([o], [v], s=12, color=color, alpha=0.35,
                            edgecolor="none", zorder=2)

        # Mean line + CI band
        axa.fill_between(overlaps, ci_lo, ci_hi, color=color,
                         alpha=BAND_ALPHA, linewidth=0, zorder=1)
        axa.plot(overlaps, means, lw=SERIES_LW, markersize=MARKER_SIZE,
                 markeredgecolor="white", markeredgewidth=0.6,
                 label=label, zorder=3, **style)

    axa.set_xlabel("Target test-subject overlap fraction")
    axa.set_ylabel("Test AUROC")
    axa.set_xlim(-0.05, 1.05)
    axa.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axa.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axa.set_ylim(0.79, 0.97)
    axa.set_title("(a) Per-seed AUROC, mean $\\pm$ 95% CI", loc="left")
    thin_y_grid(axa)

    # Reference levels are protocol results, so they carry the protocol's
    # colour. Labels sit inside Panel A rather than in the right margin,
    # where they spilled into Panel B.
    for key, level, dy, va in (("A", 0.949, 0.003, "bottom"),
                               ("C", 0.819, -0.007, "top")):
        axa.axhline(level, color=PROTOCOL_COLOR[key], lw=REF_LW,
                    linestyle=REF_STYLE, zorder=0)
        axa.text(0.02, level + dy, f"{PROTOCOL_LABEL[key]}, {level:.3f}",
                 fontsize=7, color=MUTED, va=va, ha="left")

    # ── Panel B: linear fits ──────────────────────────────────────────────
    grid = np.linspace(0, 1, 100)
    for arch, label_short, style in ARCH:
        color = style["color"]
        f = d["linear_fits"][arch]
        intercept = f["intercept"]; slope = f["slope"]; r2 = f["r2"]
        # Scatter the per-seed points
        agg = d["by_arch"][arch]
        for k, dd in agg.items():
            o = float(k)
            for v in dd["per_seed"]:
                axb.scatter([o], [v], s=12, color=color, alpha=0.35,
                            edgecolor="none", zorder=2)
        # Fit line: short legend (slope + R\u00b2 only); the full
        # intercept-slope equation belongs in the caption, not the figure.
        axb.plot(grid, intercept + slope * grid, color=color,
                 linestyle=style["linestyle"], lw=SERIES_LW, zorder=3,
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
             fontsize=7.5, color=MUTED)

    # Legend below both panels, from Panel B: the same backbones, with the
    # fitted slope and R\u00b2.
    handles_b, labels_b = axb.get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.26, wspace=0.30)
    fig.legend(handles_b, labels_b, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8,
               columnspacing=2.0, handlelength=2.2)
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
