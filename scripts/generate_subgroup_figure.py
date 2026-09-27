#!/usr/bin/env python3
"""Per-protocol × subgroup AUROC bar chart (ADNI) — Figure 8.

The female-vs-male gap of ~0.10 AUROC under both honest protocols is
one of the more striking findings of the ADNI tier; the leaky protocol
flattens it. A grouped bar chart makes this visible at a glance.

Reads:  reports/tables/adni/adni_subgroup_analysis.json
Writes: paper/figS1_subgroup_auroc.{pdf,png}

Style: shared publication-quality (STIX serif, Wong colorblind-safe,
minimal grid, no top/right spines, 600 dpi PDF, Type-42 fonts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid, reference_line,
    MUTED, PROTOCOL_COLOR, PROTOCOL_LABEL, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_subgroup_analysis.json"
OUT_STEM = PROJECT_ROOT / "paper" / "figS1_subgroup_auroc"


def main() -> int:
    d = json.loads(DATA.read_text())
    R = d["results"]

    protocols    = ["random", "subject_only", "component_safe"]
    proto_labels = [PROTOCOL_LABEL[k] for k in ("A", "B", "C")]
    proto_colors = [PROTOCOL_COLOR[k] for k in ("A", "B", "C")]

    subgroups   = ["sex_F", "sex_M", "age_young", "age_old"]
    subg_labels = ["Female", "Male", r"Age $<$ 76", r"Age $\geq$ 76"]

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.6))

    n_sub = len(subgroups)
    n_pro = len(protocols)
    group_width = 0.78
    bar_width   = group_width / n_pro
    x = np.arange(n_sub)

    for i, proto in enumerate(protocols):
        means, lo_err, hi_err = [], [], []
        for sg in subgroups:
            r = R[f"{proto}__{sg}"]
            means.append(r["point_mean"])
            lo_err.append(r["point_mean"] - r["ci_lo"])
            hi_err.append(r["ci_hi"] - r["point_mean"])
        xs = x + (i - (n_pro - 1) / 2) * bar_width
        ax.bar(xs, means, bar_width,
               label=proto_labels[i], color=proto_colors[i],
               edgecolor="none",
               yerr=[lo_err, hi_err], capsize=1.8,
               error_kw={"lw": 0.5, "ecolor": MUTED},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(subg_labels)
    ax.set_ylabel("Test AUROC")
    # Raise the y-cap so the F-M gap line has its own band above the
    # tallest error bar (Female ~0.97).
    ax.set_ylim(0.6, 1.06)
    reference_line(ax, 0.5)
    thin_y_grid(ax)
    # Below the axes, as in the other bar figures: "lower right" put the
    # frameless legend on top of the Age bars, leaving black text over a
    # filled orange bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=7.5, frameon=False)

    # Female-male gap under the two honest protocols. Top-left in axes
    # coordinates, as in the other figures: centred at x=0.5 in data
    # coordinates it ran off the left edge of the axes.
    so_f = R["subject_only__sex_F"]["point_mean"]
    so_m = R["subject_only__sex_M"]["point_mean"]
    cs_f = R["component_safe__sex_F"]["point_mean"]
    cs_m = R["component_safe__sex_M"]["point_mean"]
    ax.text(0.01, 0.99,
            "Female$-$male AUROC gap:  "
            f"Protocol B $+{so_f - so_m:.3f}$,  "
            f"Protocol C $+{cs_f - cs_m:.3f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=MUTED)

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
