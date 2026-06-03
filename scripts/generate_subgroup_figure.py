#!/usr/bin/env python3
"""Per-protocol × subgroup AUROC bar chart (ADNI) — Figure 8.

The female-vs-male gap of ~0.10 AUROC under both honest protocols is
one of the more striking findings of the ADNI tier; the leaky protocol
flattens it. A grouped bar chart makes this visible at a glance.

Reads:  reports/tables/adni/adni_subgroup_analysis.json
Writes: paper/fig7_adni_subgroup_auroc.{pdf,png}

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
    apply_publication_style, thin_y_grid,
    LEAKY, INTER, SPLIT, NEUTRAL, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_subgroup_analysis.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig7_adni_subgroup_auroc"


def main() -> int:
    d = json.loads(DATA.read_text())
    R = d["results"]

    protocols    = ["random", "subject_only", "component_safe"]
    proto_labels = ["Random (leaky)", "Subject-only", "Component-safe (SplitGuard-AD)"]
    proto_colors = [LEAKY, INTER, SPLIT]

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
               error_kw={"lw": 0.5, "ecolor": "0.25"},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(subg_labels)
    ax.set_ylabel("Test AUROC")
    # Raise the y-cap so the F-M gap line has its own band above the
    # tallest error bar (Female ~0.97).
    ax.set_ylim(0.6, 1.06)
    ax.axhline(0.5, color=NEUTRAL, lw=0.4, linestyle=(0, (1, 2)))
    thin_y_grid(ax)
    ax.legend(loc="lower right", ncol=1, fontsize=7.5)

    # F-M gap inset — quiet monospace, sits in the dedicated headroom band
    # at y ~1.03, clear of every error bar.
    so_f = R["subject_only__sex_F"]["point_mean"]
    so_m = R["subject_only__sex_M"]["point_mean"]
    cs_f = R["component_safe__sex_F"]["point_mean"]
    cs_m = R["component_safe__sex_M"]["point_mean"]
    ax.text(0.50, 1.035,
            "F$-$M AUROC gap (honest):  "
            f"Subject-only $+{so_f - so_m:.3f}$    "
            f"Component-safe $+{cs_f - cs_m:.3f}$",
            transform=ax.transData, ha="center", va="center",
            fontsize=7.5, color=NEUTRAL, family="monospace")

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png")
    plt.close(fig)
    print(f"  wrote paper/{OUT_STEM.name}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
