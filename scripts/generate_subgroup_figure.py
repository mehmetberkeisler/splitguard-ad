#!/usr/bin/env python3
"""Generate the per-protocol × subgroup AUROC bar chart for §6.6.

The female-vs-male gap of ~0.10 AUROC under both honest protocols is one
of the more striking findings of the ADNI tier; the leaky protocol
flattens it (because patient memorisation dominates everything). A
grouped bar chart makes the story visible at a glance.

Reads:  reports/tables/adni/adni_subgroup_analysis.json
Writes: paper/fig7_adni_subgroup_auroc.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_subgroup_analysis.json"
OUT_STEM = PROJECT_ROOT / "paper" / "fig7_adni_subgroup_auroc"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.loads(DATA.read_text())
    R = d["results"]

    protocols = ["random", "subject_only", "component_safe"]
    proto_labels = ["Random (leaky)", "Subject-only", "Component-safe (SplitGuard)"]
    proto_colors = ["#c44e52", "#5b8a55", "#4c72b0"]

    subgroups = ["sex_F", "sex_M", "age_young", "age_old"]
    subg_labels = ["Female", "Male", "Age < 76", "Age ≥ 76"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    n_sub = len(subgroups)
    n_pro = len(protocols)
    group_width = 0.78
    bar_width = group_width / n_pro
    x = np.arange(n_sub)

    for i, proto in enumerate(protocols):
        means = []
        lo = []
        hi = []
        for sg in subgroups:
            key = f"{proto}__{sg}"
            r = R[key]
            means.append(r["point_mean"])
            lo.append(r["point_mean"] - r["ci_lo"])
            hi.append(r["ci_hi"] - r["point_mean"])
        xs = x + (i - (n_pro - 1) / 2) * bar_width
        ax.bar(
            xs, means, bar_width,
            label=proto_labels[i], color=proto_colors[i],
            edgecolor="black", linewidth=0.6,
            yerr=[lo, hi], capsize=2.5, error_kw={"lw": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subg_labels, fontsize=10)
    ax.set_ylabel("Test AUROC", fontsize=11)
    ax.set_ylim(0.6, 1.02)
    ax.axhline(0.5, color="0.7", lw=0.6, linestyle=":")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=1)
    ax.set_title(
        "ADNI per-subgroup test AUROC (5 seeds, paired-bootstrap 95% CI)",
        fontsize=11, loc="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the female-male gap under each honest protocol — placed
    # in the bottom-left corner so the bars themselves do the talking.
    so_f = R["subject_only__sex_F"]["point_mean"]
    so_m = R["subject_only__sex_M"]["point_mean"]
    cs_f = R["component_safe__sex_F"]["point_mean"]
    cs_m = R["component_safe__sex_M"]["point_mean"]
    txt = (
        "Female − Male AUROC gap (honest protocols):\n"
        f"  Subject-only:  +{so_f - so_m:.3f}\n"
        f"  Component-safe: +{cs_f - cs_m:.3f}"
    )
    ax.text(
        0.013, 0.97, txt,
        transform=ax.transAxes,
        fontsize=8.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="0.7", linewidth=0.6),
    )

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_STEM}.pdf")
    print(f"Wrote {OUT_STEM}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
