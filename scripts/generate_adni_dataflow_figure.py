#!/usr/bin/env python3
"""CONSORT-style data-flow figure for the ADNI tier.

Produces a single matplotlib figure walking through the pipeline from
LONI IDA zip download to per-seed train/val/test partitions, with row
counts annotated at every transition.

Closes the CLAIM 2024 Item 13 gap flagged in docs/REPORTING_CHECKLISTS.md.

Output
------
* paper/fig6_adni_dataflow.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_STEM = PROJECT_ROOT / "paper" / "fig6_adni_dataflow"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    BOX_COLOR_MAIN = "#dde7f2"
    BOX_COLOR_DERIVED = "#e5e5e5"
    BOX_COLOR_EXCL = "#fbe2e2"
    EDGE_COLOR = "#3c5a78"

    def box(x, y, w, h, text, color=BOX_COLOR_MAIN, fontsize=9, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=14,
            color=EDGE_COLOR, lw=1.2,
        ))

    def excl_label(x, y, text):
        ax.text(x, y, text, ha="left", va="center", fontsize=8,
                style="italic", color="#a04040")

    # Main pipeline column at x=2..7 (centred at 4.5)
    main_x = 1.0
    main_w = 6.0
    box_h = 0.95

    # 1. LONI IDA download
    box(main_x, 12.6, main_w, box_h,
        "LONI IDA: ADNI1 Complete 3Yr 1.5T\n10 archive zips (~44 GB compressed)",
        weight="bold")

    arrow(main_x + main_w / 2, 12.6, main_x + main_w / 2, 11.95)

    # 2. Stream-extract + slice cache
    box(main_x, 11.0, main_w, box_h,
        "Stream-extract + coronal-centre slice cache\n2{,}182 NIfTI volumes -> 2{,}182 single-channel PNGs (54 MB total)",
        color=BOX_COLOR_DERIVED)

    arrow(main_x + main_w / 2, 11.0, main_x + main_w / 2, 10.35)

    # 3. Subject / session structure
    box(main_x, 9.4, main_w, box_h,
        "Manifest construction (preprocess_adni_volumes_to_slices.py)\n2{,}182 scans, 1{,}959 sessions, 382 subjects",
        color=BOX_COLOR_DERIVED)

    arrow(main_x + main_w / 2, 9.4, main_x + main_w / 2, 8.75)

    # 4. Diagnosis assignment
    box(main_x, 7.8, main_w, box_h,
        "Per-ontology diagnosis join (build_adni_manifest.py)\nCN: 756  MCI: 768  AD: 656  unknown: 2",
        color=BOX_COLOR_DERIVED)

    # Exclusion side: MCI + unknown
    arrow(main_x + main_w, 8.27, main_x + main_w + 1.0, 8.27)
    box(main_x + main_w + 1.0, 7.8, 2.3, box_h,
        "Excluded:\nMCI 768, unknown 2",
        color=BOX_COLOR_EXCL, fontsize=8)

    arrow(main_x + main_w / 2, 7.8, main_x + main_w / 2, 7.15)

    # 5. CN-vs-AD universe
    box(main_x, 6.2, main_w, box_h,
        "CN-vs-AD binary universe\n1{,}412 scans (756 CN + 656 AD), 366 subjects",
        color=BOX_COLOR_DERIVED)

    arrow(main_x + main_w / 2, 6.2, main_x + main_w / 2, 5.55)

    # 6. Leakage graph + component-safe split
    box(main_x, 4.6, main_w, box_h,
        "Leakage graph + frozen component-safe split (5 seeds)\nClass-balance constraint excludes 287 orphan-component scans",
        color=BOX_COLOR_DERIVED)

    arrow(main_x + main_w, 5.07, main_x + main_w + 1.0, 5.07)
    box(main_x + main_w + 1.0, 4.6, 2.3, box_h,
        "Excluded:\n287 orphan-component scans",
        color=BOX_COLOR_EXCL, fontsize=8)

    arrow(main_x + main_w / 2, 4.6, main_x + main_w / 2, 3.95)

    # 7. Per-seed splits
    box(main_x, 3.0, main_w, box_h,
        "Per-seed train / val / test partitions (1{,}125 scans)\n795 train / 165 val / 165 test  (zero subject overlap)",
        color=BOX_COLOR_DERIVED, weight="bold")

    arrow(main_x + main_w / 2, 3.0, main_x + main_w / 2, 2.35)

    # 8. Three protocol arms
    box(main_x, 1.4, main_w, box_h,
        "Three protocols evaluated on the same 1{,}125-scan universe\nRandom (leaky)  ·  Subject-only  ·  Component-safe (SplitGuard)",
        weight="bold")

    # Outer title
    ax.text(5.0, 13.7, "ADNI1: Complete 3Yr 1.5T Data Flow",
            ha="center", va="center", fontsize=12, weight="bold")

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
