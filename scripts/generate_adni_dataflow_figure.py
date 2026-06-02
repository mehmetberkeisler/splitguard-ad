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

    # Compact aspect ratio so the float fits inline below Table 2 on the
    # same page, rather than bumping itself (and the white space) to the
    # next page. The in-figure title is dropped because the LaTeX caption
    # already names the figure.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.1, 9.65)
    ax.axis("off")

    BOX_COLOR_MAIN = "#dde7f2"
    BOX_COLOR_DERIVED = "#e5e5e5"
    BOX_COLOR_EXCL = "#fbe2e2"
    EDGE_COLOR = "#3c5a78"

    def box(x, y, w, h, text, color=BOX_COLOR_MAIN, fontsize=9, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            color=EDGE_COLOR, lw=1.0,
        ))

    # Geometry --------------------------------------------------------------
    # 8 stages, evenly spaced.  The main column sits at x ∈ [0.5, 6.5]; the
    # right-side "Excluded" boxes start at x = 6.9 and end at x = 9.7 — well
    # inside xlim=10 (the previous 8.0+2.3 layout overflowed the axes and
    # the right edge of those boxes got clipped on export).
    main_x = 0.5
    main_w = 6.0
    excl_gap = 0.4           # arrow length between main column and excl box
    excl_w = 2.8             # width of the excl-side boxes
    box_h = 0.72
    pitch = 1.05
    # Topmost box centred at y=8.85; bottom box centred at 8.85 − 7·1.05 = 1.5.
    y_centers = [8.85 - i * pitch for i in range(8)]

    def arrow_between(i):
        """Vertical arrow from bottom of box i to top of box i+1."""
        top_next = y_centers[i + 1] + box_h
        bot_curr = y_centers[i]
        arrow(main_x + main_w / 2, bot_curr,
              main_x + main_w / 2, top_next)

    # 1. LONI IDA download (main, bold)
    box(main_x, y_centers[0], main_w, box_h,
        "LONI IDA: ADNI1 Complete 3Yr 1.5T\n10 archive zips (~44 GB compressed)",
        weight="bold")
    arrow_between(0)

    # 2. Stream-extract + slice cache
    box(main_x, y_centers[1], main_w, box_h,
        "Stream-extract + coronal-centre slice cache\n2,182 NIfTI volumes → 2,182 single-channel PNGs (54 MB total)",
        color=BOX_COLOR_DERIVED)
    arrow_between(1)

    # 3. Manifest
    box(main_x, y_centers[2], main_w, box_h,
        "Manifest construction (preprocess_adni_volumes_to_slices.py)\n2,182 scans, 1,959 sessions, 382 subjects",
        color=BOX_COLOR_DERIVED)
    arrow_between(2)

    # 4. Diagnosis assignment + first exclusion side-branch
    box(main_x, y_centers[3], main_w, box_h,
        "Per-ontology diagnosis join (build_adni_manifest.py)\nCN: 756   MCI: 768   AD: 656   unknown: 2",
        color=BOX_COLOR_DERIVED)
    excl_y3 = y_centers[3] + box_h / 2
    arrow(main_x + main_w, excl_y3,
          main_x + main_w + excl_gap, excl_y3)
    box(main_x + main_w + excl_gap, y_centers[3], excl_w, box_h,
        "Excluded:\nMCI 768, unknown 2",
        color=BOX_COLOR_EXCL, fontsize=8)
    arrow_between(3)

    # 5. CN-vs-AD universe
    box(main_x, y_centers[4], main_w, box_h,
        "CN-vs-AD binary universe\n1,412 scans (756 CN + 656 AD), 366 subjects",
        color=BOX_COLOR_DERIVED)
    arrow_between(4)

    # 6. Leakage graph + component-safe split + second exclusion side-branch
    box(main_x, y_centers[5], main_w, box_h,
        "Leakage graph + frozen component-safe split (5 seeds)\nClass-balance constraint excludes 289 orphan-component scans",
        color=BOX_COLOR_DERIVED)
    excl_y5 = y_centers[5] + box_h / 2
    arrow(main_x + main_w, excl_y5,
          main_x + main_w + excl_gap, excl_y5)
    box(main_x + main_w + excl_gap, y_centers[5], excl_w, box_h,
        "Excluded:\n289 orphan-component scans\n(216 AD + 73 CN)",
        color=BOX_COLOR_EXCL, fontsize=8)
    arrow_between(5)

    # 7. Per-seed splits (bold)
    box(main_x, y_centers[6], main_w, box_h,
        "Per-seed train / val / test partitions (1,125 scans)\n795 train / 165 val / 165 test   (zero subject overlap)",
        color=BOX_COLOR_DERIVED, weight="bold")
    arrow_between(6)

    # 8. Three protocol arms (bold, main colour)
    box(main_x, y_centers[7], main_w, box_h,
        "Three protocols evaluated on the same 1,125-scan universe\nRandom (leaky)  ·  Subject-only  ·  Component-safe (SplitGuard)",
        weight="bold")

    # No in-figure title — the LaTeX \caption already says exactly this.

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
