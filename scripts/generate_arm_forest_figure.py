#!/usr/bin/env python3
"""Draw every inflation-gap arm as a forest plot.

The arms currently reach the reader as a table of point estimates and
intervals. A table is the wrong shape for the claim being made about them:
the total gap is asserted to be stable across cohort, backbone and
acquisition filter, while the component-layer marginal is asserted to be
small and of inconsistent sign. Both are statements about where intervals
sit relative to each other and to zero, which is what a forest plot shows
and a column of numbers does not.

(a) Total gap, random minus component-safe. Every interval excludes zero.

(b) Component-layer marginal, subject-only minus component-safe. Drawn on
    its own axis because it is a different quantity an order of magnitude
    smaller; plotting it beside the total would compress it to a point.
    No interval excludes zero and one inverts, which is the manuscript's
    own reading and is easier to verify here than in the table.

Usage
-----
    python3 scripts/generate_arm_forest_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _publication_style import (  # noqa: E402
    ERROR_LW, INK, MARKER_SIZE, MUTED, NEUTRAL, SERIES_LW, TWO_COL_W,
    apply_publication_style, reference_line, thin_y_grid,
)
from _paths import display_path  # noqa: E402

TABLES = ROOT / "reports" / "tables" / "adni"
ROOT_TABLES = ROOT / "reports" / "tables"
OUTPUT = ROOT / "paper" / "fig07_arm_forest.pdf"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> list:
    """One row per arm, bottom-up.

    Each row carries the total gap and, where the artefact reports one, the
    component marginal. Both panels are drawn from this single list so a row
    occupies the same vertical position in each, which is what lets the two
    share one set of labels. The cross-cohort arms have no marginal in these
    artefacts and leave that slot empty rather than shifting the rows below
    them up.
    """
    rows = []

    # Every row is a gap between protocols, not a protocol, so no row takes a
    # protocol colour. The primary arm is in ink and the rest in grey.

    # Cross-cohort arms use a flatter schema than the ADNI ones.
    for label, path, colour in (
        ("Tier 1 (redistributed), ResNet-18", ROOT_TABLES / "jpeg_inflation_gap_bootstrap.json", NEUTRAL),
        ("Tier 2 (OASIS-1), ResNet-18", ROOT_TABLES / "oasis1_inflation_gap_bootstrap.json", NEUTRAL),
    ):
        g = load(path)["inflation_gap_leaky_minus_splitguard"]
        rows.append({"label": label, "colour": colour,
                     "total": (g["point"], g["ci_lo"], g["ci_hi"]),
                     "marginal": None})

    for label, name, colour in (
        ("Tier 3 (ADNI1), ResNet-18", "adni_inflation_gap_bootstrap.json", INK),
        ("ADNI1, DenseNet-121", "adni_inflation_gap_densenet121_bootstrap.json", NEUTRAL),
        ("ADNI1, MT1 excluded", "adni_inflation_gap_no_mt1_bootstrap.json", NEUTRAL),
        ("ADNI1, converter-inclusive", "adni_inflation_gap_with_converters_bootstrap.json", NEUTRAL),
    ):
        ig = load(TABLES / name)["inflation_gap"]
        t = ig["total_random_minus_component_safe"]
        m = ig["component_leakage_subject_only_minus_component_safe"]
        rows.append({"label": label, "colour": colour,
                     "total": (t["point_estimate"], t["ci_lo"], t["ci_hi"]),
                     "marginal": (m["point_estimate"], m["ci_lo"], m["ci_hi"])})

    return rows


def draw_panel(ax, rows, key, title, zero_label):
    """Draw one panel. Row i always sits at y = i, in both panels."""
    for y, row in enumerate(rows):
        interval = row[key]
        if interval is None:                    # arm reports no such quantity
            continue
        point, lo, hi = interval
        colour = row["colour"]
        ax.plot([lo, hi], [y, y], color=colour, linewidth=SERIES_LW,
                solid_capstyle="butt", zorder=2)
        for x in (lo, hi):                      # interval caps
            ax.plot([x, x], [y - 0.13, y + 0.13], color=colour,
                    linewidth=ERROR_LW, zorder=2)
        # Diamond: the marker every figure uses for a derived gap
        ax.plot([point], [y], marker="D", markersize=MARKER_SIZE,
                color=colour, markeredgecolor="white", markeredgewidth=0.6,
                zorder=3)
    reference_line(ax, 0.0, orientation="v")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(INK if row["colour"] == INK else MUTED)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(zero_label)
    ax.set_title(title, loc="left")
    thin_y_grid(ax, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> int:
    apply_publication_style()
    import matplotlib.pyplot as plt

    rows = collect()
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(TWO_COL_W, 2.75), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.12},
    )
    draw_panel(axa, rows, "total", "(a) Total gap (A $-$ C)", "AUROC difference")
    draw_panel(axb, rows, "marginal", "(b) Component marginal (B $-$ C)",
               "AUROC difference")
    axb.tick_params(labelleft=False)            # rows are shared with (a)

    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"  wrote {display_path(OUTPUT)} + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
