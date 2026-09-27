#!/usr/bin/env python3
"""Draw sensitivity against target specificity for the three protocols.

The manuscript translates the AUROC gap into a screening cost at one fixed
operating point (0.90 specificity). A single point invites the objection that
it was chosen to flatter the argument, and the ten-row supplementary table
that answers the objection is hard to read as a shape. Plotted across the
sweep, three things the table only implies become visible at once.

First, the leaky protocol's advantage is not an artefact of one threshold:
it holds at every target specificity. Second, the gap widens as the operating
point tightens, which is the direction that matters clinically, because
screening runs at high specificity. Third, the B-C ordering inverts at 0.95,
the same inversion the component marginal shows elsewhere; it is drawn here
rather than described, because a reader should be able to see that the
component layer's advantage over subject-only splitting is not stable.

Bands are +/- 1 SD across the five seeds.

Usage
-----
    python3 scripts/generate_operating_point_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _publication_style import (  # noqa: E402
    BAND_ALPHA, MARKER_SIZE, PROTOCOL_COLOR, PROTOCOL_LABEL,
    PROTOCOL_LINESTYLE, PROTOCOL_MARKER, SERIES_LW, apply_publication_style,
    thin_y_grid,
)
from _paths import display_path  # noqa: E402

SOURCE = ROOT / "reports" / "tables" / "adni" / "adni_operating_point_sensitivity.json"
OUTPUT = ROOT / "paper" / "fig10_operating_point.pdf"

PROTOCOLS = (("random", "A"), ("subject_only", "B"), ("component_safe", "C"))


def series(payload: dict, protocol: str):
    p = payload[protocol]
    specs = sorted(
        float(k.removeprefix("sens_at_spec_").removesuffix("_mean"))
        for k in p if k.startswith("sens_at_spec_") and k.endswith("_mean")
    )
    means = [p[f"sens_at_spec_{s:.2f}_mean"] for s in specs]
    sds = [p[f"sens_at_spec_{s:.2f}_sd"] for s in specs]
    return specs, means, sds


def main() -> int:
    apply_publication_style()
    import matplotlib.pyplot as plt

    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    # Wide enough for the three protocol names on one legend row below the
    # axes, which is where every other figure puts its legend.
    fig, ax = plt.subplots(figsize=(5.0, 2.9))

    for proto, key in PROTOCOLS:
        specs, means, sds = series(payload, proto)
        lo = [m - s for m, s in zip(means, sds)]
        hi = [m + s for m, s in zip(means, sds)]
        colour = PROTOCOL_COLOR[key]
        ax.fill_between(specs, lo, hi, color=colour, alpha=BAND_ALPHA,
                        linewidth=0)
        ax.plot(specs, means, color=colour,
                linestyle=PROTOCOL_LINESTYLE[key], linewidth=SERIES_LW,
                marker=PROTOCOL_MARKER[key], markersize=MARKER_SIZE,
                label=PROTOCOL_LABEL[key],
                markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xlabel("Target specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax.set_ylim(0.30, 1.0)
    thin_y_grid(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, fontsize=7.5, handlelength=2.2, columnspacing=1.0)

    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"  wrote {display_path(OUTPUT)} + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
