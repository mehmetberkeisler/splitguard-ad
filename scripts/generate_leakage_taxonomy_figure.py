#!/usr/bin/env python3
"""Draw Figure 1: the four leakage modes and the edge type that closes each.

This figure used to be a hand-made SVG in its own visual language (sans-serif
type, a separate colour per panel, an in-figure title and a summary band that
repeated the caption). Drawing it here puts it on the same footing as every
other figure: the shared serif type, colour reserved for protocol identity,
and a legend below the axes.

Each panel shows one participant's images, a random split that puts the same
entity on both sides, and the edge that keeps it together. The two protocol
colours carry the argument: the bracket and circle mark the leak a random
split (Protocol A) produces, the triangle marks the edge that prevents it under
component-safe splitting (Protocol C). Every image is neutral grey, because
the images themselves are not a protocol.

Usage
-----
    python3 scripts/generate_leakage_taxonomy_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _publication_style import (  # noqa: E402
    INK, MUTED, PROTOCOL_COLOR, PROTOCOL_MARKER, TWO_COL_W,
    apply_publication_style, protocol_handles,
)
from _paths import display_path  # noqa: E402

OUTPUT = ROOT / "paper" / "fig01_leakage_taxonomy.pdf"

# Canvas in data units with equal aspect, so a square image is square.
W, H = 100.0, 62.0
SQ, STEP = 3.6, 4.6            # image tile size and pitch
IMAGE_FILL = "#D9D9D9"
PANEL_RULE = "#DADADA"

PANELS = (
    dict(origin=(0, 31), title="(a) Within-volume slice leakage",
         subtitle="2D slices of one 3D scan fall on both sides of the split",
         source="Subject A: one volume, 5 slices",
         n=5, train=3, copies=False,
         leak="Same volume in\ntrain and test",
         fix="Source-volume edges keep\nall slices together"),
    dict(origin=(50, 31), title="(b) Longitudinal same-subject leakage",
         subtitle="Visits of one participant fall on both sides of the split",
         source="Subject B: 5 visits, m0 to m24",
         n=5, train=3, copies=False,
         leak="Same participant at\ndifferent visits",
         fix="Subject edges bind all\nvisits into one component"),
    dict(origin=(0, 0), title="(c) Near-duplicate leakage",
         subtitle="Augmented or rescaled copies of one slice cross the split",
         source="Subject C: 1 slice + 4 augmented copies",
         n=5, train=3, copies=True,
         leak="Copy of a training\nimage in test",
         fix="Perceptual-hash edges\n(Hamming $\\leq$ 4) flag copies"),
    dict(origin=(50, 0), title="(d) Session re-acquisition leakage",
         subtitle="Rescans from one imaging session cross the split",
         source="Subject D: session s1 scanned 3 times",
         n=3, train=2, copies=False,
         leak="Rescan of the same\nsession in test",
         fix="Session edges keep all\nrescans together"),
)


def tiles(ax, x0, y, count, copies_from=None):
    """Draw `count` image tiles left to right. Tiles at index >= copies_from
    are augmented copies and get a dashed outline."""
    from matplotlib.patches import Rectangle
    for i in range(count):
        dashed = copies_from is not None and i >= copies_from
        ax.add_patch(Rectangle(
            (x0 + i * STEP, y), SQ, SQ, facecolor=IMAGE_FILL,
            edgecolor=MUTED, linewidth=0.5,
            linestyle=(0, (2, 1.2)) if dashed else "-"))


def box(ax, x, y, w, h, face="white", edge=MUTED, lw=0.6, rounded=True):
    from matplotlib.patches import FancyBboxPatch
    style = "round,pad=0,rounding_size=0.8" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=face, edgecolor=edge, linewidth=lw))


def note(ax, x, y, text, key):
    """A protocol-coloured marker carries identity; the text stays in ink."""
    ax.plot([x], [y], linestyle="none", marker=PROTOCOL_MARKER[key],
            markersize=5.5, color=PROTOCOL_COLOR[key],
            markeredgecolor="white", markeredgewidth=0.6)
    ax.text(x + 1.6, y, text, ha="left", va="center", fontsize=7.5,
            color=INK, linespacing=1.15)


def panel(ax, spec):
    ox, oy = spec["origin"]
    n, n_train = spec["n"], spec["train"]
    n_test = n - n_train
    copies = spec["copies"]

    # Type sizes follow the other figures: 9.5 pt panel titles, nothing below
    # 7 pt, so the smallest label survives scaling to the text width.
    ax.text(ox + 1, oy + 29.6, spec["title"], ha="left", va="top",
            fontsize=9.5, color=INK)
    ax.text(ox + 1, oy + 27.0, spec["subtitle"], ha="left", va="top",
            fontsize=7.5, color=MUTED)

    # Source: every image of one entity, before splitting
    ax.text(ox + 1, oy + 24.4, spec["source"], ha="left", va="top",
            fontsize=7.5, color=INK)
    src_w = 2.0 + n * SQ + (n - 1) * (STEP - SQ)
    box(ax, ox + 1, oy + 15.6, src_w, 6.6, face="#F4F4F4")
    tiles(ax, ox + 2.0, oy + 17.1, n, copies_from=1 if copies else None)

    # Random split
    train_w = 2.0 + n_train * SQ + (n_train - 1) * (STEP - SQ)
    test_x = ox + 1 + train_w + 1.2
    test_w = 2.0 + n_test * SQ + (n_test - 1) * (STEP - SQ)
    gap_x = ox + 1 + train_w + 0.6
    ax.annotate("", xy=(gap_x, oy + 12.6), xytext=(gap_x, oy + 15.4),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6,
                                mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(gap_x + 1.0, oy + 14.0, "random split", ha="left", va="center",
            fontsize=7, color=MUTED)

    for bx, bw, label, count, first in (
        (ox + 1, train_w, "Train", n_train, 0),
        (test_x, test_w, "Test", n_test, n_train),
    ):
        box(ax, bx, oy + 4.4, bw, 8.0, rounded=False)
        ax.text(bx + 0.8, oy + 11.8, label, ha="left", va="top",
                fontsize=7, color=MUTED)
        copies_from = None
        if copies:
            copies_from = 1 if first == 0 else 0
        tiles(ax, bx + 1.0, oy + 5.3, count, copies_from=copies_from)

    # The leak: one entity on both sides of the boundary, bracketed in the
    # Protocol A colour
    left, right = ox + 1 + train_w / 2, test_x + test_w / 2
    ax.plot([left, left, right, right], [oy + 4.4, oy + 2.6, oy + 2.6, oy + 4.4],
            color=PROTOCOL_COLOR["A"], linewidth=1.0,
            solid_joinstyle="miter")
    ax.plot([(left + right) / 2], [oy + 2.6], linestyle="none",
            marker=PROTOCOL_MARKER["A"], markersize=4.5,
            color=PROTOCOL_COLOR["A"], markeredgecolor="white",
            markeredgewidth=0.6)

    note(ax, ox + 30.0, oy + 18.9, spec["leak"], "A")
    note(ax, ox + 30.0, oy + 8.4, spec["fix"], "C")


def main() -> int:
    apply_publication_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(TWO_COL_W, 4.65))
    ax = fig.add_axes([0.0, 0.07, 1.0, 0.93])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.plot([1, W - 1], [31, 31], color=PANEL_RULE, linewidth=0.6)
    ax.plot([50, 50], [1.5, H - 1.5], color=PANEL_RULE, linewidth=0.6)
    for spec in PANELS:
        panel(ax, spec)

    fig.legend(handles=protocol_handles(["A", "C"]), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
               columnspacing=2.4)

    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"  wrote {display_path(OUTPUT)} + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
