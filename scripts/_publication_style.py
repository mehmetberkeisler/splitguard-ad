"""Shared publication-quality matplotlib style for the SplitGuard-AD paper.

Loaded by every figure generator so all matplotlib outputs share one
visual identity:

  * Serif body font matching the LaTeX paper (Computer Modern / STIX)
  * Wong colorblind-safe palette (Nature Methods 2011)
  * Minimal grid (alpha 0.15 on Y only)
  * No top/right spines
  * Tight margins, single-column friendly figure widths
  * 600 dpi PDF export by default
"""

from __future__ import annotations


# Wong (2011) colorblind-safe palette, normalised to hex.  Use these
# instead of matplotlib defaults so figures read for both colour-blind
# viewers and in B/W print.
WONG = {
    "blue":          "#0072B2",  # main signal / honest protocol
    "orange":        "#E69F00",  # secondary signal / annotation
    "vermilion":     "#D55E00",  # leaky / overstated
    "bluish_green":  "#009E73",  # intermediate / subject-only
    "yellow":        "#F0E442",  # avoid as foreground
    "sky_blue":      "#56B4E9",  # tertiary
    "reddish_purple":"#CC79A7",  # tertiary
    "black":         "#000000",
    "grey":          "#7F7F7F",
}

# Convenience aliases used by figure scripts.  Map to the project's
# longstanding "leaky red vs SplitGuard blue" convention but in the
# Wong palette.
LEAKY      = WONG["vermilion"]
SPLIT      = WONG["blue"]
INTER      = WONG["bluish_green"]   # subject-only intermediate
NEUTRAL    = WONG["grey"]
DENSENET   = WONG["orange"]         # second architecture in dose-response


def apply_publication_style():
    """Apply the shared rcParams.  Call once at top of each figure script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Typography: serif to match LaTeX body text.  STIX is bundled
        # with matplotlib and renders Times-style serif at every size
        # without requiring a system font install.
        "font.family":          "serif",
        "font.serif":           ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":     "stix",
        "font.size":            9,
        "axes.titlesize":       9.5,
        "axes.labelsize":       9,
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "legend.fontsize":      8,

        # Title placement: left-aligned, normal weight (avoid the bold
        # all-caps "matplotlib title" feel)
        "axes.titlelocation":   "left",
        "axes.titleweight":     "normal",
        "axes.titlepad":        4,

        # Spines: no top/right (cleaner)
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.6,

        # Grid: minimal horizontal only
        "axes.grid":            False,   # turn on explicitly per-axes
        "grid.color":           WONG["grey"],
        "grid.alpha":           0.18,
        "grid.linewidth":       0.5,
        "grid.linestyle":       (0, (1, 2)),   # dotted

        # Ticks: outward, short, thin
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     2.5,
        "ytick.major.size":     2.5,
        "xtick.major.width":    0.5,
        "ytick.major.width":    0.5,

        # Legend: no frame, tight
        "legend.frameon":       False,
        "legend.borderaxespad": 0.3,
        "legend.handlelength":  1.6,
        "legend.handletextpad": 0.5,

        # Layout: tight by default
        "figure.constrained_layout.use": False,   # we'll use tight_layout

        # Export: print-quality
        "figure.dpi":           150,    # screen preview
        "savefig.dpi":          600,    # PDF export
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "pdf.fonttype":         42,     # editable text in PDF (Type 42)
        "ps.fonttype":          42,
    })


# Standard figure widths (inches) for single-column and two-column layouts.
# elsarticle 3p text width is roughly 6.5"; single-column 3.4"–3.5".
SINGLE_COL_W = 3.5
TWO_COL_W    = 7.0


def thin_y_grid(ax):
    """Apply the project's standard 'minimal horizontal grid' look."""
    ax.grid(axis="y", linewidth=0.5, alpha=0.18, linestyle=(0, (1, 2)))
    ax.set_axisbelow(True)
