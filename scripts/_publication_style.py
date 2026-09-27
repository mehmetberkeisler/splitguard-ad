"""Shared publication-quality matplotlib style for the SplitGuard-AD paper.

Loaded by every figure generator so all matplotlib outputs share one
visual identity:

  * Serif body font matching the LaTeX paper (Computer Modern / STIX)
  * Wong colorblind-safe palette (Nature Methods 2011)
  * Solid hairline grid on the value axis only
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

# Text never wears a series colour: the mark beside a label carries identity,
# and a light hue (orange measures 2.2:1 against white) is illegible as text.
INK   = "#1A1A1A"   # values, labels, axis text
MUTED = "#595959"   # secondary annotations (7:1 on white)

# Colour encodes protocol identity and nothing else, in every figure. Series
# that are not protocols (backbones, derived gaps, shares) are drawn in INK or
# NEUTRAL and told apart by line style and marker. Checked with the dataviz
# palette validator: adjacent CVD separation dE 11.0, normal-vision dE 18.7.
PROTOCOL_COLOR = {"A": LEAKY, "B": INTER, "C": SPLIT}
PROTOCOL_LABEL = {
    "A": "Protocol A (random)",
    "B": "Protocol B (subject-only)",
    "C": "Protocol C (component-safe)",
}
PROTOCOL_MARKER    = {"A": "o", "B": "s", "C": "^"}
PROTOCOL_LINESTYLE = {"A": "-", "B": (0, (4, 2)), "C": "-"}

# Mark specs shared by every figure.
SERIES_LW   = 1.4
ERROR_LW    = 1.0
CAPSIZE     = 2.0
MARKER_SIZE = 4.5
BAND_ALPHA  = 0.12
REF_LW      = 0.6
REF_STYLE   = (0, (3, 2))   # reference levels only; gridlines stay solid


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

        # Grid: solid hairline one step off the surface. Dotted or dashed
        # gridlines read as thresholds; reference levels use REF_STYLE.
        "axes.grid":            False,   # turn on explicitly per-axes
        "grid.color":           "#DADADA",
        "grid.alpha":           1.0,
        "grid.linewidth":       0.5,
        "grid.linestyle":       "-",

        # Text and chrome in ink rather than pure black
        "text.color":           INK,
        "axes.labelcolor":      INK,
        "axes.titlecolor":      INK,
        "axes.edgecolor":       INK,
        "xtick.color":          INK,
        "ytick.color":          INK,

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


def thin_y_grid(ax, axis="y"):
    """Solid hairline grid on the value axis, drawn beneath the data."""
    ax.grid(axis=axis)
    ax.set_axisbelow(True)


def reference_line(ax, value, orientation="h", color=NEUTRAL):
    """A reference level (zero, chance, a protocol's result): dashed hairline."""
    draw = ax.axhline if orientation == "h" else ax.axvline
    draw(value, color=color, linewidth=REF_LW, linestyle=REF_STYLE, zorder=1)


def protocol_handles(keys, kind="line"):
    """Legend handles for protocols, identical in every figure."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if kind == "patch":
        return [Patch(facecolor=PROTOCOL_COLOR[k], edgecolor="none",
                      label=PROTOCOL_LABEL[k]) for k in keys]
    return [Line2D([], [], color=PROTOCOL_COLOR[k], linewidth=SERIES_LW,
                   linestyle=PROTOCOL_LINESTYLE[k], marker=PROTOCOL_MARKER[k],
                   markersize=MARKER_SIZE, markeredgecolor="white",
                   markeredgewidth=0.6, label=PROTOCOL_LABEL[k])
            for k in keys]
