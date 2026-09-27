#!/usr/bin/env python3
"""Draw the provenance-degradation curve reported in the manuscript.

The result exists as a table in the paper, which is the wrong shape for it: the
point is that two protocols coincide at zero provenance loss and separate as
identifiers disappear, and a separation is read from a curve rather than
counted off rows. Two panels.

(a) Participants whose scans straddle a partition boundary, against the
    fraction of identifiers deleted. This is the direct measurement -- it
    counts admitted leakage without routing the question through a model, so
    it carries none of the seed-variance caveats the AUROC comparisons do.

(b) The share of that leakage the leakage graph prevents. Plotted separately
    because it is a ratio and does not belong on the same axis as a count,
    and because its shape is the argument: the benefit peaks in the regime
    where redistributed benchmarks actually sit and decays once too few
    identifiers survive for any rule to bind on.

Both protocols are assigned by the same class-stratified bin-packer, so the
difference is attributable to the grouping rather than to the assignment step.

Usage
-----
    python3 scripts/generate_provenance_figure.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _publication_style import (  # noqa: E402
    BAND_ALPHA, CAPSIZE, INK, MARKER_SIZE, MUTED, NEUTRAL, PROTOCOL_COLOR,
    PROTOCOL_LABEL, PROTOCOL_LINESTYLE, PROTOCOL_MARKER, SERIES_LW, TWO_COL_W,
    apply_publication_style, protocol_handles,
)

SOURCE = ROOT / "reports" / "tables" / "adni" / "adni_provenance_degradation.json"
OUTPUT = ROOT / "paper" / "fig12_provenance_degradation.pdf"

KEY = {"subject_only": "B", "component_safe": "C"}
LABELS = {proto: PROTOCOL_LABEL[k] for proto, k in KEY.items()}
STYLE = {
    proto: dict(color=PROTOCOL_COLOR[k], marker=PROTOCOL_MARKER[k],
                linestyle=PROTOCOL_LINESTYLE[k])
    for proto, k in KEY.items()
}


def aggregate(records: list[dict], field: str) -> dict[str, dict[float, tuple[float, float]]]:
    """protocol -> deletion level -> (mean, sd) over seeds."""
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in records:
        grouped[(r["protocol"], r["deletion_fraction"])].append(r[field])
    out: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    for (protocol, level), values in grouped.items():
        sd = st.stdev(values) if len(values) > 1 else 0.0
        out[protocol][level] = (st.mean(values), sd)
    return out


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}. Run scripts/run_provenance_degradation.py first.")
    apply_publication_style()
    import matplotlib.pyplot as plt

    records = json.loads(SOURCE.read_text())["records"]
    straddling = aggregate(records, "n_subjects_straddling_partitions")
    levels = sorted(straddling["subject_only"])
    xs = [100 * v for v in levels]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TWO_COL_W, 3.1))

    # ── (a) admitted leakage ────────────────────────────────────────────
    for protocol in ("subject_only", "component_safe"):
        ys = [straddling[protocol][v][0] for v in levels]
        es = [straddling[protocol][v][1] for v in levels]
        ax_a.errorbar(xs, ys, yerr=es, capsize=CAPSIZE, lw=SERIES_LW,
                      markersize=MARKER_SIZE, markeredgecolor="white",
                      markeredgewidth=0.6, label=LABELS[protocol],
                      **STYLE[protocol])

    ax_a.set_xlabel("Participant identifiers deleted (%)")
    ax_a.set_ylabel("Participants straddling a partition")
    ax_a.set_title("(a) Leakage admitted")
    ax_a.grid(axis="y")
    ax_a.set_xlim(-4, 104)
    ax_a.set_ylim(bottom=-6)

    # The origin is the paper's null result and is easy to miss on a curve
    # that rises steeply just after it, so it is annotated rather than left
    # for the reader to infer from two overlapping markers.
    ax_a.annotate("identical at zero loss", xy=(0, 0), xytext=(30, 8),
                  fontsize=7.5, color=MUTED, va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=NEUTRAL,
                                  shrinkA=2, shrinkB=3,
                                  connectionstyle="arc3,rad=-0.15"))

    # ── (b) prevented share ─────────────────────────────────────────────
    prevented = []
    for v in levels:
        b = straddling["subject_only"][v][0]
        c = straddling["component_safe"][v][0]
        prevented.append(100 * (b - c) / b if b else 0.0)

    # A ratio of the two protocols rather than a protocol, so it is in ink.
    ax_b.plot(xs, prevented, color=INK, marker="D", markersize=MARKER_SIZE,
              markeredgecolor="white", markeredgewidth=0.6, lw=SERIES_LW)
    ax_b.fill_between(xs, 0, prevented, color=NEUTRAL, alpha=BAND_ALPHA,
                      linewidth=0)
    ax_b.set_xlabel("Participant identifiers deleted (%)")
    ax_b.set_ylabel("Leakage prevented by the graph (%)")
    ax_b.set_title("(b) What the extra edges buy")
    ax_b.grid(axis="y")
    ax_b.set_xlim(-4, 104)
    ax_b.set_ylim(0, 62)

    # Values sit above their markers, except after a steep drop, where a
    # centred label lands on the incoming segment; those move to the right.
    for i, (x, y) in enumerate(zip(xs, prevented)):
        if y <= 0:
            continue
        steep_drop = i > 0 and prevented[i - 1] - y > 5
        ax_b.annotate(f"{y:.0f}", xy=(x, y),
                      xytext=(5, 4) if steep_drop else (0, 5),
                      textcoords="offset points",
                      ha="left" if steep_drop else "center",
                      fontsize=7.5, color=INK)

    # Legend below both panels, as in every other multi-panel figure.
    fig.tight_layout(pad=0.5, rect=(0, 0.07, 1, 1))
    fig.legend(handles=protocol_handles(["B", "C"]), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    fig.savefig(OUTPUT.with_suffix(".png"))
    plt.close(fig)
    print(f"  wrote {OUTPUT.name} + .png")
    for x, y in zip(xs, prevented):
        print(f"    {x:5.0f}% deleted -> {y:5.1f}% prevented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
