#!/usr/bin/env python3
"""Generate the four legacy paper figures with a clean publication style.

Outputs to paper/:
  fig1_learning_curves.pdf/png   — Tier-1 JPEG validation AUROC by epoch
  fig2_seed_stability.pdf/png    — Per-seed AUROC bar chart (A vs B)
  fig3_degradation_curve.pdf/png — three-point degradation A → B → C
  fig4_metric_comparison.pdf/png — Tier-1 test-metric comparison

Conventions (match scripts/generate_cross_cohort_figure.py and
scripts/generate_subgroup_figure.py for cross-figure consistency):
  - title goes inside the axes (loc="left"), so the LaTeX \\caption is
    not duplicated;
  - top and right spines hidden;
  - dotted grid only on Y;
  - savefig with bbox_inches="tight" so nothing is cropped;
  - light box-style insets for in-figure annotations, never overlapping
    bars or curves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy"); sys.exit(1)

ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style (publication-grade, ASCII-safe) ───────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 180,
})

# Single colour vocabulary across all figures
LEAKY     = "#c44e52"
SPLIT     = "#4c72b0"
NEUTRAL   = "#7f8c8d"
INTER     = "#5b8a55"

# ── Load result data ──────────────────────────────────────────────────────
with (ROOT / "reports" / "tables" / "inflation_gap_experiment.json").open() as f:
    ig = json.load(f)
with (ROOT / "reports" / "tables" / "overnight_results.json").open() as f:
    on = json.load(f)

hist_a = ig["protocol_A_leaky"]["history"]
hist_b = ig["protocol_B_safe"]["history"]


def _save(fig, stem: str) -> None:
    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote paper/{stem}.pdf + paper/{stem}.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 — Learning curves (validation AUROC by epoch)
# ─────────────────────────────────────────────────────────────────────────
def fig1():
    epochs = [h["epoch"] for h in hist_a]
    auc_a  = [h["val_auc"] for h in hist_a]
    auc_b  = [h["val_auc"] for h in hist_b]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))

    ax.fill_between(epochs, auc_b, auc_a, alpha=0.10, color=LEAKY,
                    linewidth=0, zorder=1)
    ax.plot(epochs, auc_a, color=LEAKY, lw=2.0, zorder=3,
            label="Protocol A — Random image split (leaky)")
    ax.plot(epochs, auc_b, color=SPLIT, lw=2.0, zorder=3,
            label="Protocol C — SplitGuard (subject-safe)")

    # Endpoint markers
    ax.scatter([epochs[-1]], [auc_a[-1]], s=28, color=LEAKY, zorder=4)
    ax.scatter([epochs[-1]], [auc_b[-1]], s=28, color=SPLIT, zorder=4)
    ax.annotate(f"AUROC = {auc_a[-1]:.4f}",
                xy=(epochs[-1], auc_a[-1]),
                xytext=(epochs[-1] + 0.6, auc_a[-1] + 0.005),
                ha="left", va="center", color=LEAKY, fontsize=8.5)
    ax.annotate(f"AUROC = {auc_b[-1]:.4f}",
                xy=(epochs[-1], auc_b[-1]),
                xytext=(epochs[-1] + 0.6, auc_b[-1] - 0.005),
                ha="left", va="center", color=SPLIT, fontsize=8.5)

    # Inflation-gap annotation in a clean inset
    ax.text(0.04, 0.93,
            r"$\Delta$AUROC $\approx$ 0.157",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color=LEAKY, weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=LEAKY, linewidth=0.6, alpha=0.92))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.set_xlim(0.5, max(epochs) + 4.5)
    ax.set_ylim(0.55, 1.02)
    ax.set_title("Learning curves: Tier-1 JPEG benchmark, ResNet-18, seed=42")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", ncol=1)

    _save(fig, "fig1_learning_curves")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-seed AUROC bar chart (A vs B), with inflation gaps
# ─────────────────────────────────────────────────────────────────────────
def fig2():
    seeds = [42, 0, 1, 2, 3]
    leaky_auroc = [ig["protocol_A_leaky"]["test_metrics"]["auroc"]]
    safe_auroc  = [ig["protocol_B_safe"]["test_metrics"]["auroc"]]
    for s in [0, 1, 2, 3]:
        sd = on["E1b"][f"seed{s}"]
        leaky_auroc.append(sd["leaky"]["auroc"])
        safe_auroc.append(sd["safe"]["auroc"])

    x = np.arange(len(seeds))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w/2, leaky_auroc, w, color=LEAKY,
           edgecolor="black", linewidth=0.6,
           label="Protocol A — Leaky", zorder=3)
    ax.bar(x + w/2, safe_auroc, w, color=SPLIT,
           edgecolor="black", linewidth=0.6,
           label="Protocol C — SplitGuard", zorder=3)

    # Leaky bars all hit ~1.0; print value just *inside* the bar top in white
    # so there is no overlap with the per-seed Δ label above.
    for i, v in enumerate(leaky_auroc):
        ax.text(x[i] - w/2, v - 0.008, f"{v:.3f}",
                ha="center", va="top", fontsize=7.5, color="white",
                weight="bold")
    # Safe-protocol values printed above their bars (plenty of headroom).
    for i, v in enumerate(safe_auroc):
        ax.text(x[i] + w/2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=SPLIT)

    # Per-seed gap: small italic label centred above the pair, well clear of
    # any bar top now that ylim runs to 1.06.
    for i, (la, sa) in enumerate(zip(leaky_auroc, safe_auroc)):
        ax.text(x[i], la + 0.012, fr"$\Delta$ = {la - sa:+.3f}",
                ha="center", va="bottom", fontsize=7.5, color=NEUTRAL,
                style="italic")

    summary = on["E1b"]["summary"]
    gap_mean = summary["auroc_gap_mean"]
    gap_std  = summary["auroc_gap_std"]

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylim(0.78, 1.06)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(
        fr"Per-seed inflation gap (n=5).  Mean $\Delta$AUROC "
        fr"= {gap_mean:.3f} $\pm$ {gap_std:.3f}"
    )
    # Legend below the axes so it cannot collide with any bar.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, frameon=False)

    _save(fig, "fig2_seed_stability")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3 — Three-point degradation curve A → B → C
# ─────────────────────────────────────────────────────────────────────────
def fig3():
    dc = on["E3"]["degradation_curve_auroc"]
    values = [dc["A_leaky"], dc["Aprime_subject_only"], dc["B_splitguard"]]
    colors = [LEAKY, INTER, SPLIT]
    labels = ["Protocol A\nRandom image split",
              "Protocol B\nSubject-only split",
              "Protocol C\nSplitGuard (component-safe)"]
    gaps = [values[0] - values[1], values[1] - values[2]]

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot([0, 1, 2], values, color=NEUTRAL, lw=1.6, zorder=2)
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.scatter(i, v, s=120, color=c, edgecolor="black", linewidth=0.6,
                   zorder=3)
        ax.text(i, v + 0.012, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, color=c, weight="bold")

    # Annotate the two transitions
    ax.annotate("", xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle="-|>", color=LEAKY, lw=1.3))
    ax.text(0.5, (values[0] + values[1]) / 2 + 0.005,
            fr"$\Delta$ = {gaps[0]:.3f}" + "\n(subject ID leak)",
            ha="center", va="bottom", fontsize=8.5, color=LEAKY, style="italic")

    ax.annotate("", xy=(2, values[2]), xytext=(1, values[1]),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=1.3))
    ax.text(1.5, (values[1] + values[2]) / 2 - 0.012,
            fr"$\Delta$ = {gaps[1]:.3f}" + "\n(component/near-dup)",
            ha="center", va="top", fontsize=8.5, color=NEUTRAL, style="italic")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.80, 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title("Three-point degradation: subject-identity leakage carries the gap")

    _save(fig, "fig3_degradation_curve")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 — Test-metric comparison (Tier 1)
# ─────────────────────────────────────────────────────────────────────────
def fig4():
    ma = ig["protocol_A_leaky"]["test_metrics"]
    mb = ig["protocol_B_safe"]["test_metrics"]
    metrics = ["AUROC", "Balanced\naccuracy",
               "Sensitivity\n(recall)", "Specificity", "F1\n(demented)"]
    vals_a = [ma["auroc"], ma["balanced_accuracy"],
              ma["sensitivity"], ma["specificity"], ma["f1_demented"]]
    vals_b = [mb["auroc"], mb["balanced_accuracy"],
              mb["sensitivity"], mb["specificity"], mb["f1_demented"]]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(x - w/2, vals_a, w, color=LEAKY,
           edgecolor="black", linewidth=0.6, label="Protocol A — Leaky", zorder=3)
    ax.bar(x + w/2, vals_b, w, color=SPLIT,
           edgecolor="black", linewidth=0.6, label="Protocol C — SplitGuard", zorder=3)

    for i, v in enumerate(vals_a):
        ax.text(x[i] - w/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=LEAKY, weight="bold")
    for i, v in enumerate(vals_b):
        ax.text(x[i] + w/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=SPLIT, weight="bold")

    # Sensitivity collapse annotation, placed cleanly INSIDE the axes top-left.
    ax.text(0.02, 0.97,
            f"Sensitivity collapse: {ma['sensitivity']:.1%} → {mb['sensitivity']:.1%}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=SPLIT, weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=SPLIT, linewidth=0.6, alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.50, 1.14)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title("Tier-1 test metrics: leaky vs. SplitGuard protocol")
    # Legend below the axes so it cannot collide with the F1 bars.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=2, frameon=False)

    _save(fig, "fig4_metric_comparison")


if __name__ == "__main__":
    print("Regenerating Tier-1 paper figures with clean publication style...")
    fig1(); fig2(); fig3(); fig4()
    print("Done.")
