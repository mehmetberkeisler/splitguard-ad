#!/usr/bin/env python3
"""Tier-1 figures for the SplitGuard-AD paper, publication-quality.

Produces:
  paper/fig1_learning_curves.{pdf,png}   — validation AUROC by epoch
  paper/fig2_seed_stability.{pdf,png}    — per-seed AUROC bars (Tier-1)
  paper/fig3_degradation_curve.{pdf,png} — three-point degradation A→B→C
  paper/fig4_metric_comparison.{pdf,png} — Tier-1 test-metric bars

Style is shared with the other matplotlib generators via
scripts/_publication_style.py: STIX serif (matches the LaTeX paper),
Wong colorblind-safe palette, minimal horizontal grid, no
top/right spines, 600-dpi PDF export with Type-42 embedded fonts so
the text remains editable in the published PDF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid,
    LEAKY, SPLIT, INTER, NEUTRAL,
    SINGLE_COL_W, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Data load ────────────────────────────────────────────────────────────
with (ROOT / "reports" / "tables" / "inflation_gap_experiment.json").open() as f:
    ig = json.load(f)
with (ROOT / "reports" / "tables" / "overnight_results.json").open() as f:
    on = json.load(f)

hist_a = ig["protocol_A_leaky"]["history"]
hist_b = ig["protocol_B_safe"]["history"]


def _save(fig, stem: str) -> None:
    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{fmt}")
    plt.close(fig)
    print(f"  wrote paper/{stem}.pdf + paper/{stem}.png")


# ── Figure 1 — Learning curves (validation AUROC by epoch) ───────────────
def fig1():
    epochs = [h["epoch"] for h in hist_a]
    auc_a  = [h["val_auc"] for h in hist_a]
    auc_b  = [h["val_auc"] for h in hist_b]

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.6))

    ax.fill_between(epochs, auc_b, auc_a, alpha=0.08, color=LEAKY,
                    linewidth=0, zorder=1)
    ax.plot(epochs, auc_a, color=LEAKY, lw=1.6, zorder=3,
            label="Protocol A — Random (leaky)")
    ax.plot(epochs, auc_b, color=SPLIT, lw=1.6, zorder=3,
            label="Protocol C — SplitGuard-AD")
    # endpoint dots
    ax.scatter([epochs[-1]], [auc_a[-1]], s=12, color=LEAKY, zorder=4)
    ax.scatter([epochs[-1]], [auc_b[-1]], s=12, color=SPLIT, zorder=4)
    ax.text(epochs[-1] + 0.4, auc_a[-1] - 0.005, f"{auc_a[-1]:.3f}",
            ha="left", va="center", color=LEAKY, fontsize=7.5)
    ax.text(epochs[-1] + 0.4, auc_b[-1] - 0.005, f"{auc_b[-1]:.3f}",
            ha="left", va="center", color=SPLIT, fontsize=7.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.set_xlim(0.5, max(epochs) + 4)
    ax.set_ylim(0.55, 1.02)
    thin_y_grid(ax)
    # Legend BELOW the axes, two-column, so it can never overlap data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False)
    fig.tight_layout()
    _save(fig, "fig1_learning_curves")


# ── Figure 2 — Per-seed AUROC bars (Tier 1) ───────────────────────────────
def fig2():
    seeds = [42, 0, 1, 2, 3]
    leaky_auroc = [ig["protocol_A_leaky"]["test_metrics"]["auroc"]]
    safe_auroc  = [ig["protocol_B_safe"]["test_metrics"]["auroc"]]
    for s in [0, 1, 2, 3]:
        sd = on["E1b"][f"seed{s}"]
        leaky_auroc.append(sd["leaky"]["auroc"])
        safe_auroc.append(sd["safe"]["auroc"])

    summary = on["E1b"]["summary"]
    gap_mean = summary["auroc_gap_mean"]
    gap_std  = summary["auroc_gap_std"]

    x = np.arange(len(seeds))
    w = 0.36

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.6))
    ax.bar(x - w/2, leaky_auroc, w, color=LEAKY,
           edgecolor="none", label="Protocol A (leaky)", zorder=3)
    ax.bar(x + w/2, safe_auroc, w, color=SPLIT,
           edgecolor="none", label="Protocol C (SplitGuard-AD)", zorder=3)

    # subtle value labels, only on the SafeGuard bars (leaky bars saturate at ~1)
    for i, v in enumerate(safe_auroc):
        ax.text(x[i] + w/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color=SPLIT)

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylim(0.78, 1.04)
    thin_y_grid(ax)
    ax.text(0.01, 0.99,
            rf"$\Delta$AUROC $= {gap_mean:.3f}\,\pm\,{gap_std:.3f}$ (n=5)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=NEUTRAL)
    # Legend BELOW the axes, two-column, so it can never overlap bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_seed_stability")


# ── Figure 3 — Three-point degradation curve A → B → C ───────────────────
def fig3():
    dc = on["E3"]["degradation_curve_auroc"]
    values = [dc["A_leaky"], dc["Aprime_subject_only"], dc["B_splitguard"]]
    colors = [LEAKY, INTER, SPLIT]
    labels = ["A\nRandom\n(leaky)",
              "B\nSubject‑only",
              "C\nComponent‑safe"]
    gaps = [values[0] - values[1], values[1] - values[2]]

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.6))
    ax.plot([0, 1, 2], values, color=NEUTRAL, lw=1.0, zorder=2)
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.scatter(i, v, s=55, color=c, edgecolor="white", linewidth=1.0,
                   zorder=3)
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=c)

    # Transition annotations sit on top of the connecting line; a thin
    # white halo keeps text crisp where it crosses the grey line.
    _halo = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92)
    ax.annotate(rf"$\Delta = {gaps[0]:.3f}$" + "\nsubject-ID leak",
                xy=(0.5, (values[0] + values[1]) / 2),
                ha="center", va="center", fontsize=7.5, color=LEAKY,
                bbox=_halo)
    ax.annotate(rf"$\Delta = {gaps[1]:.3f}$" + "\ncomponent/near-dup",
                xy=(1.5, (values[1] + values[2]) / 2 - 0.012),
                ha="center", va="top", fontsize=7.5, color=NEUTRAL,
                bbox=_halo)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.80, 1.02)
    thin_y_grid(ax)
    fig.tight_layout()
    _save(fig, "fig3_degradation_curve")


# ── Figure 4 — Test metric comparison (Tier 1) ───────────────────────────
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
    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.7))
    ax.bar(x - w/2, vals_a, w, color=LEAKY, edgecolor="none",
           label="Protocol A (leaky)", zorder=3)
    ax.bar(x + w/2, vals_b, w, color=SPLIT, edgecolor="none",
           label="Protocol C (SplitGuard-AD)", zorder=3)
    # value labels only on SplitGuard side (leaky saturates near 1.0)
    for i, v in enumerate(vals_b):
        ax.text(x[i] + w/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color=SPLIT)
    # Δsensitivity highlight — clinical headline as plain text below the
    # bar pair (no arrow leader: the pairing already implies the comparison,
    # and the leader was clipping the SplitGuard sensitivity bar).
    ax.text(x[2], 0.48,
            rf"$\Delta$sens $= -{(vals_a[2]-vals_b[2])*100:.1f}$ pp",
            ha="center", va="bottom", fontsize=8, color=NEUTRAL)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 1.10)
    thin_y_grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False)
    fig.tight_layout()
    _save(fig, "fig4_metric_comparison")


if __name__ == "__main__":
    print("Regenerating Tier-1 paper figures (publication-quality style)...")
    fig1(); fig2(); fig3(); fig4()
    print("Done.")
