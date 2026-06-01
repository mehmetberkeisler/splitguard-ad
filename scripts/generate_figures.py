#!/usr/bin/env python3
"""
Generate all paper figures from existing result JSON files.
Outputs directly to paper/ so the generated files match the LaTeX sources and
submission package. No new training required.

Figures produced:
  fig1_learning_curves.pdf/png   — Protocol A vs B val AUROC by epoch
  fig2_seed_stability.pdf/png    — Per-seed AUROC bar chart (A vs B)
  fig3_degradation_curve.pdf/png — 3-point degradation A → A' → B
  fig4_metric_comparison.pdf/png    — Metrics comparison
"""
import json, sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy"); sys.exit(1)

ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150,
})
RED   = "#C0392B"
BLUE  = "#2980B9"
GREEN = "#27AE60"
GREY  = "#7F8C8D"

# ── Load data ──────────────────────────────────────────────────────────────────
with open(ROOT / "reports" / "tables" / "inflation_gap_experiment.json") as f:
    ig = json.load(f)
with open(ROOT / "reports" / "tables" / "overnight_results.json") as f:
    on = json.load(f)

hist_a = ig["protocol_A_leaky"]["history"]   # 30 epochs
hist_b = ig["protocol_B_safe"]["history"]

# ── Fig 1: Learning curves ─────────────────────────────────────────────────────
def fig1():
    epochs = [h["epoch"] for h in hist_a]
    auc_a  = [h["val_auc"] for h in hist_a]
    auc_b  = [h["val_auc"] for h in hist_b]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(epochs, auc_a, color=RED,  lw=2,   label="Protocol A — Leaky (random image split)")
    ax.plot(epochs, auc_b, color=BLUE, lw=2,   label="Protocol B — SplitGuard (subject-safe)")
    ax.axhline(0.9996, color=RED,  lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.8422, color=BLUE, lw=0.8, ls="--", alpha=0.5)
    ax.annotate("AUROC = 0.9996", xy=(30, 0.9996), xytext=(22, 0.975),
                color=RED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.annotate("AUROC = 0.8422", xy=(30, 0.8422), xytext=(18, 0.78),
                color=BLUE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.set_xlim(1, 30); ax.set_ylim(0.55, 1.02)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Fig. 1 — Validation AUROC: Leaky vs. SplitGuard Protocol\n"
                 r"(ResNet-18, same architecture \& hyperparameters, seed = 42)", fontsize=10)

    # Shade difference
    ax.fill_between(epochs, auc_b, auc_a, alpha=0.08, color=RED,
                    label="Inflation region")
    ax.text(15, 0.92, r"$\Delta$AUROC $\approx$ 0.157", fontsize=10,
            color=RED, ha="center")

    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig1_learning_curves.{fmt}")
    plt.close(fig)
    print(f"  ✅ fig1_learning_curves saved")

# ── Fig 2: Seed stability bar chart ────────────────────────────────────────────
def fig2():
    # seed 42 from inflation_gap, seeds 0-3 from overnight
    seeds = [42, 0, 1, 2, 3]
    leaky_auroc = [ig["protocol_A_leaky"]["test_metrics"]["auroc"]]
    safe_auroc  = [ig["protocol_B_safe"]["test_metrics"]["auroc"]]
    for s in [0, 1, 2, 3]:
        sd = on["E1b"][f"seed{s}"]
        leaky_auroc.append(sd["leaky"]["auroc"])
        safe_auroc.append(sd["safe"]["auroc"])

    x = np.arange(len(seeds)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars_a = ax.bar(x - w/2, leaky_auroc, w, color=RED,  alpha=0.85,
                    label="Protocol A — Leaky", zorder=3)
    bars_b = ax.bar(x + w/2, safe_auroc,  w, color=BLUE, alpha=0.85,
                    label="Protocol B — SplitGuard", zorder=3)

    for bar in bars_a:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=7.5, color=RED)
    for bar in bars_b:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=7.5, color=BLUE)

    # Gap annotations
    for i, (la, sa) in enumerate(zip(leaky_auroc, safe_auroc)):
        ax.annotate("", xy=(x[i]+w/2, sa), xytext=(x[i]-w/2, la),
                    arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
        ax.text(x[i]+0.02, (la+sa)/2, f"Δ={round(la-sa,3)}", fontsize=7.5,
                color=GREY, va="center")

    summary = on["E1b"]["summary"]
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylim(0.75, 1.015)
    ax.legend(loc="lower center", fontsize=9, framealpha=0.9, ncol=2)
    ax.set_title(f"Fig. 2 — Per-Seed AUROC: "
                 f"$\\Delta$AUROC = {summary['auroc_gap_mean']} $\\pm$ {summary['auroc_gap_std']} "
                 f"($n$=5 seeds)", fontsize=10)
    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig2_seed_stability.{fmt}")
    plt.close(fig)
    print(f"  ✅ fig2_seed_stability saved")

# ── Fig 3: Degradation curve ───────────────────────────────────────────────────
def fig3():
    dc = on["E3"]["degradation_curve_auroc"]
    labels = ["Protocol A\n(Random image split)", "Protocol A'\n(Subject-only split)",
              "Protocol B\n(SplitGuard)"]
    values = [dc["A_leaky"], dc["Aprime_subject_only"], dc["B_splitguard"]]
    colors = [RED, "#E67E22", BLUE]
    gaps   = [round(values[0]-values[1],4), round(values[1]-values[2],4)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0,1,2], values, "o-", color=GREY, lw=2, zorder=2, markersize=10,
            markerfacecolor="white", markeredgewidth=2)
    for i, (v, c, l) in enumerate(zip(values, colors, labels)):
        ax.plot(i, v, "o", color=c, markersize=12, zorder=3)
        ax.text(i, v+0.007, f"{v:.4f}", ha="center", fontsize=11,
                fontweight="bold", color=c)

    # Annotate gaps
    ax.annotate("", xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(0.5, (values[0]+values[1])/2+0.005,
            f"Δ={gaps[0]}\n(subject ID leak)", ha="center", fontsize=9,
            color=RED, style="italic")

    ax.annotate("", xy=(2, values[2]), xytext=(1, values[1]),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.5))
    ax.text(1.5, (values[1]+values[2])/2-0.015,
            f"Δ={gaps[1]}\n(component/near-dup)", ha="center", fontsize=9,
            color=GREY, style="italic")

    ax.set_xticks([0,1,2]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0.80, 1.02)
    ax.set_title("Three-Point Degradation Curve\n"
                 "Subject-identity leakage explains 98.3% of the inflation gap", fontsize=10)
    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig3_degradation_curve.{fmt}")
    plt.close(fig)
    print(f"  ✅ fig3_degradation_curve saved")

# ── Fig 4: Sensitivity/Specificity radar ───────────────────────────────────────
def fig4():
    ma = ig["protocol_A_leaky"]["test_metrics"]
    mb = ig["protocol_B_safe"]["test_metrics"]
    metrics = ["AUROC", "Balanced\nAccuracy", "Sensitivity\n(Recall)", "Specificity", "F1\n(Demented)"]
    vals_a  = [ma["auroc"], ma["balanced_accuracy"], ma["sensitivity"], ma["specificity"], ma["f1_demented"]]
    vals_b  = [mb["auroc"], mb["balanced_accuracy"], mb["sensitivity"], mb["specificity"], mb["f1_demented"]]

    x  = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ba = ax.bar(x - w/2, vals_a, w, color=RED,  alpha=0.82, label="Protocol A — Leaky")
    bb = ax.bar(x + w/2, vals_b, w, color=BLUE, alpha=0.82, label="Protocol B — SplitGuard")

    for bar, v in zip(ba, vals_a):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=RED, fontweight="bold")
    for bar, v in zip(bb, vals_b):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=BLUE, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0.5, 1.08); ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Fig. 4 — Test Metrics: Leaky vs. SplitGuard Protocol\n"
                 "(Same ResNet-18, seed = 42)", fontsize=10)

    # Annotate sensitivity collapse
    ax.annotate("Sensitivity collapse:\n99.8% → 60.8%",
                xy=(x[2]+w/2, mb["sensitivity"]),
                xytext=(x[2]+0.8, 0.68),
                fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))

    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig4_metric_comparison.{fmt}")
    plt.close(fig)
    print(f"  ✅ fig4_metric_comparison saved")

# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating all paper figures...")
    fig1(); fig2(); fig3(); fig4()
    print(f"\nAll figures saved to: {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}")
