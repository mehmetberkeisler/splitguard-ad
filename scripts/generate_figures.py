#!/usr/bin/env python3
"""Tier-1 figures for the SplitGuard-AD paper, publication-quality.

Tier 1 is trained under three protocols that differ only in what they group
by: random images (A), the filename key the release ships (B'), and the
participant identity recovered in the provenance audit (C'). Every figure here
shows all three, because the point of the tier is that B' looks honest from
inside the release and is not.

Produces:
  paper/fig05_learning_curves.{pdf,png}    — validation AUROC by epoch
  paper/fig04a_seed_stability.{pdf,png}    — per-seed AUROC bars
  paper/fig08_degradation_curve.{pdf,png}  — three-point degradation A → B' → C'
  paper/fig04b_metric_comparison.{pdf,png} — test-metric bars

Input is the per-seed JSON written by scripts/run_inflation_gap.py for each
grouping rule (``tier1_<arch>_<rule>_seed<seed>.json``), so the figures follow
whichever run directory is passed with --results-dir.

Style is shared with the other matplotlib generators via
scripts/_publication_style.py: STIX serif (matches the LaTeX paper),
Wong colorblind-safe palette, minimal horizontal grid, no
top/right spines, 600-dpi PDF export with Type-42 embedded fonts so
the text remains editable in the published PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project-shared style ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _publication_style import (
    apply_publication_style, thin_y_grid,
    INK, MUTED, NEUTRAL, PROTOCOL_COLOR, PROTOCOL_MARKER, PROTOCOL_LINESTYLE,
    MARKER_SIZE, SERIES_LW, TWO_COL_W,
)
apply_publication_style()

import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1, 2, 3]
RULES = [("random", "A"), ("filename_subject", "B"), ("true_participant", "C")]
# Tier-1 protocol names differ from the ADNI ones: B groups by the release's
# filename key, C by the participant identity recovered from OASIS-1.
LABEL = {"A": "A: random images",
         "B": "B′: filename key",
         "C": "C′: recovered participant"}


def load_results(results_dir: Path, arch: str) -> dict[str, dict[int, dict]]:
    """protocol key -> seed -> the run block for that protocol."""
    out: dict[str, dict[int, dict]] = {key: {} for _, key in RULES}
    label_to_key = {rule: key for rule, key in RULES}
    for seed in SEEDS:
        for rule, _ in RULES:
            path = results_dir / f"tier1_{arch}_{rule}_seed{seed}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            for block in ("protocol_A_leaky", "protocol_B_safe"):
                run = payload.get(block)
                if run and run.get("label") in label_to_key:
                    out[label_to_key[run["label"]]][seed] = run
    if not any(out[key] for _, key in RULES):
        raise SystemExit(f"No Tier-1 results in {results_dir}. Run the GPU programme's tier1_truth stage first.")
    return out


def _save(fig, stem: str) -> None:
    for fmt in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{fmt}")
    plt.close(fig)
    print(f"  wrote paper/{stem}.pdf + paper/{stem}.png")


def _present(results, seed=None):
    """Protocol keys that have a result, optionally for one seed."""
    return [key for _, key in RULES if (seed in results[key] if seed is not None else results[key])]


def _legend_below(ax, ncol):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False)


# ── Figure — Learning curves (validation AUROC by epoch, seed 42) ─────────
def learning_curves(results):
    keys = _present(results, seed=42)
    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.8))
    for key in keys:
        history = results[key][42]["history"]
        ax.plot([h["epoch"] for h in history], [h["val_auc"] for h in history],
                color=PROTOCOL_COLOR[key], linestyle=PROTOCOL_LINESTYLE[key],
                marker=PROTOCOL_MARKER[key], markersize=MARKER_SIZE, markevery=3,
                lw=SERIES_LW, label=LABEL[key], zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    thin_y_grid(ax)
    _legend_below(ax, len(keys))
    fig.tight_layout()
    _save(fig, "fig05_learning_curves")


# ── Figure — Per-seed AUROC bars ─────────────────────────────────────────
def seed_stability(results):
    keys = _present(results)
    seeds = [s for s in SEEDS if all(s in results[key] for key in keys)]
    x = np.arange(len(seeds))
    width = 0.8 / max(1, len(keys))

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.8))
    for i, key in enumerate(keys):
        values = [results[key][s]["test_metrics"]["auroc"] for s in seeds]
        offset = (i - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, values, width, color=PROTOCOL_COLOR[key],
               edgecolor="none", label=LABEL[key], zorder=3)
        for xi, v in zip(x, values):
            ax.text(xi + offset, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)

    lowest = min(results[keys[-1]][s]["test_metrics"]["auroc"] for s in seeds)
    if len(keys) == 3:
        gaps = [results["A"][s]["test_metrics"]["auroc"] - results["C"][s]["test_metrics"]["auroc"] for s in seeds]
        ax.text(0.01, 0.99, rf"$\Delta$AUROC (A$-$C$'$) $= {np.mean(gaps):.3f}\,\pm\,{np.std(gaps):.3f}$ (n={len(seeds)})",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_ylim(max(0.0, lowest - 0.10), 1.08)
    thin_y_grid(ax)
    _legend_below(ax, len(keys))
    fig.tight_layout()
    _save(fig, "fig04a_seed_stability")


# ── Figure — Three-point degradation A → B' → C' ─────────────────────────
def degradation_curve(results):
    keys = _present(results)
    if len(keys) < 3:
        print("  skipping fig08_degradation_curve: needs all three protocols")
        return
    means = [float(np.mean([r["test_metrics"]["auroc"] for r in results[key].values()])) for key in keys]
    gaps = [means[0] - means[1], means[1] - means[2]]

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.8))
    ax.plot([0, 1, 2], means, color=NEUTRAL, lw=1.0, zorder=2)
    for i, (key, value) in enumerate(zip(keys, means)):
        ax.plot(i, value, linestyle="none", marker=PROTOCOL_MARKER[key], markersize=7,
                color=PROTOCOL_COLOR[key], markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.text(i, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)

    halo = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92)
    ax.annotate(rf"$\Delta = {gaps[0]:.3f}$" + "\nremoved by the\nfilename key",
                xy=(0.5, (means[0] + means[1]) / 2), ha="center", va="center",
                fontsize=7.5, color=INK, bbox=halo)
    ax.annotate(rf"$\Delta = {gaps[1]:.3f}$" + "\nthe key could\nnot see",
                xy=(1.5, (means[1] + means[2]) / 2), ha="center", va="center",
                fontsize=7.5, color=MUTED, bbox=halo)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([LABEL[key].replace(": ", ":\n") for key in keys])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("Test AUROC (mean over seeds)")
    ax.set_ylim(min(means) - 0.08, max(means) + 0.06)
    thin_y_grid(ax)
    fig.tight_layout()
    _save(fig, "fig08_degradation_curve")


# ── Figure — Test metric comparison (seed 42) ────────────────────────────
def metric_comparison(results):
    keys = _present(results, seed=42)
    names = ["AUROC", "Balanced\naccuracy", "Sensitivity\n(recall)", "Specificity", "F1\n(demented)"]
    fields = ["auroc", "balanced_accuracy", "sensitivity", "specificity", "f1_demented"]
    x = np.arange(len(names))
    width = 0.8 / max(1, len(keys))

    fig, ax = plt.subplots(figsize=(TWO_COL_W, 2.9))
    lowest = 1.0
    for i, key in enumerate(keys):
        metrics = results[key][42]["test_metrics"]
        values = [metrics[f] for f in fields]
        lowest = min(lowest, min(values))
        offset = (i - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, values, width, color=PROTOCOL_COLOR[key],
               edgecolor="none", label=LABEL[key], zorder=3)
        for xi, v in zip(x, values):
            ax.text(xi + offset, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)

    if "A" in keys and keys[-1] != "A":
        drop = (results["A"][42]["test_metrics"]["sensitivity"]
                - results[keys[-1]][42]["test_metrics"]["sensitivity"]) * 100
        ax.text(0.01, 0.99, rf"$\Delta$sens $= -{drop:.1f}$ pp",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_ylim(max(0.0, lowest - 0.12), 1.12)
    thin_y_grid(ax)
    _legend_below(ax, len(keys))
    fig.tight_layout()
    _save(fig, "fig04b_metric_comparison")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reports" / "gpu" / "tier1")
    ap.add_argument("--arch", default="resnet18")
    args = ap.parse_args()

    print("Regenerating Tier-1 paper figures (publication-quality style)...")
    results = load_results(args.results_dir, args.arch)
    learning_curves(results)
    seed_stability(results)
    degradation_curve(results)
    metric_comparison(results)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
