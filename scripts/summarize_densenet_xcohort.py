#!/usr/bin/env python3
"""Summarize DenseNet-121 cross-cohort sensitivity-arm results.

Reads the per-seed JSON files emitted by `run_densenet_kaggle_oasis.sh`:
  reports/tables/inflation_gap_experiment__densenet121__seed*.json
  reports/tables/oasis1_inflation_gap_experiment__densenet121__seed*.json

Outputs:
  - reports/tables/densenet_xcohort_summary.json
      (machine-readable summary: per-seed AUROC table + paired-seed mean,
       SD, bootstrap interval per cohort, plus inflation-gap interval)
  - Plain-text summary to stdout suitable for pasting into the paper

Usage:
  python scripts/summarize_densenet_xcohort.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

SEEDS = [0, 1, 2, 3, 42]
N_BOOT = 10_000


def paired_bootstrap_interval(leaky, safe, n_boot=N_BOOT, alpha=0.05, rng=None):
    """Paired-seed bootstrap. Returns (point_mean, ci_lo, ci_hi, direction_preserved_count)."""
    rng = rng or np.random.default_rng(20260603)
    leaky = np.asarray(leaky, dtype=float)
    safe = np.asarray(safe, dtype=float)
    diff = leaky - safe
    n = len(diff)
    boot_means = np.empty(n_boot)
    pos_count = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bm = diff[idx].mean()
        boot_means[b] = bm
        if bm > 0:
            pos_count += 1
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(diff.mean()), lo, hi, pos_count


def cohort_bootstrap(values, n_boot=N_BOOT, alpha=0.05, rng=None):
    """Marginal bootstrap on a single array of per-seed AUROCs."""
    rng = rng or np.random.default_rng(20260603)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[b] = values[idx].mean()
    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def load_cohort(prefix: str, label: str):
    """Read all per-seed JSONs for the named cohort. Returns dict with per-protocol AUROC arrays + per-metric leaky/safe arrays."""
    leaky_auroc, safe_auroc = [], []
    leaky_sens, safe_sens = [], []
    leaky_spec, safe_spec = [], []
    leaky_baccu, safe_baccu = [], []
    found_seeds = []
    for seed in SEEDS:
        path = TABLES_DIR / f"{prefix}__densenet121__seed{seed}.json"
        if not path.exists():
            print(f"  ! missing: {path.name}", file=sys.stderr)
            continue
        d = json.loads(path.read_text())
        a = d["protocol_A_leaky"]["test_metrics"]
        b = d["protocol_B_safe"]["test_metrics"]
        leaky_auroc.append(a["auroc"]); safe_auroc.append(b["auroc"])
        leaky_sens.append(a["sensitivity"]); safe_sens.append(b["sensitivity"])
        leaky_spec.append(a["specificity"]); safe_spec.append(b["specificity"])
        leaky_baccu.append(a["balanced_accuracy"]); safe_baccu.append(b["balanced_accuracy"])
        found_seeds.append(seed)

    if not found_seeds:
        print(f"  ! NO results found for {label}", file=sys.stderr)
        return None

    print(f"\n── {label} (DenseNet-121, n={len(found_seeds)} seeds: {found_seeds}) ──")
    summary = {"label": label, "n_seeds": len(found_seeds), "seeds": found_seeds}
    for metric_name, leaky, safe in [
        ("AUROC", leaky_auroc, safe_auroc),
        ("Balanced acc.", leaky_baccu, safe_baccu),
        ("Sensitivity", leaky_sens, safe_sens),
        ("Specificity", leaky_spec, safe_spec),
    ]:
        l_mean, l_sd, l_lo, l_hi = cohort_bootstrap(leaky)
        s_mean, s_sd, s_lo, s_hi = cohort_bootstrap(safe)
        gap_mean, gap_lo, gap_hi, dir_pos = paired_bootstrap_interval(leaky, safe)
        print(f"  {metric_name:<14}  leaky={l_mean:.3f}±{l_sd:.3f}  safe={s_mean:.3f}±{s_sd:.3f}  "
              f"gap={gap_mean:+.3f} [{gap_lo:+.3f}, {gap_hi:+.3f}]  dir+={dir_pos}/{N_BOOT}")
        summary[metric_name.lower().replace(' ', '_').replace('.', '')] = {
            "leaky_mean": l_mean, "leaky_sd": l_sd, "leaky_ci": [l_lo, l_hi],
            "safe_mean":  s_mean, "safe_sd":  s_sd, "safe_ci":  [s_lo, s_hi],
            "gap_mean":   gap_mean, "gap_ci":  [gap_lo, gap_hi],
            "gap_direction_preserved": dir_pos,
        }
    return summary


def main() -> int:
    print("=" * 72)
    print("DenseNet-121 cross-cohort sensitivity arm — summary")
    print("=" * 72)

    out = {}
    for prefix, label in [
        ("inflation_gap_experiment",        "Tier 1 — Kaggle pedagogical"),
        ("oasis1_inflation_gap_experiment", "Tier 2 — OASIS-1 replication"),
    ]:
        summary = load_cohort(prefix, label)
        if summary:
            out[label] = summary

    out_path = TABLES_DIR / "densenet_xcohort_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote: {out_path}")

    # Ready-to-paste LaTeX paragraph fragments
    print("\n" + "=" * 72)
    print("READY-TO-PASTE LaTeX fragments")
    print("=" * 72)
    n_boot_tex = f"{N_BOOT:,}".replace(",", "{,}")  # 10{,}000
    for label, summary in out.items():
        a = summary["auroc"]
        dir_tex = f"{a['gap_direction_preserved']:,}".replace(",", "{,}")
        print(f"\n% {label}")
        print(f"DenseNet-121 replication on this cohort yields an inflation gap of "
              f"$+{a['gap_mean']:.3f}$ AUROC "
              f"$[{a['gap_ci'][0]:+.3f}, {a['gap_ci'][1]:+.3f}]$, "
              f"with direction preserved in {dir_tex}$/${n_boot_tex} "
              f"paired-seed bootstrap resamples "
              f"(leaky $={a['leaky_mean']:.3f} \\pm {a['leaky_sd']:.3f}$, "
              f"safe $={a['safe_mean']:.3f} \\pm {a['safe_sd']:.3f}$ across "
              f"$n={summary['n_seeds']}$ seeds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
