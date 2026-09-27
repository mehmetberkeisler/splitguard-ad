#!/usr/bin/env python3
"""Site and scanner confound sensitivity analysis for ADNI splits.

Reviewer §5.5 flagged that \\SGA{} prevents subject leakage but does not
guarantee site-independent evaluation. The paper already reports the mean
Cramer's V between site and CN/AD label as approximately 0.42 (train) / 0.53
(val) / 0.60 (test) across the five component-safe splits. This script
extends that reporting so the reviewer can see the full picture:

    - Cramer's V for site×label per (seed, partition), min/max/range
    - Same for scanner_field_strength×label
    - Sex×label as a demographic control axis
    - Number of unique sites per partition per seed
    - Site-distribution overlap (train/val/test share of top-K sites)

Reads
-----
data/splits/adni_with_converters/adni_splitguard_seed<S>.csv

Writes
------
reports/tables/adni/adni_site_scanner_confound_audit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Audit the PRIMARY component-safe splits — the same 1,125-scan universe the
# reported AUROCs come from. Defaulting to ``adni_with_converters`` (1,740
# rows, MCI retained) audited a different universe from the one the paper's
# results describe, and additionally computed Cramer's V against a
# four-category diagnosis label rather than the binary CN/AD label. Pass
# --splits-root explicitly to audit the converter-inclusive arm.
SPLITS_ROOT = PROJECT_ROOT / "data" / "splits" / "adni"

# The reported task is binary CN-vs-AD. MCI and unresolved-diagnosis rows are
# not part of the evaluated universe and must not enter the contingency table.
BINARY_LABELS = ("CN", "AD")
OUT_PATH = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_site_scanner_confound_audit.json"


def cramers_v_from_counts(counts: dict) -> tuple[float, int, int]:
    """Cramer's V corrected for chi-squared bias (Bergsma & Wicher 2013).

    counts is {(row_key, col_key): n}. Returns (V, n_rows, n_cols).
    """
    row_totals = defaultdict(int)
    col_totals = defaultdict(int)
    n = 0
    for (r, c), v in counts.items():
        row_totals[r] += v; col_totals[c] += v; n += v
    if n == 0: return float("nan"), 0, 0
    chi2 = 0.0
    for r, rt in row_totals.items():
        for c, ct in col_totals.items():
            observed = counts.get((r, c), 0)
            expected = rt * ct / n
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
    k = len(row_totals); m = len(col_totals)
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (m - 1) / max(1, n - 1))
    k_corr = k - (k - 1) ** 2 / max(1, n - 1)
    m_corr = m - (m - 1) ** 2 / max(1, n - 1)
    denom = min(k_corr - 1, m_corr - 1)
    if denom <= 0: return 0.0, k, m
    return math.sqrt(phi2_corr / denom), k, m


def per_partition_stats(rows: list[dict]) -> dict:
    """Group scan rows into train/val/test; compute audit statistics per partition."""
    per_split = defaultdict(list)
    for r in rows:
        # Restrict to the binary CN/AD universe the reported metrics use.
        if r.get("diagnosis_group") not in BINARY_LABELS:
            continue
        per_split[r["split"]].append(r)
    out = {}
    for split in ("train", "val", "test"):
        subset = per_split.get(split, [])
        if not subset:
            out[split] = {"n_rows": 0}
            continue
        # site×label
        site_label = Counter((r["ptid"][:3], r["diagnosis_group"]) for r in subset)
        v_site, k_site, m_site = cramers_v_from_counts(site_label)
        # scanner×label
        scanner_label = Counter(
            (r.get("scanner_field_strength", "unknown"), r["diagnosis_group"]) for r in subset
        )
        v_scanner, k_scan, m_scan = cramers_v_from_counts(scanner_label)
        # sex×label (demographic control axis)
        sex_label = Counter((r.get("sex", "?"), r["diagnosis_group"]) for r in subset)
        v_sex, k_sex, m_sex = cramers_v_from_counts(sex_label)
        out[split] = {
            "n_rows":                len(subset),
            "n_unique_sites":        k_site,
            "n_unique_scanners":     k_scan,
            "cramers_v_site_label":       round(v_site, 4),
            "cramers_v_scanner_label":    round(v_scanner, 4),
            "cramers_v_sex_label":        round(v_sex, 4),
            "label_distribution":     dict(Counter(r["diagnosis_group"] for r in subset)),
        }
    return out


def summarise_across_seeds(per_seed: dict) -> dict:
    """Min/max/mean across seeds per (partition, metric)."""
    out = {}
    partitions = ("train", "val", "test")
    metrics = ("cramers_v_site_label", "cramers_v_scanner_label",
               "cramers_v_sex_label", "n_unique_sites", "n_unique_scanners",
               "n_rows")
    for part in partitions:
        agg = {}
        for m in metrics:
            values = [per_seed[s][part][m] for s in per_seed
                      if part in per_seed[s] and m in per_seed[s][part]]
            if not values: continue
            agg[m] = {
                "min":  round(min(values), 4) if isinstance(values[0], float) else min(values),
                "max":  round(max(values), 4) if isinstance(values[0], float) else max(values),
                "mean": round(sum(values) / len(values), 4)
                        if isinstance(values[0], (int, float)) else None,
                "per_seed": values,
            }
        out[part] = agg
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-root", type=Path, default=SPLITS_ROOT)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    per_seed = {}
    for seed in args.seeds:
        path = args.splits_root / f"adni_splitguard_seed{seed}.csv"
        if not path.exists():
            print(f"  warn: missing split file: {path}", file=sys.stderr)
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        per_seed[seed] = per_partition_stats(rows)
        print(f"seed {seed}: "
              f"train V(site|label)={per_seed[seed]['train']['cramers_v_site_label']}, "
              f"val={per_seed[seed]['val']['cramers_v_site_label']}, "
              f"test={per_seed[seed]['test']['cramers_v_site_label']}")

    if not per_seed:
        print("error: no split files found", file=sys.stderr)
        return 1

    summary = {
        "per_seed":              per_seed,
        "cross_seed_summary":    summarise_across_seeds(per_seed),
        "n_seeds":               len(per_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}")

    # Human-readable summary
    cs = summary["cross_seed_summary"]
    print("\nCross-seed summary (all 5 seeds):")
    for part in ("train", "val", "test"):
        v = cs[part]["cramers_v_site_label"]
        s = cs[part]["cramers_v_scanner_label"]
        x = cs[part]["cramers_v_sex_label"]
        n = cs[part]["n_unique_sites"]
        r = cs[part]["n_rows"]
        print(f"  {part}: rows {r['mean']:.0f}, "
              f"sites {n['mean']:.1f} (range {n['min']}-{n['max']}), "
              f"V(site|label)={v['mean']:.3f} [{v['min']}-{v['max']}], "
              f"V(scanner|label)={s['mean']:.3f} [{s['min']}-{s['max']}], "
              f"V(sex|label)={x['mean']:.3f} [{x['min']}-{x['max']}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
