#!/usr/bin/env python3
"""Bootstrap 95% CIs on the ADNI inflation-gap result.

The 5-seed mean ± SD reported in
reports/tables/adni/adni_inflation_gap.csv is the canonical aggregate,
but reviewers will typically ask for a non-parametric CI on the
per-protocol AUROC and on the inflation gap itself. This script does a
paired bootstrap over seeds (same resampled seed indices across all
three protocols) so the gap CIs come out from a single resample loop.

Inputs
------
* ``reports/tables/adni/adni_inflation_gap.csv`` — the canonical
  15-row table from ``run_adni_inflation_gap.py``.

Outputs
-------
* ``reports/tables/adni/adni_inflation_gap_bootstrap.json`` —
  per-protocol 95% CIs and inflation-gap CIs (paired, percentile
  method, B=10000 by default).

Caveats
-------
The bootstrap distribution is over the 5 random-effect seeds. N=5 is
small, so CIs are wide; they should be reported as "an honest
non-parametric uncertainty quantification at this evidence stage"
rather than a precise interval estimate. The intended use is reviewer
defence ("here is what we can claim with 5 seeds") not a final
publication statistic.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_inflation_gap.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_inflation_gap_bootstrap.json"

PROTOCOL_ORDER = ("random", "subject_only", "component_safe")


def percentile(values, p: float) -> float:
    if not values:
        return float("nan")
    sv = sorted(values)
    rank = (len(sv) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sv) - 1)
    frac = rank - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def mean(values) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=95.0,
                        help="Two-sided percentile CI (95 → 2.5/97.5).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"Input table not found: {args.input}")

    rows = list(csv.DictReader(args.input.open()))
    by_protocol: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_protocol[r["protocol"]][int(r["seed"])] = r

    seeds_present = sorted({int(r["seed"]) for r in rows})
    n_seeds = len(seeds_present)

    # Sanity: each protocol should have every seed.
    for proto in PROTOCOL_ORDER:
        missing = set(seeds_present) - set(by_protocol[proto].keys())
        if missing:
            raise SystemExit(
                f"Protocol {proto!r} missing seeds {sorted(missing)}; bootstrap "
                "requires the full N×P grid."
            )

    point_auroc: dict[str, list[float]] = {
        proto: [float(by_protocol[proto][s]["auroc"]) for s in seeds_present]
        for proto in PROTOCOL_ORDER
    }
    point_bal_acc: dict[str, list[float]] = {
        proto: [float(by_protocol[proto][s]["balanced_accuracy"]) for s in seeds_present]
        for proto in PROTOCOL_ORDER
    }

    # Paired bootstrap: resample seed indices once per iteration, apply
    # to every protocol so paired differences (gaps) are valid.
    boot_auroc_mean: dict[str, list[float]] = {p: [] for p in PROTOCOL_ORDER}
    boot_bal_acc_mean: dict[str, list[float]] = {p: [] for p in PROTOCOL_ORDER}
    boot_total_gap: list[float] = []
    boot_subject_leakage: list[float] = []
    boot_component_leakage: list[float] = []

    for _ in range(args.n_boot):
        idx = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        sample_auroc = {
            p: [point_auroc[p][i] for i in idx] for p in PROTOCOL_ORDER
        }
        sample_bal_acc = {
            p: [point_bal_acc[p][i] for i in idx] for p in PROTOCOL_ORDER
        }
        for p in PROTOCOL_ORDER:
            boot_auroc_mean[p].append(mean(sample_auroc[p]))
            boot_bal_acc_mean[p].append(mean(sample_bal_acc[p]))
        boot_total_gap.append(
            mean(sample_auroc["random"]) - mean(sample_auroc["component_safe"])
        )
        boot_subject_leakage.append(
            mean(sample_auroc["random"]) - mean(sample_auroc["subject_only"])
        )
        boot_component_leakage.append(
            mean(sample_auroc["subject_only"]) - mean(sample_auroc["component_safe"])
        )

    lo_q = (100.0 - args.ci) / 2.0
    hi_q = 100.0 - lo_q

    def summarize(samples: list[float]) -> dict[str, float]:
        return {
            "point_estimate": round(mean(samples), 4),
            "boot_mean": round(mean(samples), 4),
            "ci_lo": round(percentile(samples, lo_q), 4),
            "ci_hi": round(percentile(samples, hi_q), 4),
        }

    try:
        input_rel = str(args.input.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        input_rel = str(args.input)
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_table": input_rel,
        "n_seeds": n_seeds,
        "n_boot": args.n_boot,
        "ci_pct": args.ci,
        "seed_for_resample": args.seed,
        "auroc": {
            p: {
                "point_mean": round(mean(point_auroc[p]), 4),
                "point_seed_values": [round(v, 4) for v in point_auroc[p]],
                "boot_mean": round(mean(boot_auroc_mean[p]), 4),
                "ci_lo": round(percentile(boot_auroc_mean[p], lo_q), 4),
                "ci_hi": round(percentile(boot_auroc_mean[p], hi_q), 4),
            }
            for p in PROTOCOL_ORDER
        },
        "balanced_accuracy": {
            p: {
                "point_mean": round(mean(point_bal_acc[p]), 4),
                "boot_mean": round(mean(boot_bal_acc_mean[p]), 4),
                "ci_lo": round(percentile(boot_bal_acc_mean[p], lo_q), 4),
                "ci_hi": round(percentile(boot_bal_acc_mean[p], hi_q), 4),
            }
            for p in PROTOCOL_ORDER
        },
        "inflation_gap": {
            "total_random_minus_component_safe": summarize(boot_total_gap),
            "subject_leakage_random_minus_subject_only": summarize(boot_subject_leakage),
            "component_leakage_subject_only_minus_component_safe": summarize(boot_component_leakage),
        },
        "direction_preserved_share": round(
            sum(1 for g in boot_total_gap if g > 0) / len(boot_total_gap), 4
        ),
        "subject_leakage_positive_share": round(
            sum(1 for g in boot_subject_leakage if g > 0) / len(boot_subject_leakage), 4
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        output_rel = str(args.output.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        output_rel = str(args.output)

    # Print a tight reviewer-facing summary.
    print(f"=== ADNI inflation-gap bootstrap (B={args.n_boot}, {args.ci:.0f}% CI) ===")
    print(f"  protocol            point mean   {args.ci:.0f}% CI")
    for p in PROTOCOL_ORDER:
        a = out["auroc"][p]
        print(f"  {p:<18s}  {a['point_mean']:>10.4f}   [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]")
    print()
    print(f"  inflation gap (random − component_safe):")
    g = out["inflation_gap"]["total_random_minus_component_safe"]
    print(f"    point {g['point_estimate']:+.4f}   {args.ci:.0f}% CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    print(f"    bootstrap share with gap > 0: {out['direction_preserved_share']:.4f}")
    print(f"  subject-leakage component (random − subject_only):")
    g = out["inflation_gap"]["subject_leakage_random_minus_subject_only"]
    print(f"    point {g['point_estimate']:+.4f}   {args.ci:.0f}% CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    print(f"  component-leakage (subject_only − component_safe):")
    g = out["inflation_gap"]["component_leakage_subject_only_minus_component_safe"]
    print(f"    point {g['point_estimate']:+.4f}   {args.ci:.0f}% CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    print()
    print(f"  Wrote {output_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
