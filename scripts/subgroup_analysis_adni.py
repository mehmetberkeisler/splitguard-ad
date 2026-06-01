#!/usr/bin/env python3
"""Per-subgroup performance breakdown on ADNI.

Closes the TRIPOD+AI 17 gap flagged in docs/REPORTING_CHECKLISTS.md.
Stratifies test-set AUROC by age band (median split) and sex within
each protocol, then aggregates across seeds with a paired bootstrap.

Inputs (per seed × protocol)
----------------------------
* ``runs/<output_root>/inflation_gap_seed{S}/{label}/test_predictions.csv``
  with image_id, subject_id, diagnosis_group, y_true, y_prob.

Joins to ``data/manifests/adni/adni_manifest.csv`` to recover age and
sex per image (the predictions file does not carry demographics).

Output
------
* JSON summary at the specified path (default
  ``reports/tables/adni/adni_subgroup_analysis.json``).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def auroc(y_true: list[int], y_score: list[float]) -> float:
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0; wins = 0.0
    for p in pos:
        for n in neg:
            pairs += 1
            if p > n: wins += 1.0
            elif p == n: wins += 0.5
    return wins / pairs


def percentile(values, p):
    sv = sorted(v for v in values if v == v)
    if not sv: return float("nan")
    rank = (len(sv)-1)*p/100; lo=int(rank); hi=min(lo+1,len(sv)-1); frac=rank-lo
    return sv[lo]*(1-frac)+sv[hi]*frac


def mean(values):
    clean = [v for v in values if v == v]
    return sum(clean)/len(clean) if clean else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path,
                        default=PROJECT_ROOT / "runs" / "adni_with_converters")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--protocols", nargs="+",
                        default=["random", "subject_only", "component_safe"])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reports" / "tables" / "adni" /
                                "adni_subgroup_analysis.json")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load manifest -> image_id -> {age, sex, ptid}
    img_meta: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(args.manifest.open()):
        img_meta[r["image_id"]] = {
            "age": r["age"], "sex": r["sex"], "ptid": r["ptid"]
        }

    # Age median across all manifest CN+AD images for stratification.
    cn_ad_ages = [
        float(r["age"]) for r in img_meta.values()
        if r["age"] and r["sex"]
    ]
    age_median = statistics.median(cn_ad_ages)

    # Per (seed, protocol, subgroup) -> AUROC
    per_seed_aurocs: dict[tuple[int, str, str], float] = {}
    test_sizes: dict[tuple[int, str, str], int] = {}

    for seed in args.seeds:
        for proto in args.protocols:
            path = args.runs_root / f"inflation_gap_seed{seed}" / proto / "test_predictions.csv"
            if not path.exists():
                raise SystemExit(f"Missing predictions: {path}")
            rows = list(csv.DictReader(path.open()))
            by_subgroup: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for r in rows:
                meta = img_meta.get(r["image_id"], {})
                age_s = meta.get("age", ""); sex = meta.get("sex", "")
                if not age_s: continue
                try: age = float(age_s)
                except ValueError: continue
                age_band = "old" if age >= age_median else "young"
                key_age = f"age_{age_band}"
                key_sex = f"sex_{sex}" if sex else "sex_unknown"
                key_overall = "overall"
                pt = (int(r["y_true"]), float(r["y_prob"]))
                by_subgroup[key_overall].append(pt)
                by_subgroup[key_age].append(pt)
                by_subgroup[key_sex].append(pt)
            for subgroup, pts in by_subgroup.items():
                y_t, y_p = zip(*pts)
                per_seed_aurocs[(seed, proto, subgroup)] = auroc(list(y_t), list(y_p))
                test_sizes[(seed, proto, subgroup)] = len(pts)

    # Aggregate across seeds with paired bootstrap
    subgroups = sorted({k[2] for k in per_seed_aurocs})
    seeds = args.seeds
    n_seeds = len(seeds)

    point: dict[tuple[str, str], dict] = {}
    for proto in args.protocols:
        for subgroup in subgroups:
            vals = [per_seed_aurocs[(s, proto, subgroup)]
                    for s in seeds if (s, proto, subgroup) in per_seed_aurocs]
            sizes = [test_sizes[(s, proto, subgroup)]
                     for s in seeds if (s, proto, subgroup) in test_sizes]
            if not vals: continue
            # Bootstrap mean
            boot_means = []
            for _ in range(args.n_boot):
                idx = [rng.randrange(len(vals)) for _ in range(len(vals))]
                boot_means.append(mean([vals[i] for i in idx]))
            lo_q = (100.0 - args.ci) / 2; hi_q = 100.0 - lo_q
            point[(proto, subgroup)] = {
                "n_seeds": len(vals),
                "n_test_mean": round(mean(sizes), 1),
                "auroc_per_seed": [round(v, 4) for v in vals],
                "point_mean": round(mean(vals), 4),
                "ci_lo": round(percentile(boot_means, lo_q), 4),
                "ci_hi": round(percentile(boot_means, hi_q), 4),
            }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_root": str(args.runs_root.resolve().relative_to(PROJECT_ROOT))
                     if str(args.runs_root.resolve()).startswith(str(PROJECT_ROOT))
                     else str(args.runs_root),
        "manifest": str(args.manifest.resolve().relative_to(PROJECT_ROOT))
                    if str(args.manifest.resolve()).startswith(str(PROJECT_ROOT))
                    else str(args.manifest),
        "age_median_for_stratification": round(age_median, 2),
        "seeds": seeds,
        "protocols": args.protocols,
        "subgroups": subgroups,
        "n_boot": args.n_boot,
        "ci_pct": args.ci,
        "results": {
            f"{p}__{sg}": v for (p, sg), v in point.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Print compact summary
    print(f"=== ADNI subgroup analysis (sensitivity arm, B={args.n_boot}, {args.ci:.0f}% CI) ===")
    print(f"  age median for stratification: {age_median:.1f}y")
    print()
    for proto in args.protocols:
        print(f"  Protocol {proto}:")
        for sg in subgroups:
            d = point.get((proto, sg))
            if not d: continue
            print(f"    {sg:<18s}  AUROC {d['point_mean']:.4f} "
                  f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]  "
                  f"n_test~{d['n_test_mean']:.0f}")
        print()
    try:
        out_rel = str(args.output.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        out_rel = str(args.output)
    print(f"  Wrote {out_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
