#!/usr/bin/env python3
"""Two-level (subject × seed) hierarchical bootstrap on ADNI inflation gap.

The per-seed paired bootstrap already reported in the manuscript
resamples seeds only. A reviewer rightly noted that, at N=5 seeds, that
procedure can look falsely precise because all within-seed variance
(subject heterogeneity, label noise, etc.) is absorbed into the point
estimate. This script adds a second resampling level:

  outer: resample seeds with replacement (paired across protocols)
  inner: within each resampled seed × protocol, resample SUBJECTS with
         replacement from the held-out test set; recompute AUROC on the
         resampled predictions

The reported 95% CI is the percentile interval over the outer bootstrap
distribution. Subject-level resampling is enabled by the trainer's
per-image-prediction persistence (test_predictions.csv).

Inputs (per protocol, per seed)
-------------------------------
* ``runs/<output_root>/inflation_gap_seed{S}/{label}/test_predictions.csv``
  with columns: image_id, subject_id, diagnosis_group, y_true, y_prob.

Output
------
* JSON summary at the specified path (default
  ``reports/tables/adni/adni_inflation_gap_hierarchical_bootstrap.json``).

Usage
-----
    python3 scripts/hierarchical_bootstrap_adni.py \\
        --runs-root runs/adni_with_converters --seeds 0 1 2 3 4 \\
        --output reports/tables/adni/adni_inflation_gap_with_converters_hierarchical.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("random", "subject_only", "component_safe")


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def auroc(y_true: list[int], y_score: list[float]) -> float:
    """Rank-based AUROC. Returns nan if degenerate (one class)."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0
    wins = 0.0
    for p in pos:
        for n in neg:
            pairs += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "adni",
        help="Directory containing inflation_gap_seed{S}/{label}/test_predictions.csv per seed.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables" / "adni" /
                "adni_inflation_gap_hierarchical_bootstrap.json",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load every (seed, protocol) -> {subject_id: [(y_true, y_prob), ...]}
    predictions: dict[tuple[int, str], dict[str, list[tuple[int, float]]]] = {}
    point_aurocs: dict[tuple[int, str], float] = {}
    n_test_records: dict[tuple[int, str], int] = {}
    n_test_subjects: dict[tuple[int, str], int] = {}

    for seed in args.seeds:
        for proto in args.protocols:
            path = args.runs_root / f"inflation_gap_seed{seed}" / proto / "test_predictions.csv"
            if not path.exists():
                raise SystemExit(
                    f"Missing predictions for seed={seed} protocol={proto!r}: {path}\n"
                    "Ensure the inflation-gap run used the patched trainer that writes "
                    "test_predictions.csv."
                )
            rows = load_predictions(path)
            by_subject: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for r in rows:
                by_subject[r["subject_id"]].append((int(r["y_true"]), float(r["y_prob"])))
            predictions[(seed, proto)] = by_subject
            y_true_full = [t for subj_preds in by_subject.values() for (t, _) in subj_preds]
            y_prob_full = [p for subj_preds in by_subject.values() for (_, p) in subj_preds]
            point_aurocs[(seed, proto)] = auroc(y_true_full, y_prob_full)
            n_test_records[(seed, proto)] = len(rows)
            n_test_subjects[(seed, proto)] = len(by_subject)

    # Hierarchical bootstrap. Outer: resample seed indices with replacement.
    # Inner: within each resampled (seed, protocol), resample SUBJECTS with replacement
    # and recompute AUROC on the resulting prediction set. Pairing is preserved by using
    # the same outer-seed index for every protocol.
    seed_list = args.seeds
    n_seeds = len(seed_list)

    boot_means: dict[str, list[float]] = {p: [] for p in args.protocols}
    boot_gaps_total: list[float] = []
    boot_gaps_subj: list[float] = []
    boot_gaps_comp: list[float] = []

    for _ in range(args.n_boot):
        seed_idx_resample = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        per_seed_per_proto_auroc: dict[str, list[float]] = {p: [] for p in args.protocols}
        for outer_idx in seed_idx_resample:
            outer_seed = seed_list[outer_idx]
            for proto in args.protocols:
                by_subject = predictions[(outer_seed, proto)]
                subjects = list(by_subject)
                resampled_subjects = [rng.choice(subjects) for _ in subjects]
                y_t, y_p = [], []
                for s in resampled_subjects:
                    for (t, q) in by_subject[s]:
                        y_t.append(t); y_p.append(q)
                per_seed_per_proto_auroc[proto].append(auroc(y_t, y_p))
        # Mean across resampled seeds
        for proto in args.protocols:
            vals = [v for v in per_seed_per_proto_auroc[proto] if v == v]  # filter nan
            if vals:
                boot_means[proto].append(sum(vals) / len(vals))
            else:
                boot_means[proto].append(float("nan"))
        # Paired gap
        if all(boot_means[p][-1] == boot_means[p][-1] for p in args.protocols):
            r = boot_means["random"][-1]
            so = boot_means["subject_only"][-1]
            cs = boot_means["component_safe"][-1]
            boot_gaps_total.append(r - cs)
            boot_gaps_subj.append(r - so)
            boot_gaps_comp.append(so - cs)

    lo_q = (100.0 - args.ci) / 2.0
    hi_q = 100.0 - lo_q

    def pct(values, p):
        sv = sorted(v for v in values if v == v)
        if not sv:
            return float("nan")
        rank = (len(sv) - 1) * p / 100.0
        lo = int(rank); hi = min(lo + 1, len(sv) - 1); frac = rank - lo
        return sv[lo] * (1 - frac) + sv[hi] * frac

    def mean(values):
        clean = [v for v in values if v == v]
        return sum(clean) / len(clean) if clean else float("nan")

    point_protocol = {
        p: mean([point_aurocs[(s, p)] for s in seed_list])
        for p in args.protocols
    }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_root": str(args.runs_root.resolve().relative_to(PROJECT_ROOT))
                     if str(args.runs_root.resolve()).startswith(str(PROJECT_ROOT)) else str(args.runs_root),
        "seeds": seed_list,
        "protocols": args.protocols,
        "n_boot": args.n_boot,
        "ci_pct": args.ci,
        "rng_seed": args.seed,
        "n_test_records_per_seed_protocol": {
            f"seed{s}/{p}": n_test_records[(s, p)] for s in seed_list for p in args.protocols
        },
        "n_test_subjects_per_seed_protocol": {
            f"seed{s}/{p}": n_test_subjects[(s, p)] for s in seed_list for p in args.protocols
        },
        "auroc_per_seed_protocol_point": {
            f"seed{s}/{p}": round(point_aurocs[(s, p)], 4) for s in seed_list for p in args.protocols
        },
        "hierarchical_bootstrap": {
            p: {
                "point_mean_over_seeds": round(point_protocol[p], 4),
                "boot_mean": round(mean(boot_means[p]), 4),
                "ci_lo": round(pct(boot_means[p], lo_q), 4),
                "ci_hi": round(pct(boot_means[p], hi_q), 4),
            }
            for p in args.protocols
        },
        "inflation_gap": {
            "total_random_minus_component_safe": {
                "point": round(mean(boot_gaps_total), 4),
                "ci_lo": round(pct(boot_gaps_total, lo_q), 4),
                "ci_hi": round(pct(boot_gaps_total, hi_q), 4),
                "direction_preserved_share": round(
                    sum(1 for g in boot_gaps_total if g > 0) / max(1, len(boot_gaps_total)), 4
                ),
            },
            "subject_leakage_random_minus_subject_only": {
                "point": round(mean(boot_gaps_subj), 4),
                "ci_lo": round(pct(boot_gaps_subj, lo_q), 4),
                "ci_hi": round(pct(boot_gaps_subj, hi_q), 4),
            },
            "component_leakage_subject_only_minus_component_safe": {
                "point": round(mean(boot_gaps_comp), 4),
                "ci_lo": round(pct(boot_gaps_comp, lo_q), 4),
                "ci_hi": round(pct(boot_gaps_comp, hi_q), 4),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"=== Hierarchical bootstrap (subject x seed, B={args.n_boot}, {args.ci:.0f}% CI) ===")
    for p in args.protocols:
        d = out["hierarchical_bootstrap"][p]
        print(f"  {p:<15s}  AUROC {d['point_mean_over_seeds']:.4f}   "
              f"hier-{args.ci:.0f}%CI [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")
    print()
    g = out["inflation_gap"]["total_random_minus_component_safe"]
    print(f"  Total gap     {g['point']:+.4f}   {args.ci:.0f}%CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    g = out["inflation_gap"]["subject_leakage_random_minus_subject_only"]
    print(f"  Subject-leak  {g['point']:+.4f}   {args.ci:.0f}%CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    g = out["inflation_gap"]["component_leakage_subject_only_minus_component_safe"]
    print(f"  Component-lvl {g['point']:+.4f}   {args.ci:.0f}%CI [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
