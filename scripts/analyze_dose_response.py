#!/usr/bin/env python3
"""Analyse the dose-response leakage stress-test results.

For each (architecture × overlap level), compute mean test AUROC across the
five seeds and a paired-seed bootstrap 95% CI.  Then fit a functional form
AUROC = f(overlap) to extract the headline dose-response law.

Reads
-----
runs/adni_dose_response/<arch>/seed<S>_overlap<P>/baseline_seed<S>/test_predictions.csv

Writes
------
reports/tables/adni/adni_dose_response.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import display_path  # noqa: E402

RUNS_ROOT    = PROJECT_ROOT / "runs" / "adni_dose_response"
OUT_PATH     = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_dose_response.json"

# Every dose-response cell is trained for this many epochs
# (stage adni_dose_response of scripts/gpu_program.py). Runs that stopped short are aborted
# smoke tests and must never enter the fit — see collect_per_seed_aurocs.
EXPECTED_EPOCHS = 15


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
    rank = (len(sv) - 1) * p / 100
    lo = int(rank); hi = min(lo + 1, len(sv) - 1); frac = rank - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x == x]
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def parse_overlap_dir(name: str) -> tuple[int, float] | None:
    """Parse 'seed0_overlap0.50' -> (0, 0.50). Returns None on mismatch."""
    if not name.startswith("seed"):
        return None
    try:
        seed_part, overlap_part = name.split("_overlap")
        return int(seed_part.replace("seed", "")), float(overlap_part)
    except (ValueError, AttributeError):
        return None


def run_epochs(run_dir: Path, seed: int) -> int | None:
    """Completed epoch count for a run, or None if it cannot be determined."""
    metrics = run_dir / f"baseline_seed{seed}" / "metrics.json"
    if not metrics.exists():
        return None
    try:
        return int(json.loads(metrics.read_text()).get("epochs"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def collect_per_seed_aurocs(
    arch_root: Path, expected_epochs: int = EXPECTED_EPOCHS
) -> dict[float, dict[int, float]]:
    """Return overlap_level -> seed -> AUROC.

    Two directories can map to the same (seed, overlap) cell — e.g.
    ``seed0_overlap0.5`` and ``seed0_overlap0.50``. They describe the same
    *split*, but not necessarily the same *run*: an aborted smoke test leaves
    a directory that looks complete apart from its epoch count. Selecting the
    "first encountered" therefore depends on filesystem iteration order and
    silently admitted a 1-epoch run into the published ResNet-18 fit
    (AUROC 0.813 instead of 0.827 at p=0.50), dragging the slope down.

    We now (a) reject any run that did not reach ``expected_epochs``, and
    (b) resolve remaining duplicates deterministically by preferring the
    longer run, then the lexicographically larger directory name — never by
    iteration order.
    """
    candidates: dict[float, dict[int, tuple[int, str, Path]]] = defaultdict(dict)
    for entry in sorted(arch_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        parsed = parse_overlap_dir(entry.name)
        if parsed is None:
            continue
        seed, overlap = parsed
        overlap_key = round(overlap, 2)
        preds = entry / f"baseline_seed{seed}" / "test_predictions.csv"
        if not preds.exists():
            continue

        epochs = run_epochs(entry, seed)
        if epochs is None:
            print(f"  ! skipping {entry.name}: no readable metrics.json")
            continue
        if epochs < expected_epochs:
            print(
                f"  ! skipping {entry.name}: incomplete run "
                f"({epochs} epochs, expected {expected_epochs})"
            )
            continue

        key = (epochs, entry.name)
        prior = candidates[overlap_key].get(seed)
        if prior is None or key > (prior[0], prior[1]):
            candidates[overlap_key][seed] = (epochs, entry.name, entry)

    per_seed: dict[float, dict[int, float]] = defaultdict(dict)
    for overlap_key, by_seed in candidates.items():
        for seed, (_epochs, _name, entry) in by_seed.items():
            preds = entry / f"baseline_seed{seed}" / "test_predictions.csv"
            rows = list(csv.DictReader(preds.open()))
            y_true = [int(r["y_true"]) for r in rows]
            y_prob = [float(r["y_prob"]) for r in rows]
            per_seed[overlap_key][seed] = auroc(y_true, y_prob)
    return per_seed


def aggregate(per_seed: dict[float, dict[int, float]],
              seeds: list[int],
              n_boot: int,
              ci: float,
              rng_seed: int) -> dict[float, dict]:
    """Per overlap level: mean across seeds + paired-bootstrap CI."""
    rng = random.Random(rng_seed)
    lo_q = (100.0 - ci) / 2; hi_q = 100.0 - lo_q
    out = {}
    for overlap, seed_dict in sorted(per_seed.items()):
        vals = [seed_dict[s] for s in seeds if s in seed_dict]
        if not vals: continue
        boot_means = []
        for _ in range(n_boot):
            idx = [rng.randrange(len(vals)) for _ in range(len(vals))]
            boot_means.append(mean([vals[i] for i in idx]))
        out[overlap] = {
            "n_seeds":    len(vals),
            "per_seed":   [round(v, 4) for v in vals],
            "mean":       round(mean(vals), 4),
            "sd":         round(sd(vals), 4),
            "ci_lo":      round(percentile(boot_means, lo_q), 4),
            "ci_hi":      round(percentile(boot_means, hi_q), 4),
        }
    return out


def fit_linear(xs, ys):
    """OLS slope + intercept; returns (slope, intercept, r2)."""
    n = len(xs)
    if n < 2: return float("nan"), float("nan"), float("nan")
    xm = sum(xs) / n; ym = sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    if den == 0: return float("nan"), float("nan"), float("nan")
    slope = num / den
    intercept = ym - slope * xm
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ym) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--ci",     type=float, default=95.0)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    arch_dirs = {
        d.name: d for d in RUNS_ROOT.iterdir()
        if d.is_dir() and d.name in ("resnet18", "densenet121")
    }
    if not arch_dirs:
        raise SystemExit(f"No arch subdirectories under {RUNS_ROOT}")

    results = {}
    fits = {}
    for arch, arch_root in arch_dirs.items():
        per_seed = collect_per_seed_aurocs(arch_root)
        agg = aggregate(per_seed, args.seeds, args.n_boot, args.ci, args.seed)
        results[arch] = agg

        # Fit AUROC = a + b * overlap on the per-seed points (not the means)
        # so the slope/intercept reflect the underlying noise.
        xs, ys = [], []
        for overlap, seed_dict in per_seed.items():
            for s, v in seed_dict.items():
                if s in args.seeds and v == v:
                    xs.append(overlap); ys.append(v)
        slope, intercept, r2 = fit_linear(xs, ys)
        # Residual standard deviation about the fit, over the per-seed
        # observations rather than the per-overlap means. The distinction is
        # not cosmetic: the manuscript uses this quantity as a prediction band
        # for a *single* training run at a given overlap, and the spread of
        # single runs is what has to parameterise that. Taking it from the
        # five overlap means instead would understate it roughly threefold,
        # because averaging five seeds removes most of the seed variance the
        # band is supposed to describe.
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_sd = (
            (sum(r * r for r in resid) / (len(resid) - 2)) ** 0.5
            if len(resid) > 2 else float("nan")
        )
        fits[arch] = {
            "n_points":     len(xs),
            "slope":        round(slope, 4),
            "intercept":    round(intercept, 4),
            "r2":           round(r2, 4),
            "residual_sd":  round(resid_sd, 4),
            "interpretation": (
                f"AUROC ≈ {intercept:.3f} + {slope:.3f} × overlap_fraction "
                f"(R²={r2:.3f}). A 10pp increase in test-subject overlap "
                f"inflates AUROC by {slope*0.1:.3f}."
            ),
        }

    out = {
        "seeds":         args.seeds,
        "n_boot":        args.n_boot,
        "ci_pct":        args.ci,
        "by_arch":       results,
        "linear_fits":   fits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    # ── Compact human-readable summary ────────────────────────────────────
    print(f"=== ADNI dose-response (sens arm, B={args.n_boot}, {args.ci:.0f}% CI) ===\n")
    for arch, agg in results.items():
        print(f"  {arch}:")
        print(f"    {'overlap':>8s} {'mean':>8s} {'sd':>8s} {'CI lo':>8s} {'CI hi':>8s} {'n':>3s}")
        for overlap in sorted(agg):
            d = agg[overlap]
            print(f"    {overlap:>8.2f} {d['mean']:>8.4f} {d['sd']:>8.4f} "
                  f"{d['ci_lo']:>8.4f} {d['ci_hi']:>8.4f} {d['n_seeds']:>3d}")
        f = fits[arch]
        print(f"    Linear fit: AUROC = {f['intercept']:.4f} "
              f"+ {f['slope']:.4f} × overlap, R²={f['r2']:.3f} (n={f['n_points']})")
        print()
    print(f"  Wrote {display_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
