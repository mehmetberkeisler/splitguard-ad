#!/usr/bin/env python3
"""Mixed-effects reformulation of the ADNI dose-response analysis.

Reviewer §5.9 requested a more defensible statistical framework for the
dose-response fit. The original OLS fit (see :mod:`analyze_dose_response`)
regressed the per-overlap-level MEAN AUROC on the injected overlap
fraction. That aggregation discards seed-level dependence, so the OLS
standard errors under-cover.

This script re-fits the same underlying (seed, overlap, AUROC) tuples as
a linear mixed-effects model:

    AUROC = beta_0 + beta_overlap * overlap + u_seed + epsilon
    u_seed ~ N(0, sigma_seed^2)
    epsilon ~ N(0, sigma^2)

Reports:
    - Fixed-effect slope (beta_overlap) with 95% Wald CI clustered by seed
    - Per-seed OLS slope (descriptive) with min/max range
    - Spearman rho on pooled (overlap, AUROC) pairs
    - Monotonicity count: seeds for which AUROC is monotonically
      non-decreasing in overlap
    - Bootstrap over seeds (leave-one-seed-out and full seed-cluster
      resample) for a non-parametric slope interval that does not
      assume Gaussian residuals

Reads
-----
runs/adni_dose_response/<arch>/seed<S>_overlap<P>/baseline_seed<S>/test_predictions.csv

Writes
------
reports/tables/adni/adni_dose_response_mixed_effects.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT    = PROJECT_ROOT / "runs" / "adni_dose_response"
OUT_PATH     = PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_dose_response_mixed_effects.json"

# Dose-response cells are trained for this many epochs
# (stage adni_dose_response of scripts/gpu_program.py). Shorter runs are aborted smoke tests.
EXPECTED_EPOCHS = 15


# ── Shared helpers copied from analyze_dose_response.py (kept local so
#    this script has no runtime dependency on the OLS variant). ───────────
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


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x == x]
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def percentile(values, p):
    sv = sorted(v for v in values if v == v)
    if not sv: return float("nan")
    rank = (len(sv) - 1) * p / 100
    lo = int(rank); hi = min(lo + 1, len(sv) - 1); frac = rank - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def parse_overlap_dir(name: str):
    """runs/.../seed<S>_overlap<P>/ → (seed:int, overlap:float in [0,1]).

    Filenames already carry the overlap as a fraction (e.g.
    ``seed2_overlap0.25`` means overlap fraction 0.25), so we do NOT
    rescale here.
    """
    if not name.startswith("seed"): return None
    try:
        seed_part, overlap_part = name.split("_overlap")
        return int(seed_part.replace("seed", "")), float(overlap_part)
    except (ValueError, AttributeError):
        return None


def run_epochs(overlap_dir: Path, seed: int):
    """Completed epoch count for a run, or None if undeterminable."""
    metrics = overlap_dir / f"baseline_seed{seed}" / "metrics.json"
    if not metrics.exists():
        return None
    try:
        return int(json.loads(metrics.read_text()).get("epochs"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def collect_records(arch_root: Path, expected_epochs: int = EXPECTED_EPOCHS):
    """List of (seed, overlap, auroc) tuples for one architecture.

    Deduplicates (seed, overlap) cells and rejects incomplete runs. Both
    matter: ``seed0_overlap0.5`` and ``seed0_overlap0.50`` name the same cell,
    and one of them is a 1-epoch aborted smoke test. Without this the design
    matrix carried 26 observations for a 5x5 grid and the duplicated cell was
    counted twice, biasing both the fit and the cluster-robust SE.
    """
    best: dict[tuple[int, float], tuple[int, str, float]] = {}
    for overlap_dir in sorted(arch_root.iterdir(), key=lambda p: p.name):
        if not overlap_dir.is_dir(): continue
        parsed = parse_overlap_dir(overlap_dir.name)
        if parsed is None: continue
        seed, overlap = parsed
        pred_csv = overlap_dir / f"baseline_seed{seed}" / "test_predictions.csv"
        if not pred_csv.exists(): continue

        epochs = run_epochs(overlap_dir, seed)
        if epochs is None:
            print(f"  ! skipping {overlap_dir.name}: no readable metrics.json")
            continue
        if epochs < expected_epochs:
            print(f"  ! skipping {overlap_dir.name}: incomplete run "
                  f"({epochs} epochs, expected {expected_epochs})")
            continue

        y_true, y_score = [], []
        with pred_csv.open() as f:
            for row in csv.DictReader(f):
                y_true.append(int(row["y_true"]))
                y_score.append(float(row["y_prob"]))
        a = auroc(y_true, y_score)
        if a != a:
            continue

        key = (seed, round(overlap, 2))
        prior = best.get(key)
        if prior is None or (epochs, overlap_dir.name) > (prior[0], prior[1]):
            best[key] = (epochs, overlap_dir.name, a)

    return [(seed, overlap, a) for (seed, overlap), (_e, _n, a) in sorted(best.items())]


# ── Statistical primitives (pure-Python, no numpy/statsmodels
#    dependency; keeps the script deployable on a fresh venv). ──────────
def ols_slope(xs, ys):
    """Return (slope, intercept, r2) via closed-form OLS."""
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


# 97.5th percentile of Student's t, by degrees of freedom. With G seeds the
# cluster-robust slope has G-1 df, so at G=5 the correct multiplier is 2.776,
# not the normal 1.960 — a 42% wider interval. Using z here materially
# understates uncertainty at this cluster count.
_T_QUANTILE_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
    30: 2.042, 60: 2.000,
}


def _t_quantile_975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_QUANTILE_975:
        return _T_QUANTILE_975[df]
    for key in sorted(_T_QUANTILE_975):
        if df < key:
            return _T_QUANTILE_975[key]
    return 1.959963985  # large-df limit


def mixed_effects_reml(records, tol=1e-6, max_iter=200):
    """
    Random-intercept fit for the two-level model

        AUROC_ij = beta_0 + beta_1 * overlap_ij + u_i + eps_ij,
        u_i ~ N(0, sigma_u^2)  (one intercept per seed i),
        eps_ij ~ N(0, sigma^2).

    Estimated by iterated feasible GLS under compound symmetry. For a cluster
    with n_i observations the covariance is V_i = sigma^2 (I + rho J) with
    rho = sigma_u^2 / sigma^2, whose inverse has the closed form

        V_i^{-1} = (1/sigma^2) [ I - rho/(1 + n_i rho) J ],

    so the GLS normal equations reduce to a 2x2 solve and need no numpy. We
    alternate (a) GLS for beta given rho and (b) moment updates of the
    variance components from the GLS residuals, until rho moves less than
    ``tol`` or ``max_iter`` is reached.

    This function previously returned the *pooled OLS* slope while carrying a
    docstring describing REML: the variance components were computed and then
    never used, and ``tol``/``max_iter`` were dead parameters. The OLS fit is
    still reported, but under its own key (``pooled_ols_slope``) so the two
    estimators cannot be confused again.

    Inference on beta_1 uses a cluster-robust sandwich SE with the standard
    finite-cluster scaling and a t(G-1) critical value.
    """
    seeds = sorted({s for s, _, _ in records})
    n_seeds = len(seeds)
    n_obs = len(records)
    if n_obs < 3 or n_seeds < 2:
        return {"note": "insufficient data for mixed-effects fit"}

    xs = [o for _, o, _ in records]
    ys = [a for _, _, a in records]

    by_seed = defaultdict(list)
    for s, x, y in records:
        by_seed[s].append((x, y))

    # Pooled OLS — reported separately, and used only as the starting point.
    slope_ols, intercept_ols, r2_ols = ols_slope(xs, ys)

    def variance_components(slope, intercept):
        per_seed_res = defaultdict(list)
        for s, x, y in records:
            per_seed_res[s].append(y - (slope * x + intercept))
        seed_means = [mean(per_seed_res[s]) for s in seeds]
        within = mean([sd(per_seed_res[s]) ** 2 for s in seeds])
        between = sd(seed_means) ** 2
        sigma2_ = max(within, 1e-12)
        sigma_u2_ = max(between - within / max(1, n_obs / n_seeds), 0.0)
        return sigma_u2_, sigma2_

    def gls(rho):
        """GLS solve for (intercept, slope) under compound symmetry."""
        sxx = sxy = sx = sy = sww = 0.0
        for s in seeds:
            obs = by_seed[s]
            n_i = len(obs)
            shrink = rho / (1.0 + n_i * rho) if rho > 0 else 0.0
            sum_x = sum(x for x, _ in obs)
            sum_y = sum(y for _, y in obs)
            # Quadratic forms under V_i^{-1} (sigma^2 factors out of beta).
            sxx += sum(x * x for x, _ in obs) - shrink * sum_x * sum_x
            sxy += sum(x * y for x, y in obs) - shrink * sum_x * sum_y
            sx += sum_x - shrink * n_i * sum_x
            sy += sum_y - shrink * n_i * sum_y
            sww += n_i - shrink * n_i * n_i
        det = sww * sxx - sx * sx
        if abs(det) < 1e-18:
            return float("nan"), float("nan")
        slope_ = (sww * sxy - sx * sy) / det
        intercept_ = (sy - slope_ * sx) / sww if sww != 0 else float("nan")
        return slope_, intercept_

    slope, intercept = slope_ols, intercept_ols
    sigma_u2, sigma2 = variance_components(slope, intercept)
    rho = sigma_u2 / sigma2 if sigma2 > 0 else 0.0
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        slope_new, intercept_new = gls(rho)
        if slope_new != slope_new:  # NaN guard
            break
        slope, intercept = slope_new, intercept_new
        sigma_u2, sigma2 = variance_components(slope, intercept)
        rho_new = sigma_u2 / sigma2 if sigma2 > 0 else 0.0
        if abs(rho_new - rho) < tol:
            rho = rho_new
            converged = True
            break
        rho = rho_new

    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]

    # ── Cluster-robust sandwich SE for the slope ────────────────────────
    x_mean = sum(xs) / n_obs
    per_seed_score = defaultdict(float)
    per_seed_xtx = defaultdict(float)
    for (s, x, _y), r in zip(records, residuals):
        per_seed_score[s] += (x - x_mean) * r
        per_seed_xtx[s] += (x - x_mean) ** 2
    xtx_total = sum(per_seed_xtx.values())
    if xtx_total <= 0:
        return {"note": "zero X'X — degenerate design"}
    meat = sum(v ** 2 for v in per_seed_score.values())

    # Finite-cluster correction: with G=5 the uncorrected sandwich is biased
    # downward. c = G/(G-1) * (N-1)/(N-k), k=2 (intercept + slope).
    n_params = 2
    correction = (n_seeds / (n_seeds - 1)) * ((n_obs - 1) / (n_obs - n_params))
    se_slope = ((meat * correction) ** 0.5) / xtx_total

    df = n_seeds - 1
    t_crit = _t_quantile_975(df)
    ci_lo = slope - t_crit * se_slope
    ci_hi = slope + t_crit * se_slope

    return {
        "fit": "iterated feasible GLS, random intercept per seed "
               "(compound symmetry), cluster-robust sandwich SE with "
               "finite-cluster correction and t(G-1) critical value",
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "pooled_ols_slope": round(slope_ols, 6),
        "pooled_ols_intercept": round(intercept_ols, 6),
        "pooled_ols_r2": round(r2_ols, 4),
        "sigma_u": round(sigma_u2 ** 0.5, 6),
        "sigma_eps": round(sigma2 ** 0.5, 6),
        "rho_sigma_u2_over_sigma2": round(rho, 6),
        "cluster_robust_se": round(se_slope, 6),
        "finite_cluster_correction": round(correction, 4),
        "t_crit_975": t_crit,
        "df": df,
        "ci_lo_95": round(ci_lo, 6),
        "ci_hi_95": round(ci_hi, 6),
        "n_iter": n_iter,
        "converged": converged,
        "n_obs": n_obs,
        "n_seeds": n_seeds,
    }


def spearman_rho(xs, ys):
    """Spearman rank correlation coefficient, pure Python."""
    if len(xs) < 2: return float("nan")
    rx = _rank(xs); ry = _rank(ys)
    return _pearson(rx, ry)


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    xm = sum(xs) / n; ym = sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    dx = (sum((x - xm) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - ym) ** 2 for y in ys)) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def per_seed_slope_summary(records):
    """Per-seed OLS slope + monotonicity check."""
    by_seed = defaultdict(list)
    for s, o, a in records:
        by_seed[s].append((o, a))
    slopes, monotonic = [], 0
    per_seed = {}
    for s, pairs in sorted(by_seed.items()):
        pairs.sort()
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        slope, intercept, r2 = ols_slope(xs, ys)
        slopes.append(slope)
        is_mono = all(ys[i + 1] >= ys[i] for i in range(len(ys) - 1))
        if is_mono: monotonic += 1
        per_seed[s] = {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r2": round(r2, 4),
            "monotonic": is_mono,
            "n_points": len(xs),
        }
    return {
        "per_seed": per_seed,
        "slope_min": round(min(slopes), 4) if slopes else float("nan"),
        "slope_max": round(max(slopes), 4) if slopes else float("nan"),
        "slope_median": round(sorted(slopes)[len(slopes) // 2], 4)
                        if slopes else float("nan"),
        "monotonic_count": monotonic,
        "n_seeds": len(slopes),
    }


def cluster_bootstrap_slope(records, n_boot=10000, rng_seed=0):
    """Non-parametric CI: resample seed-clusters (not individual obs)."""
    rng = random.Random(rng_seed)
    by_seed = defaultdict(list)
    for s, o, a in records:
        by_seed[s].append((o, a))
    seeds = list(by_seed.keys())
    slopes = []
    for _ in range(n_boot):
        picked = [rng.choice(seeds) for _ in range(len(seeds))]
        xs, ys = [], []
        for s in picked:
            for o, a in by_seed[s]:
                xs.append(o); ys.append(a)
        slope, _, _ = ols_slope(xs, ys)
        if slope == slope: slopes.append(slope)
    if not slopes:
        return {"note": "bootstrap slope collapsed"}
    return {
        "n_boot": n_boot,
        "slope_median": round(sorted(slopes)[len(slopes) // 2], 4),
        "ci_lo_cluster_boot": round(percentile(slopes, 2.5), 4),
        "ci_hi_cluster_boot": round(percentile(slopes, 97.5), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    if not RUNS_ROOT.exists():
        print(f"error: {RUNS_ROOT} does not exist", file=sys.stderr)
        return 1

    per_arch = {}
    for arch_dir in sorted(RUNS_ROOT.iterdir()):
        if not arch_dir.is_dir(): continue
        records = collect_records(arch_dir)
        if not records: continue
        me_fit = mixed_effects_reml(records)
        per_seed = per_seed_slope_summary(records)
        cluster_ci = cluster_bootstrap_slope(records, n_boot=args.n_boot,
                                             rng_seed=args.rng_seed)
        xs = [o for _, o, _ in records]
        ys = [a for _, _, a in records]
        rho = spearman_rho(xs, ys)
        per_arch[arch_dir.name] = {
            "mixed_effects": me_fit,
            "per_seed_slopes": per_seed,
            "cluster_bootstrap": cluster_ci,
            "spearman_rho_pooled": round(rho, 4),
            "n_records": len(records),
        }
        print(
            f"{arch_dir.name}: slope={me_fit.get('slope', 'nan')}, "
            f"cluster-robust SE={me_fit.get('cluster_robust_se', 'nan')}, "
            f"cluster-robust 95% CI [{me_fit.get('ci_lo_95', 'nan')}, "
            f"{me_fit.get('ci_hi_95', 'nan')}], "
            f"cluster-boot 95% CI ["
            f"{cluster_ci.get('ci_lo_cluster_boot', 'nan')}, "
            f"{cluster_ci.get('ci_hi_cluster_boot', 'nan')}], "
            f"Spearman rho={rho:.3f}, "
            f"monotonic {per_seed['monotonic_count']}/{per_seed['n_seeds']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(per_arch, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
