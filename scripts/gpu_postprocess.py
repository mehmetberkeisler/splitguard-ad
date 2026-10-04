#!/usr/bin/env python3
"""Turn the GPU programme's predictions into the numbers the manuscript quotes.

Reads the per-image predictions written under ``--runs-root`` and produces one
summary JSON plus a LaTeX macro file, so no number is transcribed by hand.
Every arm present is analysed and every arm absent is listed, so a partial GPU
run says exactly what it could not answer.

What it computes
----------------
* Per-seed and mean test AUROC per protocol, per arm, and the gaps between
  protocols (``scripts/hierarchical_bootstrap_adni.py`` supplies the
  subject-by-seed interval).
* Participant-count-matched AUROC: the leaky protocol's test set holds more
  participants than the component-safe one, so its AUROC is recomputed on
  random participant subsets of the component-safe size. This separates the
  inflation gap from a test-set-composition difference.
* Label-join sensitivity: AUROC recomputed on the test images whose CN/AD
  label comes from an exact visit key, dropping the 180-day date-proximity
  joins recorded in the linkage audit.
* Component-size composition control: the gap under the frozen
  size-descending assignment against the size-balanced (shuffled) one.
* The identity probes, whichever hold-out variants ran.

Usage
-----
    python3 scripts/gpu_postprocess.py
    python3 scripts/gpu_postprocess.py --runs-root runs_smoke --tables reports/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splitguard_ad.metrics import auroc  # noqa: E402
from _paths import display_path  # noqa: E402
PY = sys.executable
ADNI_PROTOCOLS = ["random", "subject_only", "component_safe"]
SEEDS = [0, 1, 2, 3, 4]
TIER_SEEDS = [42, 0, 1, 2, 3]
LINKAGE = ROOT / "data" / "manifests" / "adni" / "adni_linkage_audit.csv"

# arm -> (protocols leakiest first, seeds)
ARMS = {
    "adni": (ADNI_PROTOCOLS, SEEDS),
    "adni_with_converters": (ADNI_PROTOCOLS, SEEDS),
    "adni_no_mt1": (ADNI_PROTOCOLS, SEEDS),
    "adni_densenet121": (ADNI_PROTOCOLS, SEEDS),
    "adni_size_balanced": (ADNI_PROTOCOLS, SEEDS),
    "adni_exact_linkage": (ADNI_PROTOCOLS, SEEDS),
    "adni_3d": (ADNI_PROTOCOLS, SEEDS),
    "tier1": (["random", "filename_subject", "true_participant"], TIER_SEEDS),
    "oasis1": (["random", "component_safe"], TIER_SEEDS),
    "oasis1_densenet121": (["random", "component_safe"], TIER_SEEDS),
}




def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def by_participant(rows: list[dict[str, str]]) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        out.setdefault(r["subject_id"], []).append((int(r["y_true"]), float(r["y_prob"])))
    return out


def exact_key_scans() -> set[str]:
    if not LINKAGE.is_file():
        return set()
    with LINKAGE.open(encoding="utf-8") as fh:
        return {r["scan_uid"] for r in csv.DictReader(fh) if r["join_tier"] == "exact_viscode"}


def analyse_arm(runs_root: Path, arm: str, protocols: list[str], seeds: list[int],
                exact: set[str]) -> dict | None:
    root = runs_root / arm
    per_seed: dict[str, dict[int, float]] = {p: {} for p in protocols}
    rows_cache: dict[tuple[int, str], list[dict[str, str]]] = {}
    for seed in seeds:
        for proto in protocols:
            path = root / f"inflation_gap_seed{seed}" / proto / "test_predictions.csv"
            if not path.is_file():
                continue
            rows = read_predictions(path)
            rows_cache[(seed, proto)] = rows
            per_seed[proto][seed] = round(auroc([(int(r["y_true"]), float(r["y_prob"])) for r in rows]), 4)
    if not rows_cache:
        return None

    complete = [s for s in seeds if all((s, p) in rows_cache for p in protocols)]

    def mean(values):
        clean = [v for v in values if v == v]
        return round(sum(clean) / len(clean), 4) if clean else None

    result = {
        "protocols": protocols,
        "seeds_found": sorted({s for s, _ in rows_cache}),
        "seeds_complete": complete,
        "auroc_per_seed": {p: per_seed[p] for p in protocols},
        # Over the complete seeds only: a mean over a different seed set from
        # the gaps would not reconcile with them.
        "auroc_mean": {p: mean([per_seed[p][s] for s in complete if s in per_seed[p]]) for p in protocols},
        "gap_mean": mean([per_seed[protocols[0]][s] - per_seed[protocols[-1]][s] for s in complete]),
        "gap_per_seed": {s: round(per_seed[protocols[0]][s] - per_seed[protocols[-1]][s], 4) for s in complete},
        "test_participants": {p: {s: len(by_participant(rows_cache[(s, p)]))
                                  for s in seeds if (s, p) in rows_cache} for p in protocols},
    }
    if len(protocols) == 3 and complete:
        result["gap_from_grouping"] = mean([per_seed[protocols[0]][s] - per_seed[protocols[1]][s] for s in complete])
        result["gap_beyond_grouping"] = mean([per_seed[protocols[1]][s] - per_seed[protocols[-1]][s] for s in complete])

    # Label-join sensitivity: exact visit key only (ADNI arms carry ADNI image ids).
    if exact:
        restricted: dict[str, list[float]] = {p: [] for p in protocols}
        kept, total = 0, 0
        for (seed, proto), rows in rows_cache.items():
            key = "image_id" if "image_id" in rows[0] else "image_uid"
            subset = [r for r in rows if r[key] in exact]
            total += len(rows)
            kept += len(subset)
            if len(subset) >= 10:
                restricted[proto].append(auroc([(int(r["y_true"]), float(r["y_prob"])) for r in subset]))
        if kept:
            result["exact_visit_key_only"] = {
                "test_rows_kept": kept, "test_rows_total": total,
                "auroc_mean": {p: mean(restricted[p]) for p in protocols},
                "gap_mean": (round(mean(restricted[protocols[0]]) - mean(restricted[protocols[-1]]), 4)
                             if restricted[protocols[0]] and restricted[protocols[-1]] else None),
            }
    return result


def report_card(arm_root: Path) -> dict | None:
    """Seed-mean component-safe test metrics, with predictive values at two prevalence anchors."""
    metrics = []
    for path in sorted(arm_root.glob("inflation_gap_seed*/component_safe/metrics.json")):
        metrics.append(json.loads(path.read_text())["test_metrics"])
    if not metrics:
        return None
    mean = {k: sum(m[k] for m in metrics) / len(metrics)
            for k in ("sensitivity", "specificity", "balanced_accuracy", "auroc", "n_test")}
    sens, spec = mean["sensitivity"], mean["specificity"]
    card = {"n_seeds": len(metrics), **{k: round(v, 4) for k, v in mean.items()}}
    for name, prevalence in (("population", 0.138), ("memory_clinic", 0.589)):
        card[f"ppv_{name}"] = round(sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence)), 4)
        card[f"npv_{name}"] = round(spec * (1 - prevalence) / (spec * (1 - prevalence) + (1 - sens) * prevalence), 4)
    return card


def provenance_auroc(path: Path) -> dict | None:
    """Mean test AUROC per deletion level and protocol, where the provenance arm was trained."""
    if not path.is_file():
        return None
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in json.loads(path.read_text()).get("records", []):
        if record.get("test_auroc") is None:
            continue
        level = f"{record['deletion_fraction']:.2f}"
        grouped.setdefault(level, {}).setdefault(record["protocol"], []).append(record["test_auroc"])
    # Six places, not four. The macro writer formats these to three, and
    # rounding twice moves the last digit whenever the mean lands just below a
    # midpoint: 0.86546 becomes 0.8655 becomes 0.866, where the value rounds
    # to 0.865. The manuscript published that digit.
    return {level: {proto: round(sum(v) / len(v), 6) for proto, v in protos.items()}
            for level, protos in sorted(grouped.items())} or None


def provenance_shares(path: Path) -> dict:
    """Mean residual leakage per deletion level and protocol, in both units.

    The dose-response experiment injects a share of test *scans*, so the scan
    share is the one the two curves compose through; the participant share is
    reported because it is the quantity a reader of a split audit sees.
    """
    if not path.is_file():
        return {}
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in json.loads(path.read_text()).get("records", []):
        level = f"{record['deletion_fraction']:.2f}"
        per = grouped.setdefault(level, {}).setdefault(record["protocol"], {})
        per.setdefault("participants", []).append(record["test_train_subject_overlap"])
        per.setdefault("scans", []).append(record.get("test_scan_contamination"))
        per.setdefault("straddling", []).append(record["n_subjects_straddling_partitions"])
    out: dict[str, dict[str, dict[str, float]]] = {}
    for level, protocols in grouped.items():
        for protocol, fields in protocols.items():
            clean = {k: [v for v in vals if v is not None] for k, vals in fields.items()}
            out.setdefault(level, {})[protocol] = {
                k: round(sum(v) / len(v), 4) for k, v in clean.items() if v}
    return out


def hierarchical(runs_root: Path, tables: Path, arm: str, protocols: list[str], seeds: list[int],
                 n_boot: int) -> dict | None:
    out = tables / "bootstrap" / f"{arm}_hierarchical.json"
    cmd = [PY, "scripts/hierarchical_bootstrap_adni.py", "--runs-root", str(runs_root / arm),
           "--seeds", *[str(s) for s in seeds], "--protocols", *protocols,
           "--n-boot", str(n_boot), "--output", str(out)]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr.strip()[:200]}
    return json.loads(out.read_text())["inflation_gap"]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, the convention the other bootstraps use."""
    ordered = sorted(v for v in values if v == v)
    if not ordered:
        return float("nan")
    rank = (len(ordered) - 1) * q / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (1 - (rank - low)) + ordered[high] * (rank - low)


def cross_cohort_bootstrap(analysis: dict, leaky: str, safe: str, seed_labels: list[int],
                           n_boot: int = 10000) -> dict | None:
    """Paired-seed bootstrap in the schema the cross-cohort and forest figures read."""
    seeds = [s for s in seed_labels if s in analysis["seeds_complete"]]
    if len(seeds) < 2:
        return None
    a = [analysis["auroc_per_seed"][leaky][s] for s in seeds]
    b = [analysis["auroc_per_seed"][safe][s] for s in seeds]
    rng = random.Random(0)
    means_a, means_b, gaps = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(len(seeds)) for _ in seeds]
        ma = sum(a[i] for i in idx) / len(idx)
        mb = sum(b[i] for i in idx) / len(idx)
        means_a.append(ma); means_b.append(mb); gaps.append(ma - mb)

    def interval(values):
        return round(percentile(values, 2.5), 4), round(percentile(values, 97.5), 4)

    def block(values, draws):
        lo, hi = interval(draws)
        return {"point_mean": round(sum(values) / len(values), 4), "ci_lo": lo, "ci_hi": hi,
                "seed_values": values}

    lo, hi = interval(gaps)
    return {"source": f"{leaky} versus {safe} per-image predictions, scripts/gpu_postprocess.py",
            "seed_labels": seeds, "n_seeds": len(seeds), "n_boot": n_boot, "ci_pct": 95,
            "auroc": {"leaky": block(a, means_a), "splitguard": block(b, means_b)},
            "inflation_gap_leaky_minus_splitguard": {
                "point": round(sum(a) / len(a) - sum(b) / len(b), 4), "ci_lo": lo, "ci_hi": hi,
                "direction_preserved_share": round(sum(1 for g in gaps if g > 0) / len(gaps), 4)}}


def gpu_values(summary: dict) -> dict[str, object]:
    """The numbers the manuscript quotes, keyed by LaTeX macro name.

    Every interval and gap comes from the same bootstrap artefact the tables,
    figures and scripts/verify_paper_numbers.py read, so the text cannot drift
    from them: the ADNI arms from bootstrap_adni_inflation_gap.py, the other
    arms from cross_cohort_bootstrap().
    """
    def arm(name):
        return summary["arms"].get(name) or {}

    def adni(name, protocol=None):
        payload = summary.get("adni_bootstrap", {}).get(name) or {}
        if protocol:
            return ((payload.get("auroc") or {}).get(protocol) or {}).get("point_mean")
        return (payload.get("inflation_gap") or {}).get("total_random_minus_component_safe") or {}

    def subject_part(name):
        payload = summary.get("adni_bootstrap", {}).get(name) or {}
        return ((payload.get("inflation_gap") or {})
                .get("subject_leakage_random_minus_subject_only") or {})

    def marginal(name):
        payload = summary.get("adni_bootstrap", {}).get(name) or {}
        return ((payload.get("inflation_gap") or {})
                .get("component_leakage_subject_only_minus_component_safe") or {})

    def arm_spread(summary):
        totals = [((summary.get("adni_bootstrap", {}).get(a) or {}).get("inflation_gap") or {})
                  .get("total_random_minus_component_safe", {}).get("point_estimate")
                  for a in ("adni", "adni_with_converters", "adni_no_mt1", "adni_densenet121")]
        totals = [t for t in totals if t is not None]
        return None if len(totals) < 2 else round(max(totals) - min(totals), 3)

    def paired(name, part):
        payload = summary.get("paired_bootstrap", {}).get(name) or {}
        if part in ("leaky", "splitguard"):
            return ((payload.get("auroc") or {}).get(part) or {}).get("point_mean")
        return (payload.get("inflation_gap_leaky_minus_splitguard") or {}).get(part)

    def ci(lo, hi):
        return None if lo is None or hi is None else f"[{lo:+.3f}, {hi:+.3f}]"

    def minus(x, y):
        return None if x is None or y is None else round(x - y, 4)

    def pct(value):
        return None if value is None else f"{100 * value:.1f}\\%"

    t1_random, t1_true = paired("tier1", "leaky"), paired("tier1", "splitguard")
    t1_file = (arm("tier1").get("auroc_mean") or {}).get("filename_subject")
    t1_total, t1_from_file = paired("tier1", "point"), minus(t1_random, t1_file)
    hierarchical = ((arm("adni").get("hierarchical_bootstrap") or {})
                    .get("total_random_minus_component_safe") or {})
    probes = summary.get("identity_probes", {})
    card = summary.get("report_card") or {}
    provenance = summary.get("provenance_auroc") or {}

    analysis = summary.get("analysis", {})

    def dose(field):
        fit = ((analysis.get("adni_dose_response.json") or {}).get("linear_fits") or {}).get("resnet18") or {}
        return fit.get(field)

    def dose_dense(field):
        fit = ((analysis.get("adni_dose_response.json") or {}).get("linear_fits") or {}).get("densenet121") or {}
        return fit.get(field)

    def cost(protocol):
        by = ((analysis.get("adni_cost_of_leakage.json") or {}).get("by_protocol") or {}).get(protocol) or {}
        return by.get("mean_sens_at_fixed_spec")

    def missed():
        anchors = (analysis.get("adni_cost_of_leakage.json") or {}).get("cost_of_leakage") or {}
        value = (anchors.get("prev_population") or {}).get("additional_missed_if_trusting_leaky")
        return None if value is None else round(value, 1)

    def share_subject():
        gaps = (summary.get("adni_bootstrap", {}).get("adni") or {}).get("inflation_gap") or {}
        total = (gaps.get("total_random_minus_component_safe") or {}).get("point_estimate")
        subject = (gaps.get("subject_leakage_random_minus_subject_only") or {}).get("point_estimate")
        return None if not total or subject is None else round(100 * subject / total)

    shares = summary.get("provenance_shares", {})

    def share_pct(level, protocol, unit):
        value = ((shares.get(level) or {}).get(protocol) or {}).get(unit)
        return None if value is None else f"{100 * value:.0f}\\%"

    def composed(protocol):
        value = ((shares.get("0.10") or {}).get(protocol) or {}).get("scans")
        slope = dose("slope")
        return None if value is None or slope is None else round(value * slope, 3)

    def lift(name):
        return ((probes.get(name) or {}).get("component_safe") or {}).get("lift_mean")

    return {
        "TierOneRandomAUROC": t1_random,
        "TierOneFilenameAUROC": t1_file,
        "TierOneTrueAUROC": t1_true,
        "TierOneGapTotal": t1_total,
        "TierOneGapCI": ci(paired("tier1", "ci_lo"), paired("tier1", "ci_hi")),
        "TierOneGapFilename": t1_from_file,
        "TierOneGapResidual": minus(t1_file, t1_true),
        "TierOneShareFilename": (None if not t1_total or t1_from_file is None
                                 else round(100 * t1_from_file / t1_total, 1)),
        "AdniRandomAUROC": adni("adni", "random"),
        "AdniSubjectAUROC": adni("adni", "subject_only"),
        "AdniSafeAUROC": adni("adni", "component_safe"),
        "AdniGapTotal": adni("adni").get("point_estimate"),
        "AdniGapCI": ci(adni("adni").get("ci_lo"), adni("adni").get("ci_hi")),
        "AdniHierGap": hierarchical.get("point"),
        "AdniHierCI": ci(hierarchical.get("ci_lo"), hierarchical.get("ci_hi")),
        "AdniGapExactKey": (arm("adni").get("exact_visit_key_only") or {}).get("gap_mean"),
        "AdniDenseGap": adni("adni_densenet121").get("point_estimate"),
        "AdniConvGap": adni("adni_with_converters").get("point_estimate"),
        "AdniNoMtOneGap": adni("adni_no_mt1").get("point_estimate"),
        "AdniNoMtOneCI": ci(adni("adni_no_mt1").get("ci_lo"), adni("adni_no_mt1").get("ci_hi")),
        "SizeBalancedGap": adni("adni_size_balanced").get("point_estimate"),
        "AdniSubjectPart": subject_part("adni").get("point_estimate"),
        "AdniSubjectPartCI": ci(subject_part("adni").get("ci_lo"), subject_part("adni").get("ci_hi")),
        "AdniMarginal": marginal("adni").get("point_estimate"),
        "AdniMarginalCI": ci(marginal("adni").get("ci_lo"), marginal("adni").get("ci_hi")),
        "AdniConvMarginal": marginal("adni_with_converters").get("point_estimate"),
        "AdniArmSpread": arm_spread(summary),
        "OasisRandomAUROC": paired("oasis1", "leaky"),
        "OasisSafeAUROC": paired("oasis1", "splitguard"),
        "OasisGapTotal": paired("oasis1", "point"),
        "OasisGapCI": ci(paired("oasis1", "ci_lo"), paired("oasis1", "ci_hi")),
        "OasisDenseGap": paired("oasis1_densenet121", "point"),
        "VolGapTotal": paired("adni_3d", "point"),
        "VolGapCI": ci(paired("adni_3d", "ci_lo"), paired("adni_3d", "ci_hi")),
        "ProbeLiftImage": lift("adni_biometric_probe_image"),
        "ProbeLiftSession": lift("adni_biometric_probe_session"),
        "ProvScanTenSubject": share_pct("0.10", "subject_only", "scans"),
        "ProvScanTenSafe": share_pct("0.10", "component_safe", "scans"),
        "ProvParticipantTenSubject": share_pct("0.10", "subject_only", "participants"),
        "ComposedTenSubject": composed("subject_only"),
        "ComposedTenSafe": composed("component_safe"),
        "ProvSubjectAUROCTen": (provenance.get("0.10") or {}).get("subject_only"),
        "ProvSafeAUROCTen": (provenance.get("0.10") or {}).get("component_safe"),
        "CardSens": pct(card.get("sensitivity")),
        "CardSpec": pct(card.get("specificity")),
        "CardBalAcc": pct(card.get("balanced_accuracy")),
        "CardAUROC": card.get("auroc"),
        "CardNTest": None if card.get("n_test") is None else f"{card['n_test']:.0f}",
        "CardPPVPop": pct(card.get("ppv_population")),
        "CardNPVPop": pct(card.get("npv_population")),
        "CardPPVClinic": pct(card.get("ppv_memory_clinic")),
        "CardNPVClinic": pct(card.get("npv_memory_clinic")),
        "GpuHours": None if summary.get("gpu_hours") is None else f"{summary['gpu_hours']:.1f}",
        "DoseSlope": dose("slope"),
        "DoseAtZero": None if dose("intercept") is None else round(dose("intercept"), 3),
        "DoseDenseAtZero": (None if dose_dense("intercept") is None
                            else round(dose_dense("intercept"), 3)),
        "DoseDenseSlope": dose_dense("slope"),
        "DoseAtOne": (None if dose("intercept") is None or dose("slope") is None
                      else round(dose("intercept") + dose("slope"), 3)),
        "AdniConvRandomAUROC": adni("adni_with_converters", "random"),
        "AdniConvSafeAUROC": adni("adni_with_converters", "component_safe"),
        "DoseIntercept": dose("intercept"),
        "DoseRSq": dose("r2"),
        "SensLeaky": cost("random"),
        "SensSafe": cost("component_safe"),
        "MissedPerThousand": missed(),
        "AdniShareSubject": share_subject(),
    }


def latex_macros(summary: dict, path: Path) -> None:
    """Write one \\newcommand per quoted number; a number with no run renders as ??."""
    lines = ["% Generated by scripts/gpu_postprocess.py. Do not edit by hand.",
             "% \\pending marks a number whose run has not completed.",
             "\\providecommand{\\pending}{\\textbf{??}}"]
    for name, value in gpu_values(summary).items():
        if value is None:
            rendered = "\\pending{}"
        elif isinstance(value, str):
            rendered = value
        elif name == "AdniShareSubject":
            rendered = f"{value:.0f}"
        elif name == "MissedPerThousand":
            rendered = f"{value:.1f}"
        elif "Share" in name or "Lift" in name:
            rendered = f"{value:.1f}"
        elif "Gap" in name:
            rendered = f"{value:+.3f}"
        else:
            rendered = f"{value:.3f}"
        lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_tables(seed_tables: list[Path], output: Path) -> None:
    """Concatenate per-seed ADNI tables into the multi-seed table the bootstrap reads."""
    rows, fieldnames, seen = [], None, set()
    for table in seed_tables:
        with table.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"{table.name} has different columns from the first per-seed table")
            for row in reader:
                key = (row["seed"], row["protocol"])
                if key in seen:
                    raise SystemExit(f"{table.name}: seed {key[0]} protocol {key[1]} appears twice; "
                                     "the bootstrap would silently keep only the last row")
                seen.add(key)
                rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "\\pending{}"
    mean = sum(values) / len(values)
    if len(values) < 2:
        return f"${mean:.3f}$"
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return f"${mean:.3f} \\pm {sd:.3f}$"


def oasis_table(tables: Path, path: Path, seeds: list[int], analysis: dict | None = None) -> None:
    """OASIS-1 replication table, mean +/- SD over seeds, from the per-seed run JSONs."""
    metrics = [("auroc", "AUROC"), ("balanced_accuracy", "Balanced accuracy"),
               ("f1_demented", "F1 (demented)"), ("sensitivity", "Sensitivity"),
               ("specificity", "Specificity")]
    per = {"random": {m: [] for m, _ in metrics}, "component_safe": {m: [] for m, _ in metrics}}
    paired = {m: [] for m, _ in metrics}
    overlap = []
    for seed in seeds:
        run = tables / "oasis1" / f"oasis1_resnet18_seed{seed}.json"
        if not run.is_file():
            continue
        payload = json.loads(run.read_text())
        leak = payload.get("leaky_subject_overlap") or {}
        if "overlap_pct_of_test" in leak:
            overlap.append(leak["overlap_pct_of_test"])
        # Match on the block key, not on the run's label: the OASIS-1 runner
        # names its protocols descriptively (A_LEAKY_random_slice_split), so a
        # label match silently left every metric row empty.
        blocks = {}
        for name, protocol in (("protocol_A_leaky", "random"), ("protocol_B_safe", "component_safe")):
            block = payload.get(name)
            if block:
                blocks[protocol] = block["test_metrics"]
                for metric, _ in metrics:
                    per[protocol][metric].append(block["test_metrics"][metric])
        if len(blocks) == 2:
            for metric, _ in metrics:
                paired[metric].append(blocks["random"][metric] - blocks["component_safe"][metric])

    # The AUROC row comes from the same per-image predictions the macros use,
    # so the table and the text cannot disagree about the headline metric.
    if analysis:
        for protocol in ("random", "component_safe"):
            by_seed = (analysis.get("auroc_per_seed") or {}).get(protocol) or {}
            values = [by_seed[s] for s in analysis.get("seeds_complete", []) if s in by_seed]
            if values:
                per[protocol]["auroc"] = values
        complete = analysis.get("seeds_complete", [])
        paired_auroc = [(analysis["auroc_per_seed"]["random"][s]
                         - analysis["auroc_per_seed"]["component_safe"][s]) for s in complete]
        if paired_auroc:
            paired["auroc"] = paired_auroc

    rows = []
    for metric, label in metrics:
        rows.append(f"{label} & {_mean_sd(per['random'][metric])} & "
                    f"{_mean_sd(per['component_safe'][metric])} & "
                    f"\\gap{{{_mean_sd(paired[metric])}}} \\\\")
    leaking = (f"${sum(overlap) / len(overlap):.2f}\\%$" if overlap else "\\pending{}")
    rows.append(f"Test subjects leaking & {leaking} & $0\\%$ & n/a \\\\")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "% Generated by scripts/gpu_postprocess.py. Do not edit by hand.\n"
        "\\begin{table}[!ht]\n\\centering\n"
        "\\caption{OASIS-1 replication: random slice split against the participant-safe "
        "\\SGA{} split, mean $\\pm$ SD over five ResNet-18 seeds at 20 epochs.}\n"
        "\\label{tab:oasis_replication}\n\\begin{tabular}{lrrr}\n\\toprule\n"
        "\\textbf{Metric} & \\textbf{Protocol A} & \\textbf{Protocol C} & "
        "\\textbf{Gap (A$-$C)} \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def robustness_table(summary: dict, path: Path) -> None:
    """ADNI robustness arms: total gap, subject part and component marginal, per arm."""
    arms = [("Primary", "adni"),
            ("Converter-inclusive", "adni_with_converters"),
            ("MT1 excluded", "adni_no_mt1"),
            ("DenseNet-121", "adni_densenet121"),
            ("Size-balanced", "adni_size_balanced"),
            ("Exact label linkage", "adni_exact_linkage")]
    keys = ["total_random_minus_component_safe",
            "subject_leakage_random_minus_subject_only",
            "component_leakage_subject_only_minus_component_safe"]

    def cell(gap):
        if not gap or gap.get("point_estimate") is None:
            return "\\pending{}"
        return f"${gap['point_estimate']:+.3f}$ $[{gap['ci_lo']:+.3f}, {gap['ci_hi']:+.3f}]$"

    rows = []
    for label, arm in arms:
        payload = (summary.get("adni_bootstrap", {}).get(arm) or {}).get("inflation_gap") or {}
        rows.append(f"{label} & " + " & ".join(cell(payload.get(k)) for k in keys) + " \\\\")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "% Generated by scripts/gpu_postprocess.py. Do not edit by hand.\n"
        "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
        "\\caption{ADNI robustness arms; the primary arm excludes converters. "
        "Columns are the total gap (Protocol~A$-$C), "
        "which is the paper's headline, the subject part (A$-$B) and the component "
        "marginal (B$-$C), which is its null result; all three are measured on the same "
        "runs. Intervals are paired-seed bootstrap at $B = 10{,}000$.}\n"
        "\\label{tab:adni_robustness}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "\\textbf{Arm} & \\textbf{A$-$C} & \\textbf{A$-$B} & "
        "\\textbf{B$-$C} \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def subject_level_table(path: Path, source: Path) -> None:
    """Image-level against subject-level AUROC, from the subject-aggregation artefact."""
    rows, gaps = [], []
    if source.is_file():
        payload = json.loads(source.read_text())
        names = {"random": "Random (leaky)", "subject_only": "Subject-only",
                 "component_safe": "Component-safe (\\SGA{})"}
        for key, label in names.items():
            block = (payload.get("by_protocol") or {}).get(key) or {}
            cells = []
            for level in ("image_level", "subject_level"):
                stats = block.get(level) or {}
                cells.append(f"{stats['mean']:.3f} [{stats['ci_lo']:.3f}, {stats['ci_hi']:.3f}]"
                             if stats else "\\pending{}")
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        for level, label in (("image_level", "Image-level"), ("subject_level", "Subject-level")):
            gap = ((payload.get("inflation_gap") or {}).get(level) or {}).get("total_gap_A_minus_C") or {}
            positive = sum(1 for d in gap.get("delta_per_seed", []) if d > 0) if gap else 0
            cell = (f"${gap['delta_mean']:+.3f}$ [${gap['delta_ci_lo']:+.3f}$, ${gap['delta_ci_hi']:+.3f}$], "
                    f"positive in {positive}/{len(gap['delta_per_seed'])} seeds"
                    if gap else "\\pending{}")
            gaps.append(f"{label} & \\multicolumn{{2}}{{l}}{{{cell}}} \\\\")
    else:
        rows = [f"{n} & \\pending{{}} & \\pending{{}} \\\\"
                for n in ("Random (leaky)", "Subject-only", "Component-safe (\\SGA{})")]
        gaps = [f"{n} & \\multicolumn{{2}}{{l}}{{\\pending{{}}}} \\\\"
                for n in ("Image-level", "Subject-level")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "% Generated by scripts/gpu_postprocess.py. Do not edit by hand.\n"
        "\\begin{table}[!ht]\n\\centering\n\\small\n"
        "\\caption{ADNI image-level against subject-level AUROC on the converter-inclusive "
        "arm (five seeds). Both columns are AUROC with a paired-seed bootstrap interval "
        "at $B = 10{,}000$. Subject-level numbers "
        "average each participant's per-image predictions before computing AUROC.}\n"
        "\\label{tab:adni_subject_level}\n\\begin{tabular}{lrr}\n\\toprule\n"
        "\\textbf{Protocol} & \\textbf{Image-level} & "
        "\\textbf{Subject-level} \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\midrule\n\\multicolumn{3}{l}{\\textit{Inflation gap (Protocol A $-$ Protocol C):}} "
        "\\\\\n" + "\n".join(gaps) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")


def operating_point_table(path: Path, source: Path) -> None:
    """Supplementary operating-point table, from the sensitivity-sweep artefact."""
    groups = [("Sensitivity at fixed specificity", [(f"Spec $={s}$", f"sens_at_spec_{s}") for s in
                                                    ("0.80", "0.85", "0.90", "0.95")]),
              ("Specificity at fixed sensitivity", [(f"Sens $={s}$", f"spec_at_sens_{s}") for s in
                                                    ("0.85", "0.90", "0.95")]),
              ("Youden-optimal threshold (selected on validation)",
               [("J", "youden_j"), ("Sens at J", "youden_sens"), ("Spec at J", "youden_spec")])]
    payload = json.loads(source.read_text()) if source.is_file() else {}
    lines = []
    for title, entries in groups:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for label, key in entries:
            cells = []
            for protocol in ("random", "subject_only", "component_safe"):
                block = payload.get(protocol) or {}
                mean, sd = block.get(f"{key}_mean"), block.get(f"{key}_sd")
                if mean is None:
                    mean = block.get(f"mean_{key.replace('youden_', 'youden_')}")
                cells.append("\\pending{}" if mean is None else
                             (f"${mean:.3f} \\pm {sd:.3f}$" if sd is not None else f"${mean:.3f}$"))
            delta = (payload.get("_leaky_minus_honest") or {}).get(key)
            cells.append("\\pending{}" if delta is None else f"${delta:+.3f}$")
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if title != groups[-1][0]:
            lines.append("\\midrule")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "% Generated by scripts/gpu_postprocess.py. Do not edit by hand.\n"
        "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
        "\\caption{Operating-point sensitivity table for the ADNI converter-inclusive arm. "
        "\\textbf{Every threshold is selected on the validation partition and applied to "
        "test}; no operating point is chosen by scanning the test ROC. The fixed-rate "
        "rows are mean $\\pm$ SD across five seeds and the Youden rows are seed means, "
        "with $\\Delta$ the Protocol~A minus Protocol~C difference.}\n"
        "\\label{tab:sm_operating_point}\n"
        "\\begin{tabular}{@{}lccc r@{}}\n\\toprule\n"
        "Operating point & Random & Subject-only & Component-safe & $\\Delta$ \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")


def tier1_table(summary: dict, path: Path) -> None:
    """Per-seed Tier-1 AUROC table, in the layout the manuscript inputs."""
    arm = summary["arms"].get("tier1") or {}
    per_seed = arm.get("auroc_per_seed") or {}
    seeds = [42, 0, 1, 2, 3]
    protocols = ["random", "filename_subject", "true_participant"]

    def cell(protocol, seed):
        value = (per_seed.get(protocol) or {}).get(seed) or (per_seed.get(protocol) or {}).get(str(seed))
        return f"{value:.4f}" if isinstance(value, (int, float)) else "\\pending{}"

    rows = []
    for seed in seeds:
        values = [(per_seed.get(p) or {}).get(seed) or (per_seed.get(p) or {}).get(str(seed)) for p in protocols]
        gap = (f"{values[0] - values[2]:+.3f}" if all(isinstance(v, (int, float)) for v in (values[0], values[2]))
               else "\\pending{}")
        rows.append(f"{seed} & " + " & ".join(cell(p, seed) for p in protocols) + f" & \\gap{{{gap}}} \\\\")

    def summary_cell(protocol):
        values = [v for v in ((per_seed.get(protocol) or {}).values()) if isinstance(v, (int, float))]
        if len(values) < 2:
            return "\\pending{}"
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        return f"${mean:.3f} \\pm {sd:.3f}$"

    gap_mean = arm.get("gap_mean")
    body = "\n".join(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "% Generated by scripts/gpu_postprocess.py. Do not edit by hand.\n"
        "\\begin{table}[!ht]\n\\centering\n"
        "\\caption{Tier-1 test AUROC under the three grouping rules, five seeds. "
        "Protocol~B$'$ groups by the filename key the release ships, which is the best an "
        "audit inside the release can do; Protocol~C$'$ groups by the participant identity "
        "recovered from OASIS-1. The gap column is A$-$C$'$.}\n"
        "\\label{tab:tier1_truth}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "\\textbf{Seed} & \\textbf{A: random} & \\textbf{B$'$: filename key} & "
        "\\textbf{C$'$: participant} & \\textbf{Gap (A$-$C$'$)} \\\\\n\\midrule\n"
        f"{body}\n\\midrule\n"
        f"\\textbf{{Mean $\\pm$ SD}} & {summary_cell('random')} & {summary_cell('filename_subject')} & "
        f"{summary_cell('true_participant')} & "
        f"\\gap{{{f'{gap_mean:+.3f}' if isinstance(gap_mean, (int, float)) else chr(92) + 'pending{}'}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs_gpu")
    ap.add_argument("--tables", type=Path, default=ROOT / "reports" / "gpu")
    ap.add_argument("--output", type=Path, default=None, help="Summary JSON (default <tables>/summary.json).")
    ap.add_argument("--macros", type=Path, default=ROOT / "paper" / "tables" / "gpu_numbers.tex")
    ap.add_argument("--tier1-table", type=Path, default=ROOT / "paper" / "tables" / "tier1_truth.tex")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    output = args.output or args.tables / "summary.json"

    exact = exact_key_scans()
    summary: dict = {"runs_root": args.runs_root.name, "arms": {}, "missing_arms": [],
                     "exact_visit_key_scans": len(exact)}
    for arm, (protocols, seeds) in ARMS.items():
        analysis = analyse_arm(args.runs_root, arm, protocols, seeds, exact)
        if analysis is None:
            summary["missing_arms"].append(arm)
            continue
        cached = args.tables / "bootstrap" / f"{arm}_hierarchical.json"
        if not args.skip_bootstrap and analysis["seeds_complete"]:
            analysis["hierarchical_bootstrap"] = hierarchical(
                args.runs_root, args.tables, arm, protocols, analysis["seeds_complete"], args.n_boot)
        elif cached.is_file():
            # --skip-bootstrap means "do not recompute", not "forget": the
            # manuscript quotes the hierarchical interval, so reuse the one
            # already on disk rather than letting the macro fall back to ??.
            analysis["hierarchical_bootstrap"] = json.loads(cached.read_text())["inflation_gap"]
        summary["arms"][arm] = analysis

    # ADNI arms: the multi-seed table and paired-seed bootstrap that the paper
    # tables, the figures and the verifier all read.
    summary["adni_bootstrap"] = {}
    for arm in ("adni", "adni_with_converters", "adni_no_mt1", "adni_densenet121",
                "adni_size_balanced", "adni_exact_linkage"):
        seed_tables = sorted((args.tables / "adni").glob(f"{arm}_seed?.csv"))
        if not seed_tables:
            continue
        suffix = arm[len("adni"):]
        merged = args.tables / "adni" / f"adni_inflation_gap{suffix}.csv"
        boot = args.tables / "adni" / f"adni_inflation_gap{suffix}_bootstrap.json"
        merge_tables(seed_tables, merged)
        run = subprocess.run([PY, "scripts/bootstrap_adni_inflation_gap.py", "--input", str(merged),
                              "--output", str(boot), "--n-boot", str(args.n_boot)],
                             cwd=ROOT, capture_output=True, text=True)
        if run.returncode == 0:
            summary["adni_bootstrap"][arm] = json.loads(boot.read_text())
        hier_json = args.tables / "bootstrap" / f"{arm}_hierarchical.json"
        if hier_json.is_file():
            shutil.copy(hier_json, args.tables / "adni" / f"adni_inflation_gap{suffix}_hierarchical.json")

    # Every other arm: the same paired-seed bootstrap, in the schema the
    # cross-cohort and forest figures read.
    summary["paired_bootstrap"] = {}
    for arm, leaky, safe in (("tier1", "random", "true_participant"),
                             ("oasis1", "random", "component_safe"),
                             ("oasis1_densenet121", "random", "component_safe"),
                             ("adni_3d", "random", "component_safe")):
        if arm in summary["arms"]:
            payload = cross_cohort_bootstrap(summary["arms"][arm], leaky, safe, ARMS[arm][1], args.n_boot)
            if payload:
                summary["paired_bootstrap"][arm] = payload
    for arm, name in (("tier1", "jpeg_inflation_gap_bootstrap.json"),
                      ("oasis1", "oasis1_inflation_gap_bootstrap.json")):
        if arm in summary["paired_bootstrap"]:
            target = args.tables / "cross_cohort" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(summary["paired_bootstrap"][arm], indent=1) + "\n", encoding="utf-8")

    # The volumetric arm is not in the ADNI paired-seed loop above, so its
    # hierarchical interval was the one the manuscript quotes from a path the
    # release does not carry. Promote it the same way.
    vol_hier = args.tables / "bootstrap" / "adni_3d_hierarchical.json"
    if vol_hier.is_file():
        shutil.copy(vol_hier, args.tables / "adni" / "adni_inflation_gap_3d_hierarchical.json")

    # Same for the cross-cohort arms. The manuscript quotes their hierarchical
    # intervals, the bootstrap tree is not released, and a reader checking the
    # paper against the repository would otherwise find those claims
    # unverifiable. The artefacts are seed-level aggregates with no identifiers.
    for arm in ("tier1", "oasis1", "oasis1_densenet121"):
        src = args.tables / "bootstrap" / f"{arm}_hierarchical.json"
        if src.is_file():
            dst = ROOT / "reports" / "tables" / f"{arm}_hierarchical.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    # Numbers the analysis scripts compute, so the prose can quote them as macros.
    summary["analysis"] = {}
    for name in ("adni_dose_response.json", "adni_cost_of_leakage.json"):
        for folder in (ROOT / "reports" / "tables" / "adni", args.tables / "adni"):
            if (folder / name).is_file():
                summary["analysis"][name] = json.loads((folder / name).read_text())
                break

    summary["report_card"] = report_card(args.runs_root / "adni")
    summary["provenance_auroc"] = provenance_auroc(args.tables / "adni" / "adni_provenance_degradation.json")
    for folder in (args.tables / "adni", ROOT / "reports" / "tables" / "adni"):
        shares = provenance_shares(folder / "adni_provenance_degradation.json")
        if shares:
            summary["provenance_shares"] = shares
            break
    ledger = args.runs_root / "gpu_ledger.json"
    summary["gpu_hours"] = (round(json.loads(ledger.read_text())["billed_seconds"] / 3600, 1)
                            if ledger.is_file() else None)
    summary["identity_probes"] = {
        path.stem: json.loads(path.read_text()).get("summary_per_protocol", {})
        for path in sorted((args.tables / "adni").glob("adni_biometric_probe*.json"))}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latex_macros(summary, args.macros)
    tier1_table(summary, args.tier1_table)
    oasis_table(args.tables, args.tier1_table.parent / "oasis_replication.tex", TIER_SEEDS,
                summary["arms"].get("oasis1"))
    tables_root = ROOT / "reports" / "tables" / "adni"
    subject_level_table(args.tier1_table.parent / "adni_subject_level.tex",
                        tables_root / "adni_subject_level_auroc.json")
    operating_point_table(args.tier1_table.parent / "operating_point.tex",
                          tables_root / "adni_operating_point_sensitivity.json")
    robustness_table(summary, args.tier1_table.parent / "adni_robustness.tex")

    for arm, data in summary["arms"].items():
        print(f"{arm:<22} seeds {len(data['seeds_complete'])}/{len(ARMS[arm][1])}  "
              f"AUROC " + "  ".join(f"{p}={data['auroc_mean'][p]}" for p in data["protocols"]) +
              f"  gap {data['gap_mean']}")
    if summary["missing_arms"]:
        print(f"no predictions for: {', '.join(summary['missing_arms'])}")
    pending = [name for name, value in gpu_values(summary).items() if value is None]
    print(f"wrote {display_path(output)} and {display_path(args.macros)}; "
          f"{len(pending)} manuscript number(s) pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
