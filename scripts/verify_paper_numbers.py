#!/usr/bin/env python3
"""Verify that every headline number in the manuscript matches its source artefact.

The manuscript quotes results that are produced by the analysis scripts and
written to ``reports/**/*.json``. Nothing enforces that the two stay in step:
re-running an analysis silently changes the JSON while the LaTeX keeps the old
value, and a reader has no way to tell. This script closes that loop.

For each checked claim we state where the number comes from (the JSON path),
what the paper should say, and whether it says it. A claim fails if the value
is absent from the sources at the expected precision.

Usage
-----
    python3 scripts/verify_paper_numbers.py
    python3 scripts/verify_paper_numbers.py --verbose   # also list passes

Exit code is non-zero if any check fails, so this can gate a release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import display_path  # noqa: E402 -- sys.path tweak above

PAPER = ROOT / "paper" / "splitguard_ad.tex"
SM = ROOT / "paper" / "SplitGuard-AD_Supplementary_Material.tex"
HIGHLIGHTS = ROOT / "paper" / "SplitGuard-AD_Highlights.txt"
TABLES = ROOT / "reports" / "tables" / "adni"
MACROS = ROOT / "paper" / "tables" / "gpu_numbers.tex"
PAPER_TABLES = ROOT / "paper" / "tables"
PENDING = "\\pending{}"


def load_macros(path: Path) -> dict[str, str]:
    """Macro name -> body, from the file scripts/gpu_postprocess.py writes."""
    if not path.is_file():
        return {}
    return dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", path.read_text(), re.M))


def expand(text: str, macros: dict[str, str]) -> str:
    """Replace each generated macro by its value, as LaTeX will when it compiles."""
    for name, body in macros.items():
        text = re.sub(r"\\" + name + r"(\{\})?(?![A-Za-z])", lambda _m, b=body: b, text)
    return text


# Artefacts that could not be read. A missing artefact does not make any
# number wrong, but it silently removes that number from the audit -- which
# would let the script report a clean pass over claims it never attempted.
# That is the failure mode this manuscript is about, so it is reported.
MISSING: list[str] = []


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        MISSING.append(str(path))
        return None
    except Exception as exc:
        MISSING.append(f"{path} ({type(exc).__name__})")
        return None


def fmt(value: float, places: int) -> str:
    text = f"{value:.{places}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text  # no "-0.000"


class Checker:
    def __init__(self, macros: dict[str, str] | None = None) -> None:
        self.macros = macros or {}
        self.sources: dict[str, str] = {}
        self.failures: list[tuple[str, str, str]] = []
        self.passes: list[tuple[str, str]] = []

    def add_source(self, name: str, path: Path) -> None:
        if path.exists():
            self.sources[name] = expand(path.read_text(), self.macros)

    def check_text(self, label: str, needle: str, *, origin: str) -> None:
        """Assert that a formatted quantity (e.g. a power of ten) appears verbatim."""
        source = self._present(needle)
        if source:
            self.passes.append((label, f"{needle} ({source})"))
        else:
            self.failures.append((label, needle, origin))

    def _present(self, needle: str) -> str | None:
        """Name of the first source containing `needle` as a whole number.

        Substring matching made small integers vacuous: "3" occurs inside any
        decimal, and "10000" inside the bootstrap size printed in five
        captions, so those checks passed whatever the artefact said.
        """
        pattern = re.compile(rf"(?<![0-9.,]){re.escape(needle)}(?![0-9])")
        for name, text in self.sources.items():
            if pattern.search(text):
                return name
        return None

    def check(self, label: str, value, places: int, *, origin: str,
              alt: list[str] | None = None) -> None:
        """Assert that `value` (at `places` decimals) appears in the sources."""
        if value is None:
            self.failures.append((label, "<missing from JSON>", origin))
            return
        candidates = [fmt(float(value), places)]
        # Accept the value with or without a leading +, and comma-grouped ints.
        candidates.append(candidates[0].lstrip("0") if candidates[0].startswith("0.") else candidates[0])
        if alt:
            candidates.extend(alt)
        for cand in candidates:
            source = self._present(cand)
            if source:
                self.passes.append((label, f"{cand} ({source})"))
                return
        self.failures.append((label, fmt(float(value), places), origin))

    def check_absent(self, label: str, stale: str, *, replaced_by: str,
                     origin: str, allow_near: tuple[str, ...] = ()) -> None:
        """Assert a superseded value survives nowhere except sanctioned contexts.

        ``check`` only asks whether the *correct* value is present somewhere.
        That is not sufficient: when a number is quoted in several places and
        only some are updated, the correct value is present, the check passes,
        and the stale copies survive. Exactly that happened twice here — an
        R^2 of 0.69 left behind in the Discussion after the dose-response was
        recomputed to 0.72, and a missed-diagnosis count of 39 left behind
        after the operating-point thresholds moved to validation selection.
        Both sat in sections a reviewer reads closely.

        A superseded number is not always an error, though. The manuscript
        deliberately quotes old values when it explains a correction ("falls
        from 0.594 under test selection to 0.533 under validation selection").
        ``allow_near`` lists phrases that legitimise an occurrence: if any
        appears within a window around the number, that occurrence is
        historical reporting rather than a stale leftover.
        """
        window = 240
        for name, text in self.sources.items():
            start = 0
            while True:
                idx = text.find(stale, start)
                if idx == -1:
                    break
                # Collapse whitespace: a sanctioning phrase must not depend on
                # where LaTeX happens to wrap the line it sits on.
                context = " ".join(text[max(0, idx - window): idx + window].split())
                if not any(marker in context for marker in allow_near):
                    self.failures.append(
                        (label, f"stale value {stale!r} still present "
                                f"(superseded by {replaced_by})",
                         f"{origin}; in {name}")
                    )
                    return
                start = idx + len(stale)
        self.passes.append((label, f"stale {stale!r} absent or contextualised"))

    def check_int(self, label: str, value, *, origin: str) -> None:
        if value is None:
            self.failures.append((label, "<missing from JSON>", origin))
            return
        n = int(value)
        # LaTeX writes thousands as 10{,}000
        variants = [str(n), f"{n:,}".replace(",", "{,}"), f"{n:,}"]
        for v in variants:
            source = self._present(v)
            if source:
                self.passes.append((label, f"{v} ({source})"))
                return
        self.failures.append((label, str(n), origin))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    macros = load_macros(MACROS)
    c = Checker(macros)
    c.add_source("paper", PAPER)
    c.add_source("sm", SM)
    c.add_source("highlights", HIGHLIGHTS)
    # The README quotes manuscript numbers too, and nothing used to check it:
    # a Tier-1 gap this project retracted survived there for months.
    c.add_source("readme", ROOT / "README.md")
    for table in sorted(PAPER_TABLES.glob("*.tex")):
        if table != MACROS:
            c.add_source(f"table:{table.stem}", table)
    if not c.sources:
        raise SystemExit("No manuscript sources found.")

    # ── Primary-arm inflation gap ───────────────────────────────────────
    p = TABLES / "adni_inflation_gap_bootstrap.json"
    d = load(p)
    if d:
        g = d["inflation_gap"]["total_random_minus_component_safe"]
        c.check("primary gap (point)", g.get("point_estimate", g.get("point")), 3, origin=str(p))
        c.check("primary gap CI lo", g.get("ci_lo"), 3, origin=str(p))
        c.check("primary gap CI hi", g.get("ci_hi"), 3, origin=str(p))
        s = d["inflation_gap"]["subject_leakage_random_minus_subject_only"]
        c.check("subject-leakage (point)", s.get("point_estimate", s.get("point")), 3, origin=str(p))
        m = d["inflation_gap"]["component_leakage_subject_only_minus_component_safe"]
        c.check("component marginal (point)", m.get("point_estimate", m.get("point")), 3, origin=str(p))
        c.check("component marginal CI lo", m.get("ci_lo"), 3, origin=str(p))
        c.check("component marginal CI hi", m.get("ci_hi"), 3, origin=str(p))
        # The resample direction count is 10,000/10,000 by construction when
        # every seed gap has the same sign, so the manuscript states the seed
        # count instead and must not quote the resample count.
        c.check_absent("resample direction count", "10{,}000/10{,}000",
                       replaced_by="the number of seeds sharing the sign",
                       origin=str(p))

    # ── Converter-inclusive arm ─────────────────────────────────────────
    p = TABLES / "adni_inflation_gap_with_converters_bootstrap.json"
    d = load(p)
    if d:
        g = d["inflation_gap"]["total_random_minus_component_safe"]
        c.check("converter-arm gap CI lo", g.get("ci_lo"), 3, origin=str(p))
        c.check("converter-arm gap CI hi", g.get("ci_hi"), 3, origin=str(p))

    p = TABLES / "adni_inflation_gap_hierarchical.json"
    d = load(p)
    if d:
        g = d["inflation_gap"]["total_random_minus_component_safe"]
        c.check("hierarchical gap (point)", g.get("point_estimate", g.get("point")), 3, origin=str(p))
        c.check("hierarchical CI lo", g.get("ci_lo"), 3, origin=str(p))
        c.check("hierarchical CI hi", g.get("ci_hi"), 3, origin=str(p))

    # ── Dose-response ───────────────────────────────────────────────────
    p = TABLES / "adni_dose_response.json"
    d = load(p)
    if d:
        for arch, places in (("resnet18", 3), ("densenet121", 3)):
            f = d["linear_fits"].get(arch, {})
            c.check(f"{arch} dose slope", f.get("slope"), 3, origin=str(p))
            c.check(f"{arch} dose intercept", f.get("intercept"), 3, origin=str(p))
            c.check(f"{arch} dose r2", f.get("r2"), 2, origin=str(p))
            # sigma parameterises a probabilistic claim in the manuscript, so
            # it is held to the artefact rather than quoted approximately. It
            # was previously written as "approximately 0.020", a value no
            # computation over these runs reproduces.
            c.check(f"{arch} dose residual sd", f.get("residual_sd"), 4, origin=str(p))

    p = TABLES / "adni_dose_response_mixed_effects.json"
    d = load(p)
    if d:
        for arch in ("resnet18", "densenet121"):
            m = d.get(arch, {})
            m = m.get("mixed_effects", m)
            c.check(f"{arch} GLS CI lo", m.get("ci_lo_95"), 4, origin=str(p))
            c.check(f"{arch} GLS CI hi", m.get("ci_hi_95"), 4, origin=str(p))

    # ── Operating points / cost of leakage ──────────────────────────────
    p = TABLES / "adni_operating_point_sensitivity.json"
    d = load(p)
    if d:
        for proto in ("random", "subject_only", "component_safe"):
            for spec in ("0.80", "0.85", "0.90", "0.95"):
                k = f"sens_at_spec_{spec}_mean"
                c.check(f"{proto} sens@spec{spec}", d[proto].get(k), 3, origin=str(p))

    p = TABLES / "adni_cost_of_leakage.json"
    d = load(p)
    if d:
        for proto in ("random", "component_safe"):
            v = d.get("by_protocol", {}).get(proto, {}).get("mean_sens_at_fixed_spec")
            c.check(f"cost-of-leakage sens ({proto})", v, 3, origin=str(p))

    # ── Site/scanner confound audit ─────────────────────────────────────
    p = TABLES / "adni_site_scanner_confound_audit.json"
    d = load(p)
    if d:
        summ = d.get("cross_seed_summary", {})

        def stat(block, key, field="mean"):
            v = block.get(key)
            if isinstance(v, dict):
                return v.get(field)
            return v

        for part in ("train", "val", "test"):
            s = summ.get(part, {})
            c.check(f"confound V(site) {part}",
                    stat(s, "cramers_v_site_label"), 3, origin=str(p))
            c.check_int(f"confound rows {part}",
                        stat(s, "n_rows"), origin=str(p))

    # ── Tier-1 (public benchmark) and Tier-2 (OASIS-1) ──────────────────
    # These two cohorts carry a third of the manuscript's headline claims and
    # were checked by nobody until an audit found the consolidated summary
    # table pairing Tier-1's seed-42 AUROC with its five-seed gap and interval.
    # Presence-only checking on nine of sixty artefacts is how that survived.
    for label, path, key in [
        ("tier1", TABLES.parent / "jpeg_inflation_gap_bootstrap.json",
         "inflation_gap_leaky_minus_splitguard"),
        ("oasis1", TABLES.parent / "oasis1_inflation_gap_bootstrap.json",
         "inflation_gap_leaky_minus_splitguard"),
    ]:
        d = load(path)
        if not d:
            continue
        au = d.get("auroc", {})
        c.check(f"{label} leaky AUROC (5-seed mean)",
                au.get("leaky", {}).get("point_mean"), 4, origin=str(path),
                alt=[fmt(au.get("leaky", {}).get("point_mean", 0), 3)])
        c.check(f"{label} honest AUROC (5-seed mean)",
                au.get("splitguard", {}).get("point_mean"), 3, origin=str(path))
        g = d.get(key, {})
        c.check(f"{label} gap", g.get("point"), 3, origin=str(path))
        c.check(f"{label} gap CI lo", g.get("ci_lo"), 3, origin=str(path))
        c.check(f"{label} gap CI hi", g.get("ci_hi"), 3, origin=str(path))

    # ── Architecture-breadth and acquisition-type sensitivity arms ───────
    for label, name in [("densenet121", "adni_inflation_gap_densenet121_bootstrap.json"),
                        ("no_mt1", "adni_inflation_gap_no_mt1_bootstrap.json")]:
        d = load(TABLES / name)
        if not d:
            continue
        g = d["inflation_gap"]["total_random_minus_component_safe"]
        m = d["inflation_gap"]["component_leakage_subject_only_minus_component_safe"]
        c.check(f"{label} arm gap", g.get("point_estimate"), 3, origin=name)
        c.check(f"{label} arm CI lo", g.get("ci_lo"), 3, origin=name)
        c.check(f"{label} arm CI hi", g.get("ci_hi"), 3, origin=name)
        c.check(f"{label} arm marginal", m.get("point_estimate"), 3, origin=name)

    # ── Subject-identity probe (ADNI, component-safe representations) ─────
    for name, label in (("adni_biometric_probe.json", "image hold-out"),
                        ("adni_biometric_probe_session.json", "session hold-out")):
        d = load(TABLES / name)
        if d:
            stats = (d.get("summary_per_protocol") or {}).get("component_safe") or {}
            c.check(f"ADNI probe lift ({label})", stats.get("lift_mean"), 1, origin=name)

    # ── Tier-1 provenance recovered against OASIS-1 ─────────────────────
    # These numbers replace the filename-key analysis that read broken
    # identifiers as ground truth; every one of them is quoted in the text.
    p = TABLES.parent / "tier1_ground_truth_audit.json"
    d = load(p)
    if d:
        o = str(p)
        m = d["mapping"]
        c.check("tier1 min self correlation", m["min_self_correlation"], 4, origin=o)
        c.check("tier1 max other-participant correlation", m["max_other_participant_correlation"], 4, origin=o)
        for key in ("n_images", "n_participants", "n_oasis_volumes_searched"):
            c.check_int(f"tier1 mapping {key}", m[key], origin=o)
        c.check_int("tier1 slice range start", m["axial_slice_range"][0], origin=o)
        c.check_int("tier1 slice range end", m["axial_slice_range"][1], origin=o)
        c.check_int("tier1 labels equal CDR", d["labels_vs_cdr"]["participants_with_expected_cdr"], origin=o)
        f = d["filename_identity"]
        for key in ("filename_subject_ids", "ids_that_are_one_participant", "ids_merging_participants",
                    "unparsed_images"):
            c.check_int(f"tier1 {key}", f[key], origin=o)
        per_participant = {int(k): v for k, v in f["filename_ids_per_participant"].items()}
        c.check_int("tier1 participants over several keys",
                    sum(v for k, v in per_participant.items() if k >= 2), origin=o)
        c.check_int("tier1 most keys for one participant", max(per_participant), origin=o)
        fr = d["frozen_split_true_leakage"]
        for key in ("participants_straddling_partitions", "test_images",
                    "test_images_whose_participant_is_in_train", "test_participants",
                    "test_participants_also_in_train"):
            c.check_int(f"tier1 frozen split {key}", fr[key], origin=o)
        c.check("tier1 frozen split contaminated share (%)",
                100 * fr["test_images_whose_participant_is_in_train"] / fr["test_images"], 1, origin=o)
        nd = d["near_duplicates"]
        cross = nd["cross_partition_by_filename_resolution"]
        same_cross = sum(v.get("same_participant", 0) for v in cross.values())
        diff_cross = sum(v.get("different_participants", 0) for v in cross.values())
        for label, value in (("candidates", nd["candidates"]), ("same participant", nd["same_participant"]),
                             ("adjacent slices", nd["slice_gap_of_same_participant_pairs"]["1"]),
                             ("cross partition", nd["cross_partition"]),
                             ("cross partition same participant", same_cross),
                             ("cross partition different participants", diff_cross),
                             ("key wrong, detector right", cross["both_resolved"]["same_participant"])):
            c.check_int(f"tier1 near-duplicate {label}", value, origin=o)
        c.check("tier1 near-duplicate same-participant share (%)",
                100 * nd["same_participant"] / nd["candidates"], 1, origin=o)
        c.check("tier1 cross-partition true-leakage share (%)", 100 * same_cross / nd["cross_partition"], 0, origin=o)
        cal = d["dhash_calibration"]
        c.check_int("tier1 calibration pairs", cal["pairs"], origin=o)
        c.check_int("tier1 median distance same participant", cal["median_distance"]["same_participant"], origin=o)
        c.check_int("tier1 median distance different participants",
                    cal["median_distance"]["different_participants"], origin=o)
        by_t = {row["max_hamming_distance"]: row for row in cal["by_threshold"]}
        for t in (4, 8):
            row = by_t[t]
            c.check(f"dHash d<={t} precision", row["precision"], 3, origin=o)
            c.check(f"dHash d<={t} adjacent-slice recall", row["recall_adjacent_slices"], 3, origin=o)
            mantissa, exponent = f"{row['false_positive_rate']:.1e}".split("e")
            c.check_text(f"dHash d<={t} false-positive rate",
                         f"{mantissa} \\times 10^{{{int(exponent)}}}", origin=o)
        c.check("dHash d<=4 same-participant pair coverage",
                by_t[4]["same_participant_pair_coverage"], 3, origin=o)
        for key in ("participants_split_across_folders", "n_images_agreeing_with_cdr",
                    "reliability_rescan_sessions_used"):
            c.check_int(f"tier1 {key}", m[key], origin=o)
        c.check_int("dHash d<=8 flagged pairs", by_t[8]["pairs_flagged"], origin=o)

    p = TABLES.parent / "tier1_split_rules_vs_truth.json"
    d = load(p)
    if d:
        o = str(p)
        for rule, stats in d["rules"].items():
            c.check_int(f"rule {rule} components", stats["components"], origin=o)
            c.check_int(f"rule {rule} largest component", stats["largest_component_images"], origin=o)
            c.check(f"rule {rule} contamination mean (%)", 100 * stats["test_image_contamination_mean"], 1,
                    origin=o, alt=["$100\\%$"] if stats["test_image_contamination_mean"] == 1 else
                                  ["$0\\%$"] if stats["test_image_contamination_mean"] == 0 else None)
            c.check(f"rule {rule} test participants", stats["test_participants_mean"], 1, origin=o)
        c.check_int("dHash d<=8 label-mismatched merges",
                    d["rules"]["filename_plus_dhash8"]["label_mismatched_edges_not_merged"], origin=o)

    # ── Calibration, quoted as Brier/ECE per protocol in the ADNI arm ───
    table = TABLES / "adni_inflation_gap.csv"
    if table.is_file():
        import csv as _csv
        import statistics as _stats
        rows = list(_csv.DictReader(table.open()))
        for proto in ("random", "subject_only", "component_safe"):
            values = [r for r in rows if r["protocol"] == proto]
            for field in ("brier_score", "ece"):
                c.check(f"{field} ({proto})",
                        _stats.mean(float(r[field]) for r in values) if values else None, 3,
                        origin=str(table))
    else:
        MISSING.append(str(table))

    # ── Sex-stratified subgroup contrast (Supplementary S6) ─────────────
    d = load(TABLES / "adni_subgroup_analysis.json")
    if d:
        for proto, stats in (d.get("sex_gap_paired_bootstrap") or {}).items():
            c.check(f"sex gap ({proto})", stats.get("delta_mean"), 3,
                    origin="adni_subgroup_analysis.json")
            c.check(f"sex gap CI lo ({proto})", stats.get("delta_ci_lo"), 3,
                    origin="adni_subgroup_analysis.json")
            c.check(f"sex gap CI hi ({proto})", stats.get("delta_ci_hi"), 3,
                    origin="adni_subgroup_analysis.json")
            c.check_int(f"sex gap direction preserved ({proto})",
                        stats.get("direction_preserved_iters"), origin="adni_subgroup_analysis.json")

    # ── Test-partition composition per protocol ─────────────────────────
    # The asymmetry these numbers describe (roughly 161 independent patients
    # under Protocol A against 38 under Protocol C) qualifies every
    # protocol-to-protocol AUROC comparison in the paper.
    d = load(TABLES / "adni_subject_level_auroc.json")
    if d:
        for proto, stats in (d.get("by_protocol") or {}).items():
            c.check(f"test participants ({proto})",
                    stats.get("n_subjects_per_seed_mean"), 1,
                    origin="adni_subject_level_auroc.json")

    # ── Provenance degradation (D31) ────────────────────────────────────
    # Structural measurement, no model involved: how many true participants
    # straddle a partition boundary under each protocol once subject
    # identifiers are deleted at a controlled rate. These are the numbers
    # Table "provenance_degradation" prints, and they are the framework's only
    # positive demonstration, so they are checked at the precision printed.
    p = TABLES / "adni_provenance_degradation.json"
    d = load(p)
    if d:
        import statistics
        agg: dict[tuple, list] = {}
        for record in d.get("records", []):
            key = (record["deletion_fraction"], record["protocol"])
            agg.setdefault(key, []).append(record)
        for (level, proto), records in sorted(agg.items()):
            straddling = [r["n_subjects_straddling_partitions"] for r in records]
            overlap = [r["test_train_subject_overlap"] for r in records]
            tag = f"{int(level * 100)}% {proto}"
            c.check(f"provenance straddling ({tag})",
                    statistics.mean(straddling), 1, origin=str(p))
            c.check(f"provenance overlap p ({tag})",
                    statistics.mean(overlap), 3, origin=str(p))

    # ── Split composition (audit-gate blind spot) ───────────────────────
    # The partitions are disjoint by construction and unbalanced in
    # composition anyway. These numbers are the evidence for that claim and
    # for the limitation drawn from it, so they are held to the artefact.
    d = load(TABLES / "adni_split_composition.json")
    if d:
        for phase, stats in (d.get("by_partition") or {}).items():
            c.check(f"component size mean ({phase})",
                    stats.get("component_size_mean"), 2, origin="adni_split_composition.json")
            c.check_int(f"component size min ({phase})",
                        stats.get("component_size_min"), origin="adni_split_composition.json")
            c.check_int(f"component size max ({phase})",
                        stats.get("component_size_max"), origin="adni_split_composition.json")
            c.check(f"AD share by image ({phase})",
                    stats.get("ad_share_by_image"), 3, origin="adni_split_composition.json",
                    alt=[fmt(100 * stats["ad_share_by_image"], 1)]
                        if stats.get("ad_share_by_image") is not None else None)
        for label, stats in (d.get("component_size_by_diagnosis") or {}).items():
            c.check(f"scans per participant ({label})",
                    stats.get("scans_mean"), 2, origin="adni_split_composition.json")
            c.check(f"scans per participant SD ({label})",
                    stats.get("scans_sd"), 2, origin="adni_split_composition.json")

    # ── Composition of the two non-primary split families ───────────────
    # The converter-inclusive validation partition carries a different class
    # mix from its test partition, and the size-balanced control exists to
    # show what the frozen assignment costs. Both are quoted in Limitations.
    # Only the quantity each arm is quoted for: the class mix on the converter
    # arm, the component size on the size-balanced control.
    for label, name, field in (("converter arm", "adni_split_composition_with_converters.json",
                                "ad_share_by_image"),
                               ("size-balanced control", "adni_split_composition_size_balanced.json",
                                "component_size_mean")):
        d = load(TABLES / name)
        if not d:
            continue
        for phase, stats in (d.get("by_partition") or {}).items():
            value = stats.get(field)
            c.check(f"{label} {field} ({phase})", value, 3 if field.startswith("ad") else 2,
                    origin=name,
                    alt=[fmt(100 * value, 1)] if field.startswith("ad") and value is not None else None)

    # ── Superseded values that must not survive anywhere ────────────────
    # Every entry here is a number this project once printed and later
    # corrected. They are checked by absence, because presence-only checks
    # cannot catch a stale duplicate of a value that is also correct elsewhere.
    for label, stale, replacement, why in [
        ("dose-response r2 (pre A1 fix)", "$R^2=0.69$", "0.72",
         "recomputed after excluding the 1-epoch aborted run"),
        # The 2026-09-28 rerun in the published code state. The Limitations
        # quoted this pair four sections away from the Results and survived
        # the first sweep, which is the case check_absent exists for.
        ("dose-response r2 pair (pre-rerun)", "$R^2=0.35$ against $0.72$", "0.64 against 0.85",
         "the dose matrix was rerun in the published code state"),
        ("dose-response slope (pre-rerun)", "0.1065", "0.1251",
         "the dose matrix was rerun in the published code state"),
        ("dose-response densenet slope (pre-rerun)", "0.0655", "0.0906",
         "the dose matrix was rerun in the published code state"),
        ("dose-response r2 (spaced form)", "R^2 = 0.69", "0.72",
         "recomputed after excluding the 1-epoch aborted run"),
        ("dose-response intercept (pre A1 fix)", "0.832 + 0.106", "0.833 + 0.106",
         "recomputed after excluding the 1-epoch aborted run"),
        # These two legitimately reappear in the paragraph that explains the
        # threshold-selection correction, so that passage is sanctioned.
        ("sens@spec0.90 leaky (test-selected)", "0.874", "0.865",
         "thresholds moved from test to validation selection"),
        ("sens@spec0.90 honest (test-selected)", "0.594", "0.533",
         "thresholds moved from test to validation selection"),
        ("cost-of-leakage ratio (pre C1b)", "3.2\\times", "3.5x",
         "recomputed under validation-selected thresholds"),
        ("hierarchical CI lower (mis-rounded)", "+0.072, +0.217", "+0.071, +0.218",
         "rounded the wrong way from 0.0714 / 0.2176"),
        ("component marginal sign (converter arm)", "$+0.043$", "-0.043",
         "unified on the Protocol B - Protocol C convention"),
        ("Tier-1 gap on the filename-key split", "0.157 \\pm 0.016", "the recovered-participant gap",
         "the filename-key split was 70% contaminated"),
        ("Tier-1 honest AUROC on the filename-key split", "0.8422", "the recovered-participant AUROC",
         "the filename-key split was 70% contaminated"),
        ("Tier-1 subject-only share", "98.3\\%", "the filename-key share",
         "computed against a broken identifier"),
        ("near-duplicate false-positive reading", "$95\\%$ false positives", "91% true leakage",
         "adjudicated against recovered identity"),
        ("Tier-1 labels called weak", "weak labels", "CDR-equal labels",
         "folder labels reproduce OASIS-1 CDR exactly"),
    ]:
        c.check_absent(
            label, stale, replaced_by=replacement,
            origin=f"superseded: {why}",
            # Sanctioned contexts: passages that quote the old value in order
            # to explain the correction that replaced it.
            allow_near=("under test selection", "under validation selection",
                        "superseded", "previously", "earlier test-selected",
                        "Under the earlier",
                        # Not historical reporting but a collision: the composed
                        # optimism at 10% identifier loss is itself +0.043 once
                        # the dose slope is 0.125, which is the retracted
                        # converter-arm marginal to the digit.
                        "AUROC of optimism", "AUROC of expected optimism"),
        )

    # ── Numbers whose GPU stage has not completed ───────────────────────
    # Only macros the manuscript actually uses: a generated number for an arm
    # the paper does not report is not a missing claim.
    raw = "\n".join(path.read_text(encoding="utf-8") for path in
                    (PAPER, SM, *sorted(PAPER_TABLES.glob("*.tex")))
                    if path.exists() and path != MACROS)   # the macro file defines them all
    pending = sorted(name for name, body in macros.items()
                     if PENDING in body and f"\\{name}" in raw)

    # ── Report ──────────────────────────────────────────────────────────
    total = len(c.passes) + len(c.failures)
    print(f"Checked {total} numeric claims against reports/tables/**\n")

    if args.verbose and c.passes:
        print("PASS")
        for label, val in c.passes:
            print(f"  ok   {label:38s} {val}")
        print()

    if MISSING:
        print(f"UNCHECKED ({len(MISSING)}) — artefact absent, so its claims were never tested:")
        for path in MISSING:
            print(f"  --   {display_path(path)}")
        print()

    if pending:
        print(f"PENDING ({len(pending)}) — manuscript numbers still printed as ?? "
              "because their GPU stage has not completed:")
        print("  " + ", ".join(pending))
        print()

    if c.failures:
        print(f"MISMATCH ({len(c.failures)}) — value in the JSON does not appear in the manuscript:")
        for label, val, origin in c.failures:
            print(f"  FAIL {label:38s} expected {val:>10s}   source: {origin}")
        print("\nEach line means the analysis output and the manuscript disagree, or the\n"
              "manuscript rounds differently. Reconcile before submitting.")
        return 1

    if MISSING or pending:
        print(f"{len(c.passes)} checks passed, but {len(MISSING)} artefact(s) were absent and "
              f"{len(pending)} number(s) are pending; the claims they back are unverified.")
        return 1

    print(f"All {len(c.passes)} checks passed: every source value appears in the manuscript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
