#!/usr/bin/env python3
"""Stress-test the leakage graph against structured provenance corruption.

``run_provenance_degradation.py`` deletes subject identifiers uniformly at
random. That is one failure mode, and the easiest one: a missing identifier is
at least *visibly* missing, and a splitter can treat the scan as its own group.
Real redistribution damages provenance in ways that stay invisible. The Tier-1
benchmark this paper reconstructs exhibits two of them at once -- its filename
key both **merges** distinct participants under one key and **splits** one
participant across several -- and neither shows up as a missing field.

This script therefore applies a corruption operator

    C in {drop, split, merge}

at intensity lambda, rebuilds both groupings on the corrupted manifest, and
scores the residual leakage against untouched ground truth:

    drop   each scan loses ``subject_id`` with probability lambda
    split  each participant, with probability lambda, has its scans dealt into
           two pseudo-identifiers, so one patient looks like two
    merge   participants are paired at random and each pair, with probability
           lambda, is relabelled to a single pseudo-identifier, so two patients
           look like one

Only the grouping differs between the two protocols compared here; both use the
same class-stratified assigner, for the reason documented at length in
``subject_only_split``. Mixing the assignment algorithm into the comparison
was worth about half the apparent effect when it happened before.

No model is trained. Contamination is a property of the split, not of a fitted
model, so this entire matrix runs on a CPU in seconds. The AUROC consequence of
a given contamination level is what the dose-response experiment measures, and
the two compose.

Usage
-----
    python3 scripts/run_provenance_stress_test.py
    python3 scripts/run_provenance_stress_test.py --mechanisms drop split merge
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_provenance_degradation import (  # noqa: E402
    TRUE_SUBJECT,
    component_safe_split,
    residual_subject_leakage,
    subject_only_split,
)

DEFAULT_SPLIT = ROOT / "data" / "splits" / "adni" / "adni_splitguard_seed0.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "tables" / "adni" / "adni_provenance_stress_test.json"
LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
BINARY = {"CN", "AD"}


def _tag(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy rows and record untouched ground truth before any corruption."""
    out = []
    for row in rows:
        new = dict(row)
        new[TRUE_SUBJECT] = row["subject_id"]
        out.append(new)
    return out


def corrupt_drop(rows, intensity, rng):
    """Each scan independently loses its subject identifier."""
    out = _tag(rows)
    for row in out:
        if rng.random() < intensity:
            row["subject_id"] = "unknown"
    return out


def corrupt_split(rows, intensity, rng):
    """One participant is dealt into two pseudo-identifiers.

    The scans keep an identifier, so nothing looks missing; a subject-wise
    splitter simply believes there are two patients where there is one. This is
    the failure the Tier-1 filename key exhibits for 179 of its 200
    participants.
    """
    out = _tag(rows)
    by_subject = defaultdict(list)
    for row in out:
        by_subject[row[TRUE_SUBJECT]].append(row)
    for subject, subject_rows in by_subject.items():
        if len(subject_rows) < 2 or rng.random() >= intensity:
            continue
        shuffled = subject_rows[:]
        rng.shuffle(shuffled)
        cut = max(1, len(shuffled) // 2)
        for row in shuffled[:cut]:
            row["subject_id"] = f"{subject}__a"
        for row in shuffled[cut:]:
            row["subject_id"] = f"{subject}__b"
    return out


def corrupt_merge(rows, intensity, rng):
    """Two participants are relabelled to one pseudo-identifier.

    Again nothing is missing. A subject-wise splitter over-groups, which is
    conservative for leakage but destroys the participant count; the leakage
    graph inherits the same wrong key, so this mechanism is expected to hurt
    both protocols equally. Reporting a mechanism where the graph does not help
    is the point of running three.
    """
    out = _tag(rows)
    subjects = sorted({row[TRUE_SUBJECT] for row in out})
    rng.shuffle(subjects)
    merged: dict[str, str] = {}
    for left, right in zip(subjects[0::2], subjects[1::2]):
        if rng.random() < intensity:
            merged[left] = merged[right] = f"{left}+{right}"
    for row in out:
        if row[TRUE_SUBJECT] in merged:
            row["subject_id"] = merged[row[TRUE_SUBJECT]]
    return out


MECHANISMS = {"drop": corrupt_drop, "split": corrupt_split, "merge": corrupt_merge}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--mechanisms", nargs="+", default=list(MECHANISMS))
    ap.add_argument("--levels", nargs="+", type=float, default=LEVELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--emit-splits", type=Path, default=None,
                    help="Also write one split CSV per (mechanism, intensity, seed, "
                         "protocol) so the AUROC consequence of each corruption cell "
                         "can be trained. Contamination needs no model; this does.")
    args = ap.parse_args()

    with args.split.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("diagnosis_group") in BINARY]
    if not rows:
        raise SystemExit(f"no CN/AD rows in {args.split}")

    records = []
    for mechanism in args.mechanisms:
        corrupt = MECHANISMS[mechanism]
        for level in args.levels:
            for seed in args.seeds:
                # Python hashes strings with a per-process salt, so hash()
                # here would make the matrix irreproducible between runs.
                stream = f"{mechanism}|{level}|{seed}".encode("utf-8")
                rng = random.Random(int(hashlib.sha256(stream).hexdigest()[:16], 16))
                corrupted = corrupt(rows, level, rng)
                n_keys = len({r["subject_id"] for r in corrupted if r["subject_id"] != "unknown"})
                for protocol, splitter in (("subject_only", subject_only_split),
                                           ("component_safe", component_safe_split)):
                    assignment = splitter(corrupted, seed)
                    scored = residual_subject_leakage(corrupted, assignment)
                    records.append({"mechanism": mechanism, "intensity": level, "seed": seed,
                                    "protocol": protocol, "n_surviving_keys": n_keys, **scored})
                    if args.emit_splits is not None and level > 0:
                        out = (args.emit_splits /
                               f"{mechanism}_lambda{level}_{protocol}_seed{seed}.csv")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        fields = [f for f in rows[0] if not f.startswith("_")]
                        with out.open("w", newline="", encoding="utf-8") as fh:
                            w = csv.DictWriter(fh, fieldnames=fields)
                            w.writeheader()
                            for r in corrupted:
                                w.writerow({**{k: r.get(k, "") for k in fields},
                                            "split": assignment[r["image_id"]]})

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for mechanism in args.mechanisms:
        summary[mechanism] = {}
        for level in args.levels:
            cell = {}
            for protocol in ("subject_only", "component_safe"):
                sel = [r for r in records if r["mechanism"] == mechanism
                       and r["intensity"] == level and r["protocol"] == protocol]
                cell[protocol] = {
                    "straddling_mean": round(st.mean(r["n_subjects_straddling_partitions"] for r in sel), 2),
                    "test_scan_contamination_mean": round(
                        st.mean(r["test_scan_contamination"] for r in sel), 4),
                }
            b = cell["subject_only"]["straddling_mean"]
            c = cell["component_safe"]["straddling_mean"]
            cell["graph_prevented_share"] = round(1 - c / b, 4) if b else None
            summary[mechanism][f"{level}"] = cell

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_file": str(args.split.relative_to(ROOT)),
        "n_scans": len(rows),
        "mechanisms": args.mechanisms,
        "levels": args.levels,
        "seeds": args.seeds,
        "trained": False,
        "interpretation": (
            "Contamination is a property of the split, so no model is trained here. "
            "drop removes identifiers visibly; split and merge corrupt them invisibly, "
            "which is what redistribution actually does. The graph can only help where "
            "an edge family other than same_subject survives the corruption."
        ),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{'mechanism':<9}{'lambda':>8}{'subject-only':>14}{'graph':>8}{'prevented':>11}")
    for mechanism in args.mechanisms:
        for level in args.levels:
            cell = summary[mechanism][f"{level}"]
            prevented = cell["graph_prevented_share"]
            print(f"{mechanism:<9}{level:>8}"
                  f"{cell['subject_only']['straddling_mean']:>14}"
                  f"{cell['component_safe']['straddling_mean']:>8}"
                  f"{'n/a' if prevented is None else f'{100*prevented:.1f}%':>11}")
    print(f"\nWrote {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
