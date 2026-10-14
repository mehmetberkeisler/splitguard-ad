#!/usr/bin/env python3
"""Compare the two ends of the provenance axis against the random-split baseline.

Why this exists
---------------
The identifier-deletion sweep and the protocol comparison were analysed
separately, and the quantity that connects them was never computed. It is the
one the manuscript now leads with.

At the left end of the sweep the participant identifier is complete, and
subject-wise splitting reports an honest AUROC. At the right end every
identifier has been removed, so every scan is a singleton and the same
subject-wise procedure can no longer group anything. The question is whether
what it reports there is distinguishable from random image-level splitting,
which is the failure mode the procedure exists to prevent.

This script answers it from the artefacts that already exist: the deletion
sweep for the two endpoints, and the primary-arm bootstrap for the random-split
reference. The interval is a t interval over the five seed-level AUROCs, which
is what every other five-seed quantity in this paper uses.

Usage
-----
    python3 scripts/analyze_provenance_endpoint.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables" / "adni"
SWEEP = TABLES / "adni_provenance_degradation.json"
PRIMARY = TABLES / "adni_inflation_gap_bootstrap.json"
OUT = TABLES / "adni_provenance_endpoint.json"

T95_DF4 = 2.776


def interval(values: list[float]) -> dict:
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    half = T95_DF4 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": round(mean, 4), "sd": round(sd, 4),
            "ci_lo": round(mean - half, 4), "ci_hi": round(mean + half, 4),
            "n_seeds": len(values)}


def main() -> int:
    if not SWEEP.is_file() or not PRIMARY.is_file():
        print(f"missing input: {SWEEP if not SWEEP.is_file() else PRIMARY}")
        return 1

    records = json.loads(SWEEP.read_text(encoding="utf-8"))["records"]

    def auroc_at(level: float, protocol: str) -> list[float]:
        return [r["test_auroc"] for r in records
                if r["deletion_fraction"] == level and r["protocol"] == protocol]

    intact = interval(auroc_at(0.0, "subject_only"))
    destroyed = interval(auroc_at(1.0, "subject_only"))
    destroyed_safe = interval(auroc_at(1.0, "component_safe"))

    rnd = json.loads(PRIMARY.read_text(encoding="utf-8"))["auroc"]["random"]
    random_ref = {"mean": rnd["point_mean"], "ci_lo": rnd["ci_lo"], "ci_hi": rnd["ci_hi"]}

    # Do the two intervals overlap? This is the claim the manuscript makes, and
    # it is deliberately the weak form: overlapping intervals are not evidence
    # of equality, only a failure to distinguish. The paper says so.
    overlap = not (destroyed["ci_hi"] < random_ref["ci_lo"]
                   or destroyed["ci_lo"] > random_ref["ci_hi"])

    payload = {
        "description": ("Subject-wise splitting at both ends of the identifier-deletion "
                        "sweep, against the random-split baseline from the primary arm"),
        "intact_identifiers_subject_wise": intact,
        "destroyed_identifiers_subject_wise": destroyed,
        "destroyed_identifiers_component_safe": destroyed_safe,
        "random_split_reference": random_ref,
        "inflation_intact_to_destroyed": round(destroyed["mean"] - intact["mean"], 4),
        "gap_destroyed_vs_random": round(destroyed["mean"] - random_ref["mean"], 4),
        "intervals_overlap": overlap,
        "note": ("Overlapping intervals do not establish equality; they record a "
                 "failure to distinguish the two at five seeds. The manuscript "
                 "states the claim in that form."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  identifiers intact,    subject-wise : {intact['mean']:.4f} "
          f"[{intact['ci_lo']:.4f}, {intact['ci_hi']:.4f}]")
    print(f"  identifiers destroyed, subject-wise : {destroyed['mean']:.4f} "
          f"[{destroyed['ci_lo']:.4f}, {destroyed['ci_hi']:.4f}]")
    print(f"  random split (primary arm)          : {random_ref['mean']:.4f} "
          f"[{random_ref['ci_lo']:.4f}, {random_ref['ci_hi']:.4f}]")
    print(f"  inflation across the sweep          : {payload['inflation_intact_to_destroyed']:+.4f}")
    print(f"  destroyed minus random              : {payload['gap_destroyed_vs_random']:+.4f}"
          f"   intervals overlap: {overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
