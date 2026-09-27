#!/usr/bin/env python3
"""Emit the ADNI scan-to-visit linkage-integrity audit promised by the paper.

Why this exists
---------------
A leakage-audit paper stands on the fidelity of its metadata join. The ADNI
diagnosis label is attached to each scan through a phase-aware visit lookup
with a date-proximity fallback, and the manuscript states that the per-scan
accounting for that join is released. Until now no code emitted it: the join
tier was recorded inside the manifest (``label_confidence``) but never
exported as an auditable artefact, so a reader could not check how many
labels rested on an exact visit key versus a date match.

What it emits
-------------
One row per scan, with the raw participant identifier replaced by a salted
PBKDF2 derivation — the same protection the Tier-3 hashed manifest uses,
and for the same reason: ADNI PTIDs are low-entropy, so an unsalted or
publicly-salted digest would be trivially invertible and would put the
release outside the DUA.

Columns
    scan_uid            image identifier (opaque, not participant-derived)
    ptid_hash           PBKDF2-HMAC-SHA256 of the PTID under a secret salt
    acq_date            acquisition date (retained: needed to audit the join)
    join_tier           exact_viscode | date_proximity | unresolved
    label_source        the ADNI column the diagnosis came from
    diagnosis_group     resolved CN / MCI / AD / unknown
    in_cnad_universe    whether the scan enters the CN-vs-AD task

Usage
-----
    export SPLITGUARD_RELEASE_SALT=$(python3 -c \
        'import secrets; print(secrets.token_hex(32))')
    python3 scripts/build_linkage_audit.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_linkage_audit.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_linkage_audit_summary.json"

SALT_ENV_VAR = "SPLITGUARD_RELEASE_SALT"
PBKDF2_ITERATIONS = 600_000
PBKDF2_DKLEN = 32

# Map the manifest's recorded confidence onto the join tiers the paper reports.
CONFIDENCE_TO_TIER = {
    "clinical_table": "exact_viscode",
    "date_only_match": "date_proximity",
    "no_mri_meta_match": "unresolved",
    "missing": "unresolved",
    "": "unresolved",
}

RELEASE_FIELDS = [
    "scan_uid",
    "ptid_hash",
    "acq_date",
    "join_tier",
    "label_source",
    "diagnosis_group",
    "in_cnad_universe",
]


def hash_ptid(ptid: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", ptid.encode("utf-8"), salt.encode("utf-8"),
        PBKDF2_ITERATIONS, dklen=PBKDF2_DKLEN,
    ).hex()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--salt", default=None,
                    help=f"Secret release salt; or set ${SALT_ENV_VAR}.")
    args = ap.parse_args()

    salt = args.salt or os.environ.get(SALT_ENV_VAR)
    if not salt:
        raise SystemExit(
            "No release salt supplied.\n\n"
            f"Pass --salt or set ${SALT_ENV_VAR}. This file carries one row per\n"
            "ADNI scan; without key stretching under a secret salt the PTID\n"
            "hashes are invertible by enumerating the ~10^7 PTID space, which\n"
            "would place the artefact outside the LONI/ADNI data-use agreement.\n\n"
            f"    export {SALT_ENV_VAR}=$(python3 -c "
            "'import secrets; print(secrets.token_hex(32))')"
        )
    if len(salt) < 16:
        raise SystemExit(f"Salt too short ({len(salt)} chars); use >= 16.")

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Manifest is empty: {args.manifest}")

    # Hash each distinct PTID once — PBKDF2 at 600k iterations is deliberately
    # slow, and there are far fewer subjects than scans.
    distinct = {(r.get("subject_id") or "").strip() for r in rows}
    ptid_hashes = {p: hash_ptid(p, salt) for p in distinct if p}

    out_rows = []
    tier_counts: collections.Counter = collections.Counter()
    for row in rows:
        confidence = (row.get("label_confidence") or "").strip()
        tier = CONFIDENCE_TO_TIER.get(confidence, "unresolved")
        tier_counts[tier] += 1
        dx = (row.get("diagnosis_group") or "unknown").strip()
        subject = (row.get("subject_id") or "").strip()
        out_rows.append({
            "scan_uid": row.get("image_id", ""),
            "ptid_hash": ptid_hashes.get(subject, ""),
            "acq_date": row.get("acq_date", ""),
            "join_tier": tier,
            "label_source": row.get("label_source", ""),
            "diagnosis_group": dx,
            "in_cnad_universe": "yes" if dx in ("CN", "AD") else "no",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELEASE_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    unresolved = [r for r in out_rows if r["join_tier"] == "unresolved"]
    summary = {
        "n_scans": len(out_rows),
        "n_subjects": len(ptid_hashes),
        "join_tier_counts": dict(sorted(tier_counts.items())),
        "unresolved_scans": len(unresolved),
        "unresolved_in_cnad_universe": sum(
            1 for r in unresolved if r["in_cnad_universe"] == "yes"
        ),
        "kdf": f"PBKDF2-HMAC-SHA256, {PBKDF2_ITERATIONS} iterations",
        "salt_disclosed": False,
        "released_fields": RELEASE_FIELDS,
        "note": (
            "join_tier=exact_viscode means the diagnosis came from a direct "
            "(RID, VISCODE) match; date_proximity means the visit key failed "
            "and the label was taken from the nearest clinical visit within "
            "the 180-day tolerance; unresolved means neither succeeded."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(out_rows)} rows to {args.output}")
    for tier, n in sorted(tier_counts.items()):
        print(f"  {tier:16s} {n:5d}  ({100*n/len(out_rows):.1f}%)")
    print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
