#!/usr/bin/env python3
"""Emit a DUA-safe *hashed* Tier 3 (ADNI1) component-safe split manifest.

The paper's Data Availability statement releases hashed component-safe
manifests for the ADNI tier so that downstream auditors can verify

* zero cross-partition overlap on the released components,
* class balance per split,
* per-axis Cramér's V confound audits,

**without** obtaining the underlying LONI/ADNI participant-level data.
This script consumes an ADNI split manifest produced by
``make_adni_splitguard_split.py`` and outputs one row per component with

* ``subject_id_hash``   — PBKDF2-HMAC-SHA256 of the subject_id (PTID)
                          under a **secret** per-release salt supplied
                          by the operator. The salt is NEVER committed
                          to this repository and must not be published
                          alongside the hashed manifest.

  Why key stretching and a secret salt are both mandatory here: ADNI
  PTIDs are drawn from the low-entropy space ``NNN_S_NNNN`` (at most
  10^7 candidates, realistically ~10^4 in use). A single-round digest
  under a *published* salt is therefore trivially invertible — the
  entire PTID space can be enumerated in well under a minute on a
  laptop, which would fully re-identify every released component and
  defeat the DUA-safety claim this release rests on. PBKDF2 with a
  high iteration count plus an unpublished salt is what makes the
  released hashes non-invertible in practice.
* ``component_id``      — opaque component identifier, safe to release.
* ``component_size``    — integer count of scans in the component.
* ``binary_label``      — CN / AD (from the primary CN-vs-AD universe).
* ``scanner_field_strength`` — non-identifying acquisition context.
* ``modality``          — non-identifying acquisition context.
* ``split``             — train / val / test partition assignment.

The following participant-level fields are **deliberately dropped**:
``ptid``, ``rid``, ``viscode``, ``image_uid``, ``series_uid``,
``acq_date``, ``image_path``, ``relative_path``, ``age``, ``sex``,
``session_id``, ``slice_index``, and the raw ``subject_id`` PTID (only
its salted PBKDF2 derivation is emitted).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "adni_splitguard_seed42.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "splits" / "adni_hashed_manifest_seed42.csv"
DEFAULT_AUDIT = REPO_ROOT / "data" / "splits" / "adni_hashed_manifest_seed42.audit.json"

# PBKDF2 work factor. Chosen so that a full sweep of the ~10^7 ADNI PTID
# space costs on the order of CPU-weeks rather than seconds.
PBKDF2_ITERATIONS = 600_000
PBKDF2_DKLEN = 32

# There is deliberately NO default salt. A committed default would be
# published with the code and would make every released hash invertible
# by enumerating the PTID space. The operator must supply a secret salt
# via --salt or the SPLITGUARD_RELEASE_SALT environment variable, and
# must keep it out of the repository.
SALT_ENV_VAR = "SPLITGUARD_RELEASE_SALT"

# Fields released per component (in this order).
RELEASE_FIELDS = [
    "subject_id_hash",
    "component_id",
    "component_size",
    "binary_label",
    "scanner_field_strength",
    "modality",
    "split",
]

# Fields deliberately dropped from the release for DUA compliance.
DROPPED_FIELDS = {
    "ptid", "rid", "viscode", "image_uid", "series_uid",
    "acq_date", "image_path", "relative_path", "age", "sex",
    "session_id", "slice_index", "qc_flag", "subject_id",
    "label_source", "label_confidence", "diagnosis_group",
    "source_dataset",
}


@lru_cache(maxsize=None)
def hash_subject_id(subject_id: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 of the subject_id under a secret per-release salt.

    Key stretching is load-bearing, not decorative: ADNI PTIDs occupy a
    ~10^7 space, so a single-round digest would be enumerable in seconds.

    The derivation is deterministic, so it is memoised: releasing five seeds of
    the same cohort otherwise re-derives each of the 220 participants five
    times at 600,000 iterations apiece, which costs minutes and changes
    nothing. The cache lives for one process and never touches disk.
    """
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        subject_id.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
        dklen=PBKDF2_DKLEN,
    )
    return derived.hex()


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_components(
    rows: list[dict[str, str]], salt: str
) -> list[dict[str, str]]:
    """Collapse per-image rows into one row per component.

    Follows the paper's Tier 3 CN-vs-AD binary universe: MCI and
    diagnosis="unknown" rows are filtered out before component
    collapse so the released manifest contains only components with a
    uniform CN or AD label.
    """
    binary_universe = {"CN", "AD", "Demented", "NonDemented"}
    filtered = [
        row
        for row in rows
        if row.get("binary_label", row.get("diagnosis_group", "")) in binary_universe
    ]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        grouped[row["component_id"]].append(row)

    output: list[dict[str, str]] = []
    for component_id, comp_rows in sorted(grouped.items()):
        subject_ids = {row.get("subject_id", "") for row in comp_rows}
        # Component-safe splits are subject-atomic by construction; still
        # explicitly guarded so a malformed input fails loudly instead of
        # silently emitting a mixed-subject component.
        if len(subject_ids) != 1:
            raise ValueError(
                f"Component {component_id!r} spans {len(subject_ids)} "
                f"subject IDs: {sorted(subject_ids)}. Expected exactly one."
            )
        subject_id = subject_ids.pop()

        splits = {row.get("split", "") for row in comp_rows}
        if len(splits) != 1:
            raise ValueError(
                f"Component {component_id!r} spans {len(splits)} splits: "
                f"{sorted(splits)}. Component-safe assignment violated."
            )
        split = splits.pop()

        # ADNI splits use ``diagnosis_group`` (CN / MCI / AD). For the
        # Tier 3 CN-vs-AD release we accept ``binary_label`` if present
        # (Tier 1 / OASIS convention) and otherwise derive it from
        # diagnosis_group.
        label_col = "binary_label" if "binary_label" in comp_rows[0] else "diagnosis_group"
        labels = {row.get(label_col, "") for row in comp_rows}
        if len(labels) != 1:
            raise ValueError(
                f"Component {component_id!r} has {len(labels)} label values "
                f"in column {label_col!r}: {sorted(labels)}. Expected uniform "
                f"label per component."
            )
        binary_label = labels.pop()

        scanner = comp_rows[0].get("scanner_field_strength", "")
        modality = comp_rows[0].get("modality", "")

        output.append(
            {
                "subject_id_hash": hash_subject_id(subject_id, salt),
                "component_id": component_id,
                "component_size": str(len(comp_rows)),
                "binary_label": binary_label,
                "scanner_field_strength": scanner,
                "modality": modality,
                "split": split,
            }
        )

    return output


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELEASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(
    hashed_rows: list[dict[str, str]],
    input_path: Path,
    output_path: Path,
    salt: str,
    audit_path: Path,
) -> dict[str, object]:
    split_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    for row in hashed_rows:
        split_counts[row["split"]] += 1
        label_counts[row["binary_label"]] += 1
    # NEVER write the salt itself: this audit file ships next to the hashed
    # manifest, and publishing the salt would make every hash invertible.
    # A truncated fingerprint lets two releases be told apart without
    # disclosing the secret.
    salt_fingerprint = hashlib.pbkdf2_hmac(
        "sha256", b"salt-fingerprint", salt.encode("utf-8"),
        PBKDF2_ITERATIONS, dklen=8,
    ).hex()
    audit = {
        "input": str(input_path),
        "output": str(output_path),
        "salt_fingerprint": salt_fingerprint,
        "salt_disclosed": False,
        "kdf": f"PBKDF2-HMAC-SHA256, {PBKDF2_ITERATIONS} iterations, dklen={PBKDF2_DKLEN}",
        "released_fields": RELEASE_FIELDS,
        "dropped_fields": sorted(DROPPED_FIELDS),
        "n_components": len(hashed_rows),
        "components_per_split": dict(split_counts),
        "components_per_label": dict(label_counts),
        "note": (
            "Subject identifiers are protected with PBKDF2-HMAC-SHA256 under a "
            "secret per-release salt that is deliberately not recorded here. "
            "Recomputing hashes requires the operator's salt; without it the "
            "released hashes are not invertible by enumerating the PTID space."
        ),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT,
                        help="Input per-image split manifest CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output hashed manifest CSV.")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT,
                        help="Path for the audit JSON summary.")
    parser.add_argument("--salt", default=None,
                        help=("SECRET per-release salt. Required. May instead be "
                              f"supplied via ${SALT_ENV_VAR}. Never commit this "
                              "value or publish it with the manifest."))
    args = parser.parse_args()

    salt = args.salt or os.environ.get(SALT_ENV_VAR)
    if not salt:
        raise SystemExit(
            "No release salt supplied.\n\n"
            f"Pass --salt or set ${SALT_ENV_VAR}. There is no default salt on\n"
            "purpose: ADNI PTIDs occupy a ~10^7 space, so a published salt would\n"
            "let anyone invert every released subject hash in under a minute and\n"
            "re-identify the components. Generate one with:\n\n"
            f"    export {SALT_ENV_VAR}=$(python3 -c "
            "'import secrets; print(secrets.token_hex(32))')\n\n"
            "Store it outside this repository. Losing it means future releases\n"
            "cannot be matched to this one; publishing it voids the DUA-safety\n"
            "property of the release."
        )
    if len(salt) < 16:
        raise SystemExit(
            f"Release salt is too short ({len(salt)} chars). Use at least 16 "
            "characters of high-entropy secret; a guessable salt is equivalent "
            "to no salt for a low-entropy identifier space like ADNI PTIDs."
        )

    if not args.split.exists():
        raise SystemExit(
            f"Input split manifest not found: {args.split}. Run "
            f"scripts/make_adni_splitguard_split.py first."
        )

    rows = read_split(args.split)
    if not rows:
        raise SystemExit(f"Input split manifest is empty: {args.split}")

    hashed = aggregate_components(rows, salt)
    write_manifest(hashed, args.output)
    audit = write_audit(hashed, args.split, args.output, salt, args.audit)

    print(f"Wrote {len(hashed)} hashed component rows to {args.output}")
    print(f"Audit summary: {audit['components_per_split']}, "
          f"labels {audit['components_per_label']}")
    print(f"Audit written to {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
