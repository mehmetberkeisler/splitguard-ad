#!/usr/bin/env python3
"""Materialise the redistributable artefacts the Data Availability statement promises.

Why this exists
---------------
The manuscript states that the GitHub release ships, per tier:

* Tier 1 (public JPEG benchmark) — the exact frozen component-safe split
  manifests at image-path level, plus the leakage-graph edge lists;
* Tier 2 (OASIS-1) — the frozen split manifests indexed by the public OASIS
  subject and session codes, with no derived image paths;
* Tier 3 (ADNI1) — hashed component manifests only (produced separately by
  ``build_hashed_manifest_tier3.py``).

None of that actually shipped. Everything lived under ``data/``, which is
gitignored, so a reader cloning the repository received the code and no
artefacts at all — the one thing a reproducibility paper cannot get wrong.

This script writes the Tier-1 and Tier-2 artefacts into ``release/``, which is
tracked, applying the redistribution rule for each tier rather than copying
files wholesale. It is deliberately conservative: Tier 2 drops every
filesystem path and keeps only the public OASIS identifiers, and no tier
emits participant-level clinical fields.

Usage
-----
    python3 scripts/build_release_artefacts.py
    python3 scripts/build_release_artefacts.py --check   # verify, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"

# ── Tier 1: public benchmark, fully redistributable ──────────────────────
T1_SPLIT = ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
T1_COMPONENTS = ROOT / "data" / "manifests" / "current_jpeg_leakage_components.csv"
T1_NEAR_DUPES = ROOT / "data" / "manifests" / "current_jpeg_near_duplicate_candidates.csv"

T1_SPLIT_FIELDS = [
    "image_id", "relative_path", "raw_class_label", "binary_label",
    "subject_id", "component_id", "split",
]
T1_EDGE_FIELDS = [
    "image_id", "relative_path", "subject_id", "binary_label",
    "component_id", "component_size", "component_primary_reason",
]

# ── Tier 2: OASIS-1, public identifiers only, no paths ───────────────────
T2_SPLIT = ROOT / "data" / "splits" / "oasis1_splitguard_seed42.csv"
T2_COMPONENTS = ROOT / "data" / "manifests" / "oasis1_leakage_components.csv"

# Deliberately excludes path/relative_path and every clinical column
# (cdr, mmse, educ, ses, age, sex): a third party re-derives those from their
# own OASIS download.
T2_SPLIT_FIELDS = [
    "subject_id", "session_id", "volume_id", "binary_label",
    "component_id", "split",
]
T2_COMPONENT_FIELDS = [
    "subject_id", "session_id", "volume_id", "binary_label",
    "component_id", "component_size", "component_primary_reason",
]

# Redistribution rules differ by tier and the difference is the whole point.
#
# Tier 1 is a public benchmark, so the manuscript promises manifests at
# image-path level and paths may ship. Only absolute filesystem paths are
# withheld, since those describe the author's machine, not the dataset.
#
# Tier 2 is OASIS-1: access is open but the data are not ours to redistribute,
# so nothing that reconstructs an image or carries clinical detail may ship.
# A third party re-derives those from their own download using the public
# subject/session codes we do release.
FORBIDDEN_BY_TIER = {
    "tier1": {"path", "image_path", "source_archive", "source_disc"},
    "tier2": {"path", "relative_path", "image_path", "source_archive",
              "source_disc", "cdr", "mmse", "educ", "ses", "age", "sex",
              "clinical_label"},
}


def project(rows: list[dict], fields: list[str], tier: str) -> list[dict]:
    """Keep only `fields`, refusing any column this tier may not redistribute."""
    forbidden = FORBIDDEN_BY_TIER[tier]
    leaked = forbidden.intersection(fields)
    if leaked:
        raise ValueError(
            f"Refusing to release columns restricted for {tier}: {sorted(leaked)}"
        )
    out = []
    for row in rows:
        out.append({f: row.get(f, "") for f in fields})
    return out


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict], fields: list[str], check: bool) -> int:
    if not rows:
        return 0
    if check:
        return len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Report what would be written without writing.")
    args = ap.parse_args()

    manifest: dict[str, object] = {"tiers": {}}
    missing: list[str] = []

    # ── Tier 1 ───────────────────────────────────────────────────────────
    t1: dict[str, int] = {}
    rows = read(T1_SPLIT)
    if rows:
        t1["split_seed42"] = write(
            RELEASE / "tier1_public_benchmark" / "splitguard_split_seed42.csv",
            project(rows, T1_SPLIT_FIELDS, "tier1"), T1_SPLIT_FIELDS, args.check)
    else:
        missing.append(str(T1_SPLIT))

    rows = read(T1_COMPONENTS)
    if rows:
        t1["leakage_components"] = write(
            RELEASE / "tier1_public_benchmark" / "leakage_components.csv",
            project(rows, T1_EDGE_FIELDS, "tier1"), T1_EDGE_FIELDS, args.check)
    else:
        missing.append(str(T1_COMPONENTS))

    rows = read(T1_NEAR_DUPES)
    if rows:
        fields = [f for f in rows[0].keys() if f not in FORBIDDEN_BY_TIER["tier1"]]
        t1["near_duplicate_candidates"] = write(
            RELEASE / "tier1_public_benchmark" / "near_duplicate_candidates.csv",
            project(rows, fields, "tier1"), fields, args.check)
    manifest["tiers"]["tier1_public_benchmark"] = t1

    # ── Tier 2 ───────────────────────────────────────────────────────────
    t2: dict[str, int] = {}
    rows = read(T2_SPLIT)
    if rows:
        t2["split_seed42"] = write(
            RELEASE / "tier2_oasis1" / "splitguard_split_seed42.csv",
            project(rows, T2_SPLIT_FIELDS, "tier2"), T2_SPLIT_FIELDS, args.check)
    else:
        missing.append(str(T2_SPLIT))

    rows = read(T2_COMPONENTS)
    if rows:
        t2["leakage_components"] = write(
            RELEASE / "tier2_oasis1" / "leakage_components.csv",
            project(rows, T2_COMPONENT_FIELDS, "tier2"), T2_COMPONENT_FIELDS, args.check)
    else:
        missing.append(str(T2_COMPONENTS))
    manifest["tiers"]["tier2_oasis1"] = t2

    manifest["tier3_note"] = (
        "ADNI1 artefacts are not emitted here. Under the LONI/ADNI data-use "
        "agreement only hashed component manifests may be redistributed; "
        "produce them with scripts/build_hashed_manifest_tier3.py, which "
        "requires a secret release salt."
    )
    manifest["excluded_columns_by_tier"] = {k: sorted(v) for k, v in FORBIDDEN_BY_TIER.items()}

    if not args.check:
        RELEASE.mkdir(parents=True, exist_ok=True)
        (RELEASE / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    verb = "Would write" if args.check else "Wrote"
    for tier, files in manifest["tiers"].items():
        print(f"{tier}:")
        for name, n in files.items():
            print(f"  {verb.lower()} {n:6d} rows -> {name}.csv")
    if missing:
        print("\nMissing local inputs (tier artefact skipped):")
        for path in missing:
            print(f"  - {path}")
        print("These live under data/, which is not redistributed; rebuild them "
              "locally first.")
        return 1
    if not args.check:
        print(f"\nRelease manifest: {RELEASE / 'RELEASE_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
