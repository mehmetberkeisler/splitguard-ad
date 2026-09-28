#!/usr/bin/env python3
"""Materialise the redistributable artefacts the Data Availability statement promises.

Why this exists
---------------
The manuscript states that the GitHub release ships, per tier:

* Tier 1 (public JPEG benchmark) — the exact frozen component-safe split
  manifests at image-path level, plus the leakage-graph edge lists;
* Tier 2 (OASIS-1) — the frozen split manifests indexed by the public OASIS
  subject and session codes, with no derived image paths;
* Tier 3 (ADNI1) — hashed component manifests only.

None of that actually shipped. Everything lived under ``data/``, which is
gitignored, so a reader cloning the repository received the code and no
artefacts at all — the one thing a reproducibility paper cannot get wrong.

This script writes those artefacts into ``release/``, which is tracked,
applying the redistribution rule for each tier rather than copying files
wholesale. It is deliberately conservative: Tier 2 drops every filesystem path
and keeps only the public OASIS identifiers, and no tier emits
participant-level clinical fields.

Tier 3 needs a secret per-release salt, so it is emitted only when one is
supplied. Without it the ADNI rows are skipped and the manifest records their
absence rather than implying they shipped. The hashing itself lives in
``build_hashed_manifest_tier3.py``; this script drives it so that one command
owns ``release/`` and its manifest.

Usage
-----
    python3 scripts/build_release_artefacts.py
    python3 scripts/build_release_artefacts.py --check   # verify, write nothing
    python3 scripts/build_release_artefacts.py --tier3-salt-file ~/.splitguard/release_salt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hashed_manifest_tier3 as tier3  # noqa: E402

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
#
# Tier 3 is ADNI1 under the LONI data-use agreement, where participant-level
# identifiers may not be redistributed at all. Only one row per component
# ships, and the subject identifier survives solely as a PBKDF2 derivation
# under a salt that is never published.
FORBIDDEN_BY_TIER = {
    "tier1": {"path", "image_path", "source_archive", "source_disc"},
    "tier2": {"path", "relative_path", "image_path", "source_archive",
              "source_disc", "cdr", "mmse", "educ", "ses", "age", "sex",
              "clinical_label"},
    "tier3": set(tier3.DROPPED_FIELDS),
}

# ── Tier 3: ADNI1, hashed components only ────────────────────────────────
# The primary CN-vs-AD arm, which is the arm the headline result reports. The
# converter-inclusive arm is deliberately not emitted: a converting
# participant contributes both CN and AD scans to one component, and the
# hashed-manifest aggregation refuses a component whose label is not uniform
# rather than inventing one for it.
T3_SPLIT_DIR = ROOT / "data" / "splits" / "adni"


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


def retained(path: Path) -> int:
    """Row count of an artefact already in release/, or 0 if it is not there.

    A rebuild on a checkout whose local inputs are incomplete must not quietly
    drop an artefact that already shipped. This happened: the Tier-1
    near-duplicate source under data/ was regenerated empty, so a later rebuild
    omitted 1,582 released pairs from the manifest while leaving the correct
    file on disk, and the manifest stopped describing the release. Counting
    what is actually there keeps the two in step; the caller still reports the
    stale input so it gets rebuilt.
    """
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Report what would be written without writing.")
    ap.add_argument("--tier3-salt-file", type=Path, default=None,
                    help=("File holding the SECRET per-release salt for the ADNI "
                          f"hashed manifests. May instead be supplied via "
                          f"${tier3.SALT_ENV_VAR}. Never commit the salt or "
                          "publish it beside the manifests."))
    args = ap.parse_args()

    manifest: dict[str, object] = {"tiers": {}}
    missing: list[str] = []
    retained_paths: list[str] = []

    # ── Tier 1 ───────────────────────────────────────────────────────────
    t1: dict[str, int] = {}
    rows = read(T1_SPLIT)
    if rows:
        t1["splitguard_split_seed42"] = write(
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

    near_dupes = RELEASE / "tier1_public_benchmark" / "near_duplicate_candidates.csv"
    rows = read(T1_NEAR_DUPES)
    if rows:
        fields = [f for f in rows[0].keys() if f not in FORBIDDEN_BY_TIER["tier1"]]
        t1["near_duplicate_candidates"] = write(
            near_dupes, project(rows, fields, "tier1"), fields, args.check)
    else:
        # Keep what already shipped rather than dropping it from the manifest.
        kept = retained(near_dupes)
        if kept:
            t1["near_duplicate_candidates"] = kept
            retained_paths.append(f"{near_dupes.relative_to(ROOT)} ({kept} rows kept)")
        missing.append(str(T1_NEAR_DUPES))
    manifest["tiers"]["tier1_public_benchmark"] = t1

    # ── Tier 2 ───────────────────────────────────────────────────────────
    t2: dict[str, int] = {}
    rows = read(T2_SPLIT)
    if rows:
        t2["splitguard_split_seed42"] = write(
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

    # ── Tier 3 ───────────────────────────────────────────────────────────
    salt = None
    if args.tier3_salt_file is not None:
        salt_path = args.tier3_salt_file.expanduser()
        if not salt_path.is_file():
            raise SystemExit(f"Tier-3 salt file not found: {salt_path}")
        salt = salt_path.read_text(encoding="utf-8").strip()
    else:
        salt = os.environ.get(tier3.SALT_ENV_VAR)

    if salt:
        if len(salt) < 16:
            raise SystemExit(
                f"Tier-3 salt is too short ({len(salt)} chars). ADNI PTIDs occupy "
                "a ~10^7 space, so a guessable salt is equivalent to no salt.")
        t3: dict[str, int] = {}
        splits = sorted(T3_SPLIT_DIR.glob("adni_splitguard_seed*.csv"))
        if not splits:
            missing.append(str(T3_SPLIT_DIR / "adni_splitguard_seed*.csv"))
        for split_path in splits:
            rows = read(split_path)
            if not rows:
                missing.append(str(split_path))
                continue
            hashed = tier3.aggregate_components(rows, salt)
            # The aggregation selects its own columns; assert rather than trust,
            # because this is the file that leaves the machine.
            leaked = FORBIDDEN_BY_TIER["tier3"].intersection(hashed[0].keys())
            if leaked:
                raise ValueError(
                    f"Refusing to release columns restricted for tier3: {sorted(leaked)}")
            name = split_path.stem.replace("adni_splitguard_", "")
            t3[f"hashed_components_{name}"] = write(
                RELEASE / "tier3_adni1" / f"hashed_components_{name}.csv",
                hashed, tier3.RELEASE_FIELDS, args.check)
        manifest["tiers"]["tier3_adni1"] = t3
        # A fingerprint distinguishes two releases without disclosing the salt.
        manifest["tier3_salt_fingerprint"] = hashlib.pbkdf2_hmac(
            "sha256", b"salt-fingerprint", salt.encode("utf-8"),
            tier3.PBKDF2_ITERATIONS, dklen=8).hex()
        manifest["tier3_note"] = (
            "ADNI1 ships as hashed component manifests for the five frozen "
            "primary-arm splits: one row per component, with the subject "
            "identifier present only as PBKDF2-HMAC-SHA256 "
            f"({tier3.PBKDF2_ITERATIONS} iterations) under a secret per-release "
            "salt that is not published. Every participant-level identifier, "
            "date and image path is withheld. The converter-inclusive arm is "
            "not emitted because its components carry more than one label."
        )
    else:
        manifest["tier3_note"] = (
            "ADNI1 artefacts are not emitted here. Under the LONI/ADNI data-use "
            "agreement only hashed component manifests may be redistributed; "
            "produce them by re-running this script with --tier3-salt-file, "
            "which requires a secret release salt."
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
    if retained_paths:
        print("\nKept from the existing release (local input missing or empty):")
        for path in retained_paths:
            print(f"  - {path}")
    if missing:
        print("\nMissing or empty local inputs:")
        for path in missing:
            print(f"  - {path}")
        print("These live under data/, which is not redistributed; rebuild them "
              "locally first. Anything listed as kept above is still the "
              "previously released file, not a regenerated one.")
        return 1
    if not args.check:
        print(f"\nRelease manifest: {RELEASE / 'RELEASE_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
