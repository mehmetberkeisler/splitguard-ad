#!/usr/bin/env python3
"""One-shot ADNI ingest pipeline runner.

Runs the gated sequence defined in
``docs/SPLITGUARD_AD_FORWARD_PLAN.md`` §7:

    extract → inventory → label ontology check
        → manifest → leakage graph → split → audit

Each step is invoked as a subprocess (so the underlying scripts stay
independently runnable). The runner refuses to proceed past any failed
gate and prints the exact next manual command to run.

This script never trains. After it prints **PIPELINE GATE PASS**, the
next step is manual model training:

* ``python3 scripts/train_adni_baseline.py --seed 0``
* ``python3 scripts/run_adni_inflation_gap.py --seeds 0 1 2 3 4``

Gates
-----
G1  extract_adni_downloads.py succeeds (status 0); extraction log written.
G2  build_adni_inventory.py succeeds and inventory is non-empty.
G2.5 docs/ADNI_LABEL_ONTOLOGY.md exists. The draft in the forward plan is
    not enough on its own — the user must finalize the ontology against
    the actual study-file column names.
G3a build_adni_manifest.py succeeds (status 0); summary records >0 images
    and at least one CN or AD label.
G3b build_adni_leakage_graph.py succeeds.
G3c make_adni_splitguard_split.py succeeds for every requested seed.
G3d audit_adni_splits.py succeeds (all overlap counts zero).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

EXTRACT_LOG = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_extraction_log.json"
INVENTORY_SUMMARY = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_inventory_summary.json"
ONTOLOGY = PROJECT_ROOT / "docs" / "ADNI_LABEL_ONTOLOGY.md"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv"
MANIFEST_SUMMARY = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_manifest_summary.json"
COMPONENTS = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_leakage_components.csv"


class GateFailure(SystemExit):
    def __init__(self, gate: str, message: str, next_command: str | None = None) -> None:
        body = f"\n❌ Gate {gate} failed: {message}"
        if next_command:
            body += f"\n   Next manual command: {next_command}"
        super().__init__(body)


def run(name: str, cmd: list[str]) -> None:
    print(f"\n→ {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise GateFailure(name, f"command exited with code {result.returncode}")


def check_extract_log() -> None:
    if not EXTRACT_LOG.exists():
        raise GateFailure("G1", f"extraction log not found at {EXTRACT_LOG}")
    payload = json.loads(EXTRACT_LOG.read_text(encoding="utf-8"))
    if payload.get("n_zips", 0) == 0:
        raise GateFailure(
            "G1",
            f"no zips found under {payload.get('downloads_root')}",
            "Drop the raw LONI IDA zips into data/raw/adni/downloads/ and rerun.",
        )
    error_count = payload.get("status_counts", {}).get("error", 0)
    if error_count:
        raise GateFailure("G1", f"{error_count} zip(s) failed to extract; inspect {EXTRACT_LOG}")


def check_inventory() -> None:
    if not INVENTORY_SUMMARY.exists():
        raise GateFailure("G2", f"inventory summary not found at {INVENTORY_SUMMARY}")
    payload = json.loads(INVENTORY_SUMMARY.read_text(encoding="utf-8"))
    if payload.get("n_files", 0) == 0:
        raise GateFailure(
            "G2",
            "inventory is empty",
            "Confirm extraction step produced files under data/raw/adni/study_files and data/raw/adni/images.",
        )


def check_ontology() -> None:
    if not ONTOLOGY.exists():
        raise GateFailure(
            "G2.5",
            f"label ontology not finalized at {ONTOLOGY.relative_to(PROJECT_ROOT)}",
            "Write docs/ADNI_LABEL_ONTOLOGY.md using the draft in docs/SPLITGUARD_AD_FORWARD_PLAN.md §7.4 and the actual study CSV column names.",
        )


def check_manifest() -> None:
    if not MANIFEST.exists():
        raise GateFailure("G3a", f"manifest not found at {MANIFEST.relative_to(PROJECT_ROOT)}")
    if not MANIFEST_SUMMARY.exists():
        raise GateFailure("G3a", f"manifest summary not found at {MANIFEST_SUMMARY.relative_to(PROJECT_ROOT)}")
    summary = json.loads(MANIFEST_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("n_images", 0) == 0:
        raise GateFailure("G3a", "manifest is empty")
    dx_counts = summary.get("diagnosis_counts", {})
    if not (dx_counts.get("CN") or dx_counts.get("AD")):
        raise GateFailure(
            "G3a",
            "no CN or AD rows in manifest",
            "Verify the diagnosis table is being parsed correctly in build_adni_manifest.py.",
        )


def check_components() -> None:
    if not COMPONENTS.exists():
        raise GateFailure("G3b", f"components manifest not found at {COMPONENTS.relative_to(PROJECT_ROOT)}")


def check_split_audits(seeds: list[int]) -> None:
    audit_dir = PROJECT_ROOT / "reports" / "audits" / "adni"
    for seed in seeds:
        audit_path = audit_dir / f"adni_splitguard_seed{seed}_audit.json"
        if not audit_path.exists():
            raise GateFailure("G3d", f"audit not found at {audit_path.relative_to(PROJECT_ROOT)}")
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if not payload.get("passed", False):
            raise GateFailure(
                "G3d",
                f"audit failed for seed {seed}",
                f"Inspect {audit_path.relative_to(PROJECT_ROOT)} for the offending overlap.",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip the zip-extraction step (use if zips have already been extracted).",
    )
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"=== ADNI pipeline runner — started at {started_at} ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Seeds: {args.seeds}")

    # G1
    if not args.skip_extract:
        run("G1 extract", [sys.executable, str(SCRIPTS / "extract_adni_downloads.py")])
    check_extract_log()
    print("✅ G1 extract OK")

    # G2
    run("G2 inventory", [sys.executable, str(SCRIPTS / "build_adni_inventory.py")])
    check_inventory()
    print("✅ G2 inventory OK")

    # G2.5
    check_ontology()
    print("✅ G2.5 ontology OK")

    # G3a
    run("G3a manifest", [sys.executable, str(SCRIPTS / "build_adni_manifest.py")])
    check_manifest()
    print("✅ G3a manifest OK")

    # G3b
    run("G3b leakage graph", [sys.executable, str(SCRIPTS / "build_adni_leakage_graph.py")])
    check_components()
    print("✅ G3b leakage graph OK")

    # G3c
    seed_args = [str(seed) for seed in args.seeds]
    run(
        "G3c splits",
        [sys.executable, str(SCRIPTS / "make_adni_splitguard_split.py"), "--seeds", *seed_args],
    )
    print("✅ G3c splits OK")

    # G3d
    run("G3d audit", [sys.executable, str(SCRIPTS / "audit_adni_splits.py")])
    check_split_audits(args.seeds)
    print("✅ G3d audit OK")

    print("\n================================================================")
    print("✅ PIPELINE GATE PASS — ADNI ingest sequence complete.")
    print("Next manual commands:")
    print("  python3 scripts/train_adni_baseline.py --seed 0")
    print("  python3 scripts/run_adni_inflation_gap.py --seeds " + " ".join(seed_args))
    print("================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
