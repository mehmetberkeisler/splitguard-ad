#!/usr/bin/env python3
"""Extract raw LONI IDA ADNI download zips into the project layout.

The script walks ``data/raw/adni/downloads/*.zip`` and routes each archive to
the appropriate target directory:

* zips that contain only tabular study files (.csv/.xlsx/.txt) go to
  ``data/raw/adni/study_files/``.
* zips that contain image data (.dcm/.nii/.nii.gz/.img/.hdr) go to
  ``data/raw/adni/images/``.
* zips that contain both are treated as image zips so that study CSVs remain
  alongside their image siblings; this matches the LONI "Advanced Download"
  bundle layout.

The script is intentionally conservative:

* It never deletes the source zip.
* It is idempotent — if all expected files already exist on disk it skips the
  zip and records ``skipped_already_extracted``.
* It records a per-zip SHA-256, the inferred routing, the extracted file
  count, and any errors into ``reports/audits/adni/adni_extraction_log.json``.
* No ADNI participant-level content is written outside the
  Git-ignored ``data/`` and ``reports/audits/`` directories.

Run from the repo root::

    python3 scripts/extract_adni_downloads.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOADS = PROJECT_ROOT / "data" / "raw" / "adni" / "downloads"
DEFAULT_STUDY_FILES = PROJECT_ROOT / "data" / "raw" / "adni" / "study_files"
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "raw" / "adni" / "images"
DEFAULT_LOG = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_extraction_log.json"

IMAGE_SUFFIXES = {".dcm", ".nii", ".gz", ".img", ".hdr", ".mgz", ".mnc"}
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".txt", ".tsv", ".json", ".xml"}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_zip(members: Iterable[zipfile.ZipInfo]) -> str:
    """Return one of ``images``, ``study_files``, or ``mixed``."""
    has_image = False
    has_table = False
    for member in members:
        if member.is_dir():
            continue
        suffixes = {suffix.lower() for suffix in Path(member.filename).suffixes}
        if suffixes & IMAGE_SUFFIXES:
            has_image = True
        if suffixes & TABULAR_SUFFIXES:
            has_table = True
        if has_image and has_table:
            return "mixed"
    if has_image:
        return "images"
    if has_table:
        return "study_files"
    return "mixed"


def routing_destination(
    kind: str,
    images_root: Path,
    study_files_root: Path,
    zip_path: Path,
) -> Path:
    if kind in {"images", "mixed"}:
        return images_root / zip_path.stem
    return study_files_root / zip_path.stem


def safe_extract(
    zip_path: Path,
    target: Path,
    force: bool,
) -> tuple[int, int, bool]:
    """Extract ``zip_path`` into ``target``.

    Returns ``(n_files_extracted, n_files_skipped, already_complete)``.

    Refuses any member whose path escapes the target directory (zip-slip
    protection).
    """
    target.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        # First check whether every file is already on disk.
        if not force and all(
            (target / member.filename).exists()
            for member in zf.infolist()
            if not member.is_dir()
        ):
            return 0, 0, True
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = target / member.filename
            # Zip-slip protection.
            try:
                dest.resolve().relative_to(target.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to extract suspicious member {member.filename!r} from {zip_path}"
                ) from exc
            if dest.exists() and not force:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as source, dest.open("wb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            extracted += 1
    return extracted, skipped, False


def process_one(
    zip_path: Path,
    images_root: Path,
    study_files_root: Path,
    force: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "zip": str(zip_path.relative_to(PROJECT_ROOT)),
        "zip_sha256": "",
        "zip_size_bytes": zip_path.stat().st_size,
        "classification": "",
        "destination": "",
        "n_members": 0,
        "n_extracted": 0,
        "n_skipped_existing": 0,
        "status": "",
        "error": "",
    }
    try:
        record["zip_sha256"] = file_sha256(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
        record["n_members"] = sum(1 for member in members if not member.is_dir())
        kind = classify_zip(members)
        record["classification"] = kind
        destination = routing_destination(kind, images_root, study_files_root, zip_path)
        record["destination"] = str(destination.relative_to(PROJECT_ROOT))
        n_extracted, n_skipped, complete = safe_extract(zip_path, destination, force=force)
        record["n_extracted"] = n_extracted
        record["n_skipped_existing"] = n_skipped
        if complete:
            record["status"] = "skipped_already_extracted"
        else:
            record["status"] = "extracted_ok"
    except Exception as exc:  # noqa: BLE001 — log to JSON, do not crash the batch
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=DEFAULT_DOWNLOADS)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--study-files-root", type=Path, default=DEFAULT_STUDY_FILES)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract members even if they already exist on disk.",
    )
    args = parser.parse_args()

    if not args.downloads.exists():
        raise FileNotFoundError(f"ADNI downloads directory does not exist: {args.downloads}")

    zips = sorted(args.downloads.glob("*.zip"))
    args.images_root.mkdir(parents=True, exist_ok=True)
    args.study_files_root.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for zip_path in zips:
        print(f"Processing {zip_path.relative_to(PROJECT_ROOT)} ...")
        record = process_one(
            zip_path,
            args.images_root,
            args.study_files_root,
            force=args.force,
        )
        records.append(record)
        print(f"  -> {record['status']} ({record['n_extracted']} extracted, {record['n_skipped_existing']} already present)")

    status_counter = Counter(record["status"] for record in records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "downloads_root": str(args.downloads.relative_to(PROJECT_ROOT)),
        "n_zips": len(records),
        "status_counts": dict(status_counter),
        "records": records,
    }
    args.log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.log.relative_to(PROJECT_ROOT)}")
    if not zips:
        print(
            "No zip files were found under "
            f"{args.downloads.relative_to(PROJECT_ROOT)}. Drop the LONI IDA zips there and rerun."
        )
        return 1
    if status_counter.get("error", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
