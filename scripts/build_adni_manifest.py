#!/usr/bin/env python3
"""Build the ADNI manifest from downloaded LONI IDA study files + images.

Inputs
------
* ``data/raw/adni/study_files/`` — ADNI tabular CSV/XLSX exports
  (DXSUM_PDXCONV_ADNIALL or similar diagnosis tables, PTDEMOG demographics,
  visit/session tables, MRI image-metadata tables, data dictionary).
* ``data/raw/adni/images/`` — directory tree of structural MRI files
  (.dcm and/or .nii / .nii.gz). The script walks this tree, extracts subject
  / session keys from the path, and joins to the tabular metadata.

Output
------
* ``data/manifests/adni/adni_manifest.csv`` with the canonical SplitGuard
  columns: ``image_id``, ``subject_id``, ``session_id``, ``source_dataset``,
  ``scanner_field_strength``, ``age``, ``sex``, ``diagnosis_group``,
  ``slice_index``, ``qc_flag``, plus ADNI-specific columns
  (``ptid``, ``rid``, ``viscode``, ``image_uid``, ``series_uid``,
  ``acq_date``, ``modality``, ``image_path``, ``relative_path``,
  ``label_source``, ``label_confidence``).
* ``reports/audits/adni/adni_manifest_summary.json`` with per-class counts,
  per-field missingness, QC-flag rates, and field-strength distribution.

Label policy
------------
Diagnosis labels are taken from the ADNI diagnostic summary table
(typically ``DXSUM_PDXCONV_ADNIALL.csv`` or ``DXSUM*.csv``). Codes are
mapped per the canonical ontology in ``docs/SPLITGUARD_AD_FORWARD_PLAN.md``
§7.4:

* ``1`` → ``CN``
* ``2`` → ``MCI``
* ``3`` → ``AD``

(Older / phase-specific ADNI codes such as ``DXCURREN`` or ``DIAGNOSIS``
are handled.) Rows with ambiguous or missing diagnosis are kept in the
manifest with ``diagnosis_group=unknown`` and ``label_confidence=missing``
so the splitter can later exclude them; they are NOT silently dropped.

The script intentionally does not parse DICOMs (no pydicom dependency); it
reads only file names and tabular CSVs, then leaves DICOM-header
verification (e.g. magnetic-field strength from the actual file) to a later
QC pass. Field strength is taken from the MRI image-metadata table when
present.

Run from the repo root::

    python3 scripts/build_adni_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_FILES = PROJECT_ROOT / "data" / "raw" / "adni" / "study_files"
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "raw" / "adni" / "images"
DEFAULT_SLICE_MANIFEST = PROJECT_ROOT / "data" / "preprocessed" / "adni" / "slice_manifest.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "adni" / "adni_manifest.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_manifest_summary.json"

SOURCE_DATASET = "adni"
IMAGE_SUFFIXES = (".dcm", ".nii", ".nii.gz", ".img", ".hdr", ".mgz")

# Canonical SplitGuard manifest columns plus ADNI-specific extras. Order is
# stable so downstream scripts can reference fields by position if needed.
MANIFEST_COLUMNS = [
    "image_id",
    "subject_id",
    "session_id",
    "source_dataset",
    "scanner_field_strength",
    "age",
    "sex",
    "diagnosis_group",
    "slice_index",
    "qc_flag",
    "ptid",
    "rid",
    "viscode",
    "image_uid",
    "series_uid",
    "acq_date",
    "modality",
    "image_path",
    "relative_path",
    "label_source",
    "label_confidence",
]

# ADNI diagnosis-code mapping (current and legacy fields).
DX_CODE_TO_GROUP = {
    "1": "CN",
    "2": "MCI",
    "3": "AD",
    # ADNI3 DIAGNOSIS variable uses the same coding; conversion codes (e.g.
    # 4, 5, 6, 7, 8, 9 for transitions) collapse to the destination state.
    "4": "MCI",
    "5": "AD",
    "6": "AD",
    "7": "CN",
    "8": "MCI",
    "9": "MCI",
}

# Candidate filename stems (case-insensitive substring match) for each table.
DIAGNOSIS_TABLE_HINTS = ("dxsum", "diagnosis", "dxcurren")
DEMOG_TABLE_HINTS = ("ptdemog", "demog")
MRI_METADATA_HINTS = ("mri", "imageacquisition", "image_acquisition", "imagemeta", "mri_listing", "mr_meta")


def stable_token(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def normalize_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_csv_safely(path: Path) -> list[dict[str, str]]:
    """Read a CSV with the locale-tolerant defaults ADNI tables tend to need."""
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return [
                    {normalize_key(key): (value or "").strip() for key, value in row.items() if key is not None}
                    for row in reader
                ]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not decode {path}: {last_error}")


def find_table(study_files_root: Path, hints: Iterable[str]) -> Path | None:
    csvs = list(study_files_root.rglob("*.csv"))
    for hint in hints:
        for path in csvs:
            if hint in path.name.lower():
                return path
    return None


def pick(row: dict[str, str], *candidates: str) -> str:
    """Return the first present, non-empty value among ``candidates``."""
    for key in candidates:
        value = row.get(normalize_key(key), "")
        if value:
            return value
    return ""


def load_diagnosis_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Map ``(rid, viscode)`` → diagnosis row.

    Keys are normalized to strings. Rows missing both RID and PTID are
    skipped.
    """
    rows = read_csv_safely(path)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        rid = pick(row, "rid", "RID")
        ptid = pick(row, "ptid", "PTID")
        viscode = pick(row, "viscode2", "VISCODE2", "viscode", "VISCODE")
        if not (rid or ptid):
            continue
        key = (rid or ptid, viscode)
        out[key] = row
    return out


def diagnosis_from_row(row: dict[str, str]) -> tuple[str, str, str]:
    """Return (diagnosis_group, label_source, label_confidence).

    Implements ADNI_LABEL_ONTOLOGY.md §4: modern phases prefer DIAGNOSIS /
    DXCHANGE; ADNI1 falls back to the mutually-exclusive legacy flags
    DXAD / DXMCI / DXNORM.
    """
    # DIAGNOSIS (ADNI3+) is preferred; fall back to DXCURREN, then DXCHANGE.
    for column, source in (
        ("diagnosis", "DIAGNOSIS"),
        ("dxcurren", "DXCURREN"),
        ("dxchange", "DXCHANGE"),
        ("dx_bl", "DX_BL"),
    ):
        value = pick(row, column)
        if value and value in DX_CODE_TO_GROUP:
            return DX_CODE_TO_GROUP[value], source, "clinical_table"
        if value and value.upper() in {"CN", "MCI", "AD", "LMCI", "EMCI", "SMC"}:
            collapsed = "MCI" if value.upper() in {"LMCI", "EMCI", "SMC"} else value.upper()
            return collapsed, source, "clinical_table"
    # §4.2 ADNI1 legacy flags (mutually exclusive in this export).
    if pick(row, "dxad") == "1":
        return "AD", "DXAD", "clinical_table"
    if pick(row, "dxmci") == "1":
        return "MCI", "DXMCI", "clinical_table"
    if pick(row, "dxnorm") == "1":
        return "CN", "DXNORM", "clinical_table"
    return "unknown", "", "missing"


def load_demographics(path: Path) -> dict[str, dict[str, str]]:
    """Per-subject merged PTDEMOG row.

    PTDEMOG is collected at multiple visits; PTGENDER and PTDOBYY can be
    blank on later visits. Merge by keeping the first non-empty value
    seen per field for each subject, keyed by both RID and PTID.
    """
    rows = read_csv_safely(path)
    merged: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        for key in (pick(row, "rid", "RID"), pick(row, "ptid", "PTID")):
            if not key:
                continue
            dest = merged[key]
            for field, value in row.items():
                if value and not dest.get(field):
                    dest[field] = value
    return dict(merged)


def demog_age_sex(row: dict[str, str], acq_date_obj: date | None) -> tuple[str, str]:
    """Return (age, sex). Age is computed from PTDOBYY at scan time.

    PTDOBYY is encoded as an ISO date with placeholder month/day
    (``YYYY-01-01``); only the year is reliable, so age is year-difference.
    """
    raw_sex = pick(row, "ptgender", "PTGENDER", "sex", "SEX")
    sex_map = {"1": "M", "2": "F", "M": "M", "F": "F", "Male": "M", "Female": "F"}
    sex = sex_map.get(raw_sex, raw_sex)
    age = pick(row, "age", "AGE", "ptage", "PTAGE")
    if not age:
        dob_raw = pick(row, "ptdobyy", "PTDOBYY")
        if dob_raw and acq_date_obj is not None:
            year_token = dob_raw[:4]
            if year_token.isdigit():
                age = str(acq_date_obj.year - int(year_token))
    return age, sex


def load_mri_metadata(path: Path) -> dict[str, list[dict[str, str]]]:
    """Index MRIMETA / MRI3META by PTID, with rows sorted by EXAMDATE.

    The LONI IDA exports used here (MRIMETA_*.csv, MRI3META_*.csv) do
    not include image-UID or series-UID columns, so the per-image join
    must go through (PTID, EXAMDATE) date-proximity per
    ADNI_LABEL_ONTOLOGY.md §5.
    """
    rows = read_csv_safely(path)
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ptid = pick(row, "ptid", "PTID")
        if not ptid:
            continue
        out[ptid].append(row)
    for ptid_rows in out.values():
        ptid_rows.sort(key=lambda r: pick(r, "examdate", "EXAMDATE"))
    return out


def parse_iso_date(value: str) -> date | None:
    """Best-effort ISO-date parser for ADNI EXAMDATE / acq_date strings."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def closest_row_by_date(
    candidates: list[dict[str, str]],
    target: date | None,
    date_fields: tuple[str, ...],
    tolerance_days: int,
) -> dict[str, str]:
    """Return the candidate row whose date is closest to ``target``.

    Returns ``{}`` if no candidate has a parsable date or none falls
    within ``tolerance_days``.
    """
    if target is None or not candidates:
        return {}
    best: dict[str, str] = {}
    best_delta = timedelta(days=10**6)
    for row in candidates:
        for field in date_fields:
            candidate_date = parse_iso_date(pick(row, field))
            if candidate_date is not None:
                break
        else:
            continue
        delta = abs(candidate_date - target)
        if delta < best_delta:
            best, best_delta = row, delta
    if best_delta > timedelta(days=tolerance_days):
        return {}
    return best


# Pattern used to recover IDs from ADNI image paths. ADNI image collections
# encode subject as ``NNN_S_NNNN``; image UID often appears as ``I123456``;
# series UID often as ``S123456``; acquisition date as ``YYYY-MM-DD`` in the
# session-directory name.
PTID_RE = re.compile(r"\b\d{3}_S_\d{4}\b")
IMAGE_UID_RE = re.compile(r"\bI\d+\b")
SERIES_UID_RE = re.compile(r"(?<![A-Za-z])S\d{4,}\b")
ACQ_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:_\d{2}_\d{2}_\d{2})?\b")


def walk_images(images_root: Path) -> Iterable[Path]:
    for path in sorted(images_root.rglob("*")):
        if not path.is_file():
            continue
        suffixes = "".join(path.suffixes).lower()
        if any(suffixes.endswith(suf) for suf in IMAGE_SUFFIXES):
            yield path


def read_slice_manifest(path: Path) -> list[dict[str, str]]:
    """Read the preprocessed slice manifest produced by
    scripts/preprocess_adni_volumes_to_slices.py. One row per 2D
    coronal-center-slice PNG, with PTID / image_uid / acq_date
    already parsed at preprocessing time."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {k: (v or "").strip() for k, v in row.items() if k is not None}
            for row in csv.DictReader(handle)
        ]


def parse_path_identifiers(path: Path, images_root: Path) -> dict[str, str]:
    """Best-effort identifier extraction from the ADNI image-collection path."""
    text = str(path.relative_to(images_root))
    ptid_match = PTID_RE.search(text)
    image_uid_match = IMAGE_UID_RE.search(text)
    series_uid_match = SERIES_UID_RE.search(text)
    acq_date_match = ACQ_DATE_RE.search(text)
    return {
        "ptid": ptid_match.group(0) if ptid_match else "",
        "image_uid": image_uid_match.group(0) if image_uid_match else "",
        "series_uid": series_uid_match.group(0) if series_uid_match else "",
        "acq_date": acq_date_match.group(1) if acq_date_match else "",
    }


def ptid_to_rid(ptid: str) -> str:
    if not ptid:
        return ""
    parts = ptid.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return str(int(parts[-1]))
    return ""


def infer_session_id(ptid: str, viscode: str, acq_date: str) -> str:
    if ptid and viscode:
        return f"{ptid}__{viscode}"
    if ptid and acq_date:
        return f"{ptid}__{acq_date}"
    return ptid or acq_date or "unknown"


def build_manifest_row(
    path: Path,
    images_root: Path,
    mri_meta_by_ptid: dict[str, list[dict[str, str]]],
    dx_by_key: dict[tuple[str, str], dict[str, str]],
    dx_by_ptid: dict[str, list[dict[str, str]]],
    demog_by_key: dict[str, dict[str, str]],
    match_tolerance_days: int = 180,
) -> dict[str, str]:
    ids = parse_path_identifiers(path, images_root)
    ptid = ids["ptid"]
    rid = ptid_to_rid(ptid)
    image_uid = ids["image_uid"]
    series_uid = ids["series_uid"]
    acq_date = ids["acq_date"]
    acq_date_obj = parse_iso_date(acq_date)

    # MRIMETA lookup: closest EXAMDATE for this PTID within tolerance.
    # ADNI_LABEL_ONTOLOGY.md §5.
    meta = closest_row_by_date(
        mri_meta_by_ptid.get(ptid, []),
        acq_date_obj,
        date_fields=("examdate",),
        tolerance_days=match_tolerance_days,
    )
    viscode = pick(meta, "viscode2", "VISCODE2", "viscode", "VISCODE")
    field_strength = pick(meta, "field_strength", "FIELD_STRENGTH", "fldstrngth", "FLDSTRNGTH", "magstrength")
    modality = "MR" if any(str(path).lower().endswith(s) for s in IMAGE_SUFFIXES) else ""
    motion = pick(meta, "motion", "MOTION")
    malfunc = pick(meta, "malfunc", "MALFUNC")
    if motion == "1" or malfunc == "1":
        qc_flag = "fail"
    elif meta:
        qc_flag = "pass"
    else:
        qc_flag = "unknown"

    # DXSUM lookup: prefer (rid, viscode) join; fall back to date-only
    # match per ADNI_LABEL_ONTOLOGY.md §5 step 5.
    dx_row = dx_by_key.get((rid, viscode)) or dx_by_key.get((ptid, viscode)) or {}
    if dx_row:
        diagnosis_group, label_source, label_confidence = diagnosis_from_row(dx_row)
    elif not meta:
        diagnosis_group, label_source, label_confidence = ("unknown", "", "no_mri_meta_match")
    else:
        dx_row = closest_row_by_date(
            dx_by_ptid.get(ptid, []),
            acq_date_obj,
            date_fields=("examdate",),
            tolerance_days=match_tolerance_days,
        )
        if dx_row:
            diagnosis_group, label_source, _ = diagnosis_from_row(dx_row)
            label_confidence = "date_only_match" if diagnosis_group != "unknown" else "missing"
        else:
            diagnosis_group, label_source, label_confidence = ("unknown", "", "missing")

    demog_row = demog_by_key.get(rid) or demog_by_key.get(ptid) or {}
    age, sex = demog_age_sex(demog_row, acq_date_obj) if demog_row else ("", "")

    session_id = infer_session_id(ptid, viscode, acq_date)
    relative_path = str(path.relative_to(PROJECT_ROOT))
    image_id = f"adni_{stable_token(relative_path)}"

    return {
        "image_id": image_id,
        "subject_id": ptid or "unknown",
        "session_id": session_id,
        "source_dataset": SOURCE_DATASET,
        "scanner_field_strength": field_strength,
        "age": age,
        "sex": sex,
        "diagnosis_group": diagnosis_group,
        "slice_index": "",  # ADNI manifests are volume-level by default.
        "qc_flag": qc_flag,
        "ptid": ptid,
        "rid": rid,
        "viscode": viscode,
        "image_uid": image_uid,
        "series_uid": series_uid,
        "acq_date": acq_date,
        "modality": modality,
        "image_path": str(path.resolve()),
        "relative_path": relative_path,
        "label_source": label_source,
        "label_confidence": label_confidence,
    }


def build_row_from_slice(
    slice_row: dict[str, str],
    project_root: Path,
    mri_meta_by_ptid: dict[str, list[dict[str, str]]],
    dx_by_key: dict[tuple[str, str], dict[str, str]],
    dx_by_ptid: dict[str, list[dict[str, str]]],
    demog_by_key: dict[str, dict[str, str]],
    match_tolerance_days: int = 180,
) -> dict[str, str]:
    """Same clinical join as build_manifest_row, but starts from a
    preprocessed slice-manifest row (PTID / image_uid / acq_date already
    parsed) rather than walking volumes on disk."""
    image_id = slice_row["image_id"]
    ptid = slice_row.get("ptid", "")
    rid = ptid_to_rid(ptid)
    image_uid = slice_row.get("image_uid", "")
    series_uid = slice_row.get("series_uid", "")
    acq_date = slice_row.get("acq_date", "")
    acq_date_obj = parse_iso_date(acq_date)
    slice_path = slice_row.get("slice_path", "")
    abs_path = project_root / slice_path

    # MRIMETA lookup (date-proximity, §5 of ADNI_LABEL_ONTOLOGY.md).
    meta = closest_row_by_date(
        mri_meta_by_ptid.get(ptid, []),
        acq_date_obj,
        date_fields=("examdate",),
        tolerance_days=match_tolerance_days,
    )
    viscode = pick(meta, "viscode2", "VISCODE2", "viscode", "VISCODE")
    field_strength = pick(meta, "field_strength", "FIELD_STRENGTH", "fldstrngth", "FLDSTRNGTH", "magstrength")
    motion = pick(meta, "motion", "MOTION")
    malfunc = pick(meta, "malfunc", "MALFUNC")
    if motion == "1" or malfunc == "1":
        qc_flag = "fail"
    elif meta:
        qc_flag = "pass"
    else:
        qc_flag = "unknown"

    # DXSUM lookup with date-fallback (§5 step 5).
    dx_row = dx_by_key.get((rid, viscode)) or dx_by_key.get((ptid, viscode)) or {}
    if dx_row:
        diagnosis_group, label_source, label_confidence = diagnosis_from_row(dx_row)
    elif not meta:
        diagnosis_group, label_source, label_confidence = ("unknown", "", "no_mri_meta_match")
    else:
        dx_row = closest_row_by_date(
            dx_by_ptid.get(ptid, []),
            acq_date_obj,
            date_fields=("examdate",),
            tolerance_days=match_tolerance_days,
        )
        if dx_row:
            diagnosis_group, label_source, _ = diagnosis_from_row(dx_row)
            label_confidence = "date_only_match" if diagnosis_group != "unknown" else "missing"
        else:
            diagnosis_group, label_source, label_confidence = ("unknown", "", "missing")

    demog_row = demog_by_key.get(rid) or demog_by_key.get(ptid) or {}
    age, sex = demog_age_sex(demog_row, acq_date_obj) if demog_row else ("", "")

    session_id = infer_session_id(ptid, viscode, acq_date)

    return {
        "image_id": image_id,
        "subject_id": ptid or "unknown",
        "session_id": session_id,
        "source_dataset": SOURCE_DATASET,
        "scanner_field_strength": field_strength,
        "age": age,
        "sex": sex,
        "diagnosis_group": diagnosis_group,
        "slice_index": "center_coronal",
        "qc_flag": qc_flag,
        "ptid": ptid,
        "rid": rid,
        "viscode": viscode,
        "image_uid": image_uid,
        "series_uid": series_uid,
        "acq_date": acq_date,
        "modality": "MR",
        "image_path": str(abs_path.resolve()),
        "relative_path": slice_path,
        "label_source": label_source,
        "label_confidence": label_confidence,
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_images": 0,
            "n_subjects": 0,
            "n_sessions": 0,
            "diagnosis_counts": {},
            "field_strength_counts": {},
            "qc_flag_counts": {},
            "missingness": {},
        }
    subjects = {row["subject_id"] for row in rows if row["subject_id"] and row["subject_id"] != "unknown"}
    sessions = {row["session_id"] for row in rows if row["session_id"] and row["session_id"] != "unknown"}
    diagnosis_counts = Counter(row["diagnosis_group"] for row in rows)
    field_strength_counts = Counter(row["scanner_field_strength"] or "missing" for row in rows)
    qc_flag_counts = Counter(row["qc_flag"] or "missing" for row in rows)
    missingness = {
        column: sum(1 for row in rows if not row.get(column))
        for column in MANIFEST_COLUMNS
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_images": len(rows),
        "n_subjects": len(subjects),
        "n_sessions": len(sessions),
        "diagnosis_counts": dict(diagnosis_counts),
        "field_strength_counts": dict(field_strength_counts),
        "qc_flag_counts": dict(qc_flag_counts),
        "missingness": missingness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-files-root", type=Path, default=DEFAULT_STUDY_FILES)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(
        "--slice-manifest",
        type=Path,
        default=DEFAULT_SLICE_MANIFEST,
        help="Preprocessed slice manifest from preprocess_adni_volumes_to_slices.py. "
        "If this file exists, the builder reads slices from it instead of walking --images-root.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if not args.study_files_root.exists():
        raise FileNotFoundError(
            f"ADNI study-files root does not exist: {args.study_files_root}. "
            "Run scripts/extract_adni_downloads.py first."
        )
    use_slice_mode = args.slice_manifest.exists()
    if not use_slice_mode and not args.images_root.exists():
        raise FileNotFoundError(
            f"Neither slice manifest ({args.slice_manifest}) nor images root "
            f"({args.images_root}) exists. Run "
            "scripts/preprocess_adni_volumes_to_slices.py or "
            "scripts/extract_adni_downloads.py first."
        )

    dx_path = find_table(args.study_files_root, DIAGNOSIS_TABLE_HINTS)
    demog_path = find_table(args.study_files_root, DEMOG_TABLE_HINTS)
    mri_meta_path = find_table(args.study_files_root, MRI_METADATA_HINTS)

    print(f"Diagnosis table:    {dx_path}")
    print(f"Demographics table: {demog_path}")
    print(f"MRI metadata table: {mri_meta_path}")

    dx_by_key = load_diagnosis_table(dx_path) if dx_path else {}
    demog_by_key = load_demographics(demog_path) if demog_path else {}
    mri_meta_by_ptid = load_mri_metadata(mri_meta_path) if mri_meta_path else {}

    # Per-PTID DXSUM index for date-proximity fallback (§5 step 5).
    dx_by_ptid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for dx_row in dx_by_key.values():
        ptid_key = pick(dx_row, "ptid", "PTID")
        if ptid_key:
            dx_by_ptid[ptid_key].append(dx_row)
    for ptid_rows in dx_by_ptid.values():
        ptid_rows.sort(key=lambda r: pick(r, "examdate", "EXAMDATE"))

    rows: list[dict[str, str]] = []
    if use_slice_mode:
        print(f"Slice manifest:     {args.slice_manifest}")
        slice_rows = read_slice_manifest(args.slice_manifest)
        for slice_row in slice_rows:
            rows.append(
                build_row_from_slice(
                    slice_row,
                    PROJECT_ROOT,
                    mri_meta_by_ptid=mri_meta_by_ptid,
                    dx_by_key=dx_by_key,
                    dx_by_ptid=dx_by_ptid,
                    demog_by_key=demog_by_key,
                )
            )
    else:
        for path in walk_images(args.images_root):
            rows.append(
                build_manifest_row(
                    path,
                    args.images_root,
                    mri_meta_by_ptid=mri_meta_by_ptid,
                    dx_by_key=dx_by_key,
                    dx_by_ptid=dx_by_ptid,
                    demog_by_key=demog_by_key,
                )
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["diagnosis_table"] = str(dx_path.relative_to(PROJECT_ROOT)) if dx_path else None
    summary["demographics_table"] = str(demog_path.relative_to(PROJECT_ROOT)) if demog_path else None
    summary["mri_metadata_table"] = str(mri_meta_path.relative_to(PROJECT_ROOT)) if mri_meta_path else None

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.manifest.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")
    print(f"Wrote {args.summary.relative_to(PROJECT_ROOT)}")
    if summary["n_images"] == 0:
        print(
            "WARNING: 0 ADNI images discovered. Confirm that "
            "scripts/extract_adni_downloads.py extracted the image zips and "
            "that the images root path is correct."
        )
        return 1
    if summary["diagnosis_counts"].get("unknown", 0) == summary["n_images"]:
        print(
            "WARNING: every row has diagnosis_group=unknown. Confirm the "
            "diagnosis table was located and that subject IDs in the image "
            "paths match the RID/PTID coding used in the table."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
