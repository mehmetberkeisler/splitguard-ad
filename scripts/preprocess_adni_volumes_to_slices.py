#!/usr/bin/env python3
"""Preprocess ADNI NIfTI volumes to 2D coronal center slices.

Motivation
----------
Full ADNI1: Complete 3Yr 1.5T extracts to ~77 GB of .nii volumes. The
machine running this pipeline has 36 GB free disk and 8 GB RAM. The
training pipeline only uses a single coronal center slice per volume
(see ``train_adni_baseline.py:load_volume_center_slice``), so caching
that slice once at ingest reduces storage to ~110 MB total (700×
compression) without changing any model behavior.

Slice computation is byte-identical to
``train_adni_baseline.load_volume_center_slice`` so the trainer can
swap in a precomputed PNG with no observable difference.

Inputs
------
* ``--source``: a directory tree to walk for ``*.nii`` / ``*.nii.gz``
  files (default ``data/raw/adni/images``). Idempotent: existing
  slice PNGs are not recomputed.

Outputs
-------
* ``data/preprocessed/adni/slices/<image_id>.png`` — single-channel
  uint8 PNG, shape determined by source volume's H × D after rot90.
* ``data/preprocessed/adni/slice_manifest.csv`` — append-only metadata
  index keyed by ``image_id``.

The script is structured so a future ``preprocess_one_zip.py`` runner
can call ``preprocess_volume(path)`` in a stream-extract-process-delete
loop without holding multiple zips on disk simultaneously.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "adni" / "images"
DEFAULT_SLICES = PROJECT_ROOT / "data" / "preprocessed" / "adni" / "slices"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "preprocessed" / "adni" / "slice_manifest.csv"
DEFAULT_LOG = PROJECT_ROOT / "reports" / "audits" / "adni" / "adni_slice_preprocess_log.json"

NIFTI_SUFFIXES = (".nii", ".nii.gz")

# Same identifier regexes as build_adni_manifest.py so manifest joins
# stay consistent across volume- and slice-based modes.
PTID_RE = re.compile(r"\b\d{3}_S_\d{4}\b")
IMAGE_UID_RE = re.compile(r"\bI\d+\b")
SERIES_UID_RE = re.compile(r"_S(\d{4,})_")  # tolerant of digit_underscore boundary
ACQ_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:_\d{2}_\d{2}_\d{2})?\b")
# ADNI image-collection directory names encode the processing pipeline.
# Used for the manifest's `acquisition_type` column (MPR, MPR-R, MT1, …).
ACQ_TYPE_RE = re.compile(r"/(MPR-?R?|MT1|GradWarp[^/]*|N3m?[^/]*)__")


MANIFEST_COLUMNS = [
    "image_id",
    "ptid",
    "image_uid",
    "series_uid",
    "acq_date",
    "acquisition_type",
    "slice_path",
    "slice_height",
    "slice_width",
    "source_volume_path",
    "source_volume_size_mb",
    "preprocessed_at",
]


def parse_ids(path_text: str) -> dict[str, str]:
    ptid = PTID_RE.search(path_text)
    image_uid = IMAGE_UID_RE.search(path_text)
    series_uid = SERIES_UID_RE.search(path_text)
    acq_date = ACQ_DATE_RE.search(path_text)
    acq_type = ACQ_TYPE_RE.search(path_text)
    return {
        "ptid": ptid.group(0) if ptid else "",
        "image_uid": image_uid.group(0) if image_uid else "",
        "series_uid": f"S{series_uid.group(1)}" if series_uid else "",
        "acq_date": acq_date.group(1) if acq_date else "",
        "acquisition_type": acq_type.group(1) if acq_type else "",
    }


def derive_image_id(ids: dict[str, str], fallback_path: Path) -> str:
    """Stable image_id from PTID + image UID; falls back to a hash.

    Using (ptid, image_uid) instead of relative-path hash means the id
    survives storage-format migrations (volume → slice → re-extracted
    elsewhere). Same scheme is wired into ``build_adni_manifest.py``.
    """
    ptid = ids["ptid"]
    image_uid = ids["image_uid"]
    if ptid and image_uid:
        return f"adni_{ptid}_{image_uid}"
    # Fallback (should never trigger for well-formed ADNI paths).
    import hashlib
    return "adni_" + hashlib.sha1(str(fallback_path).encode("utf-8")).hexdigest()[:12]


def compute_center_slice(volume_path: Path):
    """Coronal center slice, byte-identical to train_adni_baseline.load_volume_center_slice."""
    import numpy as np
    import nibabel as nib

    image = nib.load(str(volume_path))
    data = np.squeeze(image.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze, got shape {data.shape} for {volume_path}")
    center = data.shape[1] // 2
    slice_2d = np.take(data, center, axis=1)
    slice_2d = np.nan_to_num(slice_2d, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
    else:
        low, high = float(np.min(slice_2d)), float(np.max(slice_2d))
    if high <= low:
        high = low + 1.0
    slice_2d = np.clip((slice_2d - low) / (high - low), 0.0, 1.0)
    slice_2d = (slice_2d * 255).astype(np.uint8)
    return np.rot90(slice_2d)


def save_slice_png(slice_array, out_path: Path) -> tuple[int, int]:
    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(slice_array, mode="L").save(out_path, optimize=True)
    h, w = int(slice_array.shape[0]), int(slice_array.shape[1])
    return h, w


def walk_volumes(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        text = path.name.lower()
        if text.endswith(".nii") or text.endswith(".nii.gz"):
            yield path


def read_existing_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not manifest_path.exists():
        return [], set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row["image_id"] for row in rows}


def append_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def preprocess_volume(
    volume_path: Path,
    slices_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Compute + save the center slice; return manifest row."""
    rel_text = str(volume_path.relative_to(project_root)) if volume_path.is_relative_to(project_root) else str(volume_path)
    ids = parse_ids(rel_text)
    image_id = derive_image_id(ids, volume_path)
    out_path = slices_root / f"{image_id}.png"
    slice_array = compute_center_slice(volume_path)
    h, w = save_slice_png(slice_array, out_path)
    return {
        "image_id": image_id,
        "ptid": ids["ptid"],
        "image_uid": ids["image_uid"],
        "series_uid": ids["series_uid"],
        "acq_date": ids["acq_date"],
        "acquisition_type": ids["acquisition_type"],
        "slice_path": str(out_path.relative_to(project_root)),
        "slice_height": str(h),
        "slice_width": str(w),
        "source_volume_path": rel_text,
        "source_volume_size_mb": f"{volume_path.stat().st_size / (1024 * 1024):.2f}",
        "preprocessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Directory tree to walk for .nii / .nii.gz files.")
    parser.add_argument("--slices-root", type=Path, default=DEFAULT_SLICES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, only process the first N volumes (smoke testing).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute slices even when the PNG already exists.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.source.exists():
        raise FileNotFoundError(f"Source tree does not exist: {args.source}")

    _existing_rows, existing_ids = read_existing_manifest(args.manifest)

    new_rows: list[dict[str, str]] = []
    n_skipped_existing = 0
    n_errors = 0
    errors: list[dict[str, str]] = []

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"=== ADNI slice preprocessor — started at {started_at} ===")
    print(f"Source: {args.source}")
    print(f"Slices: {args.slices_root}")
    print(f"Manifest: {args.manifest}")

    for idx, volume_path in enumerate(walk_volumes(args.source)):
        if args.limit and idx >= args.limit:
            break
        # Stable image_id check via path parse (cheap).
        rel_text = str(volume_path.relative_to(PROJECT_ROOT)) if volume_path.is_relative_to(PROJECT_ROOT) else str(volume_path)
        ids = parse_ids(rel_text)
        image_id = derive_image_id(ids, volume_path)
        slice_png = args.slices_root / f"{image_id}.png"
        if not args.force and image_id in existing_ids and slice_png.exists():
            n_skipped_existing += 1
            continue
        try:
            row = preprocess_volume(volume_path, args.slices_root)
            new_rows.append(row)
            existing_ids.add(row["image_id"])
            if (idx + 1) % 25 == 0:
                print(f"  processed {idx + 1} volumes...")
        except Exception as exc:  # noqa: BLE001 — log and continue
            n_errors += 1
            errors.append({"path": str(volume_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR on {volume_path.name}: {exc}", file=sys.stderr)

    if new_rows:
        append_manifest(args.manifest, new_rows)

    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "source": str(args.source),
        "n_processed_this_run": len(new_rows),
        "n_skipped_already_present": n_skipped_existing,
        "n_errors": n_errors,
        "errors": errors,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nProcessed: {len(new_rows)}  Skipped (already present): {n_skipped_existing}  Errors: {n_errors}")
    print(f"Manifest:  {args.manifest.relative_to(PROJECT_ROOT)}")
    print(f"Log:       {args.log.relative_to(PROJECT_ROOT)}")
    return 0 if n_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
