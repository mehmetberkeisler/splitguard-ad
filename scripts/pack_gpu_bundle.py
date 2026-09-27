#!/usr/bin/env python3
"""Pack the data the GPU programme reads into tar files for upload.

The 2D bundle holds the split files of every stage and every image they
reference. The volumetric bundle holds the 128^3 volumes the volumetric split
files reference. Archive paths are relative to the repository root, so
``tar -xf`` at the root of a clone puts every image where the split files'
``relative_path`` column expects it and every volume under
``data/adni3d_128``, the default ``--preprocessed-root`` of
``scripts/gpu_program.py``.

Both bundles contain ADNI-derived participant data. Upload them only to a node
covered by the Data Use Agreement, and delete them there afterwards.

Usage
-----
    python3 scripts/pack_gpu_bundle.py
    python3 scripts/pack_gpu_bundle.py \\
        --volumes-root /Volumes/KIOXIA/adni_downloads/ADNI1_preprocessed_128
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
SPLIT_SOURCES = ["tier1_truth", "oasis1_splitguard_seed42.csv", "adni", "adni_with_converters",
                 "adni_no_mt1", "adni_size_balanced", "adni_dose_response", "adni_3d"]
VOLUME_DIR = Path("data") / "adni3d_128"


def write_tar(path: Path, members: list[tuple[str, Path]]) -> None:
    with tarfile.open(path, "w") as tar:
        for arcname, source in members:
            tar.add(source, arcname=arcname, recursive=False)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    print(f"{path.name}: {len(members)} files, {path.stat().st_size / 1e6:.1f} MB, sha256 {digest.hexdigest()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "gpu_bundle_2d.tar")
    ap.add_argument("--volumes-root", type=Path, default=None,
                    help="Directory of <subject_id>/<image_uid>.npz volumes; packs the volumetric bundle too.")
    ap.add_argument("--output-3d", type=Path, default=ROOT / "gpu_bundle_3d.tar")
    args = ap.parse_args()

    split_files = []
    for name in SPLIT_SOURCES:
        source = SPLITS / name
        if not source.exists():
            raise SystemExit(f"missing split source {source.relative_to(ROOT)}")
        # Only split manifests: the split directories also hold audit exports
        # (excluded_components.csv and the injection audits), which are not
        # inputs to any stage.
        split_files += (sorted(f for f in source.glob("*.csv") if "audit" not in f.name
                               and f.name != "excluded_components.csv")
                        if source.is_dir() else [source])

    images: dict[str, Path] = {}
    volumes: dict[str, Path] = {}
    for split in split_files:
        images[split.relative_to(ROOT).as_posix()] = split
        with split.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("relative_path"):
                    image = ROOT / row["relative_path"]
                    if not image.is_file():
                        raise SystemExit(f"{split.name}: missing {row['relative_path']}")
                    images[row["relative_path"]] = image
                if split.parent.name == "adni_3d" and args.volumes_root is not None:
                    volume = args.volumes_root / row["subject_id"] / f"{row['image_uid']}.npz"
                    if not volume.is_file():
                        raise SystemExit(f"{split.name}: missing volume {volume}")
                    volumes[(VOLUME_DIR / row["subject_id"] / volume.name).as_posix()] = volume

    write_tar(args.output, sorted(images.items()))
    if args.volumes_root is not None:
        write_tar(args.output_3d, sorted(volumes.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
