#!/usr/bin/env python3
"""Give each frozen ADNI split file the partitions of all three protocols.

The volumetric trainer (``train_adni_3d.py``) reads one CSV per seed and picks
the partition column by protocol: ``split_random``, ``split_subject_only`` or
``split``. The frozen split files carry only ``split`` (component-safe), and
the trainer falls back to it when a column is absent, so without this step all
three volumetric protocols would train on the same component-safe partition
and the volumetric arm could not show a gap at all.

The two extra columns are produced by the functions the 2D experiment uses
(``run_adni_inflation_gap.build_random_split`` and
``build_subject_only_split``) from the same rows in the same order, so for a
given seed a scan is in the same partition under a given protocol in the 2D
and volumetric arms. With ``--volumes-root``, scans without a preprocessed
volume are dropped after the partitions are drawn, so every remaining scan
keeps its 2D partition.

Usage
-----
    python3 scripts/make_protocol_split_columns.py \\
        --split-dir data/splits/adni --output-dir data/splits/adni_3d \\
        --volumes-root /Volumes/KIOXIA/adni_downloads/ADNI1_preprocessed_128
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import display_path  # noqa: E402
from run_adni_inflation_gap import build_random_split, build_subject_only_split  # noqa: E402
from train_adni_baseline import read_split_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits" / "adni")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data" / "splits" / "adni_3d")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--volumes-root", type=Path, default=None,
                    help="Directory of <subject_id>/<image_uid>.npz volumes; rows without one are dropped.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        path = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        rows = read_split_rows(path)                       # CN/AD rows, file order
        by_protocol = {
            "split_random": build_random_split(rows, seed),
            "split_subject_only": build_subject_only_split(rows, seed),
        }
        column = {}
        for name, splits in by_protocol.items():
            column[name] = {r["image_id"]: phase for phase, members in splits.items() for r in members}
        out_rows = []
        for r in rows:
            out_rows.append({**r, "split_random": column["split_random"][r["image_id"]],
                             "split_subject_only": column["split_subject_only"][r["image_id"]]})
        # the component-safe column is untouched and still subject-disjoint
        for name in ("split", "split_subject_only"):
            parts = {}
            for r in out_rows:
                parts.setdefault(r["subject_id"], set()).add(r[name])
            assert all(len(p) == 1 for p in parts.values()), f"{name} crosses a participant in seed {seed}"
        if args.volumes_root is not None:
            has_volume = [(args.volumes_root / r["subject_id"] / f"{r['image_uid']}.npz").is_file() for r in out_rows]
            dropped = Counter(f"{r['split']}/{r['diagnosis_group']}" for r, ok in zip(out_rows, has_volume) if not ok)
            out_rows = [r for r, ok in zip(out_rows, has_volume) if ok]
            print(f"seed {seed}: {sum(dropped.values())} scans without a volume dropped "
                  f"(component-safe partition/label: {dict(sorted(dropped.items()))})")
        out = args.output_dir / path.name
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        crossing = len({r["subject_id"] for r in out_rows if r["split_random"] == "test"}
                       & {r["subject_id"] for r in out_rows if r["split_random"] == "train"})
        print(f"seed {seed}: {len(out_rows)} rows -> {display_path(out)} "
              f"(random split puts {crossing} participants in both train and test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
