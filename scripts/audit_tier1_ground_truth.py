#!/usr/bin/env python3
"""Recover the true participants behind the Tier-1 benchmark and audit its splits.

Why this exists
---------------
The Tier-1 benchmark (the public Kaggle four-class Alzheimer's MRI set) ships
6,400 JPEG slices with no participant identifiers; the pipeline inferred
identity from the parenthesised number in each filename. That inference was
never checked, because nothing in the redistribution says where the images
came from.

They came from OASIS-1. Each JPEG is pixel-identical (up to JPEG
re-encoding) to an axial slice of an OASIS-1 atlas-registered, brain-masked,
gain-field-corrected volume (``*_t88_masked_gfc``), which is available
locally. Matching every JPEG against every axial slice of every volume
recovers the true participant for every image, and with it a ground truth the
filename convention can be judged against.

What it measures
----------------
1. The mapping itself: match quality, participants, slices per participant.
2. Label fidelity: the folder labels against OASIS-1 CDR.
3. Overlap with Tier 2 (the project's own OASIS-1 pipeline).
4. Filename-derived identity against the truth: how many filename "subjects"
   are one participant, how many merge several, how far participants fragment.
5. True leakage in the frozen Tier-1 component-safe split.
6. The near-duplicate candidates, adjudicated by true identity.
7. The dHash detector calibrated over every image pair.

Inputs are local (OASIS-1 volumes, the Kaggle images, the frozen split). The
per-image ground truth goes to ``data/`` (not redistributed); the aggregate
summary goes to ``reports/tables/`` and backs the manuscript's numbers.

Usage
-----
    python3 scripts/audit_tier1_ground_truth.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import display_path  # noqa: E402
from build_current_leakage_graph import dhash, hamming  # noqa: E402  pipeline's own hash

DEFAULTS = dict(
    kaggle_root=ROOT / "Alzheimer_MRI_4_classes_dataset",
    oasis_volumes=ROOT / "oasis1" / "selected" / "processed_t88_masked",
    oasis_metadata=ROOT / "oasis1" / "metadata",
    split=ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv",
    near_dupes=ROOT / "release" / "tier1_public_benchmark" / "near_duplicate_candidates.csv",
    tier2_split=ROOT / "data" / "splits" / "oasis1_splitguard_seed42.csv",
    ground_truth=ROOT / "data" / "manifests" / "tier1_oasis1_ground_truth.csv",
    summary=ROOT / "reports" / "tables" / "tier1_ground_truth_audit.json",
)
Z_RANGE = range(40, 137)          # axial slices searched; every match lies well inside
DOWNSAMPLE = 2                    # match on 88x104 block means
THRESHOLDS = (0, 2, 4, 8, 12, 16, 24, 32)
NEAR_DUPLICATE_ROW_CAP = 5000     # --max-near-dupe-rows in the graph builder


# ── inputs ─────────────────────────────────────────────────────────────────
def read_analyze(img_path: Path) -> np.ndarray:
    """Analyze 7.5 volume as (x, y, z) float32, without an imaging dependency."""
    hdr = img_path.with_suffix(".hdr").read_bytes()
    endian = "<" if struct.unpack("<i", hdr[:4])[0] == 348 else ">"
    dims = struct.unpack(endian + "8h", hdr[40:56])
    code = struct.unpack(endian + "h", hdr[70:72])[0]
    offset = int(struct.unpack(endian + "f", hdr[108:112])[0])
    dtype = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8"}[code]
    n = dims[1] * dims[2] * dims[3]
    data = np.fromfile(img_path, dtype=endian + dtype, count=n, offset=offset)
    return data.reshape((dims[1], dims[2], dims[3]), order="F").astype(np.float32)


def signature(img2d: np.ndarray, f: int = DOWNSAMPLE) -> np.ndarray:
    h, w = (img2d.shape[0] // f) * f, (img2d.shape[1] // f) * f
    s = img2d[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3)).ravel()
    s = s - s.mean()
    n = np.linalg.norm(s)
    return s / n if n > 0 else s


def read_cdr(metadata_dir: Path) -> dict[str, str]:
    """ID -> CDR from the OASIS-1 cross-sectional spreadsheet (xlsx)."""
    path = next(p for p in sorted(metadata_dir.glob("oasis_cross-sectional-*.xlsx"))
                if "reliability" not in p.name)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    table = []
    for row in sheet.find("m:sheetData", ns).findall("m:row", ns):
        cells = {}
        for c in row.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", ns)
            val = None if v is None else (shared[int(v.text)] if c.get("t") == "s" else v.text)
            cells[col] = val
        table.append(cells)
    header = table[0]
    col = {name: letter for letter, name in header.items()}
    return {r.get(col["ID"]): (r.get(col["CDR"]) or "") for r in table[1:] if r.get(col["ID"])}


# ── 1. content match ───────────────────────────────────────────────────────
def match_images(kaggle_root: Path, volumes_dir: Path):
    vols = sorted(volumes_dir.rglob("*_t88_masked_gfc.img"))
    keys, bank = [], []
    for v in vols:
        vol = read_analyze(v)
        session = v.name.split("_mpr")[0]
        for z in Z_RANGE:
            keys.append((session, z))
            bank.append(signature(vol[:, :, z].T))    # axial slice, transposed to 208x176
    bank = np.stack(bank).astype(np.float32)
    subject_of_key = np.array([k[0].rsplit("_MR", 1)[0] for k in keys])
    images = sorted(kaggle_root.rglob("*.jpg"))
    out = []
    for start in range(0, len(images), 400):
        chunk = images[start:start + 400]
        sig = np.stack([signature(np.asarray(Image.open(p), np.float32)) for p in chunk])
        corr = sig.astype(np.float32) @ bank.T
        for i, path in enumerate(chunk):
            j = int(corr[i].argmax())
            subject = subject_of_key[j]
            out.append({
                "image": str(path.relative_to(ROOT)),
                "kaggle_class": path.parent.name,
                "oasis_session": keys[j][0],
                "oasis_subject": subject,
                "z": keys[j][1],
                "corr": float(corr[i, j]),
                "best_other_participant": float(corr[i][subject_of_key != subject].max()),
            })
    return out, len(vols)


# ── 7. detector calibration ────────────────────────────────────────────────
def pair_distance_histograms(hashes, participant, z, session):
    """Hamming-distance histograms for same-participant, adjacent-slice and
    different-participant pairs, over all n(n-1)/2 pairs."""
    as_bytes = np.array([list(h.to_bytes(32, "little")) for h in hashes], dtype=np.uint8)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    same_h = np.zeros(257, np.int64); diff_h = np.zeros(257, np.int64); adj_h = np.zeros(257, np.int64)
    for i in range(len(hashes) - 1):
        d = popcount[as_bytes[i + 1:] ^ as_bytes[i]].sum(axis=1)
        same = participant[i + 1:] == participant[i]
        adjacent = same & (session[i + 1:] == session[i]) & (np.abs(z[i + 1:] - z[i]) == 1)
        same_h += np.bincount(d[same], minlength=257)
        diff_h += np.bincount(d[~same], minlength=257)
        adj_h += np.bincount(d[adjacent], minlength=257)
    return same_h, diff_h, adj_h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=Path, default=v)
    ap.add_argument("--reuse-mapping", action="store_true",
                    help="Recompute the audit from an existing ground-truth CSV "
                         "instead of matching every image against the volumes again.")
    ap.add_argument("--n-volumes-searched", type=int, default=436,
                    help="Recorded with --reuse-mapping, where no volume is opened.")
    args = ap.parse_args()

    if args.reuse_mapping and args.ground_truth.is_file():
        with args.ground_truth.open(encoding="utf-8") as fh:
            rows = [{**r, "corr": float(r["corr"]),
                     "best_other_participant": float(r["best_other_participant"]),
                     "z": int(r["z"])} for r in csv.DictReader(fh)]
        n_volumes = int(args.n_volumes_searched)
        print(f"reusing the recovered mapping for {len(rows)} images")
    else:
        rows, n_volumes = match_images(args.kaggle_root, args.oasis_volumes)
    cdr = read_cdr(args.oasis_metadata)
    for r in rows:
        r["cdr"] = cdr.get(r["oasis_session"], "")
    unresolved = [r["oasis_session"] for r in rows if not r["cdr"]]
    if unresolved:
        raise SystemExit(f"{len(unresolved)} images have no CDR, e.g. {unresolved[0]}; "
                         "the clinical spreadsheet did not resolve")
    args.ground_truth.parent.mkdir(parents=True, exist_ok=True)
    with args.ground_truth.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, "corr": f"{r['corr']:.5f}", "best_other_participant": f"{r['best_other_participant']:.5f}"} for r in rows)
    truth = {r["image"]: r for r in rows}
    n_unambiguous = int(((np.array([r["corr"] for r in rows]) >= 0.999)
                         & (np.array([r["best_other_participant"] for r in rows]) < 0.99)).sum())
    if n_unambiguous != len(rows):
        raise SystemExit(f"{len(rows) - n_unambiguous} of {len(rows)} images do not meet the "
                         "unambiguous-match criterion (self >= 0.999, next best < 0.99)")
    self_corr = np.array([r["corr"] for r in rows])
    other_corr = np.array([r["best_other_participant"] for r in rows])
    per_participant = Counter(r["oasis_subject"] for r in rows)
    zs = [r["z"] for r in rows]

    # 2. labels
    expected = {"NonDemented": 0.0, "VeryMildDemented": 0.5, "MildDemented": 1.0, "ModerateDemented": 2.0}
    by_participant: dict[str, list] = defaultdict(list)
    for r in rows:
        by_participant[r["oasis_subject"]].append(r)
    # Every image of a participant must sit in one folder, or "the folder label
    # is the participant's CDR" would be a statement about representatives.
    split_across_folders = {p: sorted({r["kaggle_class"] for r in rs})
                            for p, rs in by_participant.items() if len({r["kaggle_class"] for r in rs}) > 1}
    images_agreeing = sum(1 for r in rows if expected[r["kaggle_class"]] == float(r["cdr"]))
    agree = sum(1 for rs in by_participant.values()
                if all(expected[r["kaggle_class"]] == float(r["cdr"]) for r in rs))
    label_table = Counter((rs[0]["kaggle_class"], rs[0]["cdr"]) for rs in by_participant.values())

    # 3. Tier-2 overlap
    tier2 = {r["subject_id"].rsplit("_MR", 1)[0] for r in csv.DictReader(args.tier2_split.open())}

    # 4-5. filename identity and the frozen split
    split = list(csv.DictReader(args.split.open()))
    for r in split:
        t = truth[r["relative_path"]]
        r["true"], r["z"], r["session"] = t["oasis_subject"], int(t["z"]), t["oasis_session"]
    fid_true, true_fid, parts = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in split:
        fid_true[r["subject_id"]].add(r["true"])
        true_fid[r["true"]].add(r["subject_id"])
        parts[r["true"]].add(r["split"])
    train = {r["true"] for r in split if r["split"] == "train"}
    test = [r for r in split if r["split"] == "test"]
    test_participants = {r["true"] for r in test}

    # 6. near-duplicate candidates
    by_id = {r["image_id"]: r for r in split}
    pairs = list(csv.DictReader(args.near_dupes.open()))
    if len(pairs) >= NEAR_DUPLICATE_ROW_CAP:
        raise SystemExit(f"{args.near_dupes.name} holds {len(pairs)} rows, at or above the writer's "
                         f"cap of {NEAR_DUPLICATE_ROW_CAP}: the candidate list is truncated, so "
                         "every share computed from it would be over an arbitrary prefix")
    same_pairs = [p for p in pairs if by_id[p["image_id_a"]]["true"] == by_id[p["image_id_b"]]["true"]]

    def resolution(p):
        a = by_id[p["image_id_a"]]["subject_parse_status"] != "parsed_parentheses"
        b = by_id[p["image_id_b"]]["subject_parse_status"] != "parsed_parentheses"
        return "neither_resolved" if a and b else ("one_resolved" if a or b else "both_resolved")

    cross = [p for p in pairs if by_id[p["image_id_a"]]["split"] != by_id[p["image_id_b"]]["split"]]
    cross_tab = defaultdict(Counter)
    for p in cross:
        same = by_id[p["image_id_a"]]["true"] == by_id[p["image_id_b"]]["true"]
        cross_tab[resolution(p)]["same_participant" if same else "different_participants"] += 1

    # 7. calibration
    paths = [ROOT / r["relative_path"] for r in split]
    hashes = [dhash(p) for p in paths]
    assert hamming(hashes[0], hashes[1]) == bin(hashes[0] ^ hashes[1]).count("1")
    same_h, diff_h, adj_h = pair_distance_histograms(
        hashes, np.array([r["true"] for r in split]), np.array([r["z"] for r in split]),
        np.array([r["session"] for r in split]))
    cs, cd, ca = np.cumsum(same_h), np.cumsum(diff_h), np.cumsum(adj_h)
    calibration = []
    for t in THRESHOLDS:
        flagged = int(cs[t] + cd[t])
        calibration.append({
            "max_hamming_distance": t, "pairs_flagged": flagged,
            "same_participant_pair_coverage": round(float(cs[t] / same_h.sum()), 4),
            "recall_adjacent_slices": round(float(ca[t] / adj_h.sum()), 4),
            "false_positive_rate": float(cd[t] / diff_h.sum()),
            "precision": round(float(cs[t] / flagged), 4) if flagged else None,
        })

    summary = {
        "source_cohort": "OASIS-1 cross-sectional (t88_masked_gfc axial slices)",
        "mapping": {
            "n_images": len(rows), "n_oasis_volumes_searched": n_volumes,
            "min_self_correlation": round(float(self_corr.min()), 4),
            "max_other_participant_correlation": round(float(other_corr.max()), 4),
            "n_unambiguous": n_unambiguous,
            "n_participants": len(per_participant),
            "images_per_participant": sorted(set(per_participant.values())),
            "axial_slice_range": [min(zs), max(zs)],
            "reliability_rescan_sessions_used": len({r["oasis_session"] for r in rows
                                                     if r["oasis_session"].endswith("MR2")}),
            "participants_split_across_folders": len(split_across_folders),
            "n_images_agreeing_with_cdr": images_agreeing,
        },
        "labels_vs_cdr": {
            "participants_with_expected_cdr": agree, "participants": len(by_participant),
            "table": {f"{k[0]}|CDR={k[1]}": v for k, v in sorted(label_table.items())},
        },
        "tier2_overlap": {"tier1_participants": len(per_participant), "tier2_participants": len(tier2),
                          "shared": len(set(per_participant) & tier2)},
        "filename_identity": {
            "filename_subject_ids": len(fid_true),
            "ids_that_are_one_participant": sum(1 for s in fid_true.values() if len(s) == 1),
            "ids_merging_participants": sum(1 for s in fid_true.values() if len(s) > 1),
            "participants_per_filename_id": dict(sorted(Counter(len(s) for s in fid_true.values()).items())),
            "filename_ids_per_participant": dict(sorted(Counter(len(s) for s in true_fid.values()).items())),
            "unparsed_images": sum(1 for r in split if r["subject_parse_status"] != "parsed_parentheses"),
        },
        "frozen_split_true_leakage": {
            "participants_straddling_partitions": sum(1 for s in parts.values() if len(s) > 1),
            "test_images": len(test),
            "test_images_whose_participant_is_in_train": sum(1 for r in test if r["true"] in train),
            "test_participants": len(test_participants),
            "test_participants_also_in_train": len(test_participants & train),
        },
        "near_duplicates": {
            "candidates": len(pairs),
            "same_participant": len(same_pairs),
            "slice_gap_of_same_participant_pairs": dict(sorted(Counter(abs(by_id[p["image_id_a"]]["z"] - by_id[p["image_id_b"]]["z"]) for p in same_pairs).items())),
            "cross_partition": len(cross),
            "cross_partition_by_filename_resolution": {k: dict(v) for k, v in sorted(cross_tab.items())},
        },
        "dhash_calibration": {
            "hash_bits": 256, "pairs": int(same_h.sum() + diff_h.sum()),
            "same_participant_pairs": int(same_h.sum()), "adjacent_slice_pairs": int(adj_h.sum()),
            "median_distance": {"same_participant": int(np.searchsorted(cs, same_h.sum() / 2)),
                                "adjacent_slices": int(np.searchsorted(ca, adj_h.sum() / 2)),
                                "different_participants": int(np.searchsorted(cd, diff_h.sum() / 2))},
            "by_threshold": calibration,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "dhash_calibration"}, indent=1))
    print("calibration:", json.dumps(calibration))
    print(f"wrote {display_path(args.ground_truth)} and {display_path(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
