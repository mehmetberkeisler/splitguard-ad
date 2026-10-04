#!/usr/bin/env python3
"""Measure what the leakage graph buys once subject identifiers start disappearing.

The claim this tests
--------------------
Section 4.2 of the manuscript states that the non-subject edge rules are
"redundant on these cohorts rather than inert, and would become load-bearing
only where subject identity is absent, corrupted, or inferred wrongly." That
is a hypothesis, and until now it was untested: on all three cohorts the
components coincide exactly with a subject grouping, so the component layer's
marginal contribution over subject-wise splitting is indistinguishable from
zero and the framework's distinguishing layer has no positive demonstration.

The regime where it should matter is not hypothetical. Redistributed public
benchmarks routinely lose the patient index: on the Tier-1 cohort 128 of 6,400
images (2.0%) carry no recoverable subject identifier, and under subject-wise
splitting each of them is a singleton free to land opposite a near-identical
image of the same patient. What has been missing is a way to measure the
consequence, because the cohorts with complete metadata are precisely the ones
where the failure cannot arise.

This script creates the regime deliberately. Starting from ADNI1, where the
subject identifiers are complete and authoritative, it deletes a controlled
fraction of them and asks what each protocol does with the wreckage:

    Protocol B (subject-only)  -- every scan whose identifier was deleted
                                  becomes its own group and may be placed in
                                  any partition.
    Protocol C (component-safe) -- the leakage graph is rebuilt from whatever
                                  identifiers survive. session_id and
                                  series_uid are globally unique and do not
                                  take subject_id as an input, so they still
                                  bind scans that same_subject can no longer
                                  reach.

The design mirrors the leakage dose-response experiment: hold everything fixed
except one controlled quantity, sweep it, and read off the curve. Here the
swept quantity is provenance loss rather than test-set overlap.

Two things are reported per level. The AUROC difference is the headline, but
the more direct measurement is ``residual_subject_leakage``: the number of
*true* subjects (recovered from the untouched ground-truth column) whose scans
end up straddling partitions. That counts the leakage each protocol actually
admits, without routing the question through a model.

Deletion is per scan, not per subject, because that is the observed failure
mode: filenames lose their index one file at a time, so a patient's record is
typically partly recoverable rather than wholly anonymous.

Expected result, stated in advance so it cannot be rationalised afterwards:
the graph should recover *part* of the protection, not all of it. same_subject
joins far more pairs than the surviving rules do, so components will fragment
as deletion rises. A partial recovery curve is the honest outcome and is worth
reporting either way -- including if it turns out to be flat, which would mean
the extra edge rules earn nothing even in the regime built to favour them.

Usage
-----
    python3 scripts/run_provenance_degradation.py --device cuda
    python3 scripts/run_provenance_degradation.py --levels 0.0 0.5 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splitguard_ad.metrics import TRUE_SUBJECT, residual_subject_leakage  # noqa: E402

from build_adni_leakage_graph import build_graph  # noqa: E402
from make_adni_splitguard_split import (  # noqa: E402
    bucket_field_strength,
    build_assignments,
)
from train_adni_baseline import read_split_rows, train_and_eval  # noqa: E402

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "adni"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "adni_provenance_degradation"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "reports" / "tables" / "adni" / "adni_provenance_degradation.json"
)



def degrade(
    rows: list[dict[str, str]], fraction: float, rng: random.Random
) -> list[dict[str, str]]:
    """Delete ``subject_id`` on a random ``fraction`` of scans.

    The original value is preserved under ``TRUE_SUBJECT`` so that residual
    leakage can be scored against ground truth after the split is made. The
    deleted value is set to ``"unknown"``, which ``build_graph`` already skips
    when forming ``same_subject`` edges.
    """
    out = []
    for row in rows:
        new_row = dict(row)
        new_row[TRUE_SUBJECT] = row["subject_id"]
        if rng.random() < fraction:
            new_row["subject_id"] = "unknown"
        out.append(new_row)
    return out


def subject_only_split(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Group by surviving subject_id, then bin-pack with the *same* assigner
    Protocol C uses.

    Matching the assigner matters more than it looks. An earlier version of
    this comparison shuffled subject groups and cut at 70/15/15 by group
    count, while Protocol C ran a class-stratified greedy bin-pack over
    components at 70/15/15 by image count. That confounds two interventions:
    the grouping (subject vs leakage-graph component) and the assignment
    algorithm. Measured on this cohort the confound was worth roughly half the
    apparent effect --- at 10% identifier loss the graph appeared to prevent
    78% of residual leakage under the mismatched comparison and prevents 49%
    under this one. Only the grouping is the claim, so only the grouping may
    differ.

    Every scan whose identifier was deleted becomes its own group, which is
    what a ``GroupKFold``-style splitter does with a missing group key.
    Lumping all unknowns into a single group would be a different and
    accidentally safer protocol.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        subject = row["subject_id"]
        key = subject if subject != "unknown" else f"__singleton__{row['image_id']}"
        groups[key].append(row)

    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for group_id, group_rows in groups.items():
        labels = {r.get("diagnosis_group", "unknown") for r in group_rows} - {"unknown"}
        label = next(iter(labels)) if len(labels) == 1 else "mixed_label"
        by_label[label].append({
            "component_id": group_id,
            "n_images": len(group_rows),
            "bucket": bucket_field_strength(group_rows),
        })

    assignments = build_assignments(by_label, seed)
    return {
        row["image_id"]: assignments[key]
        for key, group_rows in groups.items()
        for row in group_rows
    }


def component_safe_split(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Rebuild the leakage graph on the degraded manifest and bin-pack components."""
    uf, _reasons, _edges = build_graph(rows)

    members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        members[uf.find(row["image_id"])].append(row)

    components_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for component_id, component_rows in members.items():
        labels = {r.get("diagnosis_group", "unknown") for r in component_rows} - {"unknown"}
        label = next(iter(labels)) if len(labels) == 1 else "mixed_label"
        components_by_label[label].append(
            {
                "component_id": component_id,
                "n_images": len(component_rows),
                "bucket": bucket_field_strength(component_rows),
            }
        )

    assignments = build_assignments(components_by_label, seed)
    return {
        row["image_id"]: assignments[uf.find(row["image_id"])]
        for row in rows
    }




def to_splits(
    rows: list[dict[str, str]], assignment: dict[str, str]
) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        out[assignment[row["image_id"]]].append({**row, "split": assignment[row["image_id"]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.0, 0.10, 0.25, 0.50, 1.0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--no-train", action="store_true",
                    help="Compute splits and residual leakage only; skip training. "
                         "Use to sanity-check the degradation before spending GPU time.")
    args = ap.parse_args()

    protocols = {
        "subject_only": subject_only_split,
        "component_safe": component_safe_split,
    }
    records: list[dict[str, object]] = []

    for seed in args.seeds:
        split_path = args.split_dir / f"adni_splitguard_seed{seed}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        base_rows = read_split_rows(split_path)

        for level in args.levels:
            rows = degrade(base_rows, level, random.Random(20_000 + seed))
            n_unknown = sum(1 for r in rows if r["subject_id"] == "unknown")

            for name, splitter in protocols.items():
                assignment = splitter(rows, seed)
                leakage = residual_subject_leakage(rows, assignment)
                record: dict[str, object] = {
                    "seed": seed,
                    "deletion_fraction": level,
                    "protocol": name,
                    "n_scans_without_subject_id": n_unknown,
                    **leakage,
                }

                if not args.no_train:
                    splits = to_splits(rows, assignment)
                    if not splits["train"] or not splits["test"]:
                        print(f"  skip seed={seed} level={level} {name}: empty partition")
                        continue
                    cell = args.output_root / f"seed{seed}_del{int(level*100)}" / name
                    finished = cell / "metrics.json"
                    metrics = (json.loads(finished.read_text()) if finished.exists()
                               else train_and_eval(
                        splits,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        image_size=args.image_size,
                        pretrained=True,
                        label=name,
                        output_dir=cell,
                        device_str=args.device,
                        arch=args.arch,
                    ))
                    record["test_auroc"] = metrics.get("test_metrics", {}).get("auroc")

                records.append(record)
                print(f"  seed={seed} del={level:.2f} {name:15s} "
                      f"straddling_subjects={leakage['n_subjects_straddling_partitions']:3d}"
                      + (f"  auroc={record.get('test_auroc')}" if "test_auroc" in record else ""))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deletion_unit": "per_scan",
        "levels": args.levels,
        "seeds": args.seeds,
        "trained": not args.no_train,
        "records": records,
        "interpretation": (
            "n_subjects_straddling_partitions is the direct measurement: how many "
            "true patients have scans on both sides of a partition boundary. Under "
            "subject_only it should rise with the deletion fraction, because every "
            "scan whose identifier was removed becomes a free-floating singleton. "
            "Under component_safe it should rise more slowly, because session_id "
            "and series_uid edges do not take subject_id as an input and continue "
            "to bind scans that same_subject can no longer reach. The difference "
            "between the two curves is what the leakage-graph layer buys, measured "
            "rather than argued."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
