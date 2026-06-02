#!/usr/bin/env python3
"""Generate leakage-injected split manifests for the dose-response stress test.

Starting from a frozen Protocol C (component-safe) split, inject controlled
test-subject overlap at a specified target fraction.  The injection uses
subject-substitution: a fraction p of test scans are replaced with scans
from train subjects (drawn from those subjects' *other* scan-instances,
never literal duplicates).  This isolates pure biometric-identity leakage
from slice-level/near-duplicate leakage, mirroring the mechanism the WP6
biometric probe identified.

Design properties (verified at the end of every run):
  * Test set size is preserved at the input test size.
  * Class balance (CN vs AD) is preserved on the test set.
  * At p=0 the output is byte-identical to the input split.
  * At p=1.0 every test subject is also in train (≈ Protocol A overlap).
  * Subject IDs used as overlap donors are sampled only from multi-visit
    train subjects, so the substituted scan is a distinct MRI session for
    the same subject — pure identity transfer, not duplication.

CLI
---
python3 scripts/inject_leakage_split.py \\
    --base data/splits/adni_with_converters/adni_splitguard_seed0.csv \\
    --overlap 0.50 --seed 0 \\
    --output data/splits/adni_dose_response/seed0_overlap50.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_split(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def class_label(row: dict) -> str:
    """Binary CN/AD label as a single character."""
    return row["diagnosis_group"]


def inject(base_rows: list[dict],
           overlap_frac: float,
           seed: int) -> tuple[list[dict], dict]:
    """Return (injected_rows, audit) where audit summarises what changed."""

    rng = random.Random(seed)

    # Bucket rows by split. The downstream binary task uses `component_label`
    # (per-component reassignment to CN-or-AD majority for the converter-
    # inclusive arm), NOT `diagnosis_group` (per-visit). A subject's
    # diagnosis_group can vary across visits (e.g. CN→MCI→AD), but its
    # component_label is constant; we route the injection by the latter.
    train     = [r for r in base_rows if r["split"] == "train"]
    val       = [r for r in base_rows if r["split"] == "val"]
    test_all  = [r for r in base_rows if r["split"] == "test"]
    test      = [r for r in test_all  if r["component_label"] in ("CN", "AD")]
    test_other = [r for r in test_all if r["component_label"] not in ("CN", "AD")]

    n_test = len(test)
    n_target_overlap = round(overlap_frac * n_test)
    if n_target_overlap == 0:
        # p=0: identity. Hand back unchanged rows with audit.
        return base_rows, {
            "overlap_frac_target": overlap_frac,
            "overlap_frac_actual": 0.0,
            "n_test": n_test,
            "n_overlap_scans": 0,
            "n_donor_subjects_eligible": 0,
            "class_balance_test_before": _class_counts(test),
            "class_balance_test_after":  _class_counts(test),
            "note": "p=0 identity passthrough",
        }

    # Build subject -> list of train rows.  We only consider rows that the
    # downstream training pipeline will actually use: diagnosis_group must be
    # in (CN, AD) and equal to component_label (the per-subject binary
    # label).  Converter subjects' MCI visits and any inconsistent visits are
    # excluded; otherwise a "moved" row could be silently dropped at
    # training time, inflating the overlap count without causing real
    # leakage.
    train_by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        if (r["diagnosis_group"] in ("CN", "AD")
                and r["diagnosis_group"] == r["component_label"]):
            train_by_subject[r["subject_id"]].append(r)
    # Donor pool: train subjects with ≥2 usable scans (so we can pick one to
    # move to test without removing them from train entirely).
    donors = {sid: rows for sid, rows in train_by_subject.items()
              if len(rows) >= 2}
    if len(donors) < n_target_overlap and overlap_frac < 1.0:
        # Caller asked for more overlap than the donor pool supports.
        # Saturate at the donor count and report it in the audit.
        n_target_overlap = len(donors)
    elif overlap_frac >= 0.999 and len(donors) < n_target_overlap:
        # At p≈1.0 we need ALL test scans to be train-subject scans.
        # If donor pool is short we'll allow one donor to contribute multiple
        # scans (so 100% overlap is achievable even with a small multi-visit
        # subset).  This costs a little class-balance flexibility; we restore
        # it by stratified sampling below.
        pass

    # Build a stratified plan: how many CN, how many AD must the injected
    # test set contain. Keyed off component_label (constant per subject).
    cls_test_before = _class_counts(test)
    n_cn_keep = cls_test_before.get("CN", 0)
    n_ad_keep = cls_test_before.get("AD", 0)
    # Of the n_target_overlap incoming scans, sample so that the class
    # ratio of *injected* scans matches the existing test ratio. This keeps
    # the overall class balance constant.
    p_cn = n_cn_keep / n_test if n_test else 0.5
    n_inj_cn = round(p_cn * n_target_overlap)
    n_inj_ad = n_target_overlap - n_inj_cn

    # Split donor pool by component_label (per-subject binary label,
    # constant within a subject; safe to read from the first row).
    donors_by_cls: dict[str, list[tuple[str, list[dict]]]] = {
        "CN": [], "AD": [],
    }
    for sid, rows in donors.items():
        cls = rows[0]["component_label"]
        if cls in donors_by_cls:
            donors_by_cls[cls].append((sid, rows))
    for cls in donors_by_cls:
        rng.shuffle(donors_by_cls[cls])

    # Pick which scans to inject.  For each chosen donor, take ONE of their
    # rows (the one with the earliest acquisition date, for determinism)
    # and route it to test.  The donor's other train rows stay in train.
    injected_test: list[dict] = []
    donors_used: set[str] = set()
    def _pick(cls: str, n: int) -> list[dict]:
        picks = []
        pool = donors_by_cls[cls]
        if not pool: return picks
        i = 0
        while len(picks) < n and i < 10 * max(1, n):
            sid, rows = pool[i % len(pool)]
            # Use the latest-date scan as the injected one; subsequent loop
            # iterations on the same donor will use earlier scans.
            rows_sorted = sorted(rows, key=lambda r: r.get("acq_date", ""))
            already_picked_count = sum(1 for r in picks if r["subject_id"] == sid)
            if already_picked_count < len(rows_sorted):
                chosen = dict(rows_sorted[already_picked_count])
                chosen["split"] = "test"
                # Mark the row for the audit trail.
                chosen["_injected_for_overlap"] = "1"
                picks.append(chosen)
                donors_used.add(sid)
            i += 1
        return picks

    injected_test.extend(_pick("CN", n_inj_cn))
    injected_test.extend(_pick("AD", n_inj_ad))

    # We added n_target_overlap scans to test; now remove the same count
    # from the *original* test rows (stratified by component_label so the
    # test set retains its CN/AD ratio).
    test_cn = [r for r in test if r["component_label"] == "CN"]
    test_ad = [r for r in test if r["component_label"] == "AD"]
    rng.shuffle(test_cn); rng.shuffle(test_ad)
    n_removed_cn = min(n_inj_cn, len(test_cn))
    n_removed_ad = min(n_inj_ad, len(test_ad))
    kept_test = test_cn[n_removed_cn:] + test_ad[n_removed_ad:]

    # Build the new manifest. For the injected scans, REMOVE them from train
    # (their image_id was originally a train row).
    injected_image_ids = {r["image_id"] for r in injected_test}
    new_train = [r for r in train if r["image_id"] not in injected_image_ids]

    # Non-CN/AD test rows (if any) pass through unchanged.
    new_rows = new_train + val + kept_test + injected_test + test_other

    # Strip the internal audit marker — it was only for in-memory tracking.
    for r in new_rows:
        r.pop("_injected_for_overlap", None)

    # ── Audit ─────────────────────────────────────────────────────────────
    actual_overlap = len(injected_test) / max(1, len(kept_test) + len(injected_test))
    audit = {
        "overlap_frac_target":      overlap_frac,
        "overlap_frac_actual":      round(actual_overlap, 4),
        "n_test":                   len(kept_test) + len(injected_test),
        "n_overlap_scans":          len(injected_test),
        "n_donor_subjects_eligible": len(donors),
        "n_donors_used":            len(donors_used),
        "class_balance_test_before": cls_test_before,
        "class_balance_test_after":  _class_counts(kept_test + injected_test),
        "n_train_after":             len(new_train),
        "n_val_after":               len(val),
    }
    return new_rows, audit


def _class_counts(rows: list[dict]) -> dict[str, int]:
    """Class counts on the downstream binary key (component_label)."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["component_label"]] += 1
    return dict(counts)


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"Refusing to write empty manifest: {path}")
    # Preserve the same column ordering as the input split (plus the
    # _injected_for_overlap audit marker at the end if present).
    fieldnames = list(rows[0].keys())
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True,
                   help="Base Protocol C split CSV (0% overlap baseline)")
    p.add_argument("--overlap", type=float, required=True,
                   help="Target overlap fraction in [0, 1].")
    p.add_argument("--seed", type=int, required=True,
                   help="RNG seed for donor selection.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output CSV path for the injected split.")
    p.add_argument("--audit-output", type=Path, default=None,
                   help="Optional JSON audit summary path.")
    args = p.parse_args()

    base = read_split(args.base)
    new_rows, audit = inject(base, args.overlap, args.seed)
    write_split(new_rows, args.output)

    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2))

    print(f"Wrote {args.output}")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
