#!/usr/bin/env python3
"""Measure what the injection arm actually trains on and evaluates.

Why this exists
---------------
The manuscript described this arm as injecting contamination into a *fixed*
training set. That was wrong, and the error was load-bearing: it turned a
descriptive slope into a causal price per unit of leakage.

Two things change with the overlap setting, and neither is visible in the
injector's own audit file:

1. ``inject_leakage_split.py`` moves a donor scan from train to test and
   removes it from training. The donor participant stays, because donors need
   two or more scans, so the identity leakage is genuine. The training set is
   not fixed.

2. Substitution is stratified on ``component_label``, while the binary trainer
   filters on visit-level ``diagnosis_group`` and requires it to equal
   ``component_label``. So the injector's "test size and class balance
   preserved" holds at manifest level and fails at evaluated level: the
   evaluated test set both grows and shifts its class mix.

The numbers here are therefore the honest description of the dose axis, and
the manuscript is held to them by ``verify_paper_numbers.py``. The split
manifests this reads are Tier-3 and stay local; this summary is counts and
shares only, so it is released.

Usage
-----
    python3 scripts/analyze_dose_response_composition.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits" / "adni_dose_response"
OUT = ROOT / "reports" / "tables" / "adni" / "adni_dose_response_composition.json"

# The nominal overlap settings, as the dose-response matrix was run.
SETTINGS = ["0.0", "0.25", "0.50", "0.75", "1.0"]


def evaluated(rows: list[dict], split: str) -> list[dict]:
    """The rows the trainer keeps: visit-level CN/AD agreeing with the component."""
    return [r for r in rows
            if r["split"] == split
            and r["diagnosis_group"] in ("CN", "AD")
            and r["diagnosis_group"] == r["component_label"]]


def main() -> int:
    if not SPLITS.is_dir():
        print(f"{SPLITS} not found: the injected manifests are built locally "
              f"by scripts/inject_leakage_split.py and are not redistributed.")
        return 1

    per_setting = {}
    for setting in SETTINGS:
        train, test, ad_share = [], [], []
        for path in sorted(SPLITS.glob(f"seed*_overlap{setting}.csv")):
            rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
            tst = evaluated(rows, "test")
            if not tst:
                continue
            train.append(len(evaluated(rows, "train")))
            test.append(len(tst))
            ad_share.append(sum(1 for r in tst if r["diagnosis_group"] == "AD") / len(tst))
        if not test:
            continue
        per_setting[setting] = {
            "n_seeds": len(test),
            "evaluated_train_mean": round(st.mean(train), 1),
            "evaluated_test_mean": round(st.mean(test), 1),
            "evaluated_test_ad_share_mean": round(st.mean(ad_share), 4),
        }

    if not per_setting:
        print("no injected manifests matched; nothing written")
        return 1

    lo, hi = per_setting[SETTINGS[0]], per_setting[SETTINGS[-1]]
    removed = 1 - hi["evaluated_train_mean"] / lo["evaluated_train_mean"]
    payload = {
        "description": "Evaluated train and test composition across the injection settings",
        "trainer_filter": "diagnosis_group in (CN, AD) and diagnosis_group == component_label",
        "by_setting": per_setting,
        "endpoints": {
            "evaluated_train_first": lo["evaluated_train_mean"],
            "evaluated_train_last": hi["evaluated_train_mean"],
            "evaluated_train_share_removed": round(removed, 4),
            "evaluated_test_first": lo["evaluated_test_mean"],
            "evaluated_test_last": hi["evaluated_test_mean"],
            "evaluated_test_ad_share_first": lo["evaluated_test_ad_share_mean"],
            "evaluated_test_ad_share_last": hi["evaluated_test_ad_share_mean"],
        },
        "note": ("The training set is not held fixed across the dose axis, and the "
                 "evaluated test set grows and shifts its class mix. The fitted "
                 "slope is descriptive of this substitution procedure."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  evaluated training rows {lo['evaluated_train_mean']:.0f} -> "
          f"{hi['evaluated_train_mean']:.0f} ({100 * removed:.1f}% removed)")
    print(f"  evaluated test rows     {lo['evaluated_test_mean']:.0f} -> "
          f"{hi['evaluated_test_mean']:.0f}")
    print(f"  evaluated test AD share {100 * lo['evaluated_test_ad_share_mean']:.1f}% -> "
          f"{100 * hi['evaluated_test_ad_share_mean']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
