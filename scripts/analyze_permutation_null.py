#!/usr/bin/env python3
"""Fold the paired contrasts and the training curves into the permutation-null artefact.

``run_adni_permutation_null.py`` reports each protocol's permuted-label AUROC
as a mean and SD over seeds. Two further quantities are in the run tree and
belong with it.

The first is the paired contrast. Every seed trains all three protocols on the
same permutation, so Protocol A minus Protocol C is a within-seed difference,
and summarising it unpaired discards the permutation-to-permutation variance
that cancels. The same applies to B minus C, which matters more than it looks:
both arms have nothing to learn, so whatever that contrast turns out to be is a
direct measurement of the spurious protocol-to-protocol difference this design
admits at five seeds.

The second is the final-epoch training loss beside the final-epoch validation
AUROC. A model that fits the permuted training set and still scores at chance
on held-out patients has memorised something that does not transfer; one that
fits it equally well and scores far above chance has memorised something that
does. Printing both makes the mechanism legible without a figure.

Usage
-----
    python3 scripts/analyze_permutation_null.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "runs" / "adni_permutation_null"
DEFAULT_ARTEFACT = ROOT / "reports" / "tables" / "adni" / "adni_permutation_null.json"
PROTOCOLS = ("random", "subject_only", "component_safe")
T95_DF4 = 2.776                      # five seeds, four degrees of freedom


def interval(values: list[float]) -> dict:
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    half = T95_DF4 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": round(mean, 4), "sd": round(sd, 4),
            "ci95_lo": round(mean - half, 4), "ci95_hi": round(mean + half, 4),
            "n_seeds": len(values)}


def label_agreement(runs_root: Path, split_dir: Path, n_seeds: int) -> dict:
    """How often each protocol's scored label equals the true diagnosis.

    The control's whole claim rests on the labels having been permuted. If a
    wiring change ever let the true labels through, the random arm would still
    produce a high AUROC and the result would read as a stronger version of the
    same finding rather than as a broken one. Agreement near 50% is the
    evidence that nothing about the disease survived the permutation.
    """
    truth: dict[str, int] = {}
    for seed in range(n_seeds):
        split = split_dir / f"adni_splitguard_seed{seed}.csv"
        if not split.is_file():
            return {}
        for row in csv.DictReader(split.open(encoding="utf-8")):
            if row.get("diagnosis_group") in ("CN", "AD"):
                truth[row["image_id"]] = 1 if row["diagnosis_group"] == "AD" else 0

    out = {}
    for protocol in PROTOCOLS:
        agree = total = 0
        for seed in range(n_seeds):
            pred = (runs_root / f"permutation_null_seed{seed}" / protocol /
                    "test_predictions.csv")
            if not pred.is_file():
                continue
            rows = list(csv.DictReader(pred.open(encoding="utf-8")))
            key = "image_id" if rows and "image_id" in rows[0] else "image_uid"
            for row in rows:
                if row[key] in truth:
                    total += 1
                    agree += int(float(row["y_true"])) == truth[row[key]]
        if total:
            out[protocol] = {"n_test_rows": total,
                             "agreement_with_true_diagnosis": round(100 * agree / total, 1)}
    return out


def final_epoch(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return (float(rows[-1]["loss"]), float(rows[-1]["val_auroc"])) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--artefact", type=Path, default=DEFAULT_ARTEFACT)
    ap.add_argument("--split-dir", type=Path,
                    default=ROOT / "data" / "splits" / "adni")
    args = ap.parse_args()

    payload = json.loads(args.artefact.read_text(encoding="utf-8"))
    per_seed = {p: payload["by_protocol"][p]["per_seed_auroc"] for p in PROTOCOLS}
    n = len(per_seed["random"])

    payload["paired_contrasts"] = {
        "random_minus_component_safe": interval(
            [per_seed["random"][i] - per_seed["component_safe"][i] for i in range(n)]),
        "subject_only_minus_component_safe": interval(
            [per_seed["subject_only"][i] - per_seed["component_safe"][i] for i in range(n)]),
    }
    payload["per_protocol_interval"] = {p: interval(per_seed[p]) for p in PROTOCOLS}

    curves: dict[str, dict] = {}
    for protocol in PROTOCOLS:
        losses, vals = [], []
        for seed in range(n):
            got = final_epoch(args.runs_root / f"permutation_null_seed{seed}" /
                              protocol / "training_log.csv")
            if got:
                losses.append(got[0]); vals.append(got[1])
        if losses:
            curves[protocol] = {
                "final_train_loss_mean": round(st.mean(losses), 3),
                "final_train_loss_sd": round(st.stdev(losses), 3) if len(losses) > 1 else 0.0,
                "final_val_auroc_mean": round(st.mean(vals), 3),
                "final_val_auroc_sd": round(st.stdev(vals), 3) if len(vals) > 1 else 0.0,
            }
    payload["training_curves"] = curves

    # Checkpoint selection, which is where the honest arms' sub-chance test
    # values come from. The trainer keeps the best-validation-AUROC epoch. With
    # permuted labels the honest arms' validation partitions carry no signal,
    # so the maximum over epochs is selection on noise, and the smaller the
    # partition the further it overshoots. This is the mechanism behind the
    # contrast between the two honest protocols, and it is worth reporting:
    # a validation partition holding ~22 participants cannot select a
    # checkpoint reliably, with permuted labels or real ones.
    selection = {}
    for protocol in PROTOCOLS:
        gaps, parts = [], []
        for seed in range(n):
            run = args.runs_root / f"permutation_null_seed{seed}" / protocol
            metrics = run / "metrics.json"
            if not metrics.is_file():
                continue
            m = json.loads(metrics.read_text())
            if m.get("best_val_auroc") is not None:
                gaps.append(m["best_val_auroc"] - m["test_metrics"]["auroc"])
            val = run / "val_predictions.csv"
            if val.is_file():
                rows = list(csv.DictReader(val.open(encoding="utf-8")))
                key = "image_id" if rows and "image_id" in rows[0] else None
                if key:
                    parts.append(len({r[key].rsplit("_I", 1)[0] for r in rows}))
        if gaps:
            selection[protocol] = {
                "selection_optimism_mean": round(st.mean(gaps), 3),
                "val_participants_mean": round(st.mean(parts)) if parts else None,
            }
    if selection:
        payload["checkpoint_selection"] = selection
    agreement = label_agreement(args.runs_root, args.split_dir, n)
    if agreement:
        payload["label_agreement"] = agreement
        payload["label_agreement_note"] = (
            "Percentage of scored test rows whose permuted label happens to "
            "equal the participant's true diagnosis. Near 50 means the "
            "permutation destroyed the disease association, which is the "
            "precondition for reading the random arm's AUROC as patient recall."
        )
    payload["curve_note"] = (
        "All three protocols drive the permuted training loss to a comparably "
        "low value, so they memorise the training set to a comparable degree. "
        "Only the random split's memorisation transfers to its validation "
        "partition, because that partition holds the same patients."
    )
    args.artefact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    header = ("protocol".ljust(16) + "permuted AUROC".rjust(16) + "95% CI".rjust(20)
              + "final loss".rjust(12) + "final val".rjust(11))
    print(header)
    for protocol in PROTOCOLS:
        iv = payload["per_protocol_interval"][protocol]
        cv = curves.get(protocol, {})
        ci = f"[{iv['ci95_lo']:.3f}, {iv['ci95_hi']:.3f}]"
        print(protocol.ljust(16) + f"{iv['mean']:.4f}".rjust(16) + ci.rjust(20)
              + f"{cv.get('final_train_loss_mean', float('nan')):.3f}".rjust(12)
              + f"{cv.get('final_val_auroc_mean', float('nan')):.3f}".rjust(11))
    for protocol, block in (payload.get("label_agreement") or {}).items():
        print(f"  {protocol:<16} scored label equals the true diagnosis in "
              f"{block['agreement_with_true_diagnosis']}% of "
              f"{block['n_test_rows']} test rows")
    for name, block in payload["paired_contrasts"].items():
        print(f"paired {name}: {block['mean']:+.4f} "
              f"[{block['ci95_lo']:+.4f}, {block['ci95_hi']:+.4f}]")
    print(f"\nUpdated {args.artefact.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
