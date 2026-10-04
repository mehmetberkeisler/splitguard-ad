"""Metrics and leakage counters, moved verbatim from the analysis scripts.

``auroc`` is a rank-based implementation rather than a call into scikit-learn,
which matters twice: the verification gates then import nothing outside the
standard library, and the test suite checks it against an independently written
Mann-Whitney oracle, so a metric bug cannot hide behind a plausible number.

``residual_subject_leakage`` counts what a split actually admits. Scan share
and participant share are different quantities with different denominators and
must not be read as one, which the tests assert.
"""

from __future__ import annotations

from collections import defaultdict

__all__ = ["auroc", "residual_subject_leakage", "TRUE_SUBJECT"]


def auroc(pairs: list[tuple[int, float]]) -> float:
    """Rank-based AUROC over (label, score) pairs; nan if one class is absent."""
    pos = [s for t, s in pairs if t == 1]
    neg = [s for t, s in pairs if t == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][1] == pairs[order[i]][1]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    rank_sum = sum(r for r, (t, _) in zip(ranks, pairs, strict=True) if t == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


TRUE_SUBJECT = "_true_subject_id"


def residual_subject_leakage(
    rows: list[dict[str, str]], assignment: dict[str, str]
) -> dict[str, float]:
    """Score the leakage this assignment admits, against the untouched ground truth.

    Two quantities, and the distinction between them matters.

    ``n_subjects_straddling_partitions`` counts every true patient with scans
    on both sides of any boundary. It is the natural measure of how well a
    protocol held, and it is what the degradation curve is read from.

    ``test_train_subject_overlap`` is narrower: the fraction of test-partition
    patients who also appear in training.

    ``test_scan_contamination`` is the fraction of test *scans* whose patient
    appears in training. This is the quantity the leakage dose-response
    sweeps, which injects a share of test scans rather than a share of test
    patients, so it is the one to compose the two curves through. The patient
    share is always the larger of the two, because one contaminated patient
    can carry a single scan, and composing through it would overstate the
    optimism that provenance loss buys.
    """
    phases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        phases[row[TRUE_SUBJECT]].add(assignment[row["image_id"]])
    straddling = {s for s, p in phases.items() if len(p) > 1}
    leaked_images = sum(1 for r in rows if r[TRUE_SUBJECT] in straddling)

    train_subjects = {s for s, p in phases.items() if "train" in p}
    test_subjects = {s for s, p in phases.items() if "test" in p}
    overlap = test_subjects & train_subjects
    test_rows = [r for r in rows if assignment[r["image_id"]] == "test"]
    contaminated_scans = sum(1 for r in test_rows if r[TRUE_SUBJECT] in overlap)

    return {
        "n_true_subjects": len(phases),
        "n_subjects_straddling_partitions": len(straddling),
        "n_images_in_straddling_subjects": leaked_images,
        "n_test_subjects": len(test_subjects),
        "n_test_subjects_also_in_train": len(overlap),
        "test_train_subject_overlap": (
            round(len(overlap) / len(test_subjects), 4) if test_subjects else 0.0
        ),
        "n_test_scans": len(test_rows),
        "test_scan_contamination": (
            round(contaminated_scans / len(test_rows), 4) if test_rows else 0.0
        ),
    }
