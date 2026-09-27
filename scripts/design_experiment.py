#!/usr/bin/env python3
"""Read a manifest and report what evaluation protocol it can actually support.

The audit scripts answer "is this split clean?" after a split exists. This one
answers the question that comes before it: given the identifiers this cohort
carries, which leakage channels can be closed at all, which cannot, and what
does a naive split cost here?

Three things are reported.

**Which channels are closeable.** A splitting protocol can only separate what
the metadata lets it recognise. Every channel is listed against the column it
needs, so a missing column reads as an open channel rather than as silence.

**What the leakage graph does to this cohort.** Component count against
participant count says whether the graph is doing anything a subject grouping
would not: on cohorts with intact identifiers the two coincide exactly, which
is the expected result and not a defect.

**What a naive split would cost, at minimum.** The fraction of test
participants a random image-level split returns to training is computable
directly from the component structure, and the dose-response calibration turns
that into AUROC optimism. The number is a *lower bound*, not an estimate, and
the distinction is load-bearing. That calibration was fitted by injecting
test-subject overlap while holding the training set fixed, so it prices the
participant-identity channel and nothing else; a real naive split also leaks
through slices of one volume, near-duplicates and re-acquisitions. Checked
against the two cohorts where the total gap was measured, the bound behaves as
a bound should: +0.106 predicted against +0.129 measured on ADNI1, and +0.097
against +0.157 on the Tier-1 benchmark, where the same participant contributes
far more images and the non-identity channels carry more of the gap. Reporting
it as a point estimate would understate the problem by a third on a cohort like
that one.

Usage
-----
    python3 scripts/design_experiment.py --manifest my_cohort.csv
    python3 scripts/design_experiment.py --manifest my_cohort.csv --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Fitted on the ADNI1 primary arm, ResNet-18, one 2D coronal-centre slice per
# scan: AUROC = 0.833 + 0.106 * (test-subject overlap). See the manuscript's
# dose-response section for why the slope is cohort- and
# representation-specific and is not claimed to generalise.
DOSE_SLOPE = 0.106
DOSE_RESIDUAL_SD = 0.0247

CHANNELS = [
    ("subject_id", "Longitudinal same-patient leakage",
     "different visits of one patient landing on both sides of the split"),
    ("session_id", "Session / re-acquisition leakage",
     "rescans of one session separated across partitions"),
    ("series_uid", "Within-volume slice leakage",
     "multiple slices derived from one acquisition separated across partitions"),
    ("file_sha256", "Exact-duplicate leakage",
     "byte-identical copies of an image on both sides"),
    ("relative_path", "Near-duplicate leakage (detection only)",
     "visually near-identical images; detected and reported, never merged"),
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def usable(rows: list[dict[str, str]], column: str) -> tuple[bool, int, int]:
    """(column present and populated, rows carrying a value, distinct values)."""
    if not rows or column not in rows[0]:
        return False, 0, 0
    values = [(r.get(column) or "").strip() for r in rows]
    filled = [v for v in values if v and v != "unknown"]
    return bool(filled), len(filled), len(set(filled))


def components(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Union-find over whichever identifier columns this manifest actually has."""
    parent: dict[str, str] = {r["image_id"]: r["image_id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for column in ("subject_id", "session_id", "series_uid", "file_sha256"):
        if column not in (rows[0] if rows else {}):
            continue
        groups: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            v = (r.get(column) or "").strip()
            if v and v != "unknown":
                groups[v].append(r["image_id"])
        for ids in groups.values():
            for other in ids[1:]:
                union(ids[0], other)

    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[find(r["image_id"])].append(r)
    return out


def naive_overlap(rows: list[dict[str, str]], trials: int = 200) -> float:
    """Mean fraction of test participants also in training under a random image split."""
    have_subject = "subject_id" in (rows[0] if rows else {})
    if not have_subject:
        return float("nan")
    rng = random.Random(0)
    n_test = max(1, int(len(rows) * 0.15))
    shares = []
    for _ in range(trials):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        test, train = shuffled[:n_test], shuffled[n_test:]
        test_s = {r["subject_id"] for r in test}
        train_s = {r["subject_id"] for r in train}
        if test_s:
            shares.append(len(test_s & train_s) / len(test_s))
    return sum(shares) / len(shares) if shares else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="Also write the report as JSON.")
    args = ap.parse_args()

    rows = read(args.manifest)
    if not rows:
        raise SystemExit(f"{args.manifest} is empty.")
    if "image_id" not in rows[0]:
        raise SystemExit(
            f"Manifest needs an 'image_id' column. Found: {sorted(rows[0])}"
        )

    report: dict[str, object] = {"manifest": str(args.manifest), "n_images": len(rows)}
    print(f"\n  {args.manifest.name} — {len(rows)} images\n")

    print("  Leakage channels this manifest can close")
    print("  " + "-" * 68)
    channel_state = {}
    for column, name, what in CHANNELS:
        ok, filled, distinct = usable(rows, column)
        channel_state[name] = ok
        if ok:
            print(f"  ✓ {name}")
            print(f"      {column}: {filled}/{len(rows)} rows, {distinct} distinct")
        else:
            print(f"  ✗ {name}  — OPEN")
            print(f"      needs {column}; {what}")
    report["channels"] = channel_state
    print()

    comps = components(rows)
    n_comp = len(comps)
    ok_subject, _, n_subject = usable(rows, "subject_id")
    print("  What the leakage graph does here")
    print("  " + "-" * 68)
    print(f"  {n_comp} components over {len(rows)} images")
    if ok_subject:
        print(f"  {n_subject} distinct participants")
        if n_comp == n_subject:
            print("  Components coincide exactly with participants: the graph is")
            print("  equivalent to subject-wise splitting on this cohort. That is the")
            print("  expected result when identifiers are intact, not a defect — the")
            print("  graph earns its keep when they are not.")
        elif n_comp > n_subject:
            # More components than participants means the graph could not reach
            # participant granularity, which happens when identifiers are
            # missing: the unidentified images fall out as singletons instead of
            # joining their participant. It does not mean the extra edge rules
            # found something a subject grouping would have missed -- that would
            # show up as FEWER components, not more.
            print(f"  {n_comp} components against {n_subject} participants: the graph did")
            print("  not reach participant granularity. Images whose participant is")
            print("  unknown cannot be joined to their own record and fall out as")
            print("  singletons, so the split will separate scans of one patient.")
        else:
            print(f"  {n_comp} components against {n_subject} participants: some")
            print("  components span more than one participant identifier. Check the")
            print("  session, series and hash columns for values shared across")
            print("  participants — an identifier that is not globally unique will")
            print("  merge unrelated people into one component.")
        missing = sum(1 for r in rows
                      if not (r.get("subject_id") or "").strip()
                      or (r.get("subject_id") or "").strip() == "unknown")
        if missing:
            print(f"  ⚠ {missing} images ({100*missing/len(rows):.1f}%) carry no participant")
            print("    identifier. Subject-wise splitting cannot group these at all;")
            print("    each is a singleton and may land anywhere.")
    report["n_components"] = n_comp
    report["n_participants"] = n_subject if ok_subject else None
    sizes = [len(v) for v in comps.values()]
    print(f"  Component size: mean {sum(sizes)/len(sizes):.2f}, "
          f"range {min(sizes)}-{max(sizes)}")
    report["component_size_mean"] = round(sum(sizes) / len(sizes), 2)
    print()

    print("  What a naive image-level split would cost")
    print("  " + "-" * 68)
    p = naive_overlap(rows)
    if p != p:
        print("  Not estimable without a subject_id column.")
    else:
        est = DOSE_SLOPE * p
        print(f"  A random image-level split returns {100*p:.0f}% of test participants")
        print(f"  to training. Priced through the dose-response calibration, that is")
        print(f"  at least +{est:.3f} AUROC of optimism (residual SD {DOSE_RESIDUAL_SD}).")
        print()
        print("  Read that as a floor, not a forecast. The calibration prices the")
        print("  participant-identity channel only, because it was fitted by varying")
        print("  test-subject overlap with the training set held fixed. A naive split")
        print("  also leaks through slices of one volume, near-duplicates and")
        print("  re-acquisitions, and those are not in this number. Against the two")
        print("  cohorts where the total gap was measured it behaved as a floor:")
        print("  +0.106 predicted vs +0.129 measured on ADNI1, +0.097 vs +0.157 on")
        print("  the Tier-1 benchmark, where one participant contributes many more")
        print("  images. It was fitted on ADNI1 with one 2D coronal-centre slice per")
        print("  scan and a ResNet-18; refit it on your own data if you need a")
        print("  number rather than a threshold for concern.")
        report["naive_test_subject_overlap"] = round(p, 4)
        report["estimated_naive_optimism_auroc"] = round(est, 4)
    print()

    print("  Recommended next step")
    print("  " + "-" * 68)
    if not ok_subject:
        print("  Recover participant identifiers before splitting. Without them no")
        print("  protocol here can close the dominant leakage channel, and a split")
        print("  will look clean while leaking.")
    else:
        print("  python3 scripts/build_current_leakage_graph.py \\")
        print(f"      --manifest {args.manifest} --components components.csv")
        print("  python3 scripts/make_current_splitguard_split.py \\")
        print(f"      --manifest {args.manifest} --components components.csv \\")
        print("      --output split_seed42.csv --audit split_audit.md --seed 42")
        print()
        print("  Read the audit's component-size section, not only its overlap")
        print("  checks: partitions can be disjoint and still be drawn from")
        print("  different kinds of participant.")
    print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  Wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
