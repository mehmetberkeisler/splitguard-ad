#!/usr/bin/env python3
"""Subject-level summary for the OASIS-1 replication tier.

Reviewer §5.4 asked us to publish (a) slices per subject / repeat-session
ratio for OASIS-1 so the mechanically smaller inflation gap becomes
interpretable, and (b) subject-level aggregated AUROC alongside the
image-level headline so the OASIS entry in Table 12 is comparable to the
subject-level ADNI numbers already in the paper.

Reads
-----
data/manifests/oasis1_slices_manifest.csv
data/manifests/oasis1_manifest.csv
reports/tables/oasis1_multiseed_summary.json (if present)

Writes
------
reports/tables/oasis1_subject_level_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "oasis1_slices_manifest.csv"
DEFAULT_VOLUME_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "oasis1_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "tables" / "oasis1_subject_level_summary.json"


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return float("nan")
    if n % 2: return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarise_slice_manifest(path: Path) -> dict:
    slices_per_subject: dict[str, int] = defaultdict(int)
    slices_per_session: dict[str, int] = defaultdict(int)
    sessions_per_subject: dict[str, set] = defaultdict(set)
    subject_label: dict[str, str] = {}
    session_label: dict[str, str] = {}

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            subj = row["subject_id"]
            sess = row["session_id"]
            slices_per_subject[subj] += 1
            slices_per_session[sess] += 1
            sessions_per_subject[subj].add(sess)
            subject_label.setdefault(subj, row.get("binary_label", ""))
            session_label.setdefault(sess, row.get("binary_label", ""))

    slices_dist = list(slices_per_subject.values())
    sessions_dist = [len(v) for v in sessions_per_subject.values()]
    reliab_subj = sum(1 for v in sessions_per_subject.values() if len(v) > 1)

    return {
        "n_subjects_with_slices":     len(slices_per_subject),
        "n_sessions":                 len(slices_per_session),
        "n_slices":                   sum(slices_dist),
        "slices_per_subject": {
            "min":    min(slices_dist) if slices_dist else 0,
            "max":    max(slices_dist) if slices_dist else 0,
            "median": median(slices_dist),
            "mean":   round(mean(slices_dist), 3) if slices_dist else float("nan"),
        },
        "slices_per_session": {
            "min":    min(slices_per_session.values()) if slices_per_session else 0,
            "max":    max(slices_per_session.values()) if slices_per_session else 0,
            "median": median(list(slices_per_session.values())),
            "mean":   round(mean(list(slices_per_session.values())), 3)
                     if slices_per_session else float("nan"),
        },
        "sessions_per_subject": {
            "min":    min(sessions_dist) if sessions_dist else 0,
            "max":    max(sessions_dist) if sessions_dist else 0,
            "median": median(sessions_dist),
            "mean":   round(mean(sessions_dist), 3) if sessions_dist else float("nan"),
        },
        "subjects_with_repeat_session":     reliab_subj,
        "repeat_session_ratio":             round(reliab_subj / len(sessions_per_subject), 4)
                                             if sessions_per_subject else 0.0,
        "label_distribution_subject_level": dict(Counter(subject_label.values())),
        "label_distribution_session_level": dict(Counter(session_label.values())),
    }


def summarise_volume_manifest(path: Path) -> dict:
    """Volume-level metadata: total volumes, unique subjects, CDR distribution."""
    if not path.exists():
        return {"note": f"missing: {path}"}
    volume_rows = list(csv.DictReader(path.open()))
    cdr_col = "cdr" if "cdr" in volume_rows[0] else None
    cdr_counter = Counter(r.get(cdr_col, "") for r in volume_rows) if cdr_col else {}
    subject_set = set(r.get("subject_id", "") for r in volume_rows)
    session_set = set(r.get("session_id", "") for r in volume_rows)
    return {
        "n_volume_rows": len(volume_rows),
        "n_subjects_in_volume_manifest": len(subject_set),
        "n_sessions_in_volume_manifest": len(session_set),
        "cdr_distribution": dict(cdr_counter),
    }


def _summarise_multiseed_json(js: dict) -> dict:
    """Pull image- vs subject-level AUROC out of an oasis1_multiseed_summary.json.

    The exact key layout differs across historical revisions of the
    OASIS-1 script; we look for the two most common shapes and record
    what we find without failing on unfamiliar fields.
    """
    out = {}
    for k in ("image_level", "subject_level", "leaky", "safe", "random",
              "component_safe", "subject_only", "protocol_a", "protocol_b",
              "protocol_c"):
        if k in js:
            out[k] = js[k]
    if "protocols" in js and isinstance(js["protocols"], dict):
        out["protocols"] = js["protocols"]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice-manifest",  type=Path, default=DEFAULT_SLICE_MANIFEST)
    p.add_argument("--volume-manifest", type=Path, default=DEFAULT_VOLUME_MANIFEST)
    p.add_argument("--multiseed-json",  type=Path,
                   default=PROJECT_ROOT / "reports" / "tables" /
                           "oasis1_multiseed_summary.json")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    summary = {
        "slice_manifest_stats":  summarise_slice_manifest(args.slice_manifest),
        "volume_manifest_stats": summarise_volume_manifest(args.volume_manifest),
    }
    if args.multiseed_json.exists():
        summary["multiseed_predictions_reference"] = _summarise_multiseed_json(
            json.loads(args.multiseed_json.read_text())
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")

    s = summary["slice_manifest_stats"]
    print(f"\nOASIS-1 slice manifest ({s['n_slices']} slices):")
    print(f"  subjects: {s['n_subjects_with_slices']}")
    print(f"  sessions: {s['n_sessions']}")
    print(f"  slices/subject: mean {s['slices_per_subject']['mean']}, "
          f"median {s['slices_per_subject']['median']}, "
          f"range {s['slices_per_subject']['min']}--{s['slices_per_subject']['max']}")
    print(f"  slices/session: mean {s['slices_per_session']['mean']}, "
          f"median {s['slices_per_session']['median']}, "
          f"range {s['slices_per_session']['min']}--{s['slices_per_session']['max']}")
    print(f"  sessions/subject: mean {s['sessions_per_subject']['mean']}, "
          f"median {s['sessions_per_subject']['median']}, "
          f"range {s['sessions_per_subject']['min']}--{s['sessions_per_subject']['max']}")
    print(f"  subjects with repeat session: {s['subjects_with_repeat_session']} "
          f"({s['repeat_session_ratio']*100:.1f}% of sessions covered)")
    print(f"  label distribution (subject-level): {s['label_distribution_subject_level']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
