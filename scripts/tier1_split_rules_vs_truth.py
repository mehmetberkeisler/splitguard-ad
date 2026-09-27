#!/usr/bin/env python3
"""Which split rule prevents leakage on a benchmark whose provenance was stripped?

The provenance-degradation experiment deletes identifiers at random from a
cohort that has them. This script runs the same question on a benchmark that
lost them for real: the Tier-1 redistribution of OASIS-1, for which the true
participant of every image has been recovered by content matching
(``audit_tier1_ground_truth.py``).

Each rule defines the connected components the splitter must keep together;
every rule is then assigned by the same class-stratified bin-packer used for
the published splits, so the rules differ only in what they group. Leakage is
measured against the true participants, not against the rule's own notion of
identity, which is the mistake a rule cannot detect in itself.

Rules
-----
filename_subject      the published Tier-1 grouping (parenthesised filename number)
filename_plus_dhash4  filename grouping, plus near-duplicate edges at dHash <= 4
filename_plus_dhash8  filename grouping, plus near-duplicate edges at dHash <= 8
dhash8_only           near-duplicate edges alone, as if no filename existed
true_participant      the recovered OASIS-1 participant (the reference)

Near-duplicate edges that would join two differently-labelled images are
counted and not merged, because the splitter requires label-pure components.

Also writes per-seed frozen splits for ``filename_subject`` and
``true_participant`` so the AUROC arm can train on both.

Usage
-----
    python3 scripts/tier1_split_rules_vs_truth.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import display_path  # noqa: E402
from build_current_leakage_graph import dhash  # noqa: E402
from make_current_splitguard_split import DEFAULT_RATIOS, assign_splits, build_components  # noqa: E402

DEFAULT_SPLIT = ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
DEFAULT_TRUTH = ROOT / "data" / "manifests" / "tier1_oasis1_ground_truth.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "tables" / "tier1_split_rules_vs_truth.json"
DEFAULT_SPLIT_DIR = ROOT / "data" / "splits" / "tier1_truth"
SEEDS = (42, 0, 1, 2, 3)          # the seed array the Tier-1 experiments use
RULES = ("filename_subject", "filename_plus_dhash4", "filename_plus_dhash8", "dhash8_only", "true_participant")
FROZEN = ("filename_subject", "true_participant")


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_pairs(paths, max_d):
    """All image pairs with dHash Hamming distance <= max_d, as (i, j, d)."""
    as_bytes = np.array([list(dhash(p).to_bytes(32, "little")) for p in paths], dtype=np.uint8)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    out = []
    for i in range(len(paths) - 1):
        d = popcount[as_bytes[i + 1:] ^ as_bytes[i]].sum(axis=1)
        for k in np.nonzero(d <= max_d)[0]:
            out.append((i, i + 1 + int(k), int(d[k])))
    return out


def components_for(rule, rows, pairs):
    n = len(rows)
    uf = UnionFind(n)
    blocked = 0
    if rule in ("filename_subject", "filename_plus_dhash4", "filename_plus_dhash8"):
        first = {}
        for i, r in enumerate(rows):
            first.setdefault(r["subject_id"], i)
            uf.union(first[r["subject_id"]], i)
    if rule == "true_participant":
        first = {}
        for i, r in enumerate(rows):
            first.setdefault(r["true"], i)
            uf.union(first[r["true"]], i)
    limit = {"filename_plus_dhash4": 4, "filename_plus_dhash8": 8, "dhash8_only": 8}.get(rule)
    if limit is not None:
        for i, j, d in pairs:
            if d > limit:
                continue
            if rows[i]["raw_class_label"] != rows[j]["raw_class_label"]:
                blocked += 1          # splitter needs label-pure components
                continue
            uf.union(i, j)
    return [f"{rule}_{uf.find(i):05d}" for i in range(n)], blocked


def measure(rows, component_of, seed):
    comp_rows = [{**r, "component_id": c, "component_primary_reason": "rule"} for r, c in zip(rows, component_of)]
    assignments, _ = assign_splits(build_components(comp_rows), seed, DEFAULT_RATIOS)
    split = [assignments[c] for c in component_of]
    train = {r["true"] for r, s in zip(rows, split) if s == "train"}
    test_idx = [i for i, s in enumerate(split) if s == "test"]
    test_true = {rows[i]["true"] for i in test_idx}
    parts = defaultdict(set)
    for r, s in zip(rows, split):
        parts[r["true"]].add(s)
    sizes = Counter(component_of)
    return split, {
        "components": len(sizes),
        "largest_component_images": max(sizes.values()),
        "test_images": len(test_idx),
        "test_images_contaminated": sum(1 for i in test_idx if rows[i]["true"] in train),
        "test_participants": len(test_true),
        "test_participants_in_train": len(test_true & train),
        "participants_straddling": sum(1 for s in parts.values() if len(s) > 1),
        "images_per_split": dict(Counter(split)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    args = ap.parse_args()

    truth = {r["image"]: r for r in csv.DictReader(args.truth.open())}
    rows = sorted(csv.DictReader(args.split.open()), key=lambda r: r["relative_path"])
    for r in rows:
        r["true"] = truth[r["relative_path"]]["oasis_subject"]
    pairs = near_pairs([ROOT / r["relative_path"] for r in rows], 8)

    per_rule, blocked_edges = {}, {}
    args.split_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for rule in RULES:
        component_of, blocked = components_for(rule, rows, pairs)
        blocked_edges[rule] = blocked
        runs = {}
        for seed in SEEDS:
            split, metrics = measure(rows, component_of, seed)
            runs[str(seed)] = metrics
            if rule in FROZEN:
                sizes = Counter(component_of)
                out_rows = []
                for r, s, c in zip(rows, split, component_of):
                    o = {k: r[k] for k in fieldnames if k != "true"}
                    o.update(split=s, split_seed=seed, component_id=c, component_size=sizes[c],
                             component_primary_reason=rule,
                             split_policy=f"tier1_{rule}_raw_class_stratified_v1")
                    if rule == "true_participant":
                        o["subject_id"] = r["true"]
                    o["true_participant"] = r["true"]
                    o["filename_subject_id"] = r["subject_id"]
                    out_rows.append(o)
                path = args.split_dir / f"tier1_{rule}_seed{seed}.csv"
                with path.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                    w.writeheader()
                    w.writerows(out_rows)
        contam = [m["test_images_contaminated"] / m["test_images"] for m in runs.values()]
        per_rule[rule] = {
            "components": runs[str(SEEDS[0])]["components"],
            "largest_component_images": runs[str(SEEDS[0])]["largest_component_images"],
            "label_mismatched_edges_not_merged": blocked,
            "test_image_contamination_mean": round(st.mean(contam), 4),
            "test_image_contamination_min": round(min(contam), 4),
            "test_image_contamination_max": round(max(contam), 4),
            "test_participants_in_train_mean": round(st.mean(m["test_participants_in_train"] for m in runs.values()), 1),
            "test_participants_mean": round(st.mean(m["test_participants"] for m in runs.values()), 1),
            "participants_straddling_mean": round(st.mean(m["participants_straddling"] for m in runs.values()), 1),
            "per_seed": runs,
        }

    summary = {"seeds": list(SEEDS), "near_duplicate_pairs_dhash8": len(pairs), "rules": per_rule}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'rule':<22}{'comps':>6}{'largest':>8}{'blocked':>8}{'test contam mean [min,max]':>30}{'test pts in train':>20}{'straddling':>11}")
    for rule, v in per_rule.items():
        print(f"{rule:<22}{v['components']:>6}{v['largest_component_images']:>8}{v['label_mismatched_edges_not_merged']:>8}"
              f"{v['test_image_contamination_mean']:>13.1%} [{v['test_image_contamination_min']:.1%}, {v['test_image_contamination_max']:.1%}]"
              f"{v['test_participants_in_train_mean']:>11} / {v['test_participants_mean']:<6}{v['participants_straddling_mean']:>9}")
    print(f"wrote {display_path(args.summary)} and frozen splits under {display_path(args.split_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
