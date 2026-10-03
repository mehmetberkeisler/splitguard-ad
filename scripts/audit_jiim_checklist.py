#!/usr/bin/env python3
"""Self-audit the manuscript against the JIIM classification-review checklist.

The journal publishes a checklist its reviewers use for classification papers,
covering data, partitioning, preprocessing, training, evaluation and
reproducibility. Asserting compliance is worthless; this searches the
manuscript and supplement for evidence of each item and reports what it
actually finds, including the items it cannot find.

A miss is not necessarily a defect. Some items do not apply to a study whose
subject is evaluation validity rather than a diagnostic model. The point is to
know which is which before a reviewer decides for you.

Usage
-----
    python3 scripts/audit_jiim_checklist.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "JIIM_CHECKLIST_AUDIT.md"


def corpus() -> str:
    parts = []
    for name in ("splitguard_ad.tex", "SplitGuard-AD_Supplementary_Material.tex"):
        p = ROOT / "paper" / name
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
    for p in sorted((ROOT / "paper" / "tables").glob("*.tex")):
        parts.append(p.read_text(encoding="utf-8"))
    # Flatten: LaTeX wraps wherever the line fills, and a phrase split across
    # a newline is not absent from the paper.
    return " ".join(" ".join(parts).split())


# (section, item, regex that would evidence it, note when it cannot be found)
ITEMS = [
    ("Data", "Data source named",
     r"ADNI|Alzheimer's Disease Neuroimaging Initiative", ""),
    ("Data", "Inclusion and exclusion stated",
     r"exclude[sd]? .{0,80}(converter|MCI|mixed)", ""),
    ("Data", "Participant and image counts stated",
     r"\$220\$ .{0,40}participants|1\{,\}123", ""),
    ("Data", "Repeated observations per participant stated",
     r"scans per (participant|subject)|median 5", ""),
    ("Data", "Reference standard explained",
     r"DIAGNOSIS|visit-level diagnosis|reference standard", ""),
    ("Data", "Label linkage and its failure mode stated",
     r"date-proximity|exact visit key", ""),
    ("Data", "Missing or unresolved data accounted for",
     r"unresolved|diagnosis_group=unknown|two scans whose", ""),
    ("Data", "Demographics by subset reported",
     r"tab:adni_demographics", ""),

    ("Partitioning", "Train/validation/test construction explained",
     r"train.{0,10}validation.{0,10}test|165/35/35", ""),
    ("Partitioning", "Patient independence explicitly audited",
     r"zero subject, session and component overlap|overlap_check|component overlap", ""),
    ("Partitioning", "Composition differences between partitions reported",
     r"component size|stratified by component size|class mix", ""),
    ("Partitioning", "Cross-cohort set clearly identified",
     r"OASIS-1 .{0,60}replication|cross-cohort", ""),
    ("Partitioning", "Redistributed set NOT called independent validation",
     r"not a third cohort|not independent replications", ""),

    ("Preprocessing", "Slice or volume selection stated",
     r"coronal-centre slice", ""),
    ("Preprocessing", "Normalisation stated",
     r"percentile intensity normalisation|ImageNet mean/std", ""),
    ("Preprocessing", "Resizing stated", r"224", ""),
    ("Preprocessing", "Augmentation stated", r"flip|rotation", ""),

    ("Training", "Architecture stated", r"ResNet-18|DenseNet-121", ""),
    ("Training", "Initialisation stated", r"ImageNet pre-trained|random initialisation", ""),
    ("Training", "Optimiser stated", r"AdamW", ""),
    ("Training", "Learning rate stated", r"10\^\{-4\}|3 \\times 10", ""),
    ("Training", "Epochs stated", r"15 epochs|\$15\$ epochs|epochs", ""),
    ("Training", "Batch size stated", r"batch size", ""),
    ("Training", "Checkpoint rule stated",
     r"best validation AUROC|best-val", ""),
    ("Training", "Seeds stated", r"five seeds|seed array", ""),
    ("Training", "Hardware stated", r"NVIDIA|H100|RTX", ""),

    ("Evaluation", "Primary metric defined", r"AUROC", ""),
    ("Evaluation", "Unit of analysis clear",
     r"image-level|subject-level|per participant", ""),
    ("Evaluation", "Uncertainty method explained",
     r"paired-seed bootstrap|hierarchical", ""),
    ("Evaluation", "Threshold selection explained",
     r"selected on .{0,40}validation|validation-selected", ""),
    ("Evaluation", "Participant- vs image-level metrics distinguished",
     r"subject-level number is the recommended|subject-level AUROC", ""),
    ("Evaluation", "Intervals described as descriptive where they are",
     r"descriptive", ""),

    ("Reproducibility", "Code released", r"Apache-2\.0|github\.com", ""),
    ("Reproducibility", "Split manifests released", r"frozen split manifest", ""),
    ("Reproducibility", "Environment and versions stated",
     r"requirements|software environment|one code state", ""),
    ("Reproducibility", "Random seeds reported", r"seed", ""),
    ("Reproducibility", "Restricted-data access explained",
     r"data-use agreement|LONI", ""),
    ("Reproducibility", "Numerical verification of the manuscript described",
     r"verify\\_paper\\_numbers|re-checks each of", ""),
    ("Reproducibility", "AI/LLM use declared", r"large-language-model|LLM", ""),
]


def main() -> int:
    text = corpus()
    rows, misses = [], []
    for section, item, pattern, note in ITEMS:
        found = re.search(pattern, text, re.I) is not None
        rows.append((section, item, found, note))
        if not found:
            misses.append((section, item, note))

    lines = ["# JIIM classification-checklist self-audit", "",
             "Generated by `scripts/audit_jiim_checklist.py`, which searches the "
             "manuscript, supplement and generated tables for evidence of each "
             "item rather than asserting compliance. Whitespace is flattened "
             "first, because LaTeX wraps prose wherever the line fills and a "
             "phrase split across a newline is not absent from the paper.", "",
             f"**{sum(1 for r in rows if r[2])} of {len(rows)} items evidenced.**", ""]

    current = None
    for section, item, found, note in rows:
        if section != current:
            lines += ["", f"## {section}", "", "| Item | Evidenced |", "| --- | --- |"]
            current = section
        lines.append(f"| {item} | {'yes' if found else '**not found**'} |")

    lines += ["", "---", ""]
    if misses:
        lines += ["## Items not evidenced", "",
                  "Each needs a decision: add it, or state why it does not apply "
                  "to a study whose subject is evaluation validity rather than a "
                  "diagnostic model.", ""]
        for section, item, note in misses:
            lines.append(f"- **{section} — {item}**" + (f" — {note}" if note else ""))
    else:
        lines.append("Every item in this checklist is evidenced in the submission.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{sum(1 for r in rows if r[2])}/{len(rows)} items evidenced")
    for section, item, _ in misses:
        print(f"  not found: {section} — {item}")
    print(f"\nWrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
