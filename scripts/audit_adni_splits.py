#!/usr/bin/env python3
"""Audit frozen ADNI SplitGuard splits before any model training.

For each ``data/splits/adni/adni_splitguard_seed{SEED}.csv`` the audit
verifies:

1. **Zero subject overlap** across train / val / test.
2. **Zero session overlap** across train / val / test.
3. **Zero component overlap** across train / val / test.
4. **Class balance**: per-split proportions of each ``diagnosis_group`` are
   within 10 percentage points of the global proportion. Wider gaps are
   reported as warnings, not failures (small ADNI batches will be
   imbalanced by construction).
5. **Field-strength balance**: per-split proportions of
   ``scanner_field_strength`` are reported.
6. **Demographics**: per-split age (mean, SD) and sex distribution.
7. **Orphan components**: any component whose images are not all in the
   same split is flagged (this should be impossible if
   ``make_adni_splitguard_split.py`` succeeded; this audit verifies it
   downstream regardless).

Outputs
-------
* ``reports/audits/adni/adni_splitguard_seed{SEED}_audit.md`` — human-readable
  audit report.
* ``reports/audits/adni/adni_splitguard_seed{SEED}_audit.json`` — machine-readable
  report for the pipeline runner.

Exit code 0 if every checked split passes; non-zero if any split has
non-zero subject/session/component overlap. Gate this audit in CI before
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "adni"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "reports" / "audits" / "adni"

SPLITS = ("train", "val", "test")


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "subject_id",
        "session_id",
        "component_id",
        "diagnosis_group",
        "split",
    }
    missing = required - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")
    return rows


def overlap_check(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row[key] and row[key] != "unknown":
            by_split[row["split"]].add(row[key])
    return {
        f"train_val_{key}_overlap": len(by_split["train"] & by_split["val"]),
        f"train_test_{key}_overlap": len(by_split["train"] & by_split["test"]),
        f"val_test_{key}_overlap": len(by_split["val"] & by_split["test"]),
    }


def label_balance(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_split: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["diagnosis_group"]] += 1
    global_counter = Counter(row["diagnosis_group"] for row in rows)
    total = sum(global_counter.values()) or 1
    global_share = {label: count / total for label, count in global_counter.items()}
    per_split = {}
    warnings: list[str] = []
    for split in SPLITS:
        split_total = sum(by_split[split].values()) or 1
        share = {label: count / split_total for label, count in by_split[split].items()}
        per_split[split] = {
            "counts": dict(by_split[split]),
            "share": {label: round(value, 4) for label, value in share.items()},
        }
        for label, global_value in global_share.items():
            actual = share.get(label, 0.0)
            if abs(actual - global_value) > 0.10:
                warnings.append(
                    f"{split} {label} share {actual:.2%} differs from global {global_value:.2%} by >10pp"
                )
    return {
        "global_share": {label: round(value, 4) for label, value in global_share.items()},
        "per_split": per_split,
        "warnings": warnings,
    }


def field_strength_balance(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_split: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row.get("scanner_field_strength") or "missing"] += 1
    return {split: dict(by_split[split]) for split in SPLITS}


def demographic_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        ages: list[float] = []
        for row in split_rows:
            try:
                ages.append(float(row.get("age", "")))
            except (TypeError, ValueError):
                continue
        sex_counts = Counter(row.get("sex") or "missing" for row in split_rows)
        out[split] = {
            "n": len(split_rows),
            "age_mean": round(statistics.fmean(ages), 2) if ages else None,
            "age_sd": round(statistics.pstdev(ages), 2) if len(ages) > 1 else None,
            "sex_counts": dict(sex_counts),
        }
    return out


def orphan_components(rows: list[dict[str, str]]) -> list[str]:
    components: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        components[row["component_id"]].add(row["split"])
    return [component_id for component_id, splits in components.items() if len(splits) > 1]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# ADNI SplitGuard Audit — seed {report['seed']}",
        "",
        f"- Created: {report['created_at']}",
        f"- Split file: `{report['split_file']}`",
        f"- Total rows: {report['n_rows']}",
        "",
        "## Overlap Checks (must all be 0)",
        "",
    ]
    for key, value in report["overlap"].items():
        marker = "✅" if value == 0 else "❌"
        lines.append(f"- {marker} `{key}`: {value}")
    lines.append("")
    lines.append("## Label Balance")
    lines.append("")
    lines.append(f"Global share: {report['label_balance']['global_share']}")
    for split, payload in report["label_balance"]["per_split"].items():
        lines.append(f"- `{split}` counts: {payload['counts']}, share: {payload['share']}")
    if report["label_balance"]["warnings"]:
        lines.append("")
        lines.append("**Warnings (>10pp drift):**")
        for warning in report["label_balance"]["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Field Strength Distribution")
    lines.append("")
    for split, counts in report["field_strength"].items():
        lines.append(f"- `{split}`: {counts}")
    lines.append("")
    lines.append("## Demographics")
    lines.append("")
    for split, payload in report["demographics"].items():
        lines.append(
            f"- `{split}`: n={payload['n']}, age={payload['age_mean']} ± {payload['age_sd']}, sex={payload['sex_counts']}"
        )
    lines.append("")
    lines.append("## Orphan Components")
    lines.append("")
    if not report["orphan_components"]:
        lines.append("✅ No orphan components.")
    else:
        lines.append(f"❌ {len(report['orphan_components'])} components span multiple splits:")
        for component_id in report["orphan_components"][:25]:
            lines.append(f"- `{component_id}`")
    lines.append("")
    lines.append("## Gate Decision")
    lines.append("")
    if report["passed"]:
        lines.append("✅ **PASS — safe to train under this split.**")
    else:
        lines.append("❌ **FAIL — do not train on this split until the failures above are resolved.**")
    return "\n".join(lines) + "\n"


def audit_one(path: Path, audit_dir: Path) -> dict[str, Any]:
    rows = read_split(path)
    overlap_subject = overlap_check(rows, "subject_id")
    overlap_session = overlap_check(rows, "session_id")
    overlap_component = overlap_check(rows, "component_id")
    combined_overlap = {**overlap_subject, **overlap_session, **overlap_component}
    orphans = orphan_components(rows)
    passed = all(value == 0 for value in combined_overlap.values()) and not orphans

    seed = path.stem.split("seed")[-1]
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "split_file": str(path.relative_to(PROJECT_ROOT)),
        "n_rows": len(rows),
        "overlap": combined_overlap,
        "label_balance": label_balance(rows),
        "field_strength": field_strength_balance(rows),
        "demographics": demographic_stats(rows),
        "orphan_components": orphans,
        "passed": passed,
    }

    audit_dir.mkdir(parents=True, exist_ok=True)
    md_path = audit_dir / f"adni_splitguard_seed{seed}_audit.md"
    json_path = audit_dir / f"adni_splitguard_seed{seed}_audit.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {md_path.relative_to(PROJECT_ROOT)} (passed={passed})")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()

    if not args.split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {args.split_dir}")

    split_files = sorted(args.split_dir.glob("adni_splitguard_seed*.csv"))
    if not split_files:
        print(f"No SplitGuard splits found under {args.split_dir}.")
        return 1

    failed = 0
    for path in split_files:
        report = audit_one(path, args.audit_dir)
        if not report["passed"]:
            failed += 1
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
