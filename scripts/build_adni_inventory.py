#!/usr/bin/env python3
"""Build a first-pass ADNI local inventory report.

This script is intentionally conservative: it records what has been downloaded
without parsing restricted ADNI subject-level metadata into Git-tracked outputs.
The generated reports live under ignored local paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "adni"
DEFAULT_REPORT_DIR = ROOT / "reports" / "audits" / "adni"
DEFAULT_TABLE_DIR = ROOT / "reports" / "tables" / "adni"


def file_sha256(path: Path, max_size_mb: int) -> str:
    if path.stat().st_size > max_size_mb * 1024 * 1024:
        return "skipped_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, max_hash_mb: int) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(relative),
                "suffix": path.suffix.lower() or "[none]",
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "sha256": file_sha256(path, max_hash_mb),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["relative_path", "suffix", "size_bytes", "size_mb", "sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, root: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ADNI Inventory Audit",
        "",
        f"- Created: {summary['created_at']}",
        f"- Raw root: `{root}`",
        f"- Files found: {summary['n_files']}",
        f"- Total size GB: {summary['total_size_gb']}",
        "",
        "## File Types",
        "",
        "| Suffix | Count | Size GB |",
        "|---|---:|---:|",
    ]
    for suffix, stats in summary["suffix_summary"].items():
        lines.append(f"| `{suffix}` | {stats['count']} | {stats['size_gb']} |")
    lines.extend(
        [
            "",
            "## Next Check",
            "",
            "Confirm that clinical study files, image metadata, and baseline T1/MPRAGE images are present before building the ADNI manifest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    sizes = Counter()
    counts = Counter()
    for row in rows:
        suffix = str(row["suffix"])
        counts[suffix] += 1
        sizes[suffix] += int(row["size_bytes"])
    suffix_summary = {
        suffix: {
            "count": counts[suffix],
            "size_gb": round(sizes[suffix] / (1024**3), 4),
        }
        for suffix in sorted(counts)
    }
    total_size = sum(int(row["size_bytes"]) for row in rows)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(rows),
        "total_size_gb": round(total_size / (1024**3), 4),
        "suffix_summary": suffix_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--max-hash-mb", type=int, default=100)
    parser.add_argument("--csv", type=Path, default=DEFAULT_TABLE_DIR / "adni_inventory.csv")
    parser.add_argument("--json", type=Path, default=DEFAULT_REPORT_DIR / "adni_inventory_summary.json")
    parser.add_argument("--audit", type=Path, default=DEFAULT_REPORT_DIR / "adni_inventory_audit.md")
    args = parser.parse_args()

    if not args.raw_root.exists():
        raise FileNotFoundError(f"ADNI raw root does not exist: {args.raw_root}")
    rows = inventory(args.raw_root, args.max_hash_mb)
    summary = summarize(rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.csv, rows)
    write_markdown(args.audit, args.raw_root, rows, summary)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.json}")
    print(f"Wrote {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
