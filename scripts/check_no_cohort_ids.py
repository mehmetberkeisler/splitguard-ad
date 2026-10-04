#!/usr/bin/env python3
"""Refuse a file that names a real participant from the local cohort.

The test suite checks this across everything git would ship. This is the same
rule as a commit hook, so the answer arrives before the commit exists rather
than after, which is the difference between editing a file and rewriting
history.

Deciding whether an identifier is real needs the ADNI inventory, which is
local-only and never released. So this exits 0 with a note when the inventory
is absent: on a machine that could introduce the defect the inventory is
present, and on a machine without it there is nothing to leak.

Usage
-----
    python3 scripts/check_no_cohort_ids.py [path ...]

With no paths, every file git tracks or would add is checked.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "reports" / "tables" / "adni" / "adni_inventory.csv"
PTID = re.compile(r"\b\d{3}_S_\d{4}\b")
SKIP_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".pt", ".pth", ".nii", ".gz",
                 ".zip", ".tar", ".dcm", ".img", ".hdr"}
MAX_BYTES = 8_000_000


def cohort_identifiers() -> set[str]:
    return set(PTID.findall(INVENTORY.read_text(encoding="utf-8", errors="ignore")))


def candidates(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv]
    out = subprocess.run(["git", "ls-files", "-c", "-o", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [ROOT / line for line in out.stdout.split("\n") if line.strip()]


def main(argv: list[str]) -> int:
    if not INVENTORY.is_file():
        print(f"{INVENTORY.name} is not present, so no identifier can be "
              f"recognised as real; nothing to check.")
        return 0

    real = cohort_identifiers()
    if not real:
        print(f"{INVENTORY.name} holds no identifiers in the expected shape; "
              f"the pattern may have changed.", file=sys.stderr)
        return 1

    offenders: list[str] = []
    for path in candidates(argv):
        if (not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES
                or path.stat().st_size > MAX_BYTES
                or path.resolve() == INVENTORY.resolve()):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = sorted(real.intersection(PTID.findall(text)))
        if found:
            rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            offenders.append(f"  {rel}: {', '.join(found[:3])}"
                             + (f" and {len(found) - 3} more" if len(found) > 3 else ""))

    if offenders:
        print("These files name a participant in the cohort, which discloses "
              "membership:", file=sys.stderr)
        print("\n".join(offenders), file=sys.stderr)
        print("\nUse a synthetic identifier instead. Site 999 does not exist, "
              "so 999_S_9999 cannot collide with anyone.", file=sys.stderr)
        return 1

    print(f"No cohort participant identifier found "
          f"(checked against {len(real)} real identifiers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
