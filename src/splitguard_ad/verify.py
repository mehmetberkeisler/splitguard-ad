"""One command that runs every gate and prints a single verdict.

    python -m splitguard_ad.verify

A reviewer should not have to know which of five scripts proves what. This runs
them in the order that fails fastest and prints one table, because the useful
question is whether the repository as a whole still holds together.

The environment check is reported but does not by itself decide the verdict,
and that split is deliberate. The three gates import nothing outside the
standard library, so they pass on a machine where every training script is
unrunnable. That is worth seeing next to a PASS rather than hidden behind one,
but failing the whole run on it would mean a reviewer with no GPU stack could
never get a green result from artefacts that are in fact intact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

# (label, argv, counts_towards_verdict)
STAGES: list[tuple[str, list[str], bool]] = [
    ("Tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests"], True),
    ("Paper values", [sys.executable, str(SCRIPTS / "verify_paper_numbers.py")], True),
    ("Submission structure", [sys.executable, str(SCRIPTS / "validate_submission.py")], True),
    ("Environment", [sys.executable, str(SCRIPTS / "check_environment.py")], False),
]


def run(argv: list[str], verbose: bool) -> bool:
    proc = subprocess.run(argv, cwd=ROOT,
                          capture_output=not verbose, text=True)
    if proc.returncode != 0 and not verbose and proc.stdout:
        lines = (proc.stdout + (proc.stderr or "")).splitlines()
        tail = [line for line in lines if line.strip()][-12:]
        print("\n".join("    " + line for line in tail))
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true",
                    help="stream each stage's own output instead of only failures")
    args = ap.parse_args(argv)

    print("SplitGuard-AD verification")
    print("--------------------------")
    results, blocking_failed = [], False
    for label, cmd, counts in STAGES:
        ok = run(cmd, args.verbose)
        if not ok and counts:
            blocking_failed = True
        note = "" if counts else "   (reported, not blocking)"
        results.append((label, ok, note))
        print(f"{label + ':':<22}{'PASS' if ok else 'FAIL'}{note}")

    print()
    print(f"OVERALL STATUS: {'FAIL' if blocking_failed else 'PASS'}")
    if not blocking_failed and any(not ok for _, ok, note in results if note):
        print()
        print("The environment does not match the one the published numbers came")
        print("from. The gates above do not import the training stack, so they")
        print("pass either way; retraining or regenerating a figure here would")
        print("not reproduce the released artefacts. See requirements.txt.")
    return 1 if blocking_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
