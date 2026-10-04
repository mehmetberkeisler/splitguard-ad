"""The ``splitguard`` command: one entry point for the pipeline stages.

    splitguard graph   --manifest my_cohort.csv
    splitguard split   --components leakage_components.csv
    splitguard audit   --split my_split.csv
    splitguard design  --manifest my_cohort.csv
    splitguard verify

Each subcommand delegates to the script that implements that stage and passes
your arguments through unchanged, so ``splitguard graph --help`` is the script's
own help. The scripts are the validated implementations that produced every
published number, and wrapping them is the point: a new name for a stage would
be a second implementation to keep correct.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

STAGES = {
    "graph":  ("build_current_leakage_graph.py",
               "build the leakage graph and its connected components"),
    "split":  ("make_current_splitguard_split.py",
               "allocate whole components to train, validation and test"),
    "audit":  ("audit_adni_splits.py",
               "audit an existing split for cross-partition leakage"),
    "design": ("design_experiment.py",
               "report which provenance regime a cohort is in, before training"),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="splitguard", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'splitguard <stage> --help' for a stage's own options.")
    sub = ap.add_subparsers(dest="stage", metavar="<stage>")
    for name, (_, help_text) in STAGES.items():
        sub.add_parser(name, help=help_text, add_help=False)
    sub.add_parser("verify", help="run every gate and print one verdict")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    stage, rest = argv[0], argv[1:]

    if stage == "verify":
        from . import verify
        return verify.main(rest)

    if stage not in STAGES:
        build_parser().print_help()
        print(f"\nunknown stage: {stage}", file=sys.stderr)
        return 2

    script = SCRIPTS / STAGES[stage][0]
    if not script.is_file():
        print(f"{script} is missing from this checkout", file=sys.stderr)
        return 1

    # Hand the script its own argv and run it as if invoked directly, so its
    # argparse, its --help and its exit status are the ones you see.
    sys.argv = [str(script), *rest]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
