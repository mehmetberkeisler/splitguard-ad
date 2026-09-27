#!/usr/bin/env python3
"""Compare the pre-rerun artefacts against the regenerated ones, arm by arm.

Why this is needed
------------------
Four of the five published ADNI arms were trained before the first commit of
the training code, and commit 877ca72 subsequently rewired the DataLoader
(``num_workers`` 0 -> 8, a seeded sampler generator, per-worker augmentation
seeding). Those changes alter data order and augmentation draws, so the
published numbers are not reproducible from the released code. ``gpu_program.py``
regenerates everything under one code state, ``rebuild_after_gpu.sh`` keeps the
previous runs under ``runs_frozen/``, and this script says what moved.

The output is the evidence base for two separate things: which manuscript
numbers have to be updated, and what the cover letter says about why. Both
need the same table, so it is generated rather than assembled by hand.

A shift is expected and is not a failure. What would be a failure is a shift
large enough to change a conclusion --- the sign of the inflation gap, whether
an interval excludes zero, or the ordering of the three protocols. Those are
called out separately from the raw deltas.

Usage
-----
    python3 scripts/compare_reruns.py
    python3 scripts/compare_reruns.py --old runs_frozen --new runs
    python3 scripts/compare_reruns.py --markdown reports/rerun_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A shift beyond this is worth a sentence in the manuscript rather than a
# silent update. Chosen to sit below the smallest effect the paper interprets
# (the +0.019 component marginal), so anything that could move a conclusion
# is surfaced.
NOTABLE = 0.010


def collect(root: Path) -> dict[str, float]:
    """Map 'arm/seed_dir/protocol' -> test AUROC for every run under `root`."""
    found: dict[str, float] = {}
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        auroc = payload.get("test_metrics", {}).get("auroc")
        if auroc is None:
            continue
        found[str(metrics_path.parent.relative_to(root))] = float(auroc)
    return found


def arm_of(key: str) -> str:
    return key.split("/")[0] if "/" in key else key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, default=PROJECT_ROOT / "runs_frozen")
    ap.add_argument("--new", type=Path, default=PROJECT_ROOT / "runs")
    ap.add_argument("--markdown", type=Path,
                    default=PROJECT_ROOT / "reports" / "rerun_comparison.md")
    args = ap.parse_args()

    if not args.old.exists():
        raise SystemExit(
            f"No pre-rerun snapshot at {args.old}. scripts/rebuild_after_gpu.sh "
            "creates it before promoting the GPU runs."
        )

    old, new = collect(args.old), collect(args.new)
    shared = sorted(set(old) & set(new))
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))

    lines: list[str] = [
        "# Re-run comparison: published artefacts vs regenerated",
        "",
        f"Old: `{args.old.name}/` ({len(old)} runs) — trained before the "
        "DataLoader rewiring in 877ca72, mostly before the first commit of the "
        "training code.",
        f"New: `{args.new.name}/` ({len(new)} runs) — one code state, one device.",
        "",
        f"Matched runs: {len(shared)}. New-only: {len(only_new)}. "
        f"Missing from the re-run: {len(only_old)}.",
        "",
    ]

    by_arm: dict[str, list[tuple[str, float, float]]] = {}
    for key in shared:
        by_arm.setdefault(arm_of(key), []).append((key, old[key], new[key]))

    notable: list[tuple[str, float, float, float]] = []

    for arm in sorted(by_arm):
        rows = by_arm[arm]
        deltas = [n - o for _, o, n in rows]
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        worst = max(deltas, key=abs)
        lines += [
            f"## `{arm}` — {len(rows)} runs",
            "",
            f"Mean absolute change {mean_abs:+.4f}; largest single change {worst:+.4f}.",
            "",
            "| run | published | regenerated | delta |",
            "|---|---|---|---|",
        ]
        for key, o, n in rows:
            flag = " ⚠️" if abs(n - o) >= NOTABLE else ""
            lines.append(f"| `{key}` | {o:.4f} | {n:.4f} | {n - o:+.4f}{flag} |")
            if abs(n - o) >= NOTABLE:
                notable.append((key, o, n, n - o))
        lines.append("")

    if only_new:
        lines += ["## New arms (no published counterpart)", ""]
        lines += [f"- `{k}` — {new[k]:.4f}" for k in only_new]
        lines.append("")

    if only_old:
        lines += [
            "## ⚠️ Published runs with no regenerated counterpart", "",
            "These arms did not re-run. Either the stage was skipped or it "
            "failed; the manuscript cannot cite them as reproducible until "
            "they do.", "",
        ]
        lines += [f"- `{k}` — published {old[k]:.4f}" for k in only_old]
        lines.append("")

    lines += ["## Verdict", ""]
    if not shared:
        lines.append("No matched runs — nothing to compare yet.")
    elif not notable:
        lines.append(
            f"No run moved by {NOTABLE:.3f} AUROC or more. The published numbers "
            "survive regeneration under the current code, and the manuscript "
            "needs no numeric update on this account."
        )
    else:
        lines += [
            f"{len(notable)} run(s) moved by at least {NOTABLE:.3f} AUROC. "
            "Each one has to be traced into the manuscript before submission: "
            "the per-seed tables, the bootstrap intervals derived from them, "
            "and any sentence that interprets the affected arm.",
            "",
            "| run | published | regenerated | delta |",
            "|---|---|---|---|",
        ]
        lines += [f"| `{k}` | {o:.4f} | {n:.4f} | {d:+.4f} |"
                  for k, o, n, d in sorted(notable, key=lambda r: -abs(r[3]))]
        lines += [
            "",
            "Re-derive every downstream artefact from the regenerated runs "
            "(bootstraps, operating points, cost-of-leakage, subject-level "
            "aggregation) rather than editing the manuscript by hand, then run "
            "`scripts/verify_paper_numbers.py` to confirm the two agree.",
        ]

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Matched {len(shared)} runs across {len(by_arm)} arms.")
    for arm in sorted(by_arm):
        deltas = [n - o for _, o, n in by_arm[arm]]
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        print(f"  {arm:32s} mean |delta| {mean_abs:.4f}   "
              f"worst {max(deltas, key=abs):+.4f}")
    if only_old:
        print(f"\n  {len(only_old)} published run(s) did NOT re-run.")
    print(f"\n{'No run moved notably.' if not notable else f'{len(notable)} run(s) moved >= {NOTABLE}.'}")
    print(f"Wrote {args.markdown}")
    return 1 if notable else 0


if __name__ == "__main__":
    raise SystemExit(main())
