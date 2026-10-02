#!/usr/bin/env python3
"""Run the GPU programme in one code state under a hard spending cap.

Every stage is a list of training commands, each with the split files it reads
and the files it must produce. Outputs go to ``runs_gpu/`` and ``reports/gpu/``,
apart from the frozen results, and a command whose outputs exist is skipped, so
an interrupted programme resumes where it stopped.

Spending is the billed wall-clock time of this programme, kept in
``runs_gpu/gpu_ledger.json`` across restarts and charged at ``--usd-per-hour``.
A command starts only if its projected cost fits under ``--budget-usd`` minus
``--reserve-usd``, and a command still running when spending reaches that cap
is stopped. The reserve covers what the provider bills outside this programme:
start-up, environment setup, upload and download. Projections use the minutes
per training measured on this node once a run of the same kind has finished,
and the defaults in ``MINUTES`` before that.

Before training, every split file must exist and every image or volume it
references must be on disk. ``--smoke`` runs the first command of every stage
for one epoch and one seed into ``runs_smoke/``, which finds missing data and
broken imports within minutes.

The results archive holds predictions, metrics, logs and summaries, never model
weights. Its per-image predictions carry ADNI participant identifiers: move it
only to the machine the Data Use Agreement covers.

Usage, from the repository root of the node, after ``scripts/gpu_setup.sh``:
    python scripts/gpu_program.py --list
    python scripts/gpu_program.py --smoke
    python scripts/gpu_program.py --usd-per-hour 0.69 --budget-usd 10
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS, TABLES = "runs_gpu", "reports/gpu"
LOG = ROOT / "runs" / "gpu_program_log.jsonl"

# Minutes per training on one CUDA GPU before this node has timed a run of the
# same kind. From recorded CUDA wall times of these scripts (ADNI ResNet-18
# 18 s and DenseNet-121 57 s at 15 epochs, Kaggle DenseNet-121 22 min at 30
# epochs, OASIS-1 DenseNet-121 1.5 min at 20 epochs), with margin. ResNet-18
# on Kaggle and OASIS-1 is scaled from DenseNet-121 by the ADNI ratio.
MINUTES = {
    "tier1_resnet18": 7.0,
    "tier2_resnet18": 1.0, "tier2_densenet121": 2.0,
    "adni_resnet18": 0.5, "adni_densenet121": 1.3,
    "adni_probe": 1.5, "adni_3d": 10.0,
}
SEEDS = [0, 1, 2, 3, 4]
TIER_SEEDS = [42, 0, 1, 2, 3]
PROTOCOLS = ["random", "subject_only", "component_safe"]
OVERLAPS = ["0.0", "0.25", "0.50", "0.75", "1.0"]
OUTPUT_FLAGS = ("--output", "--table", "--summary", "--output-dir", "--output-root", "--runs-root")
DEFAULT_ORDER = ["tier1_truth", "adni_primary", "adni_converters", "adni_no_mt1", "adni_densenet121",
                 "tier2_oasis1", "adni_null", "adni_size_balanced", "identity_probe", "adni_provenance",
                 "adni_3d", "adni_dose_response", "adni_exact_linkage", "adni_stress"]


# What each kind of stage imports at training time. A bare CUDA image has
# torch but rarely the rest, and discovering that per command costs one failed
# launch per command: 50 of them, in the run this check was written for.
STAGE_MODULES = {
    "adni_3d": ("torch", "monai", "nibabel", "numpy"),
    None: ("torch", "torchvision", "sklearn", "nibabel", "PIL", "numpy"),
}


def missing_modules(kinds) -> dict[str, list[str]]:
    """Modules the selected stages need that this interpreter cannot import."""
    wanted: set[str] = set()
    for kind in kinds:
        wanted.update(STAGE_MODULES.get(kind, STAGE_MODULES[None]))
    probe = "import importlib,sys;print(' '.join(m for m in sys.argv[1:] if not importlib.util.find_spec(m)))"
    out = subprocess.run([PY, "-c", probe, *sorted(wanted)], capture_output=True, text=True)
    return {"missing": out.stdout.split(), "error": out.stderr.strip()[:200]}


def cuda_is_healthy() -> str | None:
    """Return an error string when torch cannot use the GPU it is about to rent."""
    probe = ("import torch, torchvision, torchvision.ops as ops;"
             "assert torch.cuda.is_available(), 'torch.cuda.is_available() is False';"
             "ops.nms(torch.zeros(1, 4), torch.zeros(1), 0.5)")
    out = subprocess.run([PY, "-c", probe], capture_output=True, text=True)
    return None if out.returncode == 0 else out.stderr.strip().splitlines()[-1][:200]


def raise_descriptor_limit() -> None:
    """Give the run enough file descriptors for repeated DataLoader workers.

    A stage that trains many models in one process (the provenance sweep is 50)
    accumulates worker pipes faster than the default soft limit of 1024 allows,
    and dies with EMFILE part-way through. Children inherit this.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = min(65536, hard) if hard != resource.RLIM_INFINITY else 65536
    if soft < wanted:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (ValueError, OSError):
            print(f"could not raise the file-descriptor limit above {soft}")


def command(argv, outputs, kind, units, inputs):
    return {"argv": [str(a) for a in argv], "outputs": outputs, "kind": kind, "units": units, "inputs": inputs}


def adni_arm(name, split_dir, device, arch="resnet18", seeds=SEEDS):
    root = f"{RUNS}/{name}"
    return [command(
        [PY, "scripts/run_adni_inflation_gap.py", "--split-dir", split_dir, "--output-root", root,
         "--table", f"{TABLES}/adni/{name}_seed{s}.csv", "--seeds", s, "--epochs", 15,
         "--arch", arch, "--device", device, "--resume"],
        [f"{root}/inflation_gap_seed{s}/{p}/test_predictions.csv" for p in PROTOCOLS]
        + [f"{TABLES}/adni/{name}_seed{s}.csv"],
        f"adni_{arch}", 3, [f"{split_dir}/adni_splitguard_seed{s}.csv"]) for s in seeds]


def tier1(arch, rules, device):
    """``rules`` pairs a grouping rule with whether the random split is trained beside it."""
    root = f"{RUNS}/tier1" if arch == "resnet18" else f"{RUNS}/tier1_{arch}"
    out = []
    for seed in TIER_SEEDS:
        for rule, with_random in rules:
            split = f"data/splits/tier1_truth/tier1_{rule}_seed{seed}.csv"
            labels = ["random", rule] if with_random else [rule]
            out.append(command(
                [PY, "scripts/run_inflation_gap.py", "--split", split, "--safe-label", rule,
                 *(["--leaky-label", "random"] if with_random else ["--skip-leaky"]),
                 "--runs-root", root, "--output", f"{TABLES}/tier1/tier1_{arch}_{rule}_seed{seed}.json",
                 "--seed", seed, "--epochs", 30, "--arch", arch, "--num-workers", 8, "--device", device],
                [f"{root}/inflation_gap_seed{seed}/{label}/test_predictions.csv" for label in labels]
                + [f"{TABLES}/tier1/tier1_{arch}_{rule}_seed{seed}.json"],
                f"tier1_{arch}", len(labels), [split]))
    return out


def tier2(device):
    split = "data/splits/oasis1_splitguard_seed42.csv"
    out = []
    for seed in TIER_SEEDS:
        for arch in ("resnet18", "densenet121"):
            root = f"{RUNS}/oasis1" if arch == "resnet18" else f"{RUNS}/oasis1_{arch}"
            out.append(command(
                [PY, "scripts/run_oasis1_inflation_gap.py", "--split", split, "--seed", seed, "--epochs", 20,
                 "--arch", arch, "--runs-root", root, "--output", f"{TABLES}/oasis1/oasis1_{arch}_seed{seed}.json",
                 "--num-workers", 8, "--device", device],
                [f"{root}/inflation_gap_seed{seed}/{p}/test_predictions.csv" for p in ("random", "component_safe")]
                + [f"{TABLES}/oasis1/oasis1_{arch}_seed{seed}.json"],
                f"tier2_{arch}", 2, [split]))
    return out


def stages(device, volumes):
    primary = [f"data/splits/adni/adni_splitguard_seed{s}.csv" for s in SEEDS]
    probe_rows = "data/splits/adni_with_converters/adni_splitguard_seed0.csv"
    return {
        "tier1_truth": tier1("resnet18", [("filename_subject", True), ("true_participant", False)], device),
        "adni_primary": adni_arm("adni", "data/splits/adni", device),
        "adni_converters": adni_arm("adni_with_converters", "data/splits/adni_with_converters", device),
        "adni_no_mt1": adni_arm("adni_no_mt1", "data/splits/adni_no_mt1", device),
        "adni_densenet121": adni_arm("adni_densenet121", "data/splits/adni", device, arch="densenet121"),
        "tier2_oasis1": tier2(device),
        "adni_null": [command(
            [PY, "scripts/run_adni_permutation_null.py", "--output-root", f"{RUNS}/adni_permutation_null",
             "--summary", f"{TABLES}/adni/adni_permutation_null.json", "--seeds", *SEEDS, "--epochs", 15,
             "--device", device],
            [f"{TABLES}/adni/adni_permutation_null.json"], "adni_resnet18", 15, primary)],
        "adni_size_balanced": adni_arm("adni_size_balanced", "data/splits/adni_size_balanced", device),
        "adni_exact_linkage": adni_arm("adni_exact_linkage", "data/splits/adni_exact_linkage", device),
        # needs the adni_converters checkpoints; "session" holds out a whole visit per participant
        "identity_probe": [command(
            [PY, "scripts/run_biometric_probe_adni.py", "--manifest", probe_rows,
             "--checkpoint-glob", f"{RUNS}/adni_with_converters/inflation_gap_seed*/*/best_state.pt",
             "--hold-out", mode, "--output", f"{TABLES}/adni/adni_biometric_probe_{mode}.json"],
            [f"{TABLES}/adni/adni_biometric_probe_{mode}.json"], "adni_probe", 15,
            [probe_rows, f"{RUNS}/adni_with_converters/inflation_gap_seed*/*/best_state.pt"])
            for mode in ("image", "session")],
        "adni_provenance": [command(
            [PY, "scripts/run_provenance_degradation.py", "--output-root", f"{RUNS}/adni_provenance_degradation",
             "--summary", f"{TABLES}/adni/adni_provenance_degradation.json", "--seeds", *SEEDS, "--epochs", 15,
             "--device", device],
            [f"{TABLES}/adni/adni_provenance_degradation.json"], "adni_resnet18", 50, primary)],
        "adni_3d": [command(
            [PY, "scripts/train_adni_3d.py", "--manifest", split, "--split", split, "--protocol", p,
             "--seed", seed, "--output-dir", f"{RUNS}/adni_3d/inflation_gap_seed{seed}/{p}",
             "--preprocessed-root", volumes, "--epochs", 15],
            [f"{RUNS}/adni_3d/inflation_gap_seed{seed}/{p}/test_predictions.csv"], "adni_3d", 1, [split])
            for seed in SEEDS for p in PROTOCOLS
            for split in [f"data/splits/adni_3d/adni_splitguard_seed{seed}.csv"]],
        # The contamination half of the stress test needs no model and is
        # already computed; this is its AUROC consequence, one training per
        # (mechanism, intensity, protocol, seed) cell.
        "adni_stress": [command(
            [PY, "scripts/train_adni_baseline.py", "--split", str(s), "--output-root",
             f"{RUNS}/adni_stress/{s.stem}", "--seed", 0, "--epochs", 15,
             "--arch", "resnet18", "--device", device],
            [f"{RUNS}/adni_stress/{s.stem}/baseline_seed0/test_predictions.csv"],
            "adni_resnet18", 1, [str(s)])
            for s in sorted((ROOT / "data" / "splits" / "adni_stress").glob("*.csv"))],
        "adni_dose_response": [command(
            [PY, "scripts/train_adni_baseline.py", "--split", split, "--output-root", run_dir, "--seed", seed,
             "--epochs", 15, "--arch", arch, "--device", device],
            [f"{run_dir}/baseline_seed{seed}/test_predictions.csv"], f"adni_{arch}", 1, [split])
            for seed in SEEDS for overlap in OVERLAPS for arch in ("resnet18", "densenet121")
            for split, run_dir in [(f"data/splits/adni_dose_response/seed{seed}_overlap{overlap}.csv",
                                    f"{RUNS}/adni_dose_response/{arch}/seed{seed}_overlap{overlap}")]],
    }


def smoke_variants(commands: list[dict]) -> list[dict]:
    """One command per distinct variant, so smoke exercises every code path.

    Taking only the first command of a stage leaves the protocols, backbones,
    grouping rules and hold-out modes that appear later in the stage untested,
    which is exactly where a stage-specific failure hides.
    """
    def variant(c):
        argv = c["argv"]
        keys = [c["kind"]]
        for flag in ("--protocol", "--safe-label", "--hold-out", "--arch"):
            keys.append(argv[argv.index(flag) + 1] if flag in argv else "")
        keys.append("skip-leaky" if "--skip-leaky" in argv else "")
        return tuple(keys)

    chosen: dict[tuple, dict] = {}
    for c in commands:
        chosen.setdefault(variant(c), c)
    return list(chosen.values())


def smoke_version(c):
    """One epoch, first seed only, written under runs_smoke/ and reports/smoke/."""
    def move(text):
        return text.replace(RUNS, "runs_smoke").replace(TABLES, "reports/smoke")
    argv = [move(a) for a in c["argv"]]
    if "--epochs" in argv:
        argv[argv.index("--epochs") + 1] = "1"
    if "--seeds" in argv:
        first = argv.index("--seeds") + 1
        end = first + 1
        while end < len(argv) and not argv[end].startswith("--"):
            end += 1
        del argv[first + 1:end]
    if "scripts/run_provenance_degradation.py" in argv:
        argv += ["--levels", "0.5"]   # one deletion level exercises the whole code path
    return {**c, "argv": argv, "outputs": [move(o) for o in c["outputs"]]}


def done(c):
    return all(glob.glob(str(ROOT / pattern)) for pattern in c["outputs"])


def missing_inputs(commands, volumes):
    """Inputs each command needs but cannot find, keyed by stage.

    An entry containing ``*`` is a dependency another stage produces (the
    identity probe's checkpoints): it only has to match something.
    """
    missing: dict[str, list[str]] = {}
    checked_images: set[str] = set()
    checked_volumes: set[str] = set()
    for stage, c in commands:
        for split in c["inputs"]:
            if "*" in split:
                if not glob.glob(str(ROOT / split)):
                    missing.setdefault(stage, []).append(f"nothing matches {split}")
                continue
            path = ROOT / split
            if not path.is_file():
                missing.setdefault(stage, []).append(split)
                continue
            volumetric = c["kind"] == "adni_3d"
            done_set = checked_volumes if volumetric else checked_images
            if split in done_set:
                continue
            done_set.add(split)
            with path.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    needed = (ROOT / volumes / row["subject_id"] / f"{row['image_uid']}.npz"
                              if volumetric else ROOT / row["relative_path"])
                    if not needed.is_file():
                        missing.setdefault(stage, []).append(f"{split}: {needed}")
    return missing


def log(record):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}) + "\n")


def pack(archive, runs):
    with tarfile.open(archive, "w:gz") as tar:
        for base in (runs, TABLES, "runs/gpu_environment.txt", "runs/gpu_program_log.jsonl"):
            path = ROOT / base
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
            for f in files:
                if f.suffix in (".pt", ".pth", ".ckpt"):
                    continue
                arcname = f.relative_to(ROOT).as_posix()
                if arcname.startswith("runs/"):      # keep the run's own provenance with it
                    arcname = f"{runs}/{f.name}"
                tar.add(f, arcname=arcname)


SLOTS = {"adni_3d": 4}          # a 128^3 volume batch costs several times a slice batch


def launch(name: str, index: int, c: dict, runs: str) -> tuple:
    """Start one command detached, with its own log file."""
    for flag in OUTPUT_FLAGS:
        if flag in c["argv"]:
            (ROOT / c["argv"][c["argv"].index(flag) + 1]).parent.mkdir(parents=True, exist_ok=True)
    logfile = ROOT / runs / "logs" / f"{name}_{index:02d}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    handle = logfile.open("w", encoding="utf-8")
    proc = subprocess.Popen(c["argv"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return proc, handle, logfile, time.time()


def run_in_parallel(pending, args, over_cap, spent, save, measured, counts):
    """Run independent commands concurrently on one node.

    Order is preserved for admission, so the priority in DEFAULT_ORDER still
    decides what runs first when the budget is tight. A command whose inputs
    name a glob depends on outputs another stage writes (the identity probe on
    its checkpoints), so it waits for everything already running.
    """
    runs = "runs_smoke" if args.smoke else RUNS
    queue = list(pending)
    running: list[dict] = []
    used_slots = 0
    while queue or running:
        for entry in list(running):
            if entry["proc"].poll() is None:
                continue
            running.remove(entry)
            used_slots -= entry["slots"]
            entry["handle"].close()
            minutes = (time.time() - entry["t0"]) / 60
            c = entry["c"]
            ok = entry["proc"].returncode == 0 and done(c)
            if ok:
                measured.setdefault(c["kind"], []).append(minutes / c["units"])
            else:
                for pattern in c["outputs"]:
                    for stale in glob.glob(str(ROOT / pattern)):
                        if Path(stale).stat().st_mtime >= entry["t0"]:
                            Path(stale).unlink()
            counts["ok" if ok else "failed"] += 1
            log({"stage": entry["name"], "index": entry["index"], "status": "ok" if ok else "failed",
                 "rc": entry["proc"].returncode, "minutes": round(minutes, 2),
                 "spent_usd": round(spent(), 2),
                 "log": entry["logfile"].relative_to(ROOT).as_posix()})
            print(f"{entry['name']}[{entry['index']}] {'ok' if ok else 'FAILED'} in {minutes:.1f} min "
                  f"({len(running)} still running, spent ${spent():.2f})", flush=True)

        while queue and not over_cap():
            name, index, c = queue[0]
            slots = SLOTS.get(c["kind"], 1)
            if used_slots and used_slots + slots > args.workers:
                break
            if any("*" in i for i in c["inputs"]) and running:
                break                                   # wait for the stage it depends on
            queue.pop(0)
            proc, handle, logfile, t0 = launch(name, index, c, runs)
            running.append({"name": name, "index": index, "c": c, "proc": proc, "handle": handle,
                            "logfile": logfile, "t0": t0, "slots": slots})
            used_slots += slots
            print(f"{name}[{index}] started ({len(running)} running, spent ${spent():.2f})", flush=True)

        if over_cap() and running:
            for entry in running:
                os.killpg(entry["proc"].pid, signal.SIGTERM)
            for entry in running:
                try:
                    entry["proc"].wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(entry["proc"].pid, signal.SIGKILL)
                    entry["proc"].wait()
                entry["handle"].close()
                counts["stopped_at_cap"] += 1
            counts["skipped_budget"] += len(queue)
            print(f"budget cap reached: stopped {len(running)} running and skipped {len(queue)} queued")
            break
        if over_cap() and not running and queue:
            counts["skipped_budget"] += len(queue)
            print(f"cap reached with nothing running: {len(queue)} command(s) not started")
            break
        time.sleep(2)
        save()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd-per-hour", type=float, default=0.69)
    ap.add_argument("--budget-usd", type=float, default=10.0)
    ap.add_argument("--reserve-usd", type=float, default=1.5,
                    help="Held back for start-up, setup, upload and download.")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="Wall-clock cap in minutes; overrides the money cap when the node "
                         "is rented for a fixed window rather than by the hour.")
    ap.add_argument("--only", nargs="+", default=None, help="Run only these stages, in this order.")
    ap.add_argument("--preprocessed-root", default="data/adni3d_128",
                    help="Directory of <subject_id>/<image_uid>.npz volumes, relative to the repository root.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1,
                    help="Commands to run at once. A ResNet-18 at 224px needs a few GB, so a "
                         "large GPU can host many; volumetric runs take four slots each.")
    ap.add_argument("--list", action="store_true", help="Show pending work and projected cost, run nothing.")
    ap.add_argument("--smoke", action="store_true", help="First command of each stage, one epoch, one seed.")
    args = ap.parse_args()
    raise_descriptor_limit()

    catalogue = stages(args.device, args.preprocessed_root)
    order = args.only or DEFAULT_ORDER
    unknown = [name for name in order if name not in catalogue]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {sorted(catalogue)}")
    runs = "runs_smoke" if args.smoke else RUNS

    plan = []
    for name in order:
        if name == "adni_3d" and not (ROOT / args.preprocessed_root).is_dir():
            print(f"adni_3d: {args.preprocessed_root} not found, stage left out")
            continue
        commands = ([smoke_version(c) for c in smoke_variants(catalogue[name])]
                    if args.smoke else catalogue[name])
        plan += [(name, i, c) for i, c in enumerate(commands)]
    if args.smoke:
        # A second smoke run must retest, not report zero pending work.
        for stale in (ROOT / "runs_smoke", ROOT / "reports" / "smoke"):
            shutil.rmtree(stale, ignore_errors=True)
    pending = [(name, i, c) for name, i, c in plan if not done(c)]
    cap = args.budget_usd - args.reserve_usd
    cap_seconds = (args.time_budget_min * 60 if args.time_budget_min
                   else cap / args.usd_per_hour * 3600)

    if args.list:
        total = 0.0
        print(f"{'stage':<20}{'commands':>9}{'trainings':>10}{'pending':>9}{'est. min':>10}{'est. USD':>10}")
        for name in dict.fromkeys(n for n, _, _ in plan):
            cs = [c for n, _, c in plan if n == name]
            todo = [c for n, _, c in pending if n == name]
            minutes = sum(MINUTES[c["kind"]] * c["units"] for c in todo)
            total += minutes / 60 * args.usd_per_hour
            print(f"{name:<20}{len(cs):>9}{sum(c['units'] for c in cs):>10}{len(todo):>9}"
                  f"{minutes:>10.0f}{minutes / 60 * args.usd_per_hour:>10.2f}")
        print(f"projected ${total:.2f} at ${args.usd_per_hour}/h "
              f"({total / args.usd_per_hour * 60:.0f} min sequential, "
              f"{total / args.usd_per_hour * 60 / max(1, args.workers):.0f} min at "
              f"--workers {args.workers}); cap {cap_seconds / 60:.0f} min")
        return 0

    deps = missing_modules({c["kind"] for _, _, c in pending})
    if deps["missing"]:
        print(f"missing Python modules for the selected stages: {' '.join(deps['missing'])}")
        print("install them without touching the image's CUDA build of torch, for example:")
        print(f"  pip install --break-system-packages --no-deps {' '.join(deps['missing'])}")
        return 2
    if args.device.startswith("cuda"):
        broken = cuda_is_healthy()
        if broken:
            print(f"torch cannot use the GPU: {broken}")
            print("a pip install that pulled its own torch usually causes this; reinstall the "
                  "image's build, for example:\n  pip install --break-system-packages "
                  "torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128")
            return 2

    missing = missing_inputs([(name, c) for name, _, c in pending], args.preprocessed_root)
    if missing:
        for stage, items in missing.items():
            print(f"{stage}: {len(items)} input(s) missing, stage skipped. First: {items[0]}")
        pending = [(name, i, c) for name, i, c in pending if name not in missing]
        if not pending:
            print("no stage has all of its inputs; nothing to run")
            return 2

    ledger = ROOT / RUNS / "gpu_ledger.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        billed = json.loads(ledger.read_text())["billed_seconds"] if ledger.exists() else 0.0
    except (ValueError, OSError, KeyError):
        print(f"{ledger.name} is unreadable; restarting the meter from zero")
        billed = 0.0
    started = time.time()

    def seconds_spent():
        return billed + time.time() - started

    def over_cap():
        """True when this invocation must stop starting work.

        --time-budget-min is the wall clock of this run, because a node is
        rented for a window; the money cap is cumulative, because a budget is
        spent once. Mixing the two is what made a second invocation believe it
        was already over budget before it started anything.
        """
        if args.time_budget_min:
            return (time.time() - started) >= args.time_budget_min * 60
        return seconds_spent() >= cap_seconds

    def fits(projected_seconds):
        if args.time_budget_min:
            return (time.time() - started) + projected_seconds <= args.time_budget_min * 60
        return seconds_spent() + projected_seconds <= cap_seconds

    def spent():
        return seconds_spent() / 3600 * args.usd_per_hour

    def save():
        # Atomic: a kill during the write would otherwise leave JSON the next
        # invocation cannot parse, on a node that is still billing.
        temporary = ledger.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"billed_seconds": round(billed + time.time() - started, 1),
                                         "usd_per_hour": args.usd_per_hour}) + "\n")
        os.replace(temporary, ledger)

    measured: dict[str, list[float]] = {}
    counts = {"ok": 0, "failed": 0, "skipped_budget": 0, "stopped_at_cap": 0}
    if args.workers > 1:
        counts = run_in_parallel(pending, args, over_cap, spent, save, measured, counts)
        save()
        print(f"{counts} of {len(pending)} pending; spent ${spent():.2f} of cap ${cap:.2f}")
        if not args.smoke:
            archive = ROOT / "gpu_results.tar.gz"
            pack(archive, runs)
            print(f"results (no weights) in {archive.name}")
        return 1 if (counts["failed"] or counts["skipped_budget"] or counts["stopped_at_cap"]) else 0

    for name, i, c in pending:
        per = measured.get(c["kind"])
        projected = (sum(per) / len(per) if per else MINUTES[c["kind"]]) * c["units"] / 60 * args.usd_per_hour
        if not fits(projected / args.usd_per_hour * 3600):
            counts["skipped_budget"] += 1
            log({"stage": name, "index": i, "status": "skipped_budget", "spent_usd": round(spent(), 2),
                 "projected_usd": round(projected, 2)})
            print(f"{name}[{i}]: projected ${projected:.2f} does not fit (spent ${spent():.2f} of ${cap:.2f})")
            continue
        for flag in OUTPUT_FLAGS:
            if flag in c["argv"]:
                (ROOT / c["argv"][c["argv"].index(flag) + 1]).parent.mkdir(parents=True, exist_ok=True)
        logfile = ROOT / runs / "logs" / f"{name}_{i:02d}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name}[{i}] started (spent ${spent():.2f}, projected ${projected:.2f})", flush=True)
        t0, status = time.time(), None
        with logfile.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(c["argv"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            while proc.poll() is None:
                time.sleep(10)
                save()
                if over_cap():
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    status = "stopped_at_cap"
        minutes = (time.time() - t0) / 60
        ok = proc.returncode == 0 and done(c)
        status = status or ("ok" if ok else "failed")
        if not ok:
            # A command killed or failed mid-write can leave a truncated
            # predictions file, which a later run would mistake for a finished
            # stage. Remove anything this command touched after it started.
            for pattern in c["outputs"]:
                for stale in glob.glob(str(ROOT / pattern)):
                    if Path(stale).stat().st_mtime >= t0:
                        Path(stale).unlink()
        if ok:
            measured.setdefault(c["kind"], []).append(minutes / c["units"])
        counts[status] += 1
        save()
        log({"stage": name, "index": i, "status": status, "rc": proc.returncode, "minutes": round(minutes, 2),
             "spent_usd": round(spent(), 2), "log": logfile.relative_to(ROOT).as_posix()})
        print(f"{name}[{i}] {status} in {minutes:.1f} min, log {logfile.relative_to(ROOT)}", flush=True)
        if status == "stopped_at_cap":
            break

    save()
    print(f"{counts} of {len(pending)} pending; spent ${spent():.2f} of cap ${cap:.2f}")
    if not args.smoke:
        archive = ROOT / "gpu_results.tar.gz"
        pack(archive, runs)
        print(f"results (no weights) in {archive.name}")
    if counts["failed"] or counts["skipped_budget"] or counts["stopped_at_cap"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
