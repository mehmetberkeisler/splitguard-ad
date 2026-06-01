#!/usr/bin/env python3
"""WP6 biometric-persistence probe on ADNI.

Replicates the Tier-1 / Tier-2 mechanism experiment (subject-identity
decodability from frozen penultimate-layer features) on the ADNI
cohort. Uses checkpoints persisted by the patched
``scripts/train_adni_baseline.py`` (one checkpoint per
seed × protocol under
``runs/<output_root>/inflation_gap_seed{S}/{label}/best_state.pt``).

Within-subject probe split: 80% of each subject's images go to the
probe-train fold, 20% to probe-test. Same subjects appear in both
folds, so the test does not require generalising to unseen subjects.
The classifier is a multinomial LinearSVC (fast, well-suited to
hundreds of subject classes).

Inputs
------
* ``--manifest``: split CSV (CN+AD rows; one row per scan with
  ``image_path``, ``subject_id``).
* ``--checkpoint-glob``: glob-pattern picking one or more
  ``best_state.pt`` checkpoints. Each is probed independently.

Output
------
* JSON summary at the specified path with per-checkpoint
  probe accuracy, chance level, lift, and aggregate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def choose_device():
    import torch
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_model_frozen(ckpt_path: Path, dev, arch: str = "resnet18"):
    import torch
    import torch.nn as nn
    from torchvision import models
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        feature_attr = "avgpool"
    elif arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        feature_attr = "features"  # pool the final feature map manually
    else:
        raise ValueError(f"unsupported arch {arch!r}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m.to(dev), feature_attr


def extract_features(model, feature_attr: str, rows, dev, batch: int = 32):
    """Hook avgpool (resnet18) or final feature map (densenet121) and pool."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ImgDS(Dataset):
        def __getitem__(self, i):
            return tf(Image.open(rows[i]["image_path"]).convert("RGB"))
        def __len__(self): return len(rows)

    hook_out = []
    if feature_attr == "avgpool":
        hook = model.avgpool.register_forward_hook(
            lambda m, i, o: hook_out.append(o.detach().cpu())
        )
    else:
        # densenet121: hook the final features block; we'll global-pool ourselves.
        hook = model.features.register_forward_hook(
            lambda m, i, o: hook_out.append(
                torch.nn.functional.adaptive_avg_pool2d(o.detach().cpu(), 1)
            )
        )

    dl = DataLoader(ImgDS(), batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs in dl:
            model(imgs.to(dev))
    hook.remove()

    X = torch.cat(hook_out).squeeze(-1).squeeze(-1).numpy()
    return X


def probe_subject_id(X, rows, label: str, seed: int = 42):
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import LinearSVC

    by_subj: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_subj[r["subject_id"]].append(i)

    rng = random.Random(seed)
    tr_idx: list[int] = []
    te_idx: list[int] = []
    # Only include subjects with >= 2 scans so the probe-train/test split is non-trivial.
    eligible = [s for s, idxs in by_subj.items() if len(idxs) >= 2]
    for subj in eligible:
        idxs = by_subj[subj][:]
        rng.shuffle(idxs)
        n_tr = max(1, int(0.8 * len(idxs)))
        tr_idx.extend(idxs[:n_tr])
        te_idx.extend(idxs[n_tr:])

    le = LabelEncoder()
    all_subj = [r["subject_id"] for r in rows]
    le.fit(all_subj)
    y = le.transform(all_subj)
    n_classes = len(le.classes_)
    n_eligible = len(eligible)
    chance = 1.0 / n_eligible if n_eligible else float("nan")

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr_idx])
    Xte = sc.transform(X[te_idx])
    ytr = y[tr_idx]; yte = y[te_idx]

    clf = LinearSVC(max_iter=2000, C=0.1, dual="auto")
    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    lift = round(acc / chance, 2) if chance else float("nan")
    print(f"  {label}: probe_acc={acc:.4f}  chance={chance:.6f}  "
          f"lift={lift}x  n_eligible_subjects={n_eligible}  "
          f"n_train_imgs={len(tr_idx)}  n_test_imgs={len(te_idx)}")
    return {
        "checkpoint_label": label,
        "probe_acc": round(acc, 4),
        "chance": round(chance, 6),
        "lift_over_chance": lift,
        "n_eligible_subjects": n_eligible,
        "n_train_imgs": len(tr_idx),
        "n_test_imgs": len(te_idx),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "adni_with_converters" /
                "adni_splitguard_seed0.csv",
        help="Split CSV with image_path + subject_id columns.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "adni_with_converters" /
                    "inflation_gap_seed0" / "*" / "best_state.pt"),
    )
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reports" / "tables" / "adni" /
                                "adni_biometric_probe.json")
    parser.add_argument("--arch", default="resnet18",
                        choices=["resnet18", "densenet121"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = choose_device()
    print(f"Device: {dev}")

    rows = list(csv.DictReader(args.manifest.open()))
    # Restrict to CN+AD rows (the binary universe) for clean probe semantics.
    rows = [r for r in rows if r.get("diagnosis_group") in ("CN", "AD")]
    print(f"Probe corpus rows: {len(rows)}  "
          f"unique subjects: {len({r['subject_id'] for r in rows})}")

    ckpts = sorted(glob.glob(args.checkpoint_glob))
    if not ckpts:
        raise SystemExit(f"No checkpoints matched {args.checkpoint_glob!r}.")
    print(f"Checkpoints to probe: {len(ckpts)}")
    for c in ckpts:
        print(f"  {c}")

    import numpy as np  # noqa: F401 (touch numpy now for fail-fast)

    results = []
    for ckpt_path in ckpts:
        ckpt_path = Path(ckpt_path)
        label_parts = ckpt_path.parts
        # protocol name is the parent dir of best_state.pt
        protocol = ckpt_path.parent.name
        seed_dir = ckpt_path.parent.parent.name  # e.g. inflation_gap_seed3
        label = f"{seed_dir}/{protocol}"
        print(f"\n  Extracting features: {label}")
        model, feature_attr = make_model_frozen(ckpt_path, dev, arch=args.arch)
        X = extract_features(model, feature_attr, rows, dev)
        print(f"  Features shape: {X.shape}")
        r = probe_subject_id(X, rows, label=label, seed=args.seed)
        r["checkpoint_path"] = str(ckpt_path.resolve())
        results.append(r)

    # Group by protocol
    by_proto: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        proto = r["checkpoint_label"].split("/")[-1]
        by_proto[proto].append(r)
    per_proto_summary = {}
    for proto, rs in by_proto.items():
        lifts = [r["lift_over_chance"] for r in rs if r["lift_over_chance"] == r["lift_over_chance"]]
        per_proto_summary[proto] = {
            "n_checkpoints": len(rs),
            "lift_mean": round(sum(lifts) / len(lifts), 2) if lifts else None,
            "lift_min": min(lifts) if lifts else None,
            "lift_max": max(lifts) if lifts else None,
        }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "checkpoint_glob": args.checkpoint_glob,
        "arch": args.arch,
        "results_per_checkpoint": results,
        "summary_per_protocol": per_proto_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Per-protocol probe summary ===")
    for proto, s in per_proto_summary.items():
        print(f"  {proto:<18s}  mean lift {s['lift_mean']}x   "
              f"range [{s['lift_min']}x, {s['lift_max']}x] "
              f"(n_ckpt={s['n_checkpoints']})")
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
