#!/usr/bin/env python3
"""
E2 Fix: Subject-ID Probe (corrected within-subject image split, fast solver)
=============================================================================
Bug in overnight_runner: probe split test subjects that were *unseen*
by the logistic regression → probe_acc=0.0 by construction.

Fix: split images within each subject (80% train / 20% test for the probe).
Same subjects appear in both probe-train and probe-test, just different images.

Uses LinearSVC (much faster than multinomial LR for 196 classes).

This correctly tests: "do frozen features discriminate subject identity?"
Expected:
  Both models should show high subject-identity decodability if 2D MRI
  representations carry strong biometric structure. The relative leaky/safe
  magnitude is not the core claim.
"""
import csv, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT     = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
CKPT_DIR = ROOT / "runs" / "checkpoints" / "overnight"
OUT      = ROOT / "reports" / "tables" / "e2_probe_results.json"

def choose_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def make_model_frozen(ckpt_path, dev):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    ckpt = torch.load(ckpt_path, map_location=dev)
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m.to(dev)

def extract_features(model, rows, dev, batch=64):
    """Extract 512-dim avgpool features for all rows."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    class ImgDS(Dataset):
        def __getitem__(self, i):
            return tf(Image.open(rows[i]["path"]).convert("RGB"))
        def __len__(self): return len(rows)

    feats = []
    hook_out = []
    hook = model.avgpool.register_forward_hook(
        lambda m,i,o: hook_out.append(o.detach().cpu()))

    dl = DataLoader(ImgDS(), batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs in dl:
            model(imgs.to(dev))

    hook.remove()
    # Each hook_out element: (batch, 512, 1, 1)
    X = torch.cat(hook_out).squeeze(-1).squeeze(-1).numpy()  # (N, 512)
    return X

def probe_subject_id(X, rows, label="model", seed=42):
    """
    Within-subject image split:
    80% of each subject's images go to probe train.
    20% of each subject's images go to probe test.
    Same subjects appear in both sets, so the classifier can generalise across
    images from known subjects.
    Uses LinearSVC (converges in seconds vs minutes for multinomial LR).
    """
    by_subj = defaultdict(list)
    for i, r in enumerate(rows):
        by_subj[r["subject_id"]].append(i)

    rng = random.Random(seed)
    tr_idx, te_idx = [], []
    for subj, idxs in by_subj.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n = max(1, int(0.8 * len(shuffled)))
        tr_idx.extend(shuffled[:n])
        te_idx.extend(shuffled[n:])

    le = LabelEncoder()
    all_subj = [r["subject_id"] for r in rows]
    le.fit(all_subj)
    y = le.transform(all_subj)
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr_idx])
    Xte = sc.transform(X[te_idx])
    ytr = y[tr_idx]; yte = y[te_idx]

    # LinearSVC is much faster than multinomial LR for 196 classes.
    clf = LinearSVC(max_iter=2000, C=0.1, dual=True)
    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    lift = round(acc / chance, 2)

    result = dict(model=label, probe_acc=round(acc, 4),
                  chance=round(chance, 6), lift_over_chance=lift,
                  n_subjects=n_classes, n_train_imgs=len(tr_idx),
                  n_test_imgs=len(te_idx))
    print(f"  {label}: probe_acc={acc:.4f}  chance={chance:.6f}  lift={lift}x  n_subjects={n_classes}")
    return result


def main():
    dev = choose_device()
    print(f"Device: {dev}")

    # Load manifest — only high-confidence subject rows
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f)
                    if r["subject_id_confidence"] == "high_filename_parentheses"]
    print(f"High-confidence rows: {len(all_rows)}  "
          f"unique subjects: {len({r['subject_id'] for r in all_rows})}")

    results = []
    for ckpt_name, label in [
        ("leaky_seed0.pt", "A_leaky_seed0"),
        ("safe_seed0.pt",  "B_safe_seed0"),
    ]:
        ckpt = CKPT_DIR / ckpt_name
        if not ckpt.exists():
            print(f"  ⚠ {ckpt} not found — skipping"); continue
        print(f"\n  Extracting features: {label}")
        model = make_model_frozen(ckpt, dev)
        X = extract_features(model, all_rows, dev)
        print(f"  Features shape: {X.shape}")
        r = probe_subject_id(X, all_rows, label=label)
        results.append(r)

    # Gap summary
    if len(results) == 2:
        gap = round(results[0]["lift_over_chance"] - results[1]["lift_over_chance"], 2)
        print(f"\n  Lift gap (leaky - safe): {gap}x")
        print(f"  -> Leaky model encodes {results[0]['lift_over_chance']}x above-chance "
              f"subject identity; safe model {results[1]['lift_over_chance']}x")

    OUT.write_text(json.dumps({"e2_probe": results}, indent=2))
    print(f"\n  Results saved -> {OUT}")
    return results

if __name__ == "__main__":
    main()
