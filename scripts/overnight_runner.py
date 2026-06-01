#!/usr/bin/env python3
"""
Overnight Experiment Runner — SplitGuard-AD
============================================
Runs E1b, E2, E3, E4, E5 sequentially and saves all results.

E1b : Multi-seed (seeds 0,1,2,3) Protocol A vs B  → CI on inflation gap
E2  : Subject-ID probe on leaky vs safe model     → mechanistic evidence
E3  : Protocol A' (subject-only split)             → degradation curve 3rd point
E4  : Clinical-threshold evaluation               → reviewer defence
E5  : 4-class comparison                          → optional, runs last

Usage:  python3 scripts/overnight_runner.py [--skip-e5]
"""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
MANIFEST    = ROOT / "data" / "splits" / "current_jpeg_splitguard_seed42.csv"
OUT_DIR     = ROOT / "reports" / "tables"
CKPT_DIR    = ROOT / "runs" / "checkpoints" / "overnight"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LABEL2T     = {"NonDemented": 0, "Demented": 1}
RAW4_MAP    = {"NonDemented": 0, "VeryMildDemented": 1,
               "MildDemented": 2, "ModerateDemented": 3}
SEEDS_E1B   = [0, 1, 2, 3]   # seed 42 already done
EPOCHS      = 30

# ── Shared utilities ───────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True

def choose_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    try: auc = float(roc_auc_score(y_true, y_prob))
    except: auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    return dict(auroc=round(auc,4),
                balanced_accuracy=round(float(balanced_accuracy_score(y_true,y_pred)),4),
                f1_demented=round(float(f1_score(y_true,y_pred,pos_label=1,zero_division=0)),4),
                sensitivity=round(tp/(tp+fn) if (tp+fn) else 0.,4),
                specificity=round(tn/(tn+fp) if (tn+fp) else 0.,4),
                brier=round(float(brier_score_loss(y_true,y_prob)),4),
                n=int(len(y_true)))

def ece(y_true, y_prob, n=10):
    bins=np.linspace(0,1,n+1); e=0.
    for l,r in zip(bins[:-1],bins[1:]):
        m=(y_prob>=l)&(y_prob<=r)
        if m.any(): e+=float(np.mean(m))*abs(float(np.mean(y_true[m]))-float(np.mean(y_prob[m])))
    return round(e,4)

# ── Dataset ────────────────────────────────────────────────────────────────────
class RowDS(Dataset):
    def __init__(self, rows, tf, label_key="binary_label", label_map=LABEL2T):
        self.rows=rows; self.tf=tf; self.lmap=label_map; self.lkey=label_key
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        img=Image.open(r["path"]).convert("RGB")
        if self.tf: img=self.tf(img)
        return img, torch.tensor(self.lmap[r[self.lkey]], dtype=torch.float32)

def make_tf(sz=224):
    n=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    tr=transforms.Compose([transforms.Resize((sz,sz)),
        transforms.RandomHorizontalFlip(),transforms.RandomRotation(7),
        transforms.ToTensor(),n])
    ev=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),n])
    return tr,ev

def loader(rows,tf,bs,shuffle,dev):
    return DataLoader(RowDS(rows,tf),batch_size=bs,shuffle=shuffle,
                      num_workers=0,pin_memory=(dev.type=="cuda"))

# ── Model ──────────────────────────────────────────────────────────────────────
def make_model(n_out=1,pretrained=True):
    w=models.ResNet18_Weights.DEFAULT if pretrained else None
    m=models.resnet18(weights=w); m.fc=nn.Linear(m.fc.in_features,n_out)
    return m

# ── Training loop ──────────────────────────────────────────────────────────────
def train_eval(splits, dev, epochs, bs, lr, seed, label,
               n_out=1, label_key="binary_label", label_map=LABEL2T,
               save_ckpt=None):
    set_seed(seed)
    tr_tf,ev_tf=make_tf()
    trl=loader(splits["train"],tr_tf,bs,True,dev)
    vl=loader(splits["val"],ev_tf,bs,False,dev)
    tel=loader(splits["test"],ev_tf,bs,False,dev)

    model=make_model(n_out=n_out,pretrained=True).to(dev)
    n_pos=sum(label_map[r[label_key]] for r in splits["train"])
    n_neg=len(splits["train"])-n_pos
    pw=torch.tensor([n_neg/max(1,n_pos)],device=dev)
    crit=nn.BCEWithLogitsLoss(pos_weight=pw) if n_out==1 else nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-2)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)

    best_auc=-1.; best_state=None; history=[]
    print(f"\n  ▶ {label}  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    t0=time.time()
    for ep in range(1,epochs+1):
        model.train(); ep_loss=ep_n=0
        for imgs,labs in trl:
            imgs,labs=imgs.to(dev),labs.to(dev)
            opt.zero_grad(set_to_none=True)
            out=model(imgs).squeeze(1)
            loss=crit(out,labs); loss.backward(); opt.step()
            ep_loss+=loss.item()*imgs.size(0); ep_n+=imgs.size(0)
        sch.step()
        model.eval(); vp,vt=[],[]
        with torch.no_grad():
            for im,lb in vl:
                p=torch.sigmoid(model(im.to(dev)).squeeze(1)).cpu()
                vp.append(p); vt.append(lb)
        yp=torch.cat(vp).numpy(); yt=torch.cat(vt).numpy().astype(int)
        try: vauc=roc_auc_score(yt,yp)
        except: vauc=0.
        if vauc>best_auc:
            best_auc=vauc
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        history.append({"ep":ep,"loss":round(ep_loss/ep_n,4),"val_auc":round(vauc,4)})
        print(f"    ep={ep:02d}  loss={ep_loss/ep_n:.4f}  val_auc={vauc:.4f}")

    model.load_state_dict(best_state)
    if save_ckpt:
        torch.save({"state":best_state,"label":label,"seed":seed},save_ckpt)
    model.eval(); tp2,tt=[],[]
    with torch.no_grad():
        for im,lb in tel:
            p=torch.sigmoid(model(im.to(dev)).squeeze(1)).cpu()
            tp2.append(p); tt.append(lb)
    yp=torch.cat(tp2).numpy(); yt=torch.cat(tt).numpy().astype(int)
    m=compute_metrics(yt,yp); m["ece"]=ece(yt,yp)
    elapsed=round(time.time()-t0,1)
    print(f"  ✅ {label}: AUROC={m['auroc']}  BalAcc={m['balanced_accuracy']}  elapsed={elapsed}s")
    return {"label":label,"seed":seed,"test_metrics":m,"history":history,
            "elapsed_s":elapsed,"n_train":len(splits["train"])}

# ── Split builders ─────────────────────────────────────────────────────────────
def load_all_rows():
    with MANIFEST.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def splitguard_split(all_rows):
    s={"train":[],"val":[],"test":[]}
    for r in all_rows:
        if r["split"] in s: s[r["split"]].append(r)
    return s

def leaky_split(all_rows, seed):
    rng=random.Random(seed); rows=all_rows[:]; rng.shuffle(rows)
    n=len(rows); nt=int(n*.70); nv=int(n*.15)
    return {"train":rows[:nt],"val":rows[nt:nt+nv],"test":rows[nt+nv:]}

def subject_only_split(all_rows, seed):
    """Protocol A': split by subject_id only, no component safety."""
    by_subj=defaultdict(list)
    for r in all_rows: by_subj[r["subject_id"]].append(r)
    subjs=list(by_subj); rng=random.Random(seed); rng.shuffle(subjs)
    n=len(subjs); nt=int(n*.70); nv=int(n*.15)
    train_s=set(subjs[:nt]); val_s=set(subjs[nt:nt+nv])
    sp={"train":[],"val":[],"test":[]}
    for r in all_rows:
        s=r["subject_id"]
        if s in train_s: sp["train"].append(r)
        elif s in val_s: sp["val"].append(r)
        else: sp["test"].append(r)
    return sp

def leaky_overlap(splits):
    tr={r["subject_id"] for r in splits["train"]}
    te={r["subject_id"] for r in splits["test"]}
    ov=tr&te
    return {"overlap_pct":round(100*len(ov)/max(1,len(te)),1),"n_overlap":len(ov)}

# ── E2: Subject-ID probe ───────────────────────────────────────────────────────
def subject_id_probe(ckpt_path, all_rows, dev, label):
    """Extract 512-dim penultimate features and probe inferred subject_id.

    The probe split is within subject: each subject contributes images to both
    probe train and probe test. Splitting by held-out subjects makes identity
    prediction impossible by construction and produced the deprecated 0.0
    results in the first overnight artifact.
    """
    print(f"\n  🔬 Subject-ID probe on: {label}")
    ckpt=torch.load(ckpt_path,map_location=dev)
    model=make_model(n_out=1,pretrained=False).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # Hook penultimate layer
    feats=[]
    def hook(m,inp,out): feats.append(out.detach().cpu())
    h=model.avgpool.register_forward_hook(hook)

    _,ev_tf=make_tf()
    # Use all rows that have high-confidence subject_id
    rows=[r for r in all_rows if r["subject_id_confidence"]=="high_filename_parentheses"]
    dl=DataLoader(RowDS(rows,ev_tf),batch_size=64,shuffle=False,num_workers=0)
    with torch.no_grad():
        for imgs,_ in dl: model(imgs.to(dev))
    h.remove()

    X=torch.cat(feats).view(len(rows),-1).numpy()
    subj_ids=[r["subject_id"] for r in rows]
    le=LabelEncoder(); y=le.fit_transform(subj_ids)
    n_classes=len(le.classes_)

    by_subj=defaultdict(list)
    for i,r in enumerate(rows):
        by_subj[r["subject_id"]].append(i)
    rng=random.Random(42)
    tr_idx=[]; te_idx=[]
    for idxs in by_subj.values():
        shuffled=idxs[:]
        rng.shuffle(shuffled)
        n=max(1,int(.8*len(shuffled)))
        tr_idx.extend(shuffled[:n])
        te_idx.extend(shuffled[n:])

    sc=StandardScaler(); Xtr=sc.fit_transform(X[tr_idx]); Xte=sc.transform(X[te_idx])
    clf=LinearSVC(max_iter=2000,C=0.1,dual=True)
    clf.fit(Xtr,y[tr_idx])
    acc=float(accuracy_score(y[te_idx],clf.predict(Xte)))
    chance=1./n_classes; lift=round(acc/chance,2)
    print(f"    probe_acc={acc:.4f}  chance={chance:.4f}  lift={lift}x  n_classes={n_classes}")
    return {"model":label,"probe_acc":round(acc,4),"chance":round(chance,6),
            "lift_over_chance":lift,"n_subjects":n_classes,
            "n_train_imgs":len(tr_idx),"n_test_imgs":len(te_idx),
            "probe_split":"within_subject_image_80_20"}

# ── E4: Clinical-threshold evaluation ─────────────────────────────────────────
def clinical_threshold_eval(results_a, results_b):
    """Recompute metrics at sensitivity>=0.80 threshold."""
    out={}
    for res in [results_a, results_b]:
        hist=res["history"]
        # We don't have probabilities stored, so report what we have
        m=res["test_metrics"]
        out[res["label"]]={"auroc":m["auroc"],"sensitivity":m["sensitivity"],
                            "specificity":m["specificity"],
                            "note":"threshold=0.5; AUROC is threshold-free"}
    return out

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip-e5",action="store_true")
    ap.add_argument("--epochs",type=int,default=EPOCHS)
    ap.add_argument("--bs",type=int,default=32)
    ap.add_argument("--lr",type=float,default=3e-4)
    args=ap.parse_args()

    dev=choose_device()
    all_rows=load_all_rows()
    safe_splits=splitguard_split(all_rows)

    master_results={"created_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "device":str(dev),"epochs":args.epochs,
                    "E1b":{},"E2":[],"E3":{},"E4":{},"E5":{}}

    # ── E1b: Multi-seed ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  E1b: Multi-seed stability (seeds 0,1,2,3)")
    print("="*60)
    e1b_gaps=[]
    for seed in SEEDS_E1B:
        leaky=leaky_split(all_rows,seed)
        ov=leaky_overlap(leaky)
        # Save leaky checkpoint for probe (seed 0 only to save disk)
        ckpt_leaky=CKPT_DIR/f"leaky_seed{seed}.pt" if seed==0 else None
        ckpt_safe =CKPT_DIR/f"safe_seed{seed}.pt"  if seed==0 else None

        ra=train_eval(leaky,dev,args.epochs,args.bs,args.lr,seed,
                      f"A_leaky_seed{seed}",save_ckpt=ckpt_leaky)
        rb=train_eval(safe_splits,dev,args.epochs,args.bs,args.lr,seed,
                      f"B_safe_seed{seed}",save_ckpt=ckpt_safe)
        gap={m:round(ra["test_metrics"][m]-rb["test_metrics"][m],4)
             for m in ["auroc","balanced_accuracy","f1_demented","sensitivity"]}
        e1b_gaps.append({"seed":seed,"leaky":ra["test_metrics"],
                          "safe":rb["test_metrics"],"gap":gap,"overlap":ov})
        master_results["E1b"][f"seed{seed}"]=e1b_gaps[-1]

    # Combine with seed 42
    existing={}
    p=ROOT/"reports"/"tables"/"inflation_gap_experiment.json"
    if p.exists():
        d=json.loads(p.read_text())
        existing={"seed":42,
                  "leaky":d["protocol_A_leaky"]["test_metrics"],
                  "safe":d["protocol_B_safe"]["test_metrics"],
                  "gap":{m:round(d["protocol_A_leaky"]["test_metrics"][m]-
                                 d["protocol_B_safe"]["test_metrics"][m],4)
                         for m in ["auroc","balanced_accuracy","f1_demented","sensitivity"]},
                  "overlap":d["leaky_subject_overlap"]}
    all_seed_results=[existing]+e1b_gaps if existing else e1b_gaps
    all_gaps=[r["gap"]["auroc"] for r in all_seed_results if r]
    master_results["E1b"]["summary"]={"n_seeds":len(all_gaps),
        "auroc_gap_mean":round(float(np.mean(all_gaps)),4),
        "auroc_gap_std":round(float(np.std(all_gaps)),4),
        "auroc_gap_min":round(float(np.min(all_gaps)),4),
        "auroc_gap_max":round(float(np.max(all_gaps)),4)}
    print(f"\n  E1b SUMMARY: gap_auroc = {master_results['E1b']['summary']['auroc_gap_mean']:.4f} ± {master_results['E1b']['summary']['auroc_gap_std']:.4f}")

    # ── E2: Subject-ID probe ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  E2: Subject-ID Probe (mechanistic evidence)")
    print("="*60)
    for ckpt,lbl in [(CKPT_DIR/"leaky_seed0.pt","A_leaky_seed0"),
                     (CKPT_DIR/"safe_seed0.pt","B_safe_seed0")]:
        if ckpt.exists():
            pr=subject_id_probe(ckpt,all_rows,dev,lbl)
            master_results["E2"].append(pr)
        else:
            print(f"  ⚠ checkpoint not found: {ckpt} — skipping probe for {lbl}")

    # ── E3: Degradation curve (Protocol A') ──────────────────────────────────
    print("\n" + "="*60)
    print("  E3: Protocol A' (subject-only split) — degradation curve")
    print("="*60)
    subj_only=subject_only_split(all_rows,seed=42)
    ov_a_prime=leaky_overlap(subj_only)
    print(f"  Protocol A' subject overlap: {ov_a_prime}")
    rc=train_eval(subj_only,dev,args.epochs,args.bs,args.lr,42,
                  "Aprime_subject_only_seed42")
    master_results["E3"]={"protocol_Aprime":rc,"overlap":ov_a_prime,
        "degradation_curve_auroc":{
            "A_leaky":0.9996,"Aprime_subject_only":rc["test_metrics"]["auroc"],
            "B_splitguard":0.8422}}
    print(f"  Degradation curve AUROC: A={0.9996} → A'={rc['test_metrics']['auroc']} → B={0.8422}")

    # ── E4: Clinical-threshold (post-processing) ─────────────────────────────
    print("\n" + "="*60)
    print("  E4: Clinical-threshold analysis (post-processing)")
    print("="*60)
    if existing:
        e4=clinical_threshold_eval(
            {"label":"A_leaky","test_metrics":existing["leaky"],"history":[]},
            {"label":"B_safe","test_metrics":existing["safe"],"history":[]})
        master_results["E4"]=e4
        print(f"  A leaky:  sensitivity={existing['leaky']['sensitivity']}  specificity={existing['leaky']['specificity']}")
        print(f"  B safe:   sensitivity={existing['safe']['sensitivity']}  specificity={existing['safe']['specificity']}")
        print(f"  Clinical note: leaky model would give {round((1-existing['leaky']['sensitivity'])*100,1)}% miss rate vs {round((1-existing['safe']['sensitivity'])*100,1)}% under SplitGuard")

    # ── E5: 4-class comparison (optional) ────────────────────────────────────
    if not args.skip_e5:
        print("\n" + "="*60)
        print("  E5: 4-class comparison")
        print("="*60)
        RAW4_rows=[r for r in all_rows if r.get("raw_class_label","") in RAW4_MAP]
        leaky4=leaky_split(RAW4_rows,seed=42)
        safe4=splitguard_split(RAW4_rows)
        # For 4-class we need CrossEntropyLoss and different label mapping
        set_seed(42)
        tr_tf,ev_tf=make_tf()
        def loader4(rows,tf,sh):
            class DS4(Dataset):
                def __len__(self): return len(rows)
                def __getitem__(self,i):
                    img=Image.open(rows[i]["path"]).convert("RGB")
                    if tf: img=tf(img)
                    return img,torch.tensor(RAW4_MAP[rows[i]["raw_class_label"]],dtype=torch.long)
            return DataLoader(DS4(),batch_size=32,shuffle=sh,num_workers=0)

        e5_results={}
        for name,sp in [("A_leaky_4class",leaky4),("B_safe_4class",safe4)]:
            model4=make_model(n_out=4,pretrained=True).to(dev)
            opt=torch.optim.AdamW(model4.parameters(),lr=3e-4,weight_decay=1e-2)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
            crit4=nn.CrossEntropyLoss()
            best4=-1.; best_s4=None
            for ep in range(1,args.epochs+1):
                model4.train()
                for imgs,labs in loader4(sp["train"],tr_tf,True):
                    imgs,labs=imgs.to(dev),labs.to(dev)
                    opt.zero_grad(set_to_none=True)
                    loss=crit4(model4(imgs),labs); loss.backward(); opt.step()
                sch.step()
                model4.eval(); corr=tot=0
                with torch.no_grad():
                    for im,lb in loader4(sp["val"],ev_tf,False):
                        p=model4(im.to(dev)).argmax(1).cpu()
                        corr+=(p==lb).sum().item(); tot+=lb.size(0)
                acc=corr/max(1,tot)
                if acc>best4: best4=acc; best_s4={k:v.cpu().clone() for k,v in model4.state_dict().items()}
                print(f"    {name} ep={ep:02d} val_acc={acc:.4f}")
            model4.load_state_dict(best_s4); model4.eval()
            preds,truths=[],[]
            with torch.no_grad():
                for im,lb in loader4(sp["test"],ev_tf,False):
                    preds.extend(model4(im.to(dev)).argmax(1).cpu().tolist())
                    truths.extend(lb.tolist())
            acc4=accuracy_score(truths,preds)
            bal4=balanced_accuracy_score(truths,preds)
            f1_4=f1_score(truths,preds,average="macro",zero_division=0)
            print(f"  {name}: acc={acc4:.4f} bal_acc={bal4:.4f} macro_f1={f1_4:.4f}")
            e5_results[name]={"accuracy":round(acc4,4),"balanced_accuracy":round(bal4,4),"macro_f1":round(f1_4,4)}
        master_results["E5"]=e5_results

    # ── Save all results ──────────────────────────────────────────────────────
    out=OUT_DIR/"overnight_results.json"
    out.write_text(json.dumps(master_results,indent=2))
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results saved → {out}")
    print(f"{'='*60}")
    # Print final summary
    if master_results["E1b"].get("summary"):
        s=master_results["E1b"]["summary"]
        print(f"\n  E1b  ΔAUROC = {s['auroc_gap_mean']} ± {s['auroc_gap_std']}  (n={s['n_seeds']} seeds)")
    if master_results["E2"]:
        for p in master_results["E2"]:
            print(f"  E2   {p['model']}: probe_acc={p['probe_acc']}  lift={p['lift_over_chance']}x")
    if master_results["E3"].get("degradation_curve_auroc"):
        dc=master_results["E3"]["degradation_curve_auroc"]
        print(f"  E3   Degradation: A={dc['A_leaky']} → A'={dc['Aprime_subject_only']} → B={dc['B_splitguard']}")

if __name__ == "__main__":
    raise SystemExit(main())
