# RunPod execution — DenseNet-121 Kaggle + OASIS

This is a one-shot recipe to run the DenseNet-121 architecture-replication
sensitivity arm on the Kaggle (Tier 1) and OASIS-1 (Tier 2) cohorts.
The runs take ~4 hours on a single RTX 4090 (~$4 RunPod cost).

The primary ResNet-18 results are NOT touched; outputs land in new
arch-namespaced JSON files.

## Prerequisites
- A RunPod RTX 4090 instance (or any single GPU ≥16 GB VRAM)
- The Kaggle dataset + OASIS-1 frozen-split manifest pushed to the pod
  (same way you did the dose-response run)
- The repository at `~/splitguard-ad` with this branch checked out

## Step 1 — Spin up the pod and pull the branch
```bash
# On the RunPod terminal:
cd ~
git clone https://github.com/mehmetberkeisler/splitguard-ad.git
cd splitguard-ad
git checkout <branch-with-this-patch>     # or main if merged
pip install -r requirements.txt
```

## Step 2 — Verify both arch builds
```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from run_inflation_gap import make_model as k
from run_oasis1_inflation_gap import make_model as o
for fn in [k, o]:
    for arch in ['resnet18', 'densenet121']:
        m = fn(pretrained=False, arch=arch)
        n = sum(p.numel() for p in m.parameters())
        print(f'{fn.__module__} / {arch}: {n/1e6:.1f}M params')
"
```
Expected output (both cohorts):
```
... / resnet18:     11.2M params
... / densenet121:   7.0M params
```

## Step 3 — Single seed smoke test (~10 min, optional but recommended)
```bash
# Run 1 seed × 1 cohort with 2 epochs to verify wiring on the real GPU
python scripts/run_inflation_gap.py --arch densenet121 --seed 0 --epochs 2
# Expected: produces reports/tables/inflation_gap_experiment__densenet121__seed0.json
```

## Step 4 — Full sweep
```bash
# 5 seeds × 2 cohorts × 2 protocols = 20 runs (~4 h on RTX 4090)
bash scripts/run_densenet_kaggle_oasis.sh
```

Override knobs:
- `SEEDS="0 1 2" bash scripts/...` — fewer seeds for a faster pilot
- `SKIP_KAGGLE=1` or `SKIP_OASIS=1` — run one cohort at a time

## Step 5 — Pull results back to local
```bash
# From local machine (using runpodctl/scp/rsync):
rsync -av --include='*.json' --include='*/' --exclude='*' \
    runpod@<pod-ip>:~/splitguard-ad/reports/tables/ \
    ~/Downloads/irem\ gergin/reports/tables/

# Specifically the new files:
#   inflation_gap_experiment__densenet121__seed{0,1,2,3,42}.json
#   oasis1_inflation_gap_experiment__densenet121__seed{0,1,2,3,42}.json
```

## Step 6 — Local: summarize + integrate
Once the 10 JSON files are local, I (Claude) will:
1. Write `scripts/summarize_densenet_xcohort.py` to compute
   paired-seed mean ± SD and bootstrap intervals on Kaggle and OASIS
2. Add `\paragraph{DenseNet-121 architecture replication}` to §6.1 (Kaggle)
   and §6.4 (OASIS) with the new numbers
3. Update the abstract / intro / conclusion to read "architecture-
   invariance on all three tiers" (currently we say "on ADNI")
4. Optionally extend Fig 5 (cross-cohort) to overlay DenseNet error
   bars next to ResNet bars
5. Rebuild PDF, re-run final validation

## Cost estimate
- RTX 4090 on RunPod: ~$0.80/h × 4 h = ~$3.20
- Storage / bandwidth: negligible

## Total marginal value to the paper
- Cross-cohort architecture-invariance demonstrated on three independent
  cohorts (was: only ADNI)
- Pre-empts the most common MedIA reviewer complaint
  ("only one CNN architecture in your headline experiments")
- Adds ~2 short paragraphs and 1 small table block; no structural change
