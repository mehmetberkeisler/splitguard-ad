#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DenseNet-121 cross-cohort architecture-replication runs.
#
# Replicates the inflation-gap experiment on Kaggle (Tier 1) and OASIS-1
# (Tier 2) using DenseNet-121 in addition to the primary ResNet-18
# manuscript results. Outputs land in reports/tables/ namespaced by arch
# and seed; primary ResNet-18 result files are NOT touched.
#
# Wall-clock estimate on a single RTX 4090:
#   Kaggle:  5 seeds × 30 epochs × 2 protocols × ~5 min/run  ≈ 2.5 h
#   OASIS:   5 seeds × 20 epochs × 2 protocols × ~3 min/run  ≈ 1.5 h
#   Total:   ~4 h wallclock (~$4 RunPod)
#
# Usage:
#   cd <repo-root>
#   bash scripts/run_densenet_kaggle_oasis.sh
#
# Override seeds:    SEEDS="0 1 2" bash scripts/...
# Skip Kaggle:       SKIP_KAGGLE=1 bash scripts/...
# Skip OASIS:        SKIP_OASIS=1 bash scripts/...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 42}"
ARCH="densenet121"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p reports/tables

echo "═════════════════════════════════════════════════════════════════"
echo "  DenseNet-121 cross-cohort sensitivity arm"
echo "  Seeds: $SEEDS"
echo "  Repo:  $PROJECT_ROOT"
echo "═════════════════════════════════════════════════════════════════"

# ── Tier 1 (Kaggle) ──────────────────────────────────────────────────────────
if [[ "${SKIP_KAGGLE:-0}" != "1" ]]; then
    echo ""
    echo "── Tier 1: Kaggle JPEG benchmark ──"
    for seed in $SEEDS; do
        echo ""
        echo ">>> Kaggle DenseNet-121, seed=$seed"
        python scripts/run_inflation_gap.py \
            --arch "$ARCH" \
            --seed "$seed" \
            --epochs 30
    done
fi

# ── Tier 2 (OASIS-1) ─────────────────────────────────────────────────────────
if [[ "${SKIP_OASIS:-0}" != "1" ]]; then
    echo ""
    echo "── Tier 2: OASIS-1 ──"
    for seed in $SEEDS; do
        echo ""
        echo ">>> OASIS-1 DenseNet-121, seed=$seed"
        python scripts/run_oasis1_inflation_gap.py \
            --arch "$ARCH" \
            --seed "$seed" \
            --epochs 20
    done
fi

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo "  DONE. Output files:"
echo "    reports/tables/inflation_gap_experiment__densenet121__seed*.json"
echo "    reports/tables/oasis1_inflation_gap_experiment__densenet121__seed*.json"
echo ""
echo "  Next step: pull these JSONs back to the local repo, then run"
echo "    python scripts/summarize_densenet_xcohort.py"
echo "  (which I will create after seeing the first JSON files)"
echo "═════════════════════════════════════════════════════════════════"
