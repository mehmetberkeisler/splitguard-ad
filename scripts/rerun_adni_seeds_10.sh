#!/bin/bash
# Rerun the ADNI inflation-gap primary arm with seed count 5 → 10.
#
# Reviewer §5.3 flagged that n=5 seeds is insufficient for defensible
# bootstrap inference. This script extends the seed array to 10 by adding
# five NEW seeds (5..9) to the existing five (0..4). Only the new seeds
# are actually trained; existing seed 0..4 checkpoints/predictions are
# left untouched so the headline number and bootstrap can be recomputed
# against the union of 10 seeds without invalidating the current results.
#
# Expected wall-clock: ~6 hr per new seed × 5 seeds × 3 protocols = ~90h
# on Apple Silicon MPS (~30–35 min per protocol epoch × 15 epochs). Run
# overnight for several nights, or on RunPod A100 (~1–2 hr per seed).
#
# Usage
# -----
#   ./scripts/rerun_adni_seeds_10.sh                 # trains seeds 5..9
#   NEW_SEEDS="5 6 7" ./scripts/rerun_adni_seeds_10.sh  # subset
#   PROTOCOLS="component_safe" ./scripts/rerun_adni_seeds_10.sh  # one protocol only

set -euo pipefail

# Resolved from this script's own location, mirroring the Python scripts'
# PROJECT_ROOT = Path(__file__).resolve().parents[1], so a clone works from
# any checkout directory.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NEW_SEEDS="${NEW_SEEDS:-5 6 7 8 9}"
PROTOCOLS="${PROTOCOLS:-random subject_only component_safe}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/adni_with_converters}"
SPLITS_ROOT="${SPLITS_ROOT:-$PROJECT_ROOT/data/splits/adni_with_converters}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/data/manifests/adni_with_converters_manifest.csv}"
LOG_DIR="$PROJECT_ROOT/runs/logs/rerun_seeds_10"

mkdir -p "$LOG_DIR"

echo "==============================================================================="
echo "ADNI seed extension 5 → 10"
echo "NEW_SEEDS   : $NEW_SEEDS"
echo "PROTOCOLS   : $PROTOCOLS"
echo "MANIFEST    : $MANIFEST"
echo "SPLITS_ROOT : $SPLITS_ROOT"
echo "RUNS_ROOT   : $RUNS_ROOT"
echo "==============================================================================="

# ── Step 1: Generate splits for new seeds (component-safe DP is deterministic
# per (seed, tiebreak_salt), so seed=5 gives a distinct legitimate split). ──
for seed in $NEW_SEEDS; do
  SPLIT_FILE="$SPLITS_ROOT/adni_splitguard_seed${seed}.csv"
  if [ -f "$SPLIT_FILE" ]; then
    echo "SKIP split generation (exists): $SPLIT_FILE"
    continue
  fi
  echo "GENERATING split for seed $seed → $SPLIT_FILE"
  python3 "$PROJECT_ROOT/scripts/make_adni_splitguard_split.py" \
    --manifest "$MANIFEST" \
    --seed "$seed" \
    --output "$SPLIT_FILE" \
    2>&1 | tee "$LOG_DIR/split_gen_seed${seed}.log"
done

# ── Step 2: Train each (protocol × seed) combination. ─────────────────────
for seed in $NEW_SEEDS; do
  for proto in $PROTOCOLS; do
    OUT_DIR="$RUNS_ROOT/inflation_gap_seed${seed}/${proto}"
    if [ -f "$OUT_DIR/best_state.pt" ] && [ -f "$OUT_DIR/test_predictions.csv" ]; then
      echo "SKIP train (already done): $OUT_DIR"
      continue
    fi
    mkdir -p "$OUT_DIR"
    echo ">>> TRAIN seed=${seed} protocol=${proto} @ $(date '+%H:%M:%S')"
    python3 "$PROJECT_ROOT/scripts/train_adni_baseline.py" \
      --manifest "$MANIFEST" \
      --split "$SPLITS_ROOT/adni_splitguard_seed${seed}.csv" \
      --protocol "$proto" \
      --seed "$seed" \
      --output-dir "$OUT_DIR" \
      --save-predictions \
      --save-val-predictions \
      2>&1 | tee "$LOG_DIR/train_seed${seed}_${proto}.log"
    echo "<<< DONE seed=${seed} protocol=${proto} @ $(date '+%H:%M:%S')"
  done
done

echo ""
echo "==============================================================================="
echo "Seed extension COMPLETE. Next steps:"
echo "  1. Re-run bootstrap_adni_inflation_gap.py with --seeds 0 1 2 3 4 5 6 7 8 9"
echo "  2. Re-run analyze_dose_response_mixed_effects.py"
echo "  3. Update paper tables (adni_inflation_gap.csv, etc.)"
echo "==============================================================================="
