#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Drive the Phase 4 dose-response training matrix.
#
# Matrix: 5 seeds (0..4) × 5 overlap levels (0/25/50/75/100%) × 2 archs
#         (resnet18, densenet121) = 50 runs, 15 epochs each.
#
# Per-run output organisation:
#   runs/adni_dose_response/<arch>/seed<S>_overlap<P>/
#     ├─ test_predictions.csv   <- input to analyze_dose_response.py
#     ├─ metrics.json
#     └─ training_log.csv
#
# Resume-safe: skips any run whose test_predictions.csv already exists, so a
# Ctrl-C and re-run picks up where it left off.
#
# Usage:
#   bash scripts/run_dose_response_matrix.sh            # full 50-run matrix
#   bash scripts/run_dose_response_matrix.sh resnet18   # one arch only
#   ARCHS="resnet18" bash scripts/run_dose_response_matrix.sh
#   SEEDS="0 1" OVERLAPS="0.0 0.5 1.0" bash scripts/run_dose_response_matrix.sh
#
# Env vars (override the defaults):
#   SEEDS     space-separated seed list (default: 0 1 2 3 4)
#   OVERLAPS  space-separated overlap fractions (default: 0.0 0.25 0.50 0.75 1.0)
#   ARCHS     space-separated archs (default: resnet18 densenet121)
#   EPOCHS    training epochs per run (default: 15)
#   DEVICE    auto | cpu | mps | cuda (default: auto)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

SEEDS=${SEEDS:-"0 1 2 3 4"}
OVERLAPS=${OVERLAPS:-"0.0 0.25 0.50 0.75 1.0"}
ARCHS=${ARCHS:-"${1:-resnet18 densenet121}"}
EPOCHS=${EPOCHS:-15}
DEVICE=${DEVICE:-auto}

BASE_SPLIT_DIR="data/splits/adni_with_converters"
INJECTED_DIR="data/splits/adni_dose_response"
OUT_ROOT="runs/adni_dose_response"

mkdir -p "$INJECTED_DIR" "$OUT_ROOT"

total=0; done=0; skipped=0; failed=0
start_ts=$(date +%s)

# Count total runs upfront for progress reporting
for s in $SEEDS; do for p in $OVERLAPS; do for a in $ARCHS; do
  total=$((total + 1))
done; done; done
echo "Dose-response matrix: $total runs ($SEEDS) x ($OVERLAPS) x ($ARCHS), $EPOCHS epochs each"
echo "Device: $DEVICE  |  Output: $OUT_ROOT"
echo

idx=0
for seed in $SEEDS; do
  for overlap in $OVERLAPS; do
    # 1. Ensure injected split exists (deterministic given seed × overlap).
    split="$INJECTED_DIR/seed${seed}_overlap${overlap}.csv"
    audit="$INJECTED_DIR/seed${seed}_overlap${overlap}.audit.json"
    if [ ! -f "$split" ]; then
      echo "[$((idx+1))] Generating split: seed=$seed overlap=$overlap"
      python3 scripts/inject_leakage_split.py \
        --base "$BASE_SPLIT_DIR/adni_splitguard_seed${seed}.csv" \
        --overlap "$overlap" --seed "$seed" \
        --output "$split" --audit-output "$audit" > /dev/null
    fi

    # 2. Train each architecture against this split.
    for arch in $ARCHS; do
      idx=$((idx + 1))
      run_dir="$OUT_ROOT/$arch/seed${seed}_overlap${overlap}"
      preds="$run_dir/baseline_seed${seed}/test_predictions.csv"

      if [ -f "$preds" ]; then
        skipped=$((skipped + 1))
        printf "[%d/%d] SKIP  seed=%s overlap=%-4s arch=%-12s (already done)\n" \
               "$idx" "$total" "$seed" "$overlap" "$arch"
        continue
      fi

      run_start=$(date +%s)
      printf "[%d/%d] RUN   seed=%s overlap=%-4s arch=%-12s ..." \
             "$idx" "$total" "$seed" "$overlap" "$arch"
      mkdir -p "$(dirname "$run_dir")"   # parent (per-arch) directory
      if python3 scripts/train_adni_baseline.py \
           --split "$split" \
           --output-root "$run_dir" \
           --seed "$seed" \
           --epochs "$EPOCHS" \
           --arch "$arch" \
           --device "$DEVICE" > "$run_dir.log" 2>&1; then
        run_end=$(date +%s)
        done=$((done + 1))
        printf " OK  (%ds)\n" "$((run_end - run_start))"
      else
        failed=$((failed + 1))
        printf " FAIL — see %s.log\n" "$run_dir"
      fi
    done
  done
done

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
echo
printf "=== Matrix complete: %d done, %d skipped, %d failed.  Wall clock: %dm %ds ===\n" \
       "$done" "$skipped" "$failed" "$((elapsed / 60))" "$((elapsed % 60))"
exit $failed
