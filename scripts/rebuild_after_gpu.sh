#!/usr/bin/env bash
# Rebuild every analysis, table, figure and manuscript number from a finished
# GPU run, then verify the manuscript against them.
#
# Run from anywhere after unpacking gpu_results.tar.gz at the repository root:
#     bash scripts/rebuild_after_gpu.sh
#
# The GPU runs are promoted to the locations the analysis scripts already
# read (runs/<arm>, reports/tables/adni/...), so no analysis needs a special
# input path. The frozen runs are kept once, under runs_frozen/, for
# scripts/compare_reruns.py.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python3}
GPU_ADNI=reports/gpu/adni
ADNI=reports/tables/adni

[ -d runs_gpu ] || { echo "runs_gpu/ not found: unpack gpu_results.tar.gz first" >&2; exit 1; }

echo "== 1. promote the GPU runs"
if [ -d runs ] && [ ! -d runs_frozen ]; then mv runs runs_frozen; fi
mkdir -p runs
$PY - <<'CHECK'
# Refuse to promote predictions that do not come from the frozen splits: a
# fixture or a run against a different manifest would otherwise flow straight
# into the published numbers.
import csv, sys
from pathlib import Path
manifest = Path("data/splits/adni/adni_splitguard_seed0.csv")
if manifest.is_file():
    known = {r["image_id"] for r in csv.DictReader(manifest.open())}
    sample = Path("runs_gpu/adni/inflation_gap_seed0/component_safe/test_predictions.csv")
    if sample.is_file():
        rows = list(csv.DictReader(sample.open()))
        unknown = [r["image_id"] for r in rows if r["image_id"] not in known]
        if unknown:
            sys.exit(f"{sample} holds {len(unknown)} image ids that are not in {manifest.name}, "
                     f"e.g. {unknown[0]}: these predictions are not from the frozen splits")
        print(f"provenance check: {len(rows)} predictions all match {manifest.name}")
CHECK

arms=(runs_gpu/*/)
if [ ${#arms[@]} -eq 0 ]; then
  echo "runs_gpu/ holds no arm directories: did the programme run any stage?" >&2
  exit 1
fi
for arm in "${arms[@]}"; do
  name=$(basename "$arm")
  [ "$name" = logs ] && continue
  rm -rf "runs/$name"
  cp -R "$arm" "runs/$name"
done
# An arm this run did not produce keeps whatever it was: the analyses below
# read runs/<arm>, and a half-finished stage must not replace a complete
# earlier one. Anything restored this way is an arm the manuscript has to
# describe as not part of the single code state.
for arm in runs_frozen/*/; do
  name=$(basename "$arm")
  [ -d "runs/$name" ] || { cp -R "$arm" "runs/$name"; echo "  kept earlier arm: $name"; }
done

echo "== 2. analyses over the promoted predictions"
$PY scripts/analyze_dose_response.py
$PY scripts/analyze_dose_response_mixed_effects.py
$PY scripts/clinical_cost_of_leakage_adni.py \
    --prev-pop 0.138 --prev-pop-citation "Rajan et al. 2021" \
    --prev-clinic 0.589 --prev-clinic-citation "Calgary PROMPT registry, Thomas et al. 2024"
$PY scripts/subgroup_analysis_adni.py
$PY scripts/subject_level_aggregation_adni.py
$PY scripts/operating_point_sensitivity_table.py
for arm in adni_with_converters adni_size_balanced; do
  $PY scripts/split_composition_audit.py --split-dir "data/splits/$arm" \
      --output "$ADNI/adni_split_composition_${arm#adni_}.json"
done

echo "== 3. bootstraps, manuscript macros and the Tier-1 table from the GPU predictions"
$PY scripts/gpu_postprocess.py

echo "== 4. promote summaries to the paths the analyses and generators read"
for f in reports/gpu/adni/adni_inflation_gap*.csv reports/gpu/adni/adni_inflation_gap*.json; do
  cp "$f" "$ADNI/"
done
for f in reports/gpu/cross_cohort/*_inflation_gap_bootstrap.json; do
  cp "$f" reports/tables/
done
for f in adni_permutation_null.json adni_provenance_degradation.json adni_biometric_probe_session.json; do
  [ -f "$GPU_ADNI/$f" ] && cp "$GPU_ADNI/$f" "$ADNI/$f"
done
[ -f "$GPU_ADNI/adni_biometric_probe_image.json" ] && cp "$GPU_ADNI/adni_biometric_probe_image.json" "$ADNI/adni_biometric_probe.json"

echo "== 5. tables and figures"
$PY scripts/generate_adni_paper_tables.py
$PY scripts/generate_figures.py
$PY scripts/generate_cross_cohort_figure.py
$PY scripts/generate_arm_forest_figure.py
$PY scripts/generate_operating_point_figure.py
$PY scripts/generate_dose_response_figure.py
$PY scripts/generate_provenance_figure.py
$PY scripts/generate_cost_of_leakage_figure.py
$PY scripts/generate_subgroup_figure.py

echo "== 6. what moved against the frozen runs"
[ -d runs_frozen ] && $PY scripts/compare_reruns.py --old runs_frozen --new runs \
    --markdown reports/rerun_comparison.md || true

echo "== 7. verify and compile"
# Verification is expected to report drift after a genuine re-run, so it must
# not stop the compile or the pending report that tells you what is missing.
verify_rc=0
$PY scripts/verify_paper_numbers.py || verify_rc=$?
(cd paper && tectonic -X compile splitguard_ad.tex && tectonic -X compile SplitGuard-AD_Supplementary_Material.tex)
if grep -q '\\pending{}' paper/tables/gpu_numbers.tex; then
  echo "note: some manuscript numbers are still ?? because their GPU stage did not complete:"
  grep '\\pending{}' paper/tables/gpu_numbers.tex | sed 's/\\newcommand{\\\([A-Za-z]*\)}.*/  \1/'
fi
if [ "$verify_rc" -ne 0 ]; then
  echo "verify_paper_numbers.py exited $verify_rc: reconcile the numbers it listed above."
fi
exit "$verify_rc"
