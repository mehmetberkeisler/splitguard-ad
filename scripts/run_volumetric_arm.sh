#!/usr/bin/env bash
# Stage and run the volumetric arm on a rented node, end to end.
#
#     bash scripts/run_volumetric_arm.sh root@HOST -p PORT
#
# The volumetric arm is the one stage the manuscript does not report. Everything
# it needs is already built locally: the five frozen adni_3d split manifests and
# gpu_bundle_3d.tar (6.26 GB of 128^3 volumes). This script uploads the
# repository, the splits and the bundle, repairs the node's stack, smokes the
# stage, and then runs it -- 15 commands, five seeds by three protocols.
#
# Two things are deliberate rather than incidental:
#
#   --workers 4 runs ONE training at a time. SLOTS["adni_3d"] is 4, so this is
#   the floor, not a cushion: two concurrent commands reserve 29 GB of a 32 GB
#   card and the second dies in batch_norm. Measured, not guessed.
#
#   The run is detached with setsid+nohup. An earlier run died with its SSH
#   session when the connection dropped, losing 25 finished trainings.
#
# ADNI volumes are covered by the LONI Data Use Agreement. Delete them from the
# node before the pod is terminated; the last line of this script prints how.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET=${1:?usage: run_volumetric_arm.sh root@HOST [-p PORT] }
shift || true
SSH_OPTS=("$@")
SSH=(ssh "${SSH_OPTS[@]}" -o ConnectTimeout=25 -o ServerAliveInterval=30 "$TARGET")
RSH="ssh ${SSH_OPTS[*]} -o ConnectTimeout=25"
REMOTE=/root/splitguard

echo "== 1. node check"
"${SSH[@]}" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; df -h / | tail -1'

echo "== 2. repository (excludes data/, runs/, archive/ and the bundles)"
rsync -e "$RSH" -az --delete \
  --exclude 'data/' --exclude 'runs/' --exclude 'runs_gpu/' --exclude 'runs_frozen/' \
  --exclude 'runs_smoke/' --exclude 'archive/' --exclude 'obsolete_archive/' \
  --exclude '.venv-gpu/' --exclude 'gpu_bundle_*.tar' --exclude 'gpu_results*.tar.gz' \
  --exclude 'Alzheimer_MRI_4_classes_dataset/' --exclude 'oasis1/' --exclude 'release/' \
  --exclude '*.pdf' --exclude '__pycache__/' --exclude '*_22May2026.csv' \
  ./ "$TARGET:$REMOTE/"

echo "== 3. split manifests (small; the 2D bundle is not needed for this stage)"
"${SSH[@]}" "mkdir -p $REMOTE/data/splits/adni_3d"
rsync -e "$RSH" -az data/splits/adni_3d/ "$TARGET:$REMOTE/data/splits/adni_3d/"

echo "== 4. volume bundle (6.26 GB; resumable, so a dropped link costs nothing)"
rsync -e "$RSH" -av --partial --inplace --append-verify \
  gpu_bundle_3d.tar "$TARGET:/workspace/gpu_bundle_3d.tar"

echo "== 5. environment, then extract"
"${SSH[@]}" "cd $REMOTE && bash scripts/gpu_setup.sh 2>&1 | tail -3"
"${SSH[@]}" "cd $REMOTE && tar -xf /workspace/gpu_bundle_3d.tar && \
             echo volumes: \$(find data/adni3d_128 -name '*.npz' | wc -l)"

echo "== 6. smoke (all three protocols, one epoch each)"
"${SSH[@]}" "cd $REMOTE && rm -rf runs_smoke reports/smoke && \
             python3 -u scripts/gpu_program.py --smoke --only adni_3d --workers 4 2>&1 | tail -4"

echo "== 7. launch, detached"
"${SSH[@]}" "cd $REMOTE && rm -rf runs_smoke reports/smoke && \
  setsid nohup python3 -u scripts/gpu_program.py --only adni_3d --workers 4 \
  --time-budget-min 900 --usd-per-hour 0.69 --budget-usd 999 > vol.log 2>&1 < /dev/null & \
  sleep 20; head -2 vol.log"

cat <<EOF

Running detached. Watch the first command before trusting the next ten hours:

    ssh ${SSH_OPTS[*]} $TARGET "tail -f $REMOTE/runs_gpu/logs/adni_3d_00.log"

On the 'random' protocol validation AUROC must CLIMB as the training loss
falls. If it falls, stop: that is the metric-orientation signature, and the
remaining commands are worthless.

When all 15 are done:

    ssh ${SSH_OPTS[*]} $TARGET "cd $REMOTE && tar czf gpu_results.tar.gz runs_gpu reports/gpu"
    scp ${SSH_OPTS[*]} $TARGET:$REMOTE/gpu_results.tar.gz .
    tar xzf gpu_results.tar.gz && bash scripts/rebuild_after_gpu.sh

Then delete the ADNI data before terminating the pod (Data Use Agreement):

    ssh ${SSH_OPTS[*]} $TARGET "rm -rf $REMOTE/data $REMOTE/runs_gpu /workspace/gpu_bundle_3d.tar"
EOF
