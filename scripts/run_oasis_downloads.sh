#!/bin/bash
# One-command wrapper that downloads OASIS-3 and OASIS-4 in parallel,
# with persistent-restart on transient failures.
#
# Usage:
#   export XNAT_USER=your_username
#   export XNAT_PASS=your_password
#   ./scripts/run_oasis_downloads.sh
#
# Env overrides (optional):
#   OASIS3_DST         (default: data/raw/oasis3)
#   OASIS4_DST         (default: data/raw/oasis4)
#   OASIS3_MAX         (default: 800  — subject cap for OASIS-3)
#   OASIS4_MAX         (default: 500  — subject cap for OASIS-4)
#   DIAGNOSTIC_GROUPS  (default: "CN AD MCI")

set -uo pipefail

# Resolved from this script's own location, mirroring the Python scripts'
# PROJECT_ROOT = Path(__file__).resolve().parents[1], so a clone works from
# any checkout directory.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OASIS3_DST=${OASIS3_DST:-data/raw/oasis3}
OASIS4_DST=${OASIS4_DST:-data/raw/oasis4}
OASIS3_MAX=${OASIS3_MAX:-800}
OASIS4_MAX=${OASIS4_MAX:-500}
DIAGNOSTIC_GROUPS=${DIAGNOSTIC_GROUPS:-"CN AD MCI"}

if [ -z "${XNAT_USER:-}" ] || [ -z "${XNAT_PASS:-}" ]; then
  echo "ERROR: set XNAT_USER and XNAT_PASS env vars first." >&2
  echo "  export XNAT_USER=your_username" >&2
  echo "  export XNAT_PASS=your_password" >&2
  exit 1
fi

mkdir -p "$OASIS3_DST" "$OASIS4_DST"

download_with_retry() {
  local project=$1
  local dst=$2
  local max=$3
  local log=$dst/download.log
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "$(date '+%H:%M:%S') [$project] attempt $attempt starting" | tee -a "$log"
    python3 "$PROJECT_ROOT/scripts/download_oasis.py" \
      --project "$project" \
      --dst "$dst" \
      --diagnostic-groups $DIAGNOSTIC_GROUPS \
      --max-subjects "$max" \
      2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
      echo "$(date '+%H:%M:%S') [$project] DOWNLOAD COMPLETE (attempt $attempt)" | tee -a "$log"
      break
    fi
    echo "$(date '+%H:%M:%S') [$project] exited $rc, retry in 120s" | tee -a "$log"
    sleep 120
  done
}

echo "==============================================================="
echo "OASIS-3 → $OASIS3_DST  (max $OASIS3_MAX subjects)"
echo "OASIS-4 → $OASIS4_DST  (max $OASIS4_MAX subjects)"
echo "Diagnostic filter    : $DIAGNOSTIC_GROUPS"
echo "==============================================================="

# Run both in parallel; wait for both
download_with_retry OASIS3 "$OASIS3_DST" "$OASIS3_MAX" &
PID3=$!
sleep 30  # stagger to avoid XNAT session collision
download_with_retry OASIS4 "$OASIS4_DST" "$OASIS4_MAX" &
PID4=$!

wait $PID3 $PID4
echo "$(date '+%H:%M:%S') both downloads finished"
