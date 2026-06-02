#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# One-shot RunPod setup + Phase 4 dose-response matrix runner.
#
# Designed to be invoked exactly ONCE inside a RunPod pod after the
# adni_runpod_bundle.tar.gz has been extracted to /workspace.
#
# It is idempotent and resume-safe:
#   * Re-running it skips already-completed runs.
#   * Setup steps detect existing state and skip themselves.
#
# Usage (on the pod):
#   cd /workspace
#   bash scripts/runpod_setup_and_run.sh                # full pipeline
#   bash scripts/runpod_setup_and_run.sh smoke          # just the 1-epoch smoke test
#   bash scripts/runpod_setup_and_run.sh setup-only     # just install + verify, no training
#
# Designed to be invoked inside tmux:
#   tmux new -s phase4
#   bash scripts/runpod_setup_and_run.sh
#   Ctrl-B then D to detach
# ───────────────────────────────────────────────────────────────────────────
set -e   # bail on error; we WANT to halt at each broken step

MODE=${1:-full}

# Colours for the section headers (works in any modern terminal)
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'    # No Color

step() { echo -e "\n${BOLD}${GREEN}━━━ $1 ━━━${NC}\n"; }
warn() { echo -e "${YELLOW}!! $1${NC}"; }
fail() { echo -e "${RED}!! $1${NC}"; exit 1; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }

# ── Sanity: we should be in the workspace root with the bundle layout ────
[ -d data/preprocessed/adni/slices ] || fail "Missing data/preprocessed/adni/slices. Did you extract the bundle in /workspace?"
[ -d data/manifests/adni ]            || fail "Missing data/manifests/adni"
[ -d data/splits/adni_with_converters ] || fail "Missing data/splits/adni_with_converters"
[ -f scripts/train_adni_baseline.py ] || fail "Missing scripts/train_adni_baseline.py"
[ -f scripts/inject_leakage_split.py ] || fail "Missing scripts/inject_leakage_split.py"
[ -f scripts/run_dose_response_matrix.sh ] || fail "Missing scripts/run_dose_response_matrix.sh"

# ── Step 1: install all Python deps in one shot ──────────────────────────
step "Step 1/5: Installing Python dependencies"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet \
    nibabel \
    scikit-learn \
    'numpy<2.0'   # torchvision sometimes fights with numpy 2.x; pin to be safe
ok "pip installs complete"

# ── Step 2: verify every required import works for the SAME Python ───────
step "Step 2/5: Verifying environment"
python3 - << 'PYEOF'
import sys
print(f"python: {sys.executable} ({sys.version.split()[0]})")
mods = {}
try:
    import numpy as np; mods["numpy"] = np.__version__
    import torch; mods["torch"] = torch.__version__
    import torchvision; mods["torchvision"] = torchvision.__version__
    import sklearn; mods["sklearn"] = sklearn.__version__
    import nibabel; mods["nibabel"] = nibabel.__version__
    import PIL; mods["PIL"] = PIL.__version__
except ImportError as e:
    print(f"IMPORT FAILED: {e}")
    sys.exit(1)
for k, v in mods.items():
    print(f"  {k:12s} {v}")
print(f"  cuda?       {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() is False — GPU not visible.")
    sys.exit(1)
print(f"  GPU:        {torch.cuda.get_device_name(0)}")
PYEOF
ok "environment verified"

# ── Step 3: install tmux for clean detach during long runs ──────────────
step "Step 3/5: Installing tmux"
if command -v tmux >/dev/null 2>&1; then
    ok "tmux already installed: $(tmux -V)"
else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
    ok "tmux installed: $(tmux -V)"
fi

# ── Step 4: SMOKE TEST — one short run to confirm pipeline works ────────
step "Step 4/5: Smoke test (1 ResNet-18 run, 1 epoch, ~30 sec)"
rm -rf runs/adni_dose_response_smoke
DEVICE=cuda EPOCHS=1 SEEDS="0" OVERLAPS="0.50" ARCHS="resnet18" \
    bash scripts/run_dose_response_matrix.sh 2>&1 | tail -5

# Validate the smoke test actually produced metrics
SMOKE_METRICS="runs/adni_dose_response/resnet18/seed0_overlap0.50/baseline_seed0/metrics.json"
if [ ! -f "$SMOKE_METRICS" ]; then
    warn "Smoke test did NOT produce metrics.json — showing the log:"
    cat runs/adni_dose_response/resnet18/seed0_overlap0.50.log
    fail "Smoke test failed; halting before full matrix"
fi
# Extract AUROC for a quick sanity check
SMOKE_AUROC=$(python3 -c "
import json
m = json.load(open('$SMOKE_METRICS'))
print(m['test_metrics']['auroc'])
")
ok "smoke test passed (1-epoch AUROC = $SMOKE_AUROC, expected 0.6–0.9 range)"
# Clean the smoke output so it doesn't get re-counted in the full matrix
rm -rf runs/adni_dose_response/resnet18/seed0_overlap0.50
rm -f  runs/adni_dose_response/resnet18/seed0_overlap0.50.log

if [ "$MODE" = "setup-only" ]; then
    echo
    echo "Setup-only mode: stopping before full matrix."
    exit 0
fi
if [ "$MODE" = "smoke" ]; then
    echo
    echo "Smoke-only mode: stopping after smoke test."
    exit 0
fi

# ── Step 5: FULL 50-RUN MATRIX ──────────────────────────────────────────
step "Step 5/5: Full dose-response matrix (50 runs, ~1h 40min on RTX 4090)"
DEVICE=cuda bash scripts/run_dose_response_matrix.sh

# ── Step 6: bundle results for download ─────────────────────────────────
step "Bundling results for download"
RESULTS_TAR=/workspace/dose_response_results.tar.gz
tar czf "$RESULTS_TAR" \
    --exclude='*.pt' \
    runs/adni_dose_response
ls -lh "$RESULTS_TAR"
ok "results bundled to $RESULTS_TAR"
echo
echo "On your Mac, download with:"
echo "  scp -i ~/.ssh/id_ed25519 -P <PORT> root@<HOST>:$RESULTS_TAR /tmp/"
echo "Then on Mac: cd to project root and run: tar xzf /tmp/dose_response_results.tar.gz"
