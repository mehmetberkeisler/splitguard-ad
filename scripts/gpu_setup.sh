#!/usr/bin/env bash
# Prepare a fresh GPU node for the re-run and record exactly what it is.
#
# Run from the repository root on the node:
#     bash scripts/gpu_setup.sh
#
# ADNI-derived data must stay on the node for the duration of the run and be
# deleted afterwards (ADNI Data Use Agreement). Nothing here uploads anything;
# the results bundle is pulled back by the operator.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver on this node" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# The pinned wheels (numpy 1.26, scikit-learn 1.5) publish no cp313 build, so
# an image shipping a newer interpreter would fall back to a source build and
# fail after the node is already rented.
python3 -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13), sys.version' \
  || { echo "need Python 3.10-3.12; this image has $(python3 -V)" >&2; exit 1; }

python3 -m venv .venv-gpu
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
python -m pip install -r requirements-gpu.txt

mkdir -p runs
{
  echo "# recorded $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
  python -m pip freeze
} > runs/gpu_environment.txt

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
x = torch.randn(64, 3, 224, 224, device="cuda")
print("CUDA ok:", torch.cuda.get_device_name(0), "| tensor", tuple(x.shape))
PY
echo "environment recorded in runs/gpu_environment.txt"
