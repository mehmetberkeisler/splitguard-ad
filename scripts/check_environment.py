#!/usr/bin/env python3
"""Report whether this interpreter matches the pinned environment.

A pinned requirements file only helps if someone notices when the environment
has drifted from it, and this project has been bitten twice.

The first bite was ordinary drift: a machine carrying numpy 2.x against an
older pinned sklearn, which leaves `sklearn.metrics` unimportable because it is
a C extension built for the other ABI. Every training script then dies at its
first metric call, while all three verification gates keep passing, because
none of them imports sklearn. That is the worst shape a drift can take: the
checks stay green and the pipeline is quietly broken.

The second bite was subtler and is the reason the pins now read the way they
do. requirements.txt described a stack that had never run: the GPU node took
the image's own CUDA wheels and froze whatever resolved into
runs_gpu/gpu_environment.txt, so the recorded environment, not the pin file,
was the provenance of every published number. The pins are now read off that
record, which means a mismatch reported here is a real difference from the
environment the results came from.

Versions are compared on their public part, so a CUDA wheel reporting
2.8.0+cu128 matches a pin of 2.8.0: the local suffix names the build, not the
release.

Exit status is 0 when every pin matches, 1 otherwise, so CI can use it.

Usage
-----
    python3 scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import name differs from the distribution name for a few of these.
IMPORT_NAME = {"scikit-learn": "sklearn", "pillow": "PIL", "pyyaml": "yaml",
               "opencv-python": "cv2", "torch-vision": "torchvision"}


def public(version: str) -> str:
    """Drop a PEP 440 local-version suffix: 2.8.0+cu128 -> 2.8.0.

    The node installed the CUDA 12.8 wheels, whose versions carry the build in
    a local suffix. Treating that as a drift from the upstream pin would report
    a mismatch on the one machine whose environment is definitionally correct.
    """
    return version.split("+", 1)[0]


def pinned(path: Path) -> dict[str, str]:
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def main() -> int:
    wanted = pinned(ROOT / "requirements.txt")
    if not wanted:
        print("no requirements.txt with == pins found")
        return 1

    print(f"python {sys.version.split()[0]}  ({sys.executable})\n")
    print(f"  {'package':<18}{'pinned':<14}{'installed':<14}status")
    drifted, missing, broken = [], [], []
    for dist, want in sorted(wanted.items()):
        mod = IMPORT_NAME.get(dist, dist.replace("-", "_"))
        try:
            got = getattr(importlib.import_module(mod), "__version__", "?")
        except ImportError:
            missing.append(dist)
            print(f"  {dist:<18}{want:<14}{'-':<14}not installed")
            continue
        except Exception as exc:                      # a broken binary build
            broken.append((dist, type(exc).__name__))
            print(f"  {dist:<18}{want:<14}{'-':<14}IMPORT FAILS ({type(exc).__name__})")
            continue
        if public(got) != public(want):
            drifted.append((dist, want, got))
            print(f"  {dist:<18}{want:<14}{got:<14}DRIFTED")
        else:
            print(f"  {dist:<18}{want:<14}{got:<14}ok")

    print()
    if broken:
        print("Packages that are installed but fail to import. This usually means a")
        print("binary built against a different numpy. Training and probe scripts")
        print("cannot run until it is resolved; the verification gates do not import")
        print("these and will keep passing regardless, which is why this check exists.")
        for dist, exc in broken:
            print(f"  - {dist}: {exc}")
    if drifted:
        print("Drifted pins. Results produced under these versions are not the")
        print("versions the released numbers were produced under.")
        for dist, want, got in drifted:
            print(f"  - {dist}: pinned {want}, installed {got}")
    if missing:
        print(f"Not installed: {', '.join(missing)}")
    if not (broken or drifted or missing):
        print("Environment matches requirements.txt.")
        return 0
    print("\n  pip install -r requirements.txt   # restores the pinned set")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
